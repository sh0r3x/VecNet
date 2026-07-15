using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswIndexCompactUpperLayerIndependentTests
{
    private static readonly int[] LayeredLevels = [0, 3, 0, 2, 1, 2, 0, 1, 0, 0, 1, 0];

    [Fact]
    public void CompactUpperLayersAreObservableWhileLayerZeroStaysDense()
    {
        HnswIndex index = CreateLayeredIndex(initialCapacity: 32, out _);

        Assert.Equal(32, index.Capacity);
        Assert.Equal(32, index.DebugGetLayerCapacity(0));
        Assert.Equal(32, index.DebugGetLayerCountStorageLength(0));
        Assert.Equal(32 * index.Options.M * 2, index.DebugGetLayerNeighborStorageLength(0));

        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        index.Save(saved.Path);

        long actualGraphBytes = new FileInfo(Path.Combine(saved.Path, HnswIndexStorage.GraphFileName)).Length;
        long compactGraphBytes = ExpectedCompactGraphBytes(LayeredLevels, index.Options.M);
        long oldDenseGraphBytes = ExpectedOldDenseGraphBytes(LayeredLevels, index.Options.M);

        Assert.Equal(compactGraphBytes, actualGraphBytes);
        Assert.True(actualGraphBytes < oldDenseGraphBytes);
        for (int layer = 1; layer <= index.MaxLayer; layer++)
        {
            int participants = LayeredLevels.Count(level => level >= layer);
            Assert.Equal(32, index.DebugGetLayerCapacity(layer));
            Assert.Equal(participants, index.DebugGetLayerCompactRowCount(layer));
            Assert.True(index.DebugGetLayerCountStorageLength(layer) < index.DebugGetLayerCapacity(layer));
            Assert.True(index.DebugGetLayerNeighborStorageLength(layer) < index.DebugGetLayerCapacity(layer) * index.Options.M);
        }
    }

    [Fact]
    public void PublicSearchAllowlistCapacityAndOpenParitySurviveCompactUpperLayers()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex(initialCapacity: 32, out ExactFlatIndex exact);
        float[] query = [2.25f, 0.5f, 0.25f];
        ulong[] allowlist = [9_999, 2_003, 2_001, 2_005, 2_003, 2_010, 8_888];

        SearchResult[] expectedSearch = Search(exact, query, topK: 8);
        SearchResult[] expectedFiltered = Search(exact, query, allowlist, topK: 4);
        SearchResult[] sourceSearch = Search(source, query, topK: 8);
        SearchResult[] sourceFiltered = Search(source, query, allowlist, topK: 4);
        string graphBefore = GraphSnapshot(source);

        Assert.Equal(expectedSearch, sourceSearch);
        Assert.Equal(expectedFiltered, sourceFiltered);

        source.EnsureCapacity(40);

        Assert.Equal(40, source.Capacity);
        Assert.Equal(graphBefore, GraphSnapshot(source));
        Assert.Equal(expectedSearch, Search(source, query, topK: 8));
        Assert.Equal(expectedFiltered, Search(source, query, allowlist, topK: 4));

        source.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(source.Count, opened.Count);
        Assert.Equal(source.MaxLayer, opened.MaxLayer);
        Assert.Equal(opened.Count, opened.Capacity);
        Assert.Equal(expectedSearch, Search(opened, query, topK: 8));
        Assert.Equal(expectedFiltered, Search(opened, query, allowlist, topK: 4));
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(41));
    }

    [Fact]
    public void MutableCheckpointRebuildPublishesCompactUpperLayerOutputWithSearchAndAllowlistParity()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateLayeredIndex(initialCapacity: 32, out _);
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(2_100, [0.25f, 0.5f, 0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(2_101, [6.25f, 2.5f, 0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(2_003).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(2_101).Status);

        Row[] liveRows = LayeredRows()
            .Where(static row => row.Id != 2_003)
            .Append(new Row(2_100, [0.25f, 0.5f, 0.25f]))
            .ToArray();
        float[] query = [0.5f, 0.25f, 0.25f];
        ulong[] allowlist = [2_003, 2_101, 2_100, 2_001, 2_005, 9_999, 2_100];
        SearchResult[] expectedSearch = ExactTruth(liveRows, query, topK: 8);
        SearchResult[] expectedFiltered = ExactTruth(liveRows, query, allowlist, topK: 4);

        Assert.Equal(expectedSearch, Search(mutable, query, topK: 8));
        Assert.Equal(expectedFiltered, Search(mutable, query, allowlist, topK: 4));

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(liveRows.Length, result.RebuiltBaseVectorCount);
        Assert.Equal(liveRows.Length, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(2_003, [9f, 9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(2_101, [9f, 9f, 9f]).Status);
        Assert.Equal(expectedSearch, Search(mutable, query, topK: 8));
        Assert.Equal(expectedFiltered, Search(mutable, query, allowlist, topK: 4));
        Assert.Equal(expectedSearch, Search(opened, query, topK: 8));
        Assert.Equal(expectedFiltered, Search(opened, query, allowlist, topK: 4));
        Assert.True(new FileInfo(Path.Combine(checkpoint.Path, HnswIndexStorage.GraphFileName)).Length
            < ExpectedOldDenseGraphBytes(liveRows.Select(static _ => 0).ToArray(), mutable.Options.M) * 3);
    }

    [Fact]
    public void OpenReadOnlyRejectsCompactUpperLayerRowOrdinalCorruptionMatrix()
    {
        foreach ((string name, Action<string> corrupt) in new (string, Action<string>)[]
        {
            ("missing row ordinal", directory => PatchLevels(directory, bytes =>
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(HnswIndexStorage.LevelsHeaderLength), 1))),
            ("duplicate row ordinal", directory => PatchGraph(directory, bytes =>
            {
                int ordinalsOffset = Layer(bytes, 1).OrdinalsOffset;
                int first = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(ordinalsOffset));
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset + sizeof(int)), first);
            })),
            ("out-of-order row ordinal", directory => PatchGraph(directory, bytes =>
            {
                int ordinalsOffset = Layer(bytes, 1).OrdinalsOffset;
                int first = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(ordinalsOffset));
                int second = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(ordinalsOffset + sizeof(int)));
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset), second);
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset + sizeof(int)), first);
            })),
            ("out-of-range row ordinal", directory => PatchGraph(directory, bytes =>
            {
                int ordinalsOffset = Layer(bytes, 1).OrdinalsOffset;
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset), LayeredLevels.Length);
            })),
            ("nonparticipating row ordinal", directory => PatchGraph(directory, bytes =>
            {
                int ordinalsOffset = Layer(bytes, 1).OrdinalsOffset;
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset), 0);
            }))
        })
        {
            using TempIndexDirectory saved = SavedLayeredIndex();
            corrupt(saved.Path);

            InvalidDataException ex = Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(saved.Path));
            Assert.False(string.IsNullOrWhiteSpace(ex.Message), name);
        }
    }

    [Fact]
    public void OpenReadOnlyRejectsCompactUpperLayerNeighborCorruptionMatrix()
    {
        foreach ((string name, Action<byte[]> patch) in new (string, Action<byte[]>)[]
        {
            ("out-of-range neighbor", bytes => PatchUpperLayerNeighbor(bytes, count: 1, [LayeredLevels.Length])),
            ("self neighbor", bytes =>
            {
                int rowOrdinal = FirstCompactOrdinal(bytes, layer: 1);
                PatchUpperLayerNeighbor(bytes, count: 1, [rowOrdinal]);
            }),
            ("duplicate neighbor", bytes => PatchUpperLayerNeighbor(bytes, count: 2, [3, 3])),
            ("nonparticipating neighbor", bytes => PatchUpperLayerNeighbor(bytes, count: 1, [0])),
            ("count greater than stride", bytes =>
            {
                GraphLayer layer = Layer(bytes, 1);
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(layer.CountsOffset), layer.Stride + 1);
            })
        })
        {
            using TempIndexDirectory saved = SavedLayeredIndex();
            PatchGraph(saved.Path, patch);

            InvalidDataException ex = Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(saved.Path));
            Assert.False(string.IsNullOrWhiteSpace(ex.Message), name);
        }
    }

    private static HnswIndex CreateLayeredIndex(int initialCapacity, out ExactFlatIndex exact)
    {
        Row[] rows = LayeredRows();
        int nextLevel = 0;
        var options = new HnswIndexOptions(8, 32, 32, 0x1760_1AUL);
        var hnsw = new HnswIndex(
            3,
            VectorMetric.SquaredEuclidean,
            options,
            initialCapacity,
            () => LayeredLevels[nextLevel++]);
        exact = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean, initialCapacity: rows.Length);

        foreach (Row row in rows)
        {
            hnsw.Add(row.Id, row.Vector);
            exact.Add(row.Id, row.Vector);
        }

        Assert.Equal(3, hnsw.MaxLayer);
        AssertGraphInvariants(hnsw);
        return hnsw;
    }

    private static Row[] LayeredRows() =>
    [
        new(2_000, [0f, 0f, 0f]),
        new(2_001, [1f, 0f, 0f]),
        new(2_002, [2f, 0f, 0f]),
        new(2_003, [3f, 1f, 0f]),
        new(2_004, [4f, 1f, 0f]),
        new(2_005, [5f, 1f, 0f]),
        new(2_006, [6f, 2f, 0f]),
        new(2_007, [7f, 2f, 0f]),
        new(2_008, [8f, 2f, 0f]),
        new(2_009, [9f, 3f, 0f]),
        new(2_010, [10f, 3f, 0f]),
        new(2_011, [11f, 3f, 0f])
    ];

    private static TempIndexDirectory SavedLayeredIndex()
    {
        TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        CreateLayeredIndex(initialCapacity: 32, out _).Save(saved.Path);
        return saved;
    }

    private static void PatchUpperLayerNeighbor(byte[] bytes, int count, int[] neighbors)
    {
        GraphLayer layer = Layer(bytes, 1);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(layer.CountsOffset), count);
        for (int i = 0; i < neighbors.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(layer.NeighborsOffset + i * sizeof(int)), neighbors[i]);
        }
    }

    private static int FirstCompactOrdinal(byte[] graphBytes, int layer)
    {
        int ordinalsOffset = Layer(graphBytes, layer).OrdinalsOffset;
        return BinaryPrimitives.ReadInt32LittleEndian(graphBytes.AsSpan(ordinalsOffset));
    }

    private static GraphLayer Layer(byte[] graphBytes, int layer)
    {
        int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
        return new GraphLayer(
            checked((int)BinaryPrimitives.ReadUInt32LittleEndian(graphBytes.AsSpan(entryOffset + 4))),
            checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 16))),
            checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 24))),
            checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 32))));
    }

    private static void PatchGraph(string directory, Action<byte[]> patch)
    {
        PatchFile(directory, HnswIndexStorage.GraphFileName, patch);
    }

    private static void PatchLevels(string directory, Action<byte[]> patch)
    {
        PatchFile(directory, HnswIndexStorage.LevelsFileName, patch);
    }

    private static void PatchFile(string directory, string fileName, Action<byte[]> patch)
    {
        string path = Path.Combine(directory, fileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);
        RefreshManifestBinaryMetadata(directory, fileName);
    }

    private static void RefreshManifestBinaryMetadata(string directory, string fileName)
    {
        string manifestPath = Path.Combine(directory, HnswIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        JsonObject file = (JsonObject)root["files"]![FilePropertyName(fileName)]!;
        string binaryPath = Path.Combine(directory, file["path"]!.GetValue<string>());
        file["byteLength"] = new FileInfo(binaryPath).Length;
        file["sha256"] = Sha256Hex(binaryPath);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static string FilePropertyName(string fileName) =>
        fileName switch
        {
            HnswIndexStorage.GraphFileName => "graph",
            HnswIndexStorage.LevelsFileName => "levels",
            _ => throw new ArgumentOutOfRangeException(nameof(fileName))
        };

    private static string Sha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, int topK) =>
        rows
            .Select(row => new SearchResult(row.Id, SquaredEuclidean(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, ulong[] allowlist, int topK)
    {
        HashSet<ulong> allowed = allowlist.ToHashSet();
        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, SquaredEuclidean(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float SquaredEuclidean(float[] query, float[] vector)
    {
        float sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            float difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return sum;
    }

    private static string GraphSnapshot(HnswIndex index)
    {
        var parts = new List<string>
        {
            $"count={index.Count};entry={index.EntryPoint};max={index.MaxLayer}"
        };

        for (int ordinal = 0; ordinal < index.Count; ordinal++)
        {
            parts.Add($"level[{ordinal}]={index.DebugGetLevel(ordinal)}");
        }

        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                parts.Add($"l{layer}n{ordinal}={string.Join(",", GetNeighbors(index, layer, ordinal))}");
            }
        }

        return string.Join("|", parts);
    }

    private static void AssertGraphInvariants(HnswIndex index)
    {
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            int degreeLimit = layer == 0 ? index.Options.M * 2 : index.Options.M;
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                int[] neighbors = GetNeighbors(index, layer, ordinal);
                if (index.DebugGetLevel(ordinal) < layer)
                {
                    Assert.Empty(neighbors);
                    continue;
                }

                Assert.InRange(neighbors.Length, 0, degreeLimit);
                Assert.DoesNotContain(ordinal, neighbors);
                Assert.Equal(neighbors.Length, neighbors.Distinct().Count());
                foreach (int neighbor in neighbors)
                {
                    Assert.InRange(neighbor, 0, index.Count - 1);
                    Assert.True(index.DebugGetLevel(neighbor) >= layer);
                }
            }
        }
    }

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }

    private static long ExpectedCompactGraphBytes(int[] levels, int m)
    {
        int layerCount = levels.Max() + 1;
        long bytes = HnswIndexStorage.GraphHeaderLength + ((long)layerCount * HnswIndexStorage.GraphLayerDirectoryEntryLength);
        bytes += (long)levels.Length * sizeof(int);
        bytes += (long)levels.Length * (m * 2) * sizeof(int);

        for (int layer = 1; layer < layerCount; layer++)
        {
            int participants = levels.Count(level => level >= layer);
            bytes += (long)participants * sizeof(int);
            bytes += (long)participants * sizeof(int);
            bytes += (long)participants * m * sizeof(int);
        }

        return bytes;
    }

    private static long ExpectedOldDenseGraphBytes(int[] levels, int m)
    {
        int layerCount = levels.Max() + 1;
        long bytes = HnswIndexStorage.GraphHeaderLength + ((long)layerCount * HnswIndexStorage.GraphLayerDirectoryEntryLength);
        bytes += (long)levels.Length * sizeof(int);
        bytes += (long)levels.Length * (m * 2) * sizeof(int);
        for (int layer = 1; layer < layerCount; layer++)
        {
            bytes += (long)levels.Length * sizeof(int);
            bytes += (long)levels.Length * m * sizeof(int);
        }

        return bytes;
    }

    private sealed record Row(ulong Id, float[] Vector);

    private readonly record struct GraphLayer(int Stride, int OrdinalsOffset, int CountsOffset, int NeighborsOffset);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() => new(CreatePath());

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswIndexCompactUpperLayerIndependentTests-" + Guid.NewGuid().ToString("N"));
    }
}
