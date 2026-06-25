using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswIndexStorageTests
{
    [Fact]
    public void SaveAndOpenReadOnly_ZeroRowIndexRoundTripsAndRejectsMutation()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new HnswIndex(3, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 8, 123));

        index.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(index.Dimension, opened.Dimension);
        Assert.Equal(index.Metric, opened.Metric);
        Assert.Equal(0, opened.Count);
        Assert.Equal(-1, opened.EntryPoint);
        Assert.Equal(-1, opened.MaxLayer);
        Assert.Equal(index.Options, opened.Options);
        Assert.Equal(0, opened.Search([1f, 2f, 3f], new SearchResult[5], new HnswSearchWorkspace(0, 8)));
        Assert.Throws<InvalidOperationException>(() => opened.Add(1, [1f, 2f, 3f]));
    }

    [Fact]
    public void Save_WritesPinnedManifestAndBinaryHeaders()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex index = CreateSmallDeterministicIndex();

        index.Save(temp.Path);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(temp.Path, HnswIndexStorage.ManifestFileName)));
        JsonElement root = document.RootElement;
        Assert.Equal(HnswIndexStorage.ManifestSchemaName, root.GetProperty("schemaName").GetString());
        Assert.Equal("1.0", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("hnsw", root.GetProperty("formatFamily").GetString());
        Assert.Equal("VEC-072", root.GetProperty("createdByTask").GetString());
        Assert.Equal("squared-euclidean", root.GetProperty("index").GetProperty("metric").GetString());
        Assert.Equal("read-only", root.GetProperty("semantics").GetProperty("openedLifecycle").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());

        AssertFileHeader(temp.Path, HnswIndexStorage.IdsFileName, "VNETHI01"u8.ToArray(), HnswIndexStorage.IdsHeaderLength);
        AssertFileHeader(temp.Path, HnswIndexStorage.VectorsFileName, "VNETHV01"u8.ToArray(), HnswIndexStorage.VectorsHeaderLength);
        AssertFileHeader(temp.Path, HnswIndexStorage.LevelsFileName, "VNETHL01"u8.ToArray(), HnswIndexStorage.LevelsHeaderLength);
        AssertFileHeader(temp.Path, HnswIndexStorage.GraphFileName, "VNETHG01"u8.ToArray(), HnswIndexStorage.GraphHeaderLength);

        JsonElement files = root.GetProperty("files");
        Assert.Equal(HnswIndexStorage.IdsFileName, files.GetProperty("ids").GetProperty("path").GetString());
        Assert.Equal(HnswIndexStorage.VectorsFileName, files.GetProperty("vectors").GetProperty("path").GetString());
        Assert.Equal(HnswIndexStorage.LevelsFileName, files.GetProperty("levels").GetProperty("path").GetString());
        Assert.Equal(HnswIndexStorage.GraphFileName, files.GetProperty("graph").GetProperty("path").GetString());
    }

    [Fact]
    public void OpenReadOnly_DeterministicGraphRoundTripsExactly()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex source = CreateSmallDeterministicIndex();
        string expectedGraph = CreateGraphSnapshot(source);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(expectedGraph, CreateGraphSnapshot(opened));
        Assert.Equal(source.Options, opened.Options);
        Assert.Equal(source.EntryPoint, opened.EntryPoint);
        Assert.Equal(source.MaxLayer, opened.MaxLayer);
    }

    [Fact]
    public void OpenReadOnly_SearchParityAndRecallEquivalenceAgainstExactTruth()
    {
        foreach (int dimension in new[] { 32, 128, 386 })
        {
            using TempIndexDirectory temp = TempIndexDirectory.Create();
            const int count = 96;
            const int topK = 10;
            var options = new HnswIndexOptions(12, 80, 80, (ulong)(0x7200 + dimension));
            var hnsw = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);
            var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
            var random = new Random(0x7200 + dimension);

            for (int i = 0; i < count; i++)
            {
                float[] vector = CreateClusteredVector(random, dimension, i % 8);
                ulong id = (ulong)(10_000 + i);
                hnsw.Add(id, vector);
                exact.Add(id, vector);
            }

            hnsw.Save(temp.Path);
            HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

            for (int q = 0; q < 6; q++)
            {
                float[] query = CreateClusteredVector(random, dimension, q % 8);
                SearchResult[] sourceResults = Search(hnsw, query, topK, options.EfSearch);
                SearchResult[] openedResults = Search(opened, query, topK, options.EfSearch);
                SearchResult[] truth = Search(exact, query, topK);

                Assert.Equal(sourceResults, openedResults);
                Assert.Equal(RecallAtK(sourceResults, truth), RecallAtK(openedResults, truth));
                AssertReturnedResultsAreValid(openedResults, exact, query, count);
            }
        }
    }

    [Fact]
    public void OpenReadOnly_AddRejectsReadOnlyBeforeVectorOrDuplicateValidation()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex index = CreateSmallDeterministicIndex();
        index.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Throws<InvalidOperationException>(() => opened.Add(10, [1f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(10, [float.NaN, 0f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(10, [0f, 0f]));
    }

    [Fact]
    public void OpenReadOnly_RejectsMissingMalformedPartialAndIncompatibleAssets()
    {
        Assert.Throws<ArgumentNullException>(() => HnswIndex.OpenReadOnly(null!));
        Assert.Throws<ArgumentException>(() => HnswIndex.OpenReadOnly(""));

        using (TempIndexDirectory missingManifest = TempIndexDirectory.Create())
        {
            Assert.Throws<FileNotFoundException>(() => HnswIndex.OpenReadOnly(missingManifest.Path));
        }

        using (TempIndexDirectory temp = SavedIndex())
        {
            File.WriteAllText(Path.Combine(temp.Path, HnswIndexStorage.ManifestFileName), "{");
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
        }

        AssertManifestMutationRejected(root => root["schemaName"] = "VecNet.OtherManifest");
        AssertManifestMutationRejected(root => root["schemaVersion"] = "2.0");
        AssertManifestMutationRejected(root => root["formatFamily"] = "other");
        AssertManifestMutationRejected(root => root["index"]!["metric"] = "cosine");
        AssertManifestMutationRejected(root => ((JsonArray)root["compatibility"]!["requiredFeatures"]!).Add("future.required"));
        AssertManifestMutationRejected(root => root["compatibility"]!["minimumReaderMajorVersion"] = 2);
        AssertManifestMutationRejected(root => ((JsonArray)root["compatibility"]!["optionalFeatures"]!).Add("future.filtering"));
        AssertManifestMutationRejected(root => ((JsonArray)root["compatibility"]!["optionalFeatures"]!).Add("graph-layout-v2"));
        AssertManifestMutationRejected(root => root["files"]!["ids"]!["path"] = "../hnsw.ids.u64");

        foreach (string fileName in HnswFileNames())
        {
            using TempIndexDirectory missingFile = SavedIndex();
            File.Delete(Path.Combine(missingFile.Path, fileName));
            Assert.Throws<FileNotFoundException>(() => HnswIndex.OpenReadOnly(missingFile.Path));

            using TempIndexDirectory lengthMismatch = SavedIndex();
            MutateManifestFile(lengthMismatch.Path, FilePropertyName(fileName), file =>
                file["byteLength"] = file["byteLength"]!.GetValue<long>() + 1);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(lengthMismatch.Path));

            using TempIndexDirectory checksumMismatch = SavedIndex();
            PatchFile(checksumMismatch.Path, fileName, bytes => bytes[^1] ^= 0x5A, refreshManifest: false);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(checksumMismatch.Path));

            using TempIndexDirectory badMagic = SavedIndex();
            PatchFile(badMagic.Path, fileName, bytes => bytes[0] = (byte)'X', refreshManifest: true);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(badMagic.Path));

            using TempIndexDirectory badVersion = SavedIndex();
            PatchFile(badVersion.Path, fileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 2), refreshManifest: true);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(badVersion.Path));

            using TempIndexDirectory truncated = SavedIndex();
            TruncateFileAndRefresh(truncated.Path, fileName, HeaderLength(fileName) - 1);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(truncated.Path));
        }

        using (TempIndexDirectory tempOnly = TempIndexDirectory.Create())
        {
            File.WriteAllText(Path.Combine(tempOnly.Path, HnswIndexStorage.IdsFileName + ".tmp-abc"), "temp");
            Assert.Throws<FileNotFoundException>(() => HnswIndex.OpenReadOnly(tempOnly.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsTruncatedPayloadsAndMismatchedBinaryMetadata()
    {
        foreach (string fileName in HnswFileNames())
        {
            using TempIndexDirectory truncatedPayload = SavedIndex();
            string path = Path.Combine(truncatedPayload.Path, fileName);
            TruncateFileAndRefresh(truncatedPayload.Path, fileName, new FileInfo(path).Length - 1);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(truncatedPayload.Path));
        }

        AssertBinaryPatchRejected(HnswIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(16), 999));
        AssertBinaryPatchRejected(HnswIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(24), 1));
        AssertBinaryPatchRejected(HnswIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), 999));
        AssertBinaryPatchRejected(HnswIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28), 999));
        AssertBinaryPatchRejected(HnswIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), 999));
        AssertBinaryPatchRejected(HnswIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), 1));
        AssertBinaryPatchRejected(HnswIndexStorage.LevelsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(24), 1));
        AssertBinaryPatchRejected(HnswIndexStorage.GraphFileName, bytes => BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), -1));
        AssertBinaryPatchRejected(HnswIndexStorage.GraphFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(56), 1));

        AssertBinaryPatchRejected(HnswIndexStorage.IdsFileName, bytes =>
        {
            ulong id = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(HnswIndexStorage.IdsHeaderLength));
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(HnswIndexStorage.IdsHeaderLength + sizeof(ulong)), id);
        });
        AssertBinaryPatchRejected(HnswIndexStorage.VectorsFileName, bytes =>
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(HnswIndexStorage.VectorsHeaderLength),
                BitConverter.SingleToInt32Bits(float.PositiveInfinity)));
    }

    [Fact]
    public void OpenReadOnly_RejectsInvalidGraphAssets()
    {
        AssertGraphPatchRejected(bytes =>
        {
            (int stride, int countsOffset, int neighborsOffset) = Layer(bytes, 0);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset), 999);
        });
        AssertGraphPatchRejected(bytes =>
        {
            (int stride, int countsOffset, int neighborsOffset) = Layer(bytes, 0);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset), 0);
        });
        AssertGraphPatchRejected(bytes =>
        {
            (int stride, int countsOffset, int neighborsOffset) = Layer(bytes, 0);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), 2);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset + sizeof(int)), 1);
        });
        AssertGraphPatchRejected(bytes =>
        {
            (int stride, int countsOffset, _) = Layer(bytes, 0);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), stride + 1);
        });
        AssertGraphPatchRejected(bytes =>
        {
            (_, int countsOffset, int neighborsOffset) = Layer(bytes, 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset), 1);
        });

        using TempIndexDirectory badEntryLevel = SavedIndex();
        PatchFile(badEntryLevel.Path, HnswIndexStorage.LevelsFileName, bytes =>
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(HnswIndexStorage.LevelsHeaderLength), 0), refreshManifest: true);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(badEntryLevel.Path));

        AssertGraphPatchRejected(bytes =>
        {
            (_, int countsOffset, _) = Layer(bytes, 0);
            long originalOffset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(64 + 8));
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(64 + 8), checked((ulong)(originalOffset + sizeof(int))));
        });
    }

    [Fact]
    public void OpenReadOnly_SupportsConcurrentSearchWithIndependentWorkspaces()
    {
        using TempIndexDirectory temp = SavedIndex();
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        float[][] queries =
        [
            [0f, 0f],
            [2f, 0f],
            [4f, 1f]
        ];
        SearchResult[][] expected = queries.Select(query => Search(opened, query, 3, opened.Options.EfSearch)).ToArray();

        Parallel.For(0, 300, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            SearchResult[] actual = Search(opened, queries[queryIndex], 3, opened.Options.EfSearch);
            Assert.Equal(expected[queryIndex], actual);
        });
    }

    private static HnswIndex CreateSmallDeterministicIndex()
    {
        int[] levels = [1, 0, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x7201),
            () => levels[nextLevel++]);
        index.Add(10, [0f, 0f]);
        index.Add(20, [1f, 0f]);
        index.Add(30, [2f, 0f]);
        index.Add(40, [0f, 2f]);
        index.Add(50, [4f, 1f]);
        return index;
    }

    private static TempIndexDirectory SavedIndex()
    {
        TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateSmallDeterministicIndex().Save(temp.Path);
        return temp;
    }

    private static void AssertManifestMutationRejected(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = SavedIndex();
        MutateManifest(temp.Path, mutate);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertBinaryPatchRejected(string fileName, Action<byte[]> patch)
    {
        using TempIndexDirectory temp = SavedIndex();
        PatchFile(temp.Path, fileName, patch, refreshManifest: true);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertGraphPatchRejected(Action<byte[]> patch) =>
        AssertBinaryPatchRejected(HnswIndexStorage.GraphFileName, patch);

    private static void PatchFile(string directory, string fileName, Action<byte[]> patch, bool refreshManifest)
    {
        string path = Path.Combine(directory, fileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);
        if (refreshManifest)
        {
            RefreshManifestBinaryMetadata(directory, fileName);
        }
    }

    private static void TruncateFileAndRefresh(string directory, string fileName, long length)
    {
        string path = Path.Combine(directory, fileName);
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.None))
        {
            stream.SetLength(length);
        }

        RefreshManifestBinaryMetadata(directory, fileName);
    }

    private static void RefreshManifestBinaryMetadata(string directory, string fileName)
    {
        MutateManifestFile(directory, FilePropertyName(fileName), file =>
        {
            string relativePath = file["path"]!.GetValue<string>();
            string binaryPath = Path.Combine(directory, relativePath);
            file["byteLength"] = new FileInfo(binaryPath).Length;
            file["sha256"] = Sha256Hex(binaryPath);
        });
    }

    private static void MutateManifestFile(string directory, string filePropertyName, Action<JsonObject> mutate) =>
        MutateManifest(directory, root => mutate((JsonObject)root["files"]![filePropertyName]!));

    private static void MutateManifest(string directory, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, HnswIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static (int Stride, int CountsOffset, int NeighborsOffset) Layer(byte[] graphBytes, int layer)
    {
        int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
        int stride = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(graphBytes.AsSpan(entryOffset + 4)));
        int countsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 8)));
        int neighborsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 16)));
        return (stride, countsOffset, neighborsOffset);
    }

    private static void AssertFileHeader(string directory, string fileName, byte[] magic, int headerLength)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, fileName));
        Assert.Equal(magic, bytes[..8]);
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8)));
        Assert.Equal(0, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(10)));
        Assert.Equal((uint)headerLength, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(12)));
    }

    private static string[] HnswFileNames() =>
    [
        HnswIndexStorage.IdsFileName,
        HnswIndexStorage.VectorsFileName,
        HnswIndexStorage.LevelsFileName,
        HnswIndexStorage.GraphFileName
    ];

    private static string FilePropertyName(string fileName) =>
        fileName switch
        {
            HnswIndexStorage.IdsFileName => "ids",
            HnswIndexStorage.VectorsFileName => "vectors",
            HnswIndexStorage.LevelsFileName => "levels",
            HnswIndexStorage.GraphFileName => "graph",
            _ => throw new ArgumentOutOfRangeException(nameof(fileName))
        };

    private static int HeaderLength(string fileName) =>
        fileName switch
        {
            HnswIndexStorage.IdsFileName => HnswIndexStorage.IdsHeaderLength,
            HnswIndexStorage.VectorsFileName => HnswIndexStorage.VectorsHeaderLength,
            HnswIndexStorage.LevelsFileName => HnswIndexStorage.LevelsHeaderLength,
            HnswIndexStorage.GraphFileName => HnswIndexStorage.GraphHeaderLength,
            _ => throw new ArgumentOutOfRangeException(nameof(fileName))
        };

    private static string Sha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, efSearch));
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static double RecallAtK(SearchResult[] actual, SearchResult[] expected)
    {
        HashSet<ulong> truth = expected.Select(static result => result.Id).ToHashSet();
        return actual.Count(result => truth.Contains(result.Id)) / (double)expected.Length;
    }

    private static void AssertReturnedResultsAreValid(SearchResult[] actual, ExactFlatIndex exact, float[] query, int count)
    {
        var allExact = new SearchResult[count];
        int written = exact.Search(query, allExact);
        Dictionary<ulong, float> distanceById = allExact[..written].ToDictionary(static result => result.Id, static result => result.Distance);

        Assert.Equal(actual.Length, actual.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in actual)
        {
            Assert.True(float.IsFinite(result.Distance));
            Assert.True(distanceById.TryGetValue(result.Id, out float expectedDistance));
            Assert.Equal(expectedDistance, result.Distance);
        }
    }

    private static float[] CreateClusteredVector(Random random, int dimension, int cluster)
    {
        var vector = new float[dimension];
        float center = cluster * 8f;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = center + (i % 7 - 3) * 0.5f + ((random.NextSingle() - 0.5f) * 0.25f);
        }

        return vector;
    }

    private static string CreateGraphSnapshot(HnswIndex index)
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

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
