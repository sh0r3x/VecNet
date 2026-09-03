using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswIndexBoundedPersistenceTestsIndependent
{
    [Fact]
    public void SaveOpen_AcceptedSparseUpperLayoutRoundTripsSearchAllowlistAndGraphShape()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex();
        SearchResult[] expectedUnfiltered = Search(source, [1.75f, 0.5f, 0.25f], topK: 6);
        SearchResult[] expectedAllowed = Search(
            source,
            [1.75f, 0.5f, 0.25f],
            [105UL, 101UL, 105UL, 999UL, 100UL, 108UL],
            topK: 4);
        string expectedGraph = GraphSnapshot(source);

        source.Save(saved.Path);

        using JsonDocument manifest = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(saved.Path, HnswIndexStorage.ManifestFileName)));
        Assert.False(manifest.RootElement.TryGetProperty("createdByTask", out _));
        Assert.False(manifest.RootElement.TryGetProperty("evidence", out _));
        Assert.Equal(
            "dense-layer0-sparse-upper-v1",
            manifest.RootElement.GetProperty("hnsw").GetProperty("graph").GetProperty("adjacencyLayout").GetString());
        Assert.Equal(
            "SplitMix64",
            manifest.RootElement.GetProperty("hnsw").GetProperty("graph").GetProperty("levelGenerator").GetString());
        AssertSparseUpperGraphDirectory(saved.Path, source.Count);

        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(source.InternalIds.ToArray(), opened.InternalIds.ToArray());
        Assert.Equal(expectedGraph, GraphSnapshot(opened));
        Assert.Equal(expectedUnfiltered, Search(opened, [1.75f, 0.5f, 0.25f], topK: 6));
        Assert.Equal(
            expectedAllowed,
            Search(opened, [1.75f, 0.5f, 0.25f], [105UL, 101UL, 105UL, 999UL, 100UL, 108UL], topK: 4));
    }

    [Fact]
    public void OpenReadOnly_RejectsRehashedSparseUpperGraphRowOrdinalCorruption()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex();
        source.Save(saved.Path);

        string graphPath = Path.Combine(saved.Path, HnswIndexStorage.GraphFileName);
        byte[] graph = File.ReadAllBytes(graphPath);
        int layerOneEntryOffset = HnswIndexStorage.GraphHeaderLength + HnswIndexStorage.GraphLayerDirectoryEntryLength;
        int layerOneStoredRows = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(layerOneEntryOffset + 8)));
        int layerOneOrdinalsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graph.AsSpan(layerOneEntryOffset + 16)));
        Assert.True(layerOneStoredRows > 0);

        BinaryPrimitives.WriteInt32LittleEndian(graph.AsSpan(layerOneOrdinalsOffset), 1);
        File.WriteAllBytes(graphPath, graph);
        RepairManifestFileMetadata(saved.Path, "graph", HnswIndexStorage.GraphFileName);

        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(saved.Path));
    }

    [Fact]
    public void ValidateSavedIndex_RejectsFiniteVectorPayloadDriftEvenWhenRehashedOutputOpens()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex();
        source.Save(saved.Path);

        string vectorsPath = Path.Combine(saved.Path, HnswIndexStorage.VectorsFileName);
        byte[] vectors = File.ReadAllBytes(vectorsPath);
        BinaryPrimitives.WriteInt32LittleEndian(
            vectors.AsSpan(HnswIndexStorage.VectorsHeaderLength),
            BitConverter.SingleToInt32Bits(123.5f));
        File.WriteAllBytes(vectorsPath, vectors);
        RepairManifestFileMetadata(saved.Path, "vectors", HnswIndexStorage.VectorsFileName);

        HnswIndex openedWithDrift = HnswIndex.OpenReadOnly(saved.Path);
        Assert.Equal(source.Count, openedWithDrift.Count);
        Assert.NotEqual(source.InternalVectors.ToArray(), openedWithDrift.InternalVectors.ToArray());
        Assert.Throws<InvalidDataException>(() => HnswIndexStorage.ValidateSavedIndex(saved.Path, source));
    }

    [Fact]
    public void PublicMutableCheckpoint_FoldsMixedTombstonesPreservesReservationsAndOpenedAllowlistParity()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory noChanges = TempIndexDirectory.CreateMissing();
        var mutable = new HnswMutableIndex(CreatePublicBaseIndex());

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(12, [0.25f, 0f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(45, [4.5f, 0f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(45).Status);
        SearchResult[] preCheckpoint = Search(
            mutable,
            [0f, 0f, 0f],
            [45UL, 20UL, 12UL, 10UL, 30UL, 999UL, 12UL, 40UL],
            topK: 4);

        HnswMutableCheckpointResult published = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, published.Status);
        Assert.Equal(1, published.FoldedDeltaVectorCount);
        Assert.Equal(1, published.FoldedBaseTombstoneCount);
        Assert.Equal(1, published.FoldedDeltaTombstoneCount);
        Assert.Equal(2, published.DeletedReservedIdCount);
        Assert.Equal(0, published.TombstoneCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [9f, 9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(45, [9f, 9f, 9f]).Status);
        Assert.DoesNotContain(
            Directory.EnumerateFileSystemEntries(checkpoint.Path),
            path => Path.GetFileName(path).Contains(".tmp-", StringComparison.Ordinal));

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        SearchResult[] postCheckpoint = Search(
            mutable,
            [0f, 0f, 0f],
            [45UL, 20UL, 12UL, 10UL, 30UL, 999UL, 12UL, 40UL],
            topK: 4);
        SearchResult[] openedResults = Search(
            opened,
            [0f, 0f, 0f],
            [45UL, 20UL, 12UL, 10UL, 30UL, 999UL, 12UL, 40UL],
            topK: 4);

        Assert.Equal(preCheckpoint, postCheckpoint);
        Assert.Equal(postCheckpoint, openedResults);
        Assert.DoesNotContain(openedResults, static result => result.Id is 20UL or 45UL or 999UL);

        var reusableWorkspace = new HnswMutableSearchWorkspace(mutable, maxResults: 4);
        HnswMutableCheckpointResult unchanged = mutable.Checkpoint(noChanges.Path);

        Assert.Equal(HnswMutableCheckpointStatus.NoChanges, unchanged.Status);
        Assert.False(Directory.Exists(noChanges.Path));
        var destination = new SearchResult[4];
        Assert.Equal(4, mutable.Search([0f, 0f, 0f], destination, reusableWorkspace));
    }

    private static HnswIndex CreateLayeredIndex()
    {
        int[] levels = [2, 0, 1, 0, 2, 1, 0, 0, 1, 2];
        int nextLevel = 0;
        var index = new HnswIndex(
            3,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 12, 12, 0x180AUL),
            levels.Length,
            () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(100 + i), [(float)i, i % 4, (i % 3) * 0.5f]);
        }

        return index;
    }

    private static HnswIndex CreatePublicBaseIndex()
    {
        var index = new HnswIndex(
            3,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 12, 12, 0x180BUL),
            initialCapacity: 5);
        index.Add(10, [0f, 0f, 0f]);
        index.Add(20, [1f, 0f, 0f]);
        index.Add(30, [2f, 0f, 0f]);
        index.Add(40, [4f, 0f, 0f]);
        index.Add(50, [8f, 0f, 0f]);
        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowedIds, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(
            query,
            allowedIds,
            results,
            new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowedIds, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowedIds, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static string GraphSnapshot(HnswIndex index)
    {
        using var writer = new StringWriter();
        int[] neighbors = new int[index.Options.M * 2];
        writer.Write(index.EntryPoint);
        writer.Write('|');
        writer.Write(index.MaxLayer);
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            writer.Write("|L");
            writer.Write(layer);
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                int count = index.DebugGetNeighbors(layer, ordinal, neighbors);
                writer.Write(':');
                writer.Write(ordinal);
                writer.Write('[');
                for (int i = 0; i < count; i++)
                {
                    if (i > 0)
                    {
                        writer.Write(',');
                    }

                    writer.Write(neighbors[i]);
                }

                writer.Write(']');
            }
        }

        return writer.ToString();
    }

    private static void AssertSparseUpperGraphDirectory(string directoryPath, int vectorCount)
    {
        byte[] graph = File.ReadAllBytes(Path.Combine(directoryPath, HnswIndexStorage.GraphFileName));
        int layerCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(24)));
        Assert.True(layerCount >= 2);

        int layerZeroEntryOffset = HnswIndexStorage.GraphHeaderLength;
        Assert.Equal(0U, BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(layerZeroEntryOffset + 12)));
        Assert.Equal((uint)vectorCount, BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(layerZeroEntryOffset + 8)));

        bool foundSparseUpper = false;
        for (int layer = 1; layer < layerCount; layer++)
        {
            int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
            uint storedRows = BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(entryOffset + 8));
            Assert.Equal(1U, BinaryPrimitives.ReadUInt32LittleEndian(graph.AsSpan(entryOffset + 12)));
            Assert.True(storedRows < vectorCount);
            foundSparseUpper = true;
        }

        Assert.True(foundSparseUpper);
    }

    private static void RepairManifestFileMetadata(string directoryPath, string fileMetadataName, string fileName)
    {
        string filePath = Path.Combine(directoryPath, fileName);
        string manifestPath = Path.Combine(directoryPath, HnswIndexStorage.ManifestFileName);
        JsonObject root = JsonNode.Parse(File.ReadAllText(manifestPath))!.AsObject();
        JsonObject file = root["files"]![fileMetadataName]!.AsObject();
        var info = new FileInfo(filePath);
        file["byteLength"] = info.Length;
        file["sha256"] = ComputeSha256Hex(filePath);
        root.Remove("contentDigest");
        File.WriteAllText(manifestPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
    }

    private static string ComputeSha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

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
            else if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswBoundedPersistenceIndependent-" + Guid.NewGuid().ToString("N"));
    }
}
