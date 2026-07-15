using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswIndexCompactUpperLayerStorageTests
{
    [Fact]
    public void PlannedBuild_KeepsLayerZeroDenseAndUpperLayersCompact()
    {
        int[] levels = [0, 3, 0, 1, 0, 2, 0, 1, 0, 0];
        HnswIndex index = CreateIndex(levels, initialCapacity: 20);

        Assert.Equal(20, index.Capacity);
        Assert.Equal(20, index.DebugGetLayerCapacity(0));
        Assert.Equal(20, index.DebugGetLayerCountStorageLength(0));
        Assert.Equal(20 * index.Options.M * 2, index.DebugGetLayerNeighborStorageLength(0));

        for (int layer = 1; layer <= index.MaxLayer; layer++)
        {
            int participants = levels.Count(level => level >= layer);

            Assert.Equal(20, index.DebugGetLayerCapacity(layer));
            Assert.Equal(participants, index.DebugGetLayerCompactRowCount(layer));
            Assert.True(index.DebugGetLayerCountStorageLength(layer) < index.DebugGetLayerCapacity(layer));
            Assert.True(index.DebugGetLayerNeighborStorageLength(layer) < index.DebugGetLayerCapacity(layer) * index.Options.M);
        }
    }

    [Fact]
    public void Save_WritesCompactUpperLayerGraphPayloadAndOpenPreservesSearch()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        int[] levels = [0, 3, 0, 1, 0, 2, 0, 1, 0, 0];
        HnswIndex source = CreateIndex(levels, initialCapacity: 20);
        SearchResult[] expected = Search(source, [2f, 0.5f], topK: 6);

        source.Save(temp.Path);

        long expectedGraphBytes = ExpectedGraphBytes(levels, source.Options.M);
        Assert.Equal(expectedGraphBytes, new FileInfo(Path.Combine(temp.Path, HnswIndexStorage.GraphFileName)).Length);

        JsonNode manifest = JsonNode.Parse(File.ReadAllText(Path.Combine(temp.Path, HnswIndexStorage.ManifestFileName)))!;
        Assert.Equal(
            "dense-layer0-sparse-upper-v1",
            manifest["hnsw"]!["graph"]!["adjacencyLayout"]!.GetValue<string>());

        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(source.Count, opened.Count);
        Assert.Equal(source.MaxLayer, opened.MaxLayer);
        Assert.Equal(expected, Search(opened, [2f, 0.5f], topK: 6));
    }

    [Fact]
    public void OpenReadOnly_RejectsCompactUpperLayerRowOrdinalCorruption()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        CreateIndex([0, 2, 0, 1, 0, 0], initialCapacity: 8).Save(temp.Path);

        PatchGraph(temp.Path, bytes =>
        {
            int ordinalsOffset = CompactOrdinalsOffset(bytes, layer: 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset), 0);
        });

        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static HnswIndex CreateIndex(int[] levels, int initialCapacity)
    {
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1760UL),
            initialCapacity,
            () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(100 + i), [i, i % 3]);
        }

        return index;
    }

    private static long ExpectedGraphBytes(int[] levels, int m)
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

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static void PatchGraph(string directory, Action<byte[]> patch)
    {
        string path = Path.Combine(directory, HnswIndexStorage.GraphFileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);
        RefreshManifestBinaryMetadata(directory);
    }

    private static int CompactOrdinalsOffset(byte[] graphBytes, int layer)
    {
        int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
        return checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 16)));
    }

    private static void RefreshManifestBinaryMetadata(string directory)
    {
        string manifestPath = Path.Combine(directory, HnswIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        JsonObject graphFile = (JsonObject)root["files"]!["graph"]!;
        string graphPath = Path.Combine(directory, graphFile["path"]!.GetValue<string>());
        graphFile["byteLength"] = new FileInfo(graphPath).Length;
        graphFile["sha256"] = Sha256Hex(graphPath);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static string Sha256Hex(string path)
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
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswIndexCompactUpperLayerStorageTests-" + Guid.NewGuid().ToString("N"));
    }
}
