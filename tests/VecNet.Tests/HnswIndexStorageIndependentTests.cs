using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswIndexStorageIndependentTests
{
    [Fact]
    public void OpenReadOnly_FailsClosedForAnyCompatibilityFeatureAdvertisement()
    {
        AssertManifestMutationRejected(root =>
            ((JsonArray)root["compatibility"]!["optionalFeatures"]!).Add("opaque-hnsw-sidecar-metadata"));

        AssertManifestMutationRejected(root =>
            ((JsonArray)root["compatibility"]!["optionalFeatures"]!).Add("ann-layout-hint"));

        AssertManifestMutationRejected(root =>
            ((JsonArray)root["compatibility"]!["requiredFeatures"]!).Add("future-required-graph-reader"));

        AssertManifestMutationRejected(root =>
            ((JsonArray)root["compatibility"]!["requiredFeatures"]!).Add(17));
    }

    [Fact]
    public void OpenReadOnly_PreservesSavedSearchAndRecallOnIndependentDeterministicGraph()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex source = CreateIndependentIndex(out ExactFlatIndex exact);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(source.Dimension, opened.Dimension);
        Assert.Equal(source.Metric, opened.Metric);
        Assert.Equal(source.Count, opened.Count);
        Assert.Equal(source.Options, opened.Options);
        Assert.Equal(source.EntryPoint, opened.EntryPoint);
        Assert.Equal(source.MaxLayer, opened.MaxLayer);
        Assert.Equal(GraphSnapshot(source), GraphSnapshot(opened));

        float[][] queries =
        [
            CreateIndependentVector(9, 0, 0x7202_1001),
            CreateIndependentVector(9, 3, 0x7202_1002),
            CreateIndependentVector(9, 7, 0x7202_1003),
            [1.25f, -2.5f, 3.75f, -4.5f, 5.25f, -6f, 6.75f, -7.5f, 8.25f]
        ];

        foreach (int topK in new[] { 1, 7, 19 })
        {
            foreach (float[] query in queries)
            {
                SearchResult[] sourceResults = Search(source, query, topK);
                SearchResult[] openedResults = Search(opened, query, topK);
                SearchResult[] truth = Search(exact, query, topK);

                Assert.Equal(sourceResults.Length, openedResults.Length);
                Assert.Equal(sourceResults, openedResults);
                Assert.Equal(RecallAtK(sourceResults, truth), RecallAtK(openedResults, truth));
                AssertDistancesMatchExact(openedResults, exact, query, source.Count);
            }
        }
    }

    [Fact]
    public void OpenReadOnly_AddRejectsReadOnlyBeforeDuplicateIdAndVectorValidation()
    {
        using TempIndexDirectory temp = SavedIndependentIndex();
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Throws<InvalidOperationException>(() => opened.Add(7_001, [float.NaN]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(7_001, [0f, 1f, 2f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(99_001, [0f, 1f, 2f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(99_001, [float.PositiveInfinity]));
    }

    [Fact]
    public void OpenReadOnly_RejectsManifestIncompatibleSemanticsAndTempNamedFileReferences()
    {
        AssertManifestMutationRejected(root => root["semantics"]!["mutationPolicy"] = "append-delta");
        AssertManifestMutationRejected(root => root["semantics"]!["workspacePolicy"] = "shared-workspace");
        AssertManifestMutationRejected(root => root["hnsw"]!["graph"]!["adjacencyLayout"] = "compressed-postings");
        AssertManifestMutationRejected(root => root["evidence"]!["publicClaimEligible"] = true);

        using TempIndexDirectory temp = SavedIndependentIndex();
        string tempIds = HnswIndexStorage.IdsFileName + ".tmp-independent";
        File.Copy(
            Path.Combine(temp.Path, HnswIndexStorage.IdsFileName),
            Path.Combine(temp.Path, tempIds));
        MutateManifest(temp.Path, root => root["files"]!["ids"]!["path"] = tempIds);

        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsGraphDirectoryAndLayerReferenceCorruption()
    {
        AssertGraphPatchRejected(bytes =>
        {
            int layerOneEntry = HnswIndexStorage.GraphHeaderLength + HnswIndexStorage.GraphLayerDirectoryEntryLength;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(layerOneEntry), 3);
        });

        AssertGraphPatchRejected(bytes =>
        {
            int layerOneEntry = HnswIndexStorage.GraphHeaderLength + HnswIndexStorage.GraphLayerDirectoryEntryLength;
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(layerOneEntry + 40), 1);
        });

        AssertGraphPatchRejected(bytes =>
        {
            int ordinalsOffset = CompactOrdinalsOffset(bytes, 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ordinalsOffset), 2);
        });

        AssertGraphPatchRejected(bytes =>
        {
            (int stride, int countsOffset, int neighborsOffset) = Layer(bytes, 0);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset + 2 * sizeof(int)), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset + 2 * stride * sizeof(int)), 2);
        });
    }

    [Fact]
    public void OpenReadOnly_ParallelSearchUsesOnlyIndependentBuffersAndWorkspaces()
    {
        using TempIndexDirectory temp = SavedIndependentIndex();
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        float[][] queries = Enumerable.Range(0, 12)
            .Select(i => CreateIndependentVector(9, i % 8, 0x7202_3000 + i))
            .ToArray();
        SearchResult[][] expected = queries
            .Select(query => Search(opened, query, 11))
            .ToArray();

        Parallel.For(0, 240, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            float[] query = queries[queryIndex].ToArray();
            var results = new SearchResult[11];
            var workspace = new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch);

            int written = opened.Search(query, results, workspace);

            Assert.Equal(expected[queryIndex].Length, written);
            Assert.Equal(expected[queryIndex], results[..written]);
        });
    }

    private static HnswIndex CreateIndependentIndex(out ExactFlatIndex exact)
    {
        int[] levels =
        [
            2, 1, 0, 1, 0, 2, 0, 1,
            0, 0, 1, 0, 2, 0, 1, 0,
            0, 1, 0, 0, 1, 0, 0, 0
        ];
        int nextLevel = 0;
        var options = new HnswIndexOptions(4, 24, 32, 0x7202_0001UL);
        var hnsw = new HnswIndex(
            9,
            VectorMetric.SquaredEuclidean,
            options,
            () => levels[nextLevel++]);
        exact = new ExactFlatIndex(9, VectorMetric.SquaredEuclidean);

        for (int i = 0; i < levels.Length; i++)
        {
            float[] vector = i switch
            {
                6 => CreateIndependentVector(9, 2, 0x7202_0006),
                18 => CreateIndependentVector(9, 2, 0x7202_0006),
                _ => CreateIndependentVector(9, i % 8, 0x7202_0000 + i)
            };
            ulong id = (ulong)(7_001 + i * 29);
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        return hnsw;
    }

    private static TempIndexDirectory SavedIndependentIndex()
    {
        TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateIndependentIndex(out _).Save(temp.Path);
        return temp;
    }

    private static void AssertManifestMutationRejected(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = SavedIndependentIndex();
        MutateManifest(temp.Path, mutate);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertGraphPatchRejected(Action<byte[]> patch)
    {
        using TempIndexDirectory temp = SavedIndependentIndex();
        PatchFile(temp.Path, HnswIndexStorage.GraphFileName, patch);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
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
        MutateManifest(directory, root =>
        {
            JsonObject file = (JsonObject)root["files"]![FilePropertyName(fileName)]!;
            string relativePath = file["path"]!.GetValue<string>();
            string binaryPath = Path.Combine(directory, relativePath);
            file["byteLength"] = new FileInfo(binaryPath).Length;
            file["sha256"] = Sha256Hex(binaryPath);
        });
    }

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
        int countsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 24)));
        int neighborsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 32)));
        return (stride, countsOffset, neighborsOffset);
    }

    private static int CompactOrdinalsOffset(byte[] graphBytes, int layer)
    {
        int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
        return checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 16)));
    }

    private static string FilePropertyName(string fileName) =>
        fileName switch
        {
            HnswIndexStorage.IdsFileName => "ids",
            HnswIndexStorage.VectorsFileName => "vectors",
            HnswIndexStorage.LevelsFileName => "levels",
            HnswIndexStorage.GraphFileName => "graph",
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

    private static void AssertDistancesMatchExact(SearchResult[] actual, ExactFlatIndex exact, float[] query, int count)
    {
        var allExact = new SearchResult[count];
        int written = exact.Search(query, allExact);
        Dictionary<ulong, float> distanceById = allExact[..written]
            .ToDictionary(static result => result.Id, static result => result.Distance);

        Assert.Equal(actual.Length, actual.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in actual)
        {
            Assert.True(float.IsFinite(result.Distance));
            Assert.Equal(distanceById[result.Id], result.Distance);
        }
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

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }

    private static float[] CreateIndependentVector(int dimension, int cluster, int seed)
    {
        var random = new Random(seed);
        var vector = new float[dimension];
        float center = (cluster - 3) * 6.25f;
        for (int i = 0; i < vector.Length; i++)
        {
            float lane = ((i * 3) % 11) - 5f;
            vector[i] = center + lane * 0.375f + ((random.NextSingle() - 0.5f) * 0.2f);
        }

        return vector;
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
