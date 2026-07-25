using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswCosineMetricIndependentTests
{
    [Fact]
    public void CosineExactParity_CoversRawNonUnitVectorsAndAwkwardDimensions()
    {
        foreach (int dimension in new[] { 5, 9, 17 })
        {
            (ulong Id, float[] Vector)[] rows = CreateCosineRows(dimension);
            var options = new HnswIndexOptions(16, 64, 64, (ulong)(0x2310_0000 + dimension));
            HnswIndex hnsw = CreateHnsw(rows, options);
            ExactFlatIndex exact = CreateExact(rows);

            float[][] queries =
            [
                CreateRawVector(dimension, scale: 3.5f, phase: 1),
                CreateRawVector(dimension, scale: 0.25f, phase: 4),
                CreateRawVector(dimension, scale: -2.0f, phase: 7)
            ];

            foreach (float[] query in queries)
            {
                AssertResultsEqual(Search(exact, query, rows.Length), Search(hnsw, query, rows.Length));
            }
        }
    }

    [Fact]
    public void CosineOrdering_UsesCanonicalDistanceThenAscendingExternalIdTies()
    {
        (ulong Id, float[] Vector)[] rows =
        [
            (90, [7f, 0f, 0f, 0f, 0f]),
            (10, [1f, 0f, 0f, 0f, 0f]),
            (50, [100f, 0f, 0f, 0f, 0f]),
            (30, [3f, 0f, 0f, 0f, 0f]),
            (20, [0f, 2f, 0f, 0f, 0f]),
            (40, [-5f, 0f, 0f, 0f, 0f])
        ];
        HnswIndex hnsw = CreateHnsw(rows, new HnswIndexOptions(8, 32, 32, 0x2310_0100UL));

        SearchResult[] results = Search(hnsw, [4f, 0f, 0f, 0f, 0f], topK: rows.Length);

        Assert.Equal([10UL, 30UL, 50UL, 90UL, 20UL, 40UL], results.Select(static result => result.Id).ToArray());
        Assert.Equal(0f, results[0].Distance, precision: 6);
        Assert.Equal(1f, results[4].Distance, precision: 6);
        Assert.Equal(2f, results[5].Distance, precision: 6);
    }

    [Fact]
    public void CosineAllowlist_ExactFallbackMatchesTruthAndBroadEmissionSuppressesNonAllowedCandidates()
    {
        (ulong Id, float[] Vector)[] rows =
        [
            (10, [10f, 0f, 0f]),
            (20, [9f, 1f, 0f]),
            (30, [8f, 2f, 0f]),
            (40, [0f, 6f, 0f]),
            (50, [-3f, 0f, 0f]),
            (60, [0f, -4f, 0f])
        ];
        var options = new HnswIndexOptions(6, 24, 3, 0x2310_0200UL);
        HnswIndex hnsw = CreateHnsw(rows, options);
        ExactFlatIndex exact = CreateExact(rows);
        float[] query = [1f, 0f, 0f];

        ulong[] exactFallbackAllowlist = [999, 40, 20, 20, 60];
        SearchResult[] expectedExactFallback = Search(exact, query, exactFallbackAllowlist, topK: 3);
        SearchResult[] actualExactFallback = Search(hnsw, query, exactFallbackAllowlist, topK: 3);
        AssertResultsEqual(expectedExactFallback, actualExactFallback);

        ulong[] broadAllowlist = [999, 20, 30, 40, 50, 60, 60];
        SearchResult[] unfilteredCandidates = Search(hnsw, query, topK: options.EfSearch);
        SearchResult[] expectedBroad = unfilteredCandidates
            .Where(static result => result.Id != 10)
            .Take(3)
            .ToArray();

        SearchResult[] actualBroad = Search(hnsw, query, broadAllowlist, topK: 3);

        Assert.Equal(5, broadAllowlist.Distinct().Count(id => rows.Any(row => row.Id == id)));
        AssertResultsEqual(expectedBroad, actualBroad);
        Assert.DoesNotContain(actualBroad, static result => result.Id == 10);
        Assert.All(actualBroad, result => Assert.Contains(result.Id, broadAllowlist));
    }

    [Fact]
    public void CosineDurableOpen_PreservesSquaredL2CompatibilityAndCosineEmptySmallSnapshots()
    {
        using TempIndexDirectory squared = TempIndexDirectory.Create();
        var squaredL2 = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 16, 16, 0x2310_0300UL));
        squaredL2.Add(10, [0f, 0f]);
        squaredL2.Add(20, [2f, 0f]);
        squaredL2.Save(squared.Path);

        HnswIndex openedSquared = HnswIndex.OpenReadOnly(squared.Path);
        Assert.Equal(VectorMetric.SquaredEuclidean, openedSquared.Metric);
        Assert.Equal(Search(squaredL2, [0f, 0f], topK: 2), Search(openedSquared, [0f, 0f], topK: 2));

        using TempIndexDirectory emptyCosine = TempIndexDirectory.Create();
        var empty = new HnswIndex(3, VectorMetric.Cosine, new HnswIndexOptions(4, 16, 16, 0x2310_0301UL));
        empty.Save(emptyCosine.Path);
        HnswIndex openedEmpty = HnswIndex.OpenReadOnly(emptyCosine.Path);

        Assert.Equal(VectorMetric.Cosine, openedEmpty.Metric);
        Assert.Equal(0, openedEmpty.Count);
        Assert.Equal(0, openedEmpty.Search([1f, 0f, 0f], new SearchResult[3], openedEmpty.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(() => openedEmpty.Search([0f, 0f, 0f], new SearchResult[0], openedEmpty.CreateSearchWorkspace()));

        using TempIndexDirectory singleCosine = TempIndexDirectory.Create();
        var single = new HnswIndex(2, VectorMetric.Cosine, new HnswIndexOptions(4, 16, 16, 0x2310_0302UL));
        single.Add(70, [3f, 4f]);
        single.Save(singleCosine.Path);
        HnswIndex openedSingle = HnswIndex.OpenReadOnly(singleCosine.Path);

        Assert.Equal(VectorMetric.Cosine, openedSingle.Metric);
        Assert.Equal([0.6f, 0.8f], openedSingle.InternalVectors.ToArray());
        SearchResult[] singleResult = Search(openedSingle, [6f, 8f], topK: 1);
        Assert.Equal(70UL, singleResult[0].Id);
        Assert.Equal(0f, singleResult[0].Distance, precision: 6);
        AssertManifestMetric(singleCosine.Path, "cosine", "cosine-unit-normalized");
    }

    [Fact]
    public void CosineDurableOpen_RejectsHostileMetricAndNormalizationPermutations()
    {
        AssertCosineOpenRejected(temp => MutateManifest(temp.Path, root => root["index"]!["metric"] = "squared-euclidean"));
        AssertCosineOpenRejected(temp => MutateManifest(temp.Path, root => root["index"]!["metric"] = "inner-product"));
        AssertCosineOpenRejected(temp => MutateManifest(temp.Path, root => root["index"]!["normalizationState"] = "future-normalized"));
        AssertCosineOpenRejected(temp => PatchFile(
            temp.Path,
            HnswIndexStorage.VectorsFileName,
            bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), HnswIndexStorage.InnerProductMetricCode)));
        AssertCosineOpenRejected(temp => PatchFile(
            temp.Path,
            HnswIndexStorage.VectorsFileName,
            bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), HnswIndexStorage.NoNormalizationCode)));

        using TempIndexDirectory squared = TempIndexDirectory.Create();
        var squaredIndex = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 16, 16, 0x2310_0400UL));
        squaredIndex.Add(10, [1f, 0f]);
        squaredIndex.Add(20, [0f, 1f]);
        squaredIndex.Save(squared.Path);
        MutateManifest(squared.Path, root => root["index"]!["normalizationState"] = "cosine-unit-normalized");
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(squared.Path));
    }

    [Fact]
    public void CosineMutableSearch_MergesBaseAndDeltaTiesByExternalId()
    {
        HnswIndex baseIndex = CreateHnsw(
            [(40, [8f, 0f, 0f]), (10, [1f, 0f, 0f])],
            new HnswIndexOptions(4, 16, 16, 0x2310_0500UL));
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(30, [300f, 0f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(20, [20f, 0f, 0f]).Status);

        SearchResult[] results = Search(mutable, [2f, 0f, 0f], topK: 4);

        Assert.Equal([10UL, 20UL, 30UL, 40UL], results.Select(static result => result.Id).ToArray());
        Assert.All(results, static result => Assert.Equal(0f, result.Distance, precision: 6));
    }

    [Fact]
    public void CosineMutableChurn_CheckpointReopenPreservesResultsAndRejectsStaleWorkspace()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateHnsw(
            [(10, [1f, 0f, 0f]), (20, [0f, 5f, 0f]), (30, [-1f, 0f, 0f])],
            new HnswIndexOptions(8, 32, 32, 0x2310_0600UL));
        var mutable = new HnswMutableIndex(baseIndex);
        var staleBeforeMutation = new HnswMutableSearchWorkspace(mutable, maxResults: 3);

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [10f, 10f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(25, [0f, 50f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(25).Status);

        Assert.Throws<InvalidOperationException>(() => mutable.Search([1f, 1f, 0f], new SearchResult[3], staleBeforeMutation));

        SearchResult[] beforeCheckpoint = Search(mutable, [1f, 1f, 0f], topK: 3);
        Assert.Equal([15UL, 20UL, 30UL], beforeCheckpoint.Select(static result => result.Id).ToArray());
        Assert.Equal(0f, beforeCheckpoint[0].Distance, precision: 6);

        var staleBeforeCheckpoint = new HnswMutableSearchWorkspace(mutable, maxResults: 3);
        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(VectorMetric.Cosine, mutable.Metric);
        Assert.Equal(3, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([1f, 1f, 0f], new SearchResult[3], staleBeforeCheckpoint));

        SearchResult[] afterCheckpoint = Search(mutable, [1f, 1f, 0f], topK: 3);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);

        AssertResultsEqual(beforeCheckpoint, afterCheckpoint);
        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        AssertResultsEqual(beforeCheckpoint, Search(opened, [1f, 1f, 0f], topK: 3));
        Assert.DoesNotContain(10UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(25UL, opened.InternalIds.ToArray());
    }

    [Fact]
    public void CosineValidation_RejectsZeroAndNonFiniteAtEarlyReturnCheckpointAndOpenBoundaries()
    {
        var index = new HnswIndex(3, VectorMetric.Cosine, new HnswIndexOptions(4, 16, 16, 0x2310_0700UL));
        Assert.Throws<ArgumentException>(() => index.Add(10, [0f, 0f, 0f]));
        Assert.Throws<ArgumentException>(() => index.Add(10, [1f, float.NaN, 0f]));

        index.Add(10, [1f, 0f, 0f]);
        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f, 0f], new SearchResult[0], index.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(() => index.Search([float.NegativeInfinity, 0f, 0f], [], new SearchResult[0], index.CreateSearchWorkspace()));

        var mutable = new HnswMutableIndex(index);
        Assert.Throws<ArgumentException>(() => mutable.TryAdd(20, [0f, 0f, 0f]));
        Assert.Throws<ArgumentException>(() => mutable.Search([float.NaN, 0f, 0f], new SearchResult[0], new HnswMutableSearchWorkspace(mutable, 0)));

        AssertCheckpointSourceRejected(corruptBase: true, corruptDelta: false);
        AssertCheckpointSourceRejected(corruptBase: false, corruptDelta: true);

        AssertCosineOpenRejected(temp => PatchVectorValue(temp.Path, valueIndex: 0, float.NaN));
        AssertCosineOpenRejected(temp =>
        {
            PatchVectorValue(temp.Path, valueIndex: 0, 0f);
            PatchVectorValue(temp.Path, valueIndex: 1, 0f);
            PatchVectorValue(temp.Path, valueIndex: 2, 0f);
        });
        AssertCosineOpenRejected(temp => PatchVectorValue(temp.Path, valueIndex: 0, 0.25f));
    }

    private static (ulong Id, float[] Vector)[] CreateCosineRows(int dimension)
    {
        var rows = new (ulong Id, float[] Vector)[12];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = ((ulong)(1_000 + i * 37), CreateRawVector(dimension, scale: 1.25f + i, phase: i));
        }

        return rows;
    }

    private static float[] CreateRawVector(int dimension, float scale, int phase)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            float lane = (((i + 1) * (phase + 3)) % 11) - 5f;
            vector[i] = scale * (lane + ((phase % 3) * 0.125f));
        }

        if (vector.All(static value => value == 0f))
        {
            vector[0] = scale == 0 ? 1f : scale;
        }

        return vector;
    }

    private static HnswIndex CreateHnsw(
        IEnumerable<(ulong Id, float[] Vector)> rows,
        HnswIndexOptions options)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(dimension, VectorMetric.Cosine, options, () => 0);
        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static ExactFlatIndex CreateExact(IEnumerable<(ulong Id, float[] Vector)> rows)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new ExactFlatIndex(dimension, VectorMetric.Cosine);
        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace());
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
        int written = index.Search(query, allowlist, results, index.CreateSearchFilterWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static void AssertResultsEqual(SearchResult[] expected, SearchResult[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            Assert.Equal(expected[i].Distance, actual[i].Distance, precision: 6);
        }
    }

    private static void AssertManifestMetric(string directory, string metric, string normalizationState)
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, HnswIndexStorage.ManifestFileName)));
        JsonElement index = manifest.RootElement.GetProperty("index");
        Assert.Equal(metric, index.GetProperty("metric").GetString());
        Assert.Equal(normalizationState, index.GetProperty("normalizationState").GetString());
    }

    private static void AssertCosineOpenRejected(Action<TempIndexDirectory> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateHnsw(
            [(10, [3f, 4f, 0f]), (20, [0f, 5f, 0f]), (30, [-2f, 0f, 0f])],
            new HnswIndexOptions(4, 16, 16, 0x2310_0800UL)).Save(temp.Path);
        mutate(temp);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertCheckpointSourceRejected(bool corruptBase, bool corruptDelta)
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateHnsw(
            [(10, [1f, 0f]), (20, [0f, 1f])],
            new HnswIndexOptions(4, 16, 16, 0x2310_0900UL));
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(30, [1f, 1f]).Status);

        if (corruptBase)
        {
            GetPrivateField<float[]>(baseIndex, "_vectors")[0] = float.PositiveInfinity;
        }

        if (corruptDelta)
        {
            object inner = GetPrivateField<object>(mutable, "_inner");
            GetPrivateField<float[]>(inner, "_deltaVectors")[0] = 0f;
            GetPrivateField<float[]>(inner, "_deltaVectors")[1] = 0f;
        }

        Assert.Throws<InvalidOperationException>(() => mutable.Checkpoint(checkpoint.Path));
    }

    private static T GetPrivateField<T>(object instance, string name)
    {
        FieldInfo field = instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field '{name}' was not found.");
        return (T)field.GetValue(instance)!;
    }

    private static void PatchVectorValue(string directory, int valueIndex, float value)
    {
        PatchFile(
            directory,
            HnswIndexStorage.VectorsFileName,
            bytes => BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(HnswIndexStorage.VectorsHeaderLength + valueIndex * sizeof(float)),
                BitConverter.SingleToInt32Bits(value)));
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

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = CreatePath();
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

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
                "VecNet-HnswCosineIndependent-" + Guid.NewGuid().ToString("N"));
    }
}
