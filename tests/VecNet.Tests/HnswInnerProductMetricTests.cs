using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswInnerProductMetricTests
{
    [Fact]
    public void Constructor_AcceptsInnerProductThroughPublicShapes()
    {
        var defaults = new HnswIndex(3, VectorMetric.InnerProduct);
        var capacity = new HnswIndex(3, VectorMetric.InnerProduct, initialCapacity: 8);
        var options = new HnswIndex(3, VectorMetric.InnerProduct, new HnswIndexOptions(4, 16, 16, 0x3420UL));
        var optionsCapacity = new HnswIndex(
            3,
            VectorMetric.InnerProduct,
            new HnswIndexOptions(4, 16, 16, 0x3421UL),
            initialCapacity: 8);

        Assert.Equal(VectorMetric.InnerProduct, defaults.Metric);
        Assert.Equal(VectorMetric.InnerProduct, capacity.Metric);
        Assert.Equal(VectorMetric.InnerProduct, options.Metric);
        Assert.Equal(VectorMetric.InnerProduct, optionsCapacity.Metric);
        Assert.True(capacity.Capacity >= 8);
        Assert.True(optionsCapacity.Capacity >= 8);
    }

    [Fact]
    public void ImmutableSearch_UsesRawNegativeDotAndAcceptsZeroVectors()
    {
        (ulong Id, float[] Vector)[] rows =
        [
            (70, [100f, 0f, 0f]),
            (10, [0f, 0f, 0f]),
            (30, [0f, 2f, 0f]),
            (20, [1f, 0f, 0f]),
            (40, [-1f, 0f, 0f]),
            (50, [0f, 0f, 3f]),
            (60, [1f, 1f, 1f])
        ];
        HnswIndex hnsw = CreateHnsw(rows, new HnswIndexOptions(8, 32, 32, 0x3422UL));
        ExactFlatIndex exact = CreateExact(rows);

        SearchResult[] actual = Search(hnsw, [2f, 1f, 0f], topK: rows.Length, efSearch: rows.Length);
        SearchResult[] expected = Search(exact, [2f, 1f, 0f], topK: rows.Length);

        AssertResultsEqual(expected, actual);
        Assert.Equal([100f, 0f, 0f], hnsw.InternalVectors.Slice(0, 3).ToArray());
        Assert.Equal([0f, 0f, 0f], hnsw.InternalVectors.Slice(3, 3).ToArray());
        Assert.Equal(new SearchResult(70, -200f), actual[0]);

        SearchResult[] zeroQuery = Search(hnsw, [0f, 0f, 0f], topK: rows.Length, efSearch: rows.Length);
        Assert.All(zeroQuery, static result => Assert.Equal(0f, result.Distance));
        Assert.Equal(rows.Select(static row => row.Id).Order().ToArray(), zeroQuery.Select(static result => result.Id).ToArray());
    }

    [Fact]
    public void ValidationAndPerSearchEfSearch_AreMetricNeutralForInnerProduct()
    {
        var index = new HnswIndex(2, VectorMetric.InnerProduct, new HnswIndexOptions(2, 8, 2, 0x3423UL));

        Assert.Throws<ArgumentException>(() => index.Add(10, [1f]));
        Assert.Throws<ArgumentException>(() => index.Add(10, [float.NaN, 1f]));

        index.Add(10, [0f, 0f]);
        index.Add(20, [2f, 0f]);

        SearchResult[] destination = [new(999, 999), new(998, 998)];
        Assert.Throws<ArgumentException>(() => index.Search([float.PositiveInfinity, 0f], destination, index.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(() => index.Search([1f], destination, index.CreateSearchWorkspace()));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search([1f, 0f], destination, index.CreateSearchWorkspace(1), efSearch: 1));
        Assert.Throws<ArgumentException>(() => index.Search([1f, 0f], destination, new HnswSearchWorkspace(index.Count, 1), efSearch: 2));

        int written = index.Search([1f, 0f], destination, index.CreateSearchWorkspace(2), efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([20UL, 10UL], destination.Select(static result => result.Id).ToArray());
        Assert.Equal([-2f, 0f], destination.Select(static result => result.Distance).ToArray());
    }

    [Fact]
    public void AllowlistSearch_UsesInnerProductForExactFallbackAndBroadEmission()
    {
        (ulong Id, float[] Vector)[] rows =
        [
            (10, [0f, 0f]),
            (20, [10f, 0f]),
            (30, [0f, 4f]),
            (40, [-3f, 0f]),
            (50, [2f, 2f]),
            (60, [1f, -5f])
        ];
        var options = new HnswIndexOptions(6, 24, 3, 0x3424UL);
        HnswIndex hnsw = CreateHnsw(rows, options);
        ExactFlatIndex exact = CreateExact(rows);
        float[] query = [1f, 1f];

        ulong[] selectiveAllowlist = [999, 30, 50, 50];
        AssertResultsEqual(
            Search(exact, query, selectiveAllowlist, topK: 3),
            Search(hnsw, query, selectiveAllowlist, topK: 3, efSearch: options.EfSearch));

        ulong[] broadAllowlist = [999, 10, 30, 40, 50, 60, 60];
        SearchResult[] unfilteredCandidates = Search(hnsw, query, topK: options.EfSearch, efSearch: options.EfSearch);
        SearchResult[] expectedBroad = unfilteredCandidates
            .Where(static result => result.Id != 20)
            .Take(3)
            .ToArray();

        SearchResult[] actualBroad = Search(hnsw, query, broadAllowlist, topK: 3, efSearch: options.EfSearch);

        AssertResultsEqual(expectedBroad, actualBroad);
        Assert.DoesNotContain(actualBroad, static result => result.Id == 20);
        Assert.All(actualBroad, result => Assert.Contains(result.Id, broadAllowlist));
    }

    [Fact]
    public void SaveOpenReadOnly_PreservesInnerProductManifestHeaderRawVectorsAndSearch()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        (ulong Id, float[] Vector)[] rows =
        [
            (10, [0f, 0f, 0f]),
            (20, [2f, 0f, 0f]),
            (30, [0f, -3f, 0f]),
            (40, [1f, 1f, 1f])
        ];
        HnswIndex source = CreateHnsw(rows, new HnswIndexOptions(4, 16, 16, 0x3425UL));
        float[] query = [1f, -1f, 2f];
        SearchResult[] expected = Search(source, query, topK: rows.Length, efSearch: rows.Length);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(VectorMetric.InnerProduct, opened.Metric);
        Assert.Equal(source.InternalVectors.ToArray(), opened.InternalVectors.ToArray());
        AssertResultsEqual(expected, Search(opened, query, topK: rows.Length, efSearch: rows.Length));
        AssertResultsEqual(
            Search(source, query, [999, 10, 40, 40], topK: 2, efSearch: rows.Length),
            Search(opened, query, [999, 10, 40, 40], topK: 2, efSearch: rows.Length));

        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(temp.Path, HnswIndexStorage.ManifestFileName)));
        JsonElement index = manifest.RootElement.GetProperty("index");
        Assert.Equal("inner-product", index.GetProperty("metric").GetString());
        Assert.Equal("none", index.GetProperty("normalizationState").GetString());

        byte[] vectorBytes = File.ReadAllBytes(Path.Combine(temp.Path, HnswIndexStorage.VectorsFileName));
        Assert.Equal(HnswIndexStorage.InnerProductMetricCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(32)));
        Assert.Equal(HnswIndexStorage.NoNormalizationCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(36)));
    }

    [Fact]
    public void OpenReadOnly_RejectsInnerProductNonFiniteAndMetricNormalizationMismatches()
    {
        AssertInnerProductOpenRejected(temp =>
            PatchVectorValue(temp.Path, valueIndex: 0, float.NegativeInfinity));
        AssertInnerProductOpenRejected(temp =>
            MutateManifest(temp.Path, root => root["index"]!["normalizationState"] = "cosine-unit-normalized"));
        AssertInnerProductOpenRejected(temp =>
            PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), HnswIndexStorage.CosineUnitNormalizedCode)));
        AssertInnerProductOpenRejected(temp =>
            PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), HnswIndexStorage.SquaredEuclideanMetricCode)));
    }

    [Fact]
    public void MutableInnerProduct_MergesDeltaTombstonesAllowlistCheckpointAndReopen()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateHnsw(
            [(10, [0f, 0f]), (20, [1f, 0f]), (30, [-2f, 0f])],
            new HnswIndexOptions(4, 16, 16, 0x3426UL));
        var mutable = new HnswMutableIndex(baseIndex);
        HnswMutableSearchWorkspace staleBeforeMutation = mutable.CreateSearchWorkspace(maxResults: 4, maxEfSearch: 16);

        Assert.Equal(VectorMetric.InnerProduct, mutable.Metric);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [3f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(25, [0f, 5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(25).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [7f, 0f]).Status);

        Assert.Throws<InvalidOperationException>(() => mutable.Search([1f, 0f], new SearchResult[4], staleBeforeMutation, efSearch: 16));

        SearchResult[] beforeCheckpoint = Search(mutable, [1f, 0f], topK: 4, efSearch: 16);
        AssertResultsEqual(
            [new SearchResult(15, -3f), new SearchResult(10, 0f), new SearchResult(30, 2f)],
            beforeCheckpoint);

        SearchResult[] allowed = Search(mutable, [1f, 0f], [999, 15, 20, 30, 15], topK: 3, efSearch: 16);
        AssertResultsEqual([new SearchResult(15, -3f), new SearchResult(30, 2f)], allowed);

        HnswMutableSearchWorkspace staleBeforeCheckpoint = mutable.CreateSearchWorkspace(maxResults: 4, maxEfSearch: 16);
        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(3, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([1f, 0f], new SearchResult[4], staleBeforeCheckpoint, efSearch: 16));

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.InnerProduct, opened.Metric);
        Assert.DoesNotContain(20UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(25UL, opened.InternalIds.ToArray());
        AssertResultsEqual(beforeCheckpoint, Search(mutable, [1f, 0f], topK: 4, efSearch: 16));
        AssertResultsEqual(beforeCheckpoint, Search(opened, [1f, 0f], topK: 4, efSearch: 16));
    }

    private static HnswIndex CreateHnsw(
        IEnumerable<(ulong Id, float[] Vector)> rows,
        HnswIndexOptions options)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(dimension, VectorMetric.InnerProduct, options, () => 0);
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
        var index = new ExactFlatIndex(dimension, VectorMetric.InnerProduct);
        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(efSearch), efSearch);
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

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
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

    private static void AssertInnerProductOpenRejected(Action<TempIndexDirectory> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateHnsw(
            [(10, [0f, 0f]), (20, [2f, 0f]), (30, [-1f, 1f])],
            new HnswIndexOptions(4, 16, 16, 0x3427UL)).Save(temp.Path);
        mutate(temp);
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
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
                "VecNet-HnswInnerProductMetricTests-" + Guid.NewGuid().ToString("N"));
    }
}
