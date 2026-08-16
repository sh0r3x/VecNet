using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswCosineMetricTests
{
    [Fact]
    public void Constructor_AcceptsCosineAndInnerProduct()
    {
        var cosine = new HnswIndex(3, VectorMetric.Cosine, new HnswIndexOptions(2, 8, 8, 0x227UL));
        var innerProduct = new HnswIndex(3, VectorMetric.InnerProduct, new HnswIndexOptions(2, 8, 8, 0x228UL));

        Assert.Equal(VectorMetric.Cosine, cosine.Metric);
        Assert.Equal(VectorMetric.InnerProduct, innerProduct.Metric);
    }

    [Fact]
    public void CosineRejectsZeroAndNonFiniteVectorsAtImmutableAndMutableBoundaries()
    {
        var index = new HnswIndex(2, VectorMetric.Cosine, new HnswIndexOptions(2, 8, 8, 0x2271UL));

        Assert.Throws<ArgumentException>(() => index.Add(1, [0f, 0f]));
        Assert.Throws<ArgumentException>(() => index.Add(1, [float.NaN, 1f]));

        index.Add(10, [1f, 0f]);
        index.Add(20, [0f, 1f]);

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], new SearchResult[1], index.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(
            () => index.Search([float.PositiveInfinity, 0f], new SearchResult[1], index.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], [10], new SearchResult[1], index.CreateSearchWorkspace()));

        var mutable = new HnswMutableIndex(index);
        Assert.Throws<ArgumentException>(() => mutable.TryAdd(30, [0f, 0f]));
        Assert.Throws<ArgumentException>(() => mutable.TryAdd(30, [1f, float.NegativeInfinity]));

        var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: 1);
        Assert.Throws<ArgumentException>(() => mutable.Search([0f, 0f], new SearchResult[1], workspace));
        Assert.Throws<ArgumentException>(() => mutable.Search([float.NaN, 1f], [10], new SearchResult[1], workspace));
    }

    [Fact]
    public void CosineSearchAndAllowlistMatchExactFlatForSmallDeterministicCase()
    {
        var options = new HnswIndexOptions(8, 16, 16, 0x2272UL);
        (ulong Id, float[] Vector)[] rows =
        [
            (40, [10f, 0f, 0f]),
            (10, [1f, 1f, 0f]),
            (30, [0f, 2f, 0f]),
            (20, [-1f, 0f, 0f]),
            (50, [0f, 0f, 5f]),
            (60, [1f, 1f, 1f])
        ];
        HnswIndex hnsw = CreateHnsw(VectorMetric.Cosine, rows, options);
        ExactFlatIndex exact = CreateExact(VectorMetric.Cosine, rows);
        float[] query = [2f, 1f, 0f];

        SearchResult[] expected = Search(exact, query, topK: rows.Length);
        SearchResult[] actual = Search(hnsw, query, topK: rows.Length);

        Assert.Equal(expected, actual);

        ulong[] allowlist = [999, 50, 10, 30, 10, 777];
        SearchResult[] expectedAllowed = Search(exact, query, allowlist, topK: 3);
        SearchResult[] actualAllowed = Search(hnsw, query, allowlist, topK: 3);
        Assert.Equal(expectedAllowed, actualAllowed);

        Assert.Equal([1f, 0f, 0f], hnsw.InternalVectors.Slice(0, 3).ToArray());
    }

    [Fact]
    public void CosineReturnedDistanceIsCanonicalOneMinusNormalizedDot()
    {
        var index = new HnswIndex(2, VectorMetric.Cosine, new HnswIndexOptions(2, 8, 8, 0x2273UL), () => 0);
        index.Add(10, [3f, 4f]);
        index.Add(20, [0f, 5f]);

        SearchResult[] results = Search(index, [4f, 3f], topK: 2);

        Assert.Equal(10UL, results[0].Id);
        Assert.Equal(1f - (24f / 25f), results[0].Distance, precision: 6);
    }

    [Fact]
    public void CosineSaveOpenRoundTripPreservesMetricNormalizationAndSearch()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(8, 16, 16, 0x2274UL);
        HnswIndex source = CreateHnsw(
            VectorMetric.Cosine,
            [(10, [3f, 4f]), (20, [0f, 2f]), (30, [-5f, 0f])],
            options);
        float[] query = [4f, 3f];
        SearchResult[] expected = Search(source, query, topK: 3);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.Equal(source.InternalVectors.ToArray(), opened.InternalVectors.ToArray());
        Assert.Equal(expected, Search(opened, query, topK: 3));

        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(temp.Path, HnswIndexStorage.ManifestFileName)));
        JsonElement index = manifest.RootElement.GetProperty("index");
        Assert.Equal("cosine", index.GetProperty("metric").GetString());
        Assert.Equal("cosine-unit-normalized", index.GetProperty("normalizationState").GetString());

        byte[] vectorBytes = File.ReadAllBytes(Path.Combine(temp.Path, HnswIndexStorage.VectorsFileName));
        Assert.Equal(HnswIndexStorage.CosineMetricCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(32)));
        Assert.Equal(HnswIndexStorage.CosineUnitNormalizedCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(36)));
    }

    [Fact]
    public void OpenReadOnlyRejectsMetricNormalizationMismatchesAndInnerProductPayloads()
    {
        AssertOpenRejected(temp =>
            MutateManifest(temp.Path, root => root["index"]!["normalizationState"] = "none"));

        AssertOpenRejected(temp =>
            PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), HnswIndexStorage.SquaredEuclideanMetricCode)));

        AssertOpenRejected(temp =>
            PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), HnswIndexStorage.NoNormalizationCode)));

        using TempIndexDirectory squared = TempIndexDirectory.Create();
        CreateHnsw(VectorMetric.SquaredEuclidean, [(10, [1f, 0f]), (20, [0f, 1f])]).Save(squared.Path);
        MutateManifest(squared.Path, root =>
        {
            root["index"]!["metric"] = "cosine";
            root["index"]!["normalizationState"] = "cosine-unit-normalized";
        });
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(squared.Path));

        using TempIndexDirectory innerProductManifest = TempIndexDirectory.Create();
        CreateHnsw(VectorMetric.SquaredEuclidean, [(10, [1f, 0f]), (20, [0f, 1f])]).Save(innerProductManifest.Path);
        MutateManifest(innerProductManifest.Path, root => root["index"]!["metric"] = "inner-product");
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(innerProductManifest.Path));

        using TempIndexDirectory innerProductHeader = TempIndexDirectory.Create();
        CreateHnsw(VectorMetric.SquaredEuclidean, [(10, [1f, 0f]), (20, [0f, 1f])]).Save(innerProductHeader.Path);
        PatchFile(
            innerProductHeader.Path,
            HnswIndexStorage.VectorsFileName,
            bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), HnswIndexStorage.InnerProductMetricCode));
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(innerProductHeader.Path));
    }

    [Fact]
    public void OpenReadOnlyRejectsCosineNonFiniteZeroAndNonUnitStoredRows()
    {
        AssertOpenRejected(temp =>
            PatchVectorValue(temp.Path, valueIndex: 0, float.PositiveInfinity));
        AssertOpenRejected(temp =>
            PatchVectorValue(temp.Path, valueIndex: 0, 0f));
        AssertOpenRejected(temp =>
            PatchVectorValue(temp.Path, valueIndex: 1, 0f));
        AssertOpenRejected(temp =>
            PatchVectorValue(temp.Path, valueIndex: 0, 0.5f));
    }

    [Fact]
    public void MutableCosineSupportsDeltaTombstonesCheckpointAndReopenedSearch()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateHnsw(
            VectorMetric.Cosine,
            [(10, [1f, 0f]), (20, [0f, 1f]), (30, [-1f, 0f])],
            new HnswIndexOptions(8, 16, 16, 0x2275UL));
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Equal(VectorMetric.Cosine, mutable.Metric);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [2f, 2f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(10, [1f, 0f]).Status);

        SearchResult[] results = Search(mutable, [1f, 1f], topK: 3);
        Assert.Equal([15UL, 20UL, 30UL], results.Select(static result => result.Id).ToArray());
        Assert.Equal(0f, results[0].Distance, precision: 6);

        SearchResult[] allowed = Search(mutable, [1f, 1f], [10, 15, 20], topK: 3);
        Assert.Equal([15UL, 20UL], allowed.Select(static result => result.Id).ToArray());

        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(VectorMetric.Cosine, mutable.Metric);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.Equal(Search(mutable, [1f, 1f], topK: 3), Search(opened, [1f, 1f], topK: 3));
        Assert.DoesNotContain(10UL, opened.InternalIds.ToArray());
    }

    private static HnswIndex CreateHnsw(
        VectorMetric metric,
        IEnumerable<(ulong Id, float[] Vector)> rows,
        HnswIndexOptions? options = null)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            metric,
            options ?? new HnswIndexOptions(8, 16, 16, 0x2276UL),
            () => 0);

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static ExactFlatIndex CreateExact(VectorMetric metric, IEnumerable<(ulong Id, float[] Vector)> rows)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new ExactFlatIndex(dimension, metric);
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

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static void AssertOpenRejected(Action<TempIndexDirectory> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateHnsw(VectorMetric.Cosine, [(10, [3f, 4f]), (20, [0f, 2f])]).Save(temp.Path);
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
                "VecNet-HnswCosineMetricTests-" + Guid.NewGuid().ToString("N"));
    }
}
