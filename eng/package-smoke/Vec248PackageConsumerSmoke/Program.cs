using Microsoft.Extensions.VectorData;
using VecNet;
using VecNet.Integration.VectorData;

string artifactRoot = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.Combine(Path.GetTempPath(), "vec248-package-consumer-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(artifactRoot);

RunHnswCosinePackageSmoke(Path.Combine(artifactRoot, "hnsw-cosine"));
await RunVectorDataAdapterSmoke();

Console.WriteLine("VEC248_PACKAGE_CONSUMER_SMOKE_PASSED");

static void RunHnswCosinePackageSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 24, EfSearch: 24, RandomSeed: 0x564543_248UL);
    var index = new HnswIndex(dimension: 3, VectorMetric.Cosine, options, initialCapacity: 6);

    index.Add(40, [10f, 0f, 0f]);
    index.Add(10, [1f, 1f, 0f]);
    index.Add(30, [0f, 2f, 0f]);
    index.Add(20, [-1f, 0f, 0f]);
    index.Add(50, [0f, 0f, 5f]);
    index.Add(60, [1f, 1f, 1f]);

    Span<SearchResult> results = stackalloc SearchResult[6];
    int written = index.Search([2f, 1f, 0f], results, index.CreateSearchWorkspace());
    Require(written > 0, "HNSW cosine search should return at least one result in this small smoke.");
    for (int i = 0; i < written; i++)
    {
        Require(float.IsFinite(results[i].Distance), "HNSW cosine search should return finite distances.");
        if (i > 0)
        {
            Require(results[i - 1].Distance <= results[i].Distance, "HNSW cosine search should return results sorted by distance.");
        }
    }

    ulong[] allowlist = [999, 50, 10, 30, 10, 777];
    written = index.Search([2f, 1f, 0f], allowlist, results[..3], index.CreateSearchWorkspace());
    Require(written == 3, "HNSW cosine allowlist search should return allowed known IDs.");
    Require(
        results[..written].ToArray().Select(static result => result.Id).SequenceEqual([10UL, 30UL, 50UL]),
        "HNSW cosine allowlist search should rank only allowed external IDs.");

    ExpectThrows<ArgumentException>(() => index.Add(70, [0f, 0f, 0f]), "HNSW cosine should reject zero vectors during build.");
    ExpectThrows<ArgumentException>(
        () => index.Search([0f, 0f, 0f], new SearchResult[1], index.CreateSearchWorkspace()),
        "HNSW cosine should reject zero query vectors.");
    ExpectThrows<NotSupportedException>(
        () => new HnswIndex(3, VectorMetric.InnerProduct),
        "HNSW inner product should remain unsupported.");

    SearchResult[] sourceResults = results[..written].ToArray();
    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    HnswIndex opened = HnswIndex.OpenReadOnly(savePath);
    Require(opened.Metric == VectorMetric.Cosine, "Opened HNSW index should preserve cosine metric.");

    SearchResult[] openedResults = new SearchResult[3];
    int openedWritten = opened.Search([2f, 1f, 0f], allowlist, openedResults, opened.CreateSearchWorkspace());
    Require(openedWritten == sourceResults.Length, "Opened HNSW cosine search should preserve result count.");
    for (int i = 0; i < openedWritten; i++)
    {
        Require(openedResults[i].Id == sourceResults[i].Id, "Opened HNSW cosine search should preserve result IDs.");
        Require(
            Math.Abs(openedResults[i].Distance - sourceResults[i].Distance) <= 0.000001f,
            "Opened HNSW cosine search should preserve distances.");
    }
}

static async Task RunVectorDataAdapterSmoke()
{
    var store = new VecNetVectorStore();
    VectorStoreCollection<string, SmokeRecord> collection =
        store.GetCollection<string, SmokeRecord>("vec248-records", CreateDefinition());

    await collection.EnsureCollectionExistsAsync();
    await collection.UpsertAsync(CreateRecord("a", [1f, 0f], "red"));
    await collection.UpsertAsync(CreateRecord("b", [2f, 0f], "blue"));
    await collection.UpsertAsync(CreateRecord("c", [4f, 0f], "blue"));

    List<VectorSearchResult<SmokeRecord>> results = await Search(collection, [0f, 0f], top: 3);
    Require(
        results.Select(static result => result.Record.Id).SequenceEqual(["a", "b", "c"]),
        "VectorData adapter should restore, build, and search the exact-flat package surface.");

    ExpectThrows<NotSupportedException>(
        () => store.GetCollection<string, SmokeRecord>(
            "unsupported-hnsw",
            CreateDefinition(indexKind: Microsoft.Extensions.VectorData.IndexKind.Hnsw)),
        "VectorData adapter should remain exact-flat-only.");
}

static VectorStoreCollectionDefinition CreateDefinition(
    string distanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance,
    string indexKind = Microsoft.Extensions.VectorData.IndexKind.Flat)
{
    return new VectorStoreCollectionDefinition
    {
        Properties =
        [
            new VectorStoreKeyProperty(nameof(SmokeRecord.Id), typeof(string)),
            new VectorStoreVectorProperty(
                nameof(SmokeRecord.Vector),
                typeof(ReadOnlyMemory<float>),
                dimensions: 2)
            {
                IndexKind = indexKind,
                DistanceFunction = distanceFunction
            },
            new VectorStoreDataProperty(nameof(SmokeRecord.Tag), typeof(string))
        ]
    };
}

static SmokeRecord CreateRecord(string id, float[] vector, string tag) =>
    new() { Id = id, Vector = vector, Tag = tag };

static Task<List<VectorSearchResult<SmokeRecord>>> Search(
    VectorStoreCollection<string, SmokeRecord> collection,
    float[] query,
    int top,
    VectorSearchOptions<SmokeRecord>? options = null) =>
    ToListAsync(collection.SearchAsync(query, top, options));

static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
{
    var results = new List<T>();
    await foreach (T item in source)
    {
        results.Add(item);
    }

    return results;
}

static void ResetDirectory(string path)
{
    if (Directory.Exists(path))
    {
        Directory.Delete(path, recursive: true);
    }

    Directory.CreateDirectory(path);
}

static void ExpectThrows<TException>(Action action, string message)
    where TException : Exception
{
    try
    {
        action();
    }
    catch (TException)
    {
        return;
    }

    throw new InvalidOperationException(message);
}

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

internal sealed class SmokeRecord
{
    public string Id { get; init; } = string.Empty;

    public ReadOnlyMemory<float> Vector { get; init; }

    public string Tag { get; init; } = string.Empty;
}
