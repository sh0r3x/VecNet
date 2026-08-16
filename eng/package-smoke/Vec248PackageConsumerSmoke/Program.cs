using Microsoft.Extensions.VectorData;
using VecNet;
using VecNet.Integration.VectorData;

string artifactRoot = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.Combine(Path.GetTempPath(), "vec248-package-consumer-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(artifactRoot);

RunHnswCosinePackageSmoke(Path.Combine(artifactRoot, "hnsw-cosine"));
RunHnswInnerProductPackageSmoke(Path.Combine(artifactRoot, "hnsw-inner-product"));
RunMutableHnswCosinePackageSmoke(Path.Combine(artifactRoot, "hnsw-mutable-cosine"));
RunMutableHnswInnerProductPackageSmoke(Path.Combine(artifactRoot, "hnsw-mutable-inner-product"));
await RunVectorDataAdapterSmoke();

Console.WriteLine("PACKAGE_CONSUMER_SMOKE_PASSED");

static void RunHnswCosinePackageSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 24, EfSearch: 24, RandomSeed: 0x91E4_2D7C_A5B9_1031UL);
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

static void RunHnswInnerProductPackageSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 32, EfSearch: 32, RandomSeed: 0x1A2B_3C4D_5E6F_7081UL);
    var index = new HnswIndex(dimension: 3, VectorMetric.InnerProduct, options, initialCapacity: 6);

    index.Add(10, [5f, 0f, 0f]);
    index.Add(20, [0f, 4f, 0f]);
    index.Add(30, [1f, 1f, 0f]);
    index.Add(40, [-10f, 0f, 0f]);
    index.Add(50, [0f, 0f, 0f]);
    index.Add(60, [0f, 0f, 3f]);

    SearchResult[] results = SearchHnsw(index, [1f, 1f, 0f], 4);
    Require(
        results.Select(static result => result.Id).SequenceEqual([10UL, 20UL, 30UL, 50UL]),
        "HNSW inner-product search should rank larger dot products as lower distances.");
    Require(
        results.Select(static result => result.Distance).SequenceEqual([-5f, -4f, -2f, -0f]),
        "HNSW inner-product search should expose negative-dot distances.");

    ulong[] allowlist = [999, 60, 30, 20, 20, 777];
    SearchResult[] allowedResults = SearchHnswAllowed(index, [1f, 1f, 0f], allowlist, 3);
    Require(
        allowedResults.Select(static result => result.Id).SequenceEqual([20UL, 30UL, 60UL]),
        "HNSW inner-product allowlist search should rank only allowed external IDs.");

    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    HnswIndex opened = HnswIndex.OpenReadOnly(savePath);
    Require(opened.Metric == VectorMetric.InnerProduct, "Opened HNSW index should preserve inner-product metric.");
    ExpectThrows<InvalidOperationException>(
        () => opened.Add(70, [1f, 0f, 0f]),
        "Opened HNSW inner-product indexes should reject build ingestion.");

    SearchResult[] openedResults = SearchHnswAllowed(opened, [1f, 1f, 0f], allowlist, 3);
    Require(openedResults.Length == allowedResults.Length, "Opened HNSW inner-product search should preserve result count.");
    for (int i = 0; i < openedResults.Length; i++)
    {
        Require(openedResults[i].Id == allowedResults[i].Id, "Opened HNSW inner-product search should preserve result IDs.");
        Require(
            Math.Abs(openedResults[i].Distance - allowedResults[i].Distance) <= 0.000001f,
            "Opened HNSW inner-product search should preserve distances.");
    }
}

static void RunMutableHnswCosinePackageSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 24, EfSearch: 24, RandomSeed: 0x6A09_E667_F3BC_C909UL);
    var baseIndex = new HnswIndex(dimension: 3, VectorMetric.Cosine, options, initialCapacity: 6);

    baseIndex.Add(40, [10f, 0f, 0f]);
    baseIndex.Add(10, [1f, 1f, 0f]);
    baseIndex.Add(30, [0f, 2f, 0f]);
    baseIndex.Add(20, [-1f, 0f, 0f]);
    baseIndex.Add(50, [0f, 0f, 5f]);
    baseIndex.Add(60, [1f, 1f, 1f]);

    var mutable = new HnswMutableIndex(baseIndex);
    Require(mutable.Metric == VectorMetric.Cosine, "Mutable HNSW wrapper should preserve cosine metric.");

    VectorMutationResult add = mutable.TryAdd(15, [2f, 2f, 0f]);
    Require(add.Status == VectorMutationStatus.Committed, "Mutable HNSW cosine should accept nonzero delta vectors.");
    Require(mutable.DeltaLiveVectorCount == 1, "Mutable HNSW cosine should expose the live delta row.");

    var staleAfterAdd = new HnswMutableSearchWorkspace(mutable, maxResults: 4);
    VectorMutationResult delete = mutable.TryDelete(10);
    Require(delete.Status == VectorMutationStatus.Committed, "Mutable HNSW cosine should tombstone a base ID.");
    ExpectThrows<InvalidOperationException>(
        () => mutable.Search([1f, 1f, 0f], new SearchResult[4], staleAfterAdd),
        "Mutable HNSW cosine should reject stale workspaces after mutation.");

    SearchResult[] liveResults = SearchMutableHnsw(mutable, [1f, 1f, 0f], 4);
    Require(liveResults.Length == 4, "Mutable HNSW cosine search should return live results.");
    Require(
        liveResults.All(static result => float.IsFinite(result.Distance)),
        "Mutable HNSW cosine search should return finite distances.");
    Require(
        !liveResults.Any(static result => result.Id == 10),
        "Mutable HNSW cosine search should not return tombstoned base IDs.");

    ulong[] allowed = [999, 10, 15, 30, 15, 20, 777];
    SearchResult[] allowedResults = SearchMutableHnswAllowed(mutable, [1f, 1f, 0f], allowed, 4);
    Require(
        allowedResults.Select(static result => result.Id).SequenceEqual([15UL, 30UL, 20UL]),
        "Mutable HNSW cosine allowlist search should exclude tombstones and rank live allowed IDs.");

    string checkpointPath = Path.Combine(artifactRoot, "checkpoint");
    SearchResult[] beforeCheckpoint = SearchMutableHnsw(mutable, [1f, 1f, 0f], 4);
    HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
    Require(
        checkpoint.Status == HnswMutableCheckpointStatus.Published,
        "Mutable HNSW cosine checkpoint should publish a rebuilt immutable HNSW output.");
    Require(
        Directory.Exists(checkpointPath),
        "Mutable HNSW cosine checkpoint should write the checkpoint directory.");
    Require(
        checkpoint.RebuiltBaseVectorCount == mutable.LiveVectorCount,
        "Mutable HNSW cosine checkpoint should fold live rows into the rebuilt base.");

    HnswIndex opened = HnswIndex.OpenReadOnly(checkpointPath);
    Require(opened.Metric == VectorMetric.Cosine, "Opened mutable HNSW cosine checkpoint should preserve cosine metric.");

    SearchResult[] openedResults = SearchHnsw(opened, [1f, 1f, 0f], beforeCheckpoint.Length);
    Require(openedResults.Length == beforeCheckpoint.Length, "Opened mutable HNSW cosine checkpoint should preserve result count.");
    for (int i = 0; i < openedResults.Length; i++)
    {
        Require(openedResults[i].Id == beforeCheckpoint[i].Id, "Opened mutable HNSW cosine checkpoint should preserve result IDs.");
        Require(
            Math.Abs(openedResults[i].Distance - beforeCheckpoint[i].Distance) <= 0.000001f,
            "Opened mutable HNSW cosine checkpoint should preserve distances.");
    }
}

static void RunMutableHnswInnerProductPackageSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 32, EfSearch: 32, RandomSeed: 0x2233_4455_6677_8899UL);
    var baseIndex = new HnswIndex(dimension: 2, VectorMetric.InnerProduct, options, initialCapacity: 4);

    baseIndex.Add(10, [5f, 0f]);
    baseIndex.Add(20, [0f, 4f]);
    baseIndex.Add(30, [-3f, 0f]);
    baseIndex.Add(40, [0f, 0f]);

    var mutable = new HnswMutableIndex(baseIndex);
    Require(mutable.Metric == VectorMetric.InnerProduct, "Mutable HNSW wrapper should preserve inner-product metric.");

    VectorMutationResult add = mutable.TryAdd(15, [6f, 0f]);
    Require(add.Status == VectorMutationStatus.Committed, "Mutable HNSW inner product should accept delta vectors.");
    Require(mutable.DeltaLiveVectorCount == 1, "Mutable HNSW inner product should expose the live delta row.");

    var staleAfterAdd = new HnswMutableSearchWorkspace(mutable, maxResults: 4);
    VectorMutationResult delete = mutable.TryDelete(10);
    Require(delete.Status == VectorMutationStatus.Committed, "Mutable HNSW inner product should tombstone a base ID.");
    ExpectThrows<InvalidOperationException>(
        () => mutable.Search([1f, 1f], new SearchResult[4], staleAfterAdd),
        "Mutable HNSW inner product should reject stale workspaces after mutation.");

    SearchResult[] liveResults = SearchMutableHnsw(mutable, [1f, 1f], 4);
    Require(
        liveResults.Select(static result => result.Id).SequenceEqual([15UL, 20UL, 40UL, 30UL]),
        "Mutable HNSW inner-product search should merge live base and delta rows by negative-dot distance.");
    Require(
        !liveResults.Any(static result => result.Id == 10),
        "Mutable HNSW inner-product search should not return tombstoned base IDs.");

    ulong[] allowed = [999, 10, 15, 20, 40, 15, 777];
    SearchResult[] allowedResults = SearchMutableHnswAllowed(mutable, [1f, 1f], allowed, 4);
    Require(
        allowedResults.Select(static result => result.Id).SequenceEqual([15UL, 20UL, 40UL]),
        "Mutable HNSW inner-product allowlist search should exclude tombstones and rank live allowed IDs.");

    string checkpointPath = Path.Combine(artifactRoot, "checkpoint");
    SearchResult[] beforeCheckpoint = SearchMutableHnsw(mutable, [1f, 1f], 4);
    HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
    Require(
        checkpoint.Status == HnswMutableCheckpointStatus.Published,
        "Mutable HNSW inner-product checkpoint should publish a rebuilt immutable HNSW output.");
    Require(
        checkpoint.RebuiltBaseVectorCount == mutable.LiveVectorCount,
        "Mutable HNSW inner-product checkpoint should fold live rows into the rebuilt base.");

    HnswIndex opened = HnswIndex.OpenReadOnly(checkpointPath);
    Require(opened.Metric == VectorMetric.InnerProduct, "Opened mutable HNSW inner-product checkpoint should preserve metric.");

    SearchResult[] openedResults = SearchHnsw(opened, [1f, 1f], beforeCheckpoint.Length);
    Require(
        openedResults.Select(static result => result.Id).SequenceEqual(beforeCheckpoint.Select(static result => result.Id)),
        "Opened mutable HNSW inner-product checkpoint should preserve result IDs.");
    for (int i = 0; i < openedResults.Length; i++)
    {
        Require(
            Math.Abs(openedResults[i].Distance - beforeCheckpoint[i].Distance) <= 0.000001f,
            "Opened mutable HNSW inner-product checkpoint should preserve distances.");
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

static SearchResult[] SearchMutableHnsw(HnswMutableIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, top));
    return results[..written];
}

static SearchResult[] SearchMutableHnswAllowed(HnswMutableIndex index, float[] query, ulong[] allowedIds, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, allowedIds, results, new HnswMutableSearchWorkspace(index, top));
    return results[..written];
}

static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results, index.CreateSearchWorkspace());
    return results[..written];
}

static SearchResult[] SearchHnswAllowed(HnswIndex index, float[] query, ulong[] allowedIds, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, allowedIds, results, index.CreateSearchWorkspace());
    return results[..written];
}

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
