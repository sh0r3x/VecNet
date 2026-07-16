using Microsoft.Extensions.VectorData;
using VecNet;
using VecNet.Integration.VectorData;

string artifactRoot = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.Combine(Path.GetTempPath(), "vec191-docs-consumer-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(artifactRoot);

RunExactFlatDocsSmoke(Path.Combine(artifactRoot, "exact-flat"));
RunHnswDocsSmoke(Path.Combine(artifactRoot, "hnsw"));
RunMutableHnswDocsSmoke(Path.Combine(artifactRoot, "mutable-hnsw"));
await RunVectorDataAdapterDocsSmoke();

Console.WriteLine("VEC191_PACKAGE_CONSUMER_DOCS_SMOKE_PASSED");

static void RunExactFlatDocsSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var keyToVecNetId = new Dictionary<string, ulong>(StringComparer.Ordinal)
    {
        ["tenant-a/doc-1/chunk-0"] = 1001,
        ["tenant-a/doc-2/chunk-0"] = 1002,
        ["tenant-b/doc-3/chunk-0"] = 2001,
        ["tenant-a/doc-4/chunk-0"] = 1004
    };
    var metadata = new Dictionary<ulong, AppMetadata>
    {
        [1001] = new("tenant-a", "public"),
        [1002] = new("tenant-a", "private"),
        [2001] = new("tenant-b", "public"),
        [1004] = new("tenant-a", "public")
    };

    var index = new ExactFlatIndex(dimension: 3, VectorMetric.SquaredEuclidean);
    index.Add(keyToVecNetId["tenant-a/doc-1/chunk-0"], [1.0f, 0.0f, 0.0f]);
    index.Add(keyToVecNetId["tenant-a/doc-2/chunk-0"], [0.0f, 1.0f, 0.0f]);
    index.Add(keyToVecNetId["tenant-b/doc-3/chunk-0"], [0.0f, 0.0f, 1.0f]);
    index.Add(keyToVecNetId["tenant-a/doc-4/chunk-0"], [0.8f, 0.1f, 0.0f]);

    Span<SearchResult> results = stackalloc SearchResult[2];
    int written = index.Search([0.9f, 0.1f, 0.0f], results);
    Require(written == 2, "Exact flat unfiltered search should return two results.");
    Require(results[0].Id == 1004 && results[1].Id == 1001, "Exact flat nearest IDs did not match expected ordering.");

    ulong[] tenantPublicAllowlist = metadata
        .Where(pair => pair.Value.Tenant == "tenant-a" && pair.Value.Category == "public")
        .Select(pair => pair.Key)
        .ToArray();
    var filterWorkspace = new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount);
    written = index.Search([0.0f, 1.0f, 0.0f], tenantPublicAllowlist, results, filterWorkspace);
    Require(written == 2, "Exact allowlist search should return the tenant public rows.");
    Require(results[0].Id == 1004 && results[1].Id == 1001, "Exact allowlist should apply caller-owned metadata filtering.");

    ExactFlatCandidateSet candidates = index.CreateCandidateSet(tenantPublicAllowlist);
    written = index.Search([1.0f, 0.0f, 0.0f], candidates, results);
    Require(written == 2, "Exact candidate-set search should return candidate rows.");
    Require(results[0].Id == 1001 && results[1].Id == 1004, "Exact candidate-set search should rank only allowed rows.");

    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(savePath);
    written = opened.Search([0.9f, 0.1f, 0.0f], results);
    Require(written == 2 && results[0].Id == 1004, "Opened exact flat index should preserve saved search behavior.");
}

static void RunHnswDocsSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 4, EfConstruction: 12, EfSearch: 8, RandomSeed: 0x564543_191UL);
    var index = new HnswIndex(dimension: 3, VectorMetric.SquaredEuclidean, options, initialCapacity: 5);

    index.Add(1001, [1.0f, 0.0f, 0.0f]);
    index.Add(1002, [0.0f, 1.0f, 0.0f]);
    index.Add(1003, [0.0f, 0.0f, 1.0f]);
    index.Add(1004, [0.8f, 0.1f, 0.0f]);
    index.Add(1005, [0.7f, 0.2f, 0.0f]);

    var workspace = new HnswSearchWorkspace(index.Count, index.Options.EfSearch);
    Span<SearchResult> results = stackalloc SearchResult[2];
    int written = index.Search([0.9f, 0.1f, 0.0f], results, workspace);
    Require(written == 2, "HNSW search should fill the caller-owned result buffer for this small smoke.");
    Require(results[0].Id is 1004 or 1005 or 1001, "HNSW search should return a nearby squared-L2 candidate.");

    ulong[] allowedIds = [1001, 1003, 9999, 1003];
    written = index.Search([0.0f, 0.0f, 1.0f], allowedIds, results, workspace);
    Require(written == 2, "HNSW allowlist fallback should return known allowed rows in this small smoke.");
    Require(results[0].Id == 1003 && results[1].Id == 1001, "HNSW allowlist should use caller-owned external IDs.");

    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    HnswIndex opened = HnswIndex.OpenReadOnly(savePath);
    var openedWorkspace = new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch);
    written = opened.Search([0.9f, 0.1f, 0.0f], results, openedWorkspace);
    Require(written == 2, "Opened HNSW index should search with a caller-owned workspace.");
    ExpectThrows<InvalidOperationException>(() => opened.Add(9999, [1.0f, 1.0f, 1.0f]), "Opened HNSW index should reject Add.");
}

static void RunMutableHnswDocsSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 4, EfConstruction: 12, EfSearch: 8, RandomSeed: 0x564543_192UL);
    var baseIndex = new HnswIndex(dimension: 3, VectorMetric.SquaredEuclidean, options);
    baseIndex.Add(1001, [1.0f, 0.0f, 0.0f]);
    baseIndex.Add(1002, [0.0f, 1.0f, 0.0f]);
    baseIndex.Add(1003, [0.0f, 0.0f, 1.0f]);

    var mutable = new HnswMutableIndex(baseIndex);
    VectorMutationResult add = mutable.TryAdd(1004, [0.9f, 0.1f, 0.0f]);
    Require(add.Status == VectorMutationStatus.Committed, "Mutable HNSW TryAdd should commit a delta row.");
    VectorMutationResult delete = mutable.TryDelete(1002);
    Require(delete.Status == VectorMutationStatus.Committed, "Mutable HNSW TryDelete should tombstone a base row.");

    var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: 3);
    Span<SearchResult> results = stackalloc SearchResult[3];
    int written = mutable.Search([0.9f, 0.1f, 0.0f], results, workspace);
    Require(written == 3, "Mutable HNSW search should merge base and exact delta rows.");
    Require(results[0].Id == 1004, "Mutable HNSW exact delta row should be searchable.");
    Require(!ContainsId(results[..written], 1002), "Mutable HNSW tombstoned base row should be hidden.");

    string checkpointPath = Path.Combine(artifactRoot, "checkpoint");
    HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
    Require(checkpoint.Status == HnswMutableCheckpointStatus.Published, "Mutable HNSW checkpoint should publish after mutations.");

    workspace = new HnswMutableSearchWorkspace(mutable, maxResults: 3);
    written = mutable.Search([0.9f, 0.1f, 0.0f], results, workspace);
    Require(written == 3 && results[0].Id == 1004, "Mutable HNSW search should work after checkpoint workspace recreation.");
    Require(!ContainsId(results[..written], 1002), "Mutable HNSW checkpoint should preserve deleted-ID visibility.");
}

static async Task RunVectorDataAdapterDocsSmoke()
{
    var store = new VecNetVectorStore();
    VectorStoreCollection<string, SmokeRecord> collection =
        store.GetCollection<string, SmokeRecord>("vec191-records", CreateDefinition());

    Require(!await collection.CollectionExistsAsync(), "VectorData collection should not exist before creation.");
    await collection.EnsureCollectionExistsAsync();
    Require(await collection.CollectionExistsAsync(), "VectorData collection should exist after creation.");

    await collection.UpsertAsync(CreateRecord("doc-a", [0f, 0f], "tenant-a", "public"));
    await collection.UpsertAsync(CreateRecord("doc-b", [2f, 0f], "tenant-a", "private"));
    await collection.UpsertAsync(CreateRecord("doc-c", [4f, 0f], "tenant-b", "public"));
    await collection.UpsertAsync(CreateRecord("doc-a", [1f, 0f], "tenant-a", "public"));

    SmokeRecord? fetched = await collection.GetAsync("doc-a");
    Require(fetched?.Vector.Span[0] == 1f, "VectorData adapter should expose replacement upsert records.");

    List<VectorSearchResult<SmokeRecord>> unfiltered = await Search(collection, [0f, 0f], top: 3);
    Require(unfiltered.Select(result => result.Record.Id).SequenceEqual(["doc-a", "doc-b", "doc-c"]),
        "VectorData adapter should search exact-flat records.");
    Require(unfiltered[0].Score == 1, "VectorData adapter should project Euclidean-squared scores.");

    var options = new VectorSearchOptions<SmokeRecord>
    {
        Filter = record => record.Tenant == "tenant-a" && record.Category == "public",
        ScoreThreshold = 10
    };
    List<VectorSearchResult<SmokeRecord>> filtered = await Search(collection, [0f, 0f], top: 3, options);
    Require(filtered.Select(result => result.Record.Id).SequenceEqual(["doc-a"]),
        "VectorData adapter filter should restrict search through an exact allowlist.");

    await collection.DeleteAsync("doc-a");
    Require(await collection.GetAsync("doc-a") is null, "VectorData adapter delete should remove the record.");
    List<VectorSearchResult<SmokeRecord>> afterDelete = await Search(collection, [0f, 0f], top: 3);
    Require(afterDelete.Select(result => result.Record.Id).SequenceEqual(["doc-b", "doc-c"]),
        "VectorData adapter delete should hide the vector from search.");
}

static VectorStoreCollectionDefinition CreateDefinition()
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
                IndexKind = Microsoft.Extensions.VectorData.IndexKind.Flat,
                DistanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance
            },
            new VectorStoreDataProperty(nameof(SmokeRecord.Tenant), typeof(string)),
            new VectorStoreDataProperty(nameof(SmokeRecord.Category), typeof(string))
        ]
    };
}

static SmokeRecord CreateRecord(string id, float[] vector, string tenant, string category) =>
    new() { Id = id, Vector = vector, Tenant = tenant, Category = category };

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

static bool ContainsId(ReadOnlySpan<SearchResult> results, ulong id)
{
    foreach (SearchResult result in results)
    {
        if (result.Id == id)
        {
            return true;
        }
    }

    return false;
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

internal sealed record AppMetadata(string Tenant, string Category);

internal sealed class SmokeRecord
{
    public string Id { get; init; } = string.Empty;

    public ReadOnlyMemory<float> Vector { get; init; }

    public string Tenant { get; init; } = string.Empty;

    public string Category { get; init; } = string.Empty;
}
