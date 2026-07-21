using Microsoft.Extensions.VectorData;
using VecNet;
using VecNet.Integration.VectorData;

string artifactRoot = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.Combine(Path.GetTempPath(), "vec212-package-consumer-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(artifactRoot);

RunExactFlatSmoke(Path.Combine(artifactRoot, "exact-flat"));
RunHnswSmoke(Path.Combine(artifactRoot, "hnsw"));
RunMutableHnswSmoke(Path.Combine(artifactRoot, "mutable-hnsw"));
await RunVectorDataIncludeVectorsSmoke();

Console.WriteLine("VEC212_PACKAGE_CONSUMER_SMOKE_PASSED");

static void RunExactFlatSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var index = new ExactFlatIndex(dimension: 3, VectorMetric.SquaredEuclidean);
    index.Add(1001, [1.0f, 0.0f, 0.0f]);
    index.Add(1002, [0.0f, 1.0f, 0.0f]);
    index.Add(1003, [0.0f, 0.0f, 1.0f]);
    index.Add(1004, [0.8f, 0.1f, 0.0f]);

    Span<SearchResult> results = stackalloc SearchResult[2];
    int written = index.Search([0.9f, 0.1f, 0.0f], results);
    Require(written == 2, "Exact search should fill the requested top-k.");
    Require(results[0].Id == 1004 && results[1].Id == 1001, "Exact search nearest IDs did not match expected ordering.");

    ulong[] allowedIds = [1001, 1003, 9999, 1003];
    ExactFlatSearchFilterWorkspace filterWorkspace = index.CreateSearchFilterWorkspace();
    written = index.Search([0.0f, 0.0f, 1.0f], allowedIds, results, filterWorkspace);
    Require(written == 2, "Exact allowlist filtering should return allowed known IDs.");
    Require(results[0].Id == 1003 && results[1].Id == 1001, "Exact allowlist filtering should rank only allowed rows.");

    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(savePath);
    written = opened.Search([0.9f, 0.1f, 0.0f], results);
    Require(written == 2 && results[0].Id == 1004 && results[1].Id == 1001, "Opened exact index should preserve search results.");
}

static void RunHnswSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 4, EfConstruction: 12, EfSearch: 8, RandomSeed: 0x564543_212UL);
    var index = new HnswIndex(dimension: 3, VectorMetric.SquaredEuclidean, options, initialCapacity: 5);

    index.Add(1001, [1.0f, 0.0f, 0.0f]);
    index.Add(1002, [0.0f, 1.0f, 0.0f]);
    index.Add(1003, [0.0f, 0.0f, 1.0f]);
    index.Add(1004, [0.8f, 0.1f, 0.0f]);
    index.Add(1005, [0.7f, 0.2f, 0.0f]);

    HnswSearchWorkspace workspace = index.CreateSearchWorkspace();
    Span<SearchResult> results = stackalloc SearchResult[2];
    int written = index.Search([0.9f, 0.1f, 0.0f], results, workspace);
    Require(written == 2, "HNSW search should fill the requested top-k for this small smoke.");
    Require(results[0].Id is 1004 or 1005 or 1001, "HNSW search should return a nearby squared-L2 candidate.");

    string savePath = Path.Combine(artifactRoot, "saved");
    index.Save(savePath);
    HnswIndex opened = HnswIndex.OpenReadOnly(savePath);
    HnswSearchWorkspace openedWorkspace = opened.CreateSearchWorkspace();
    written = opened.Search([0.9f, 0.1f, 0.0f], results, openedWorkspace);
    Require(written == 2, "Opened HNSW index should search with a caller-owned workspace.");
    ExpectThrows<InvalidOperationException>(() => opened.Add(9999, [1.0f, 1.0f, 1.0f]), "Opened HNSW index should reject Add.");
}

static void RunMutableHnswSmoke(string artifactRoot)
{
    ResetDirectory(artifactRoot);

    var options = new HnswIndexOptions(M: 4, EfConstruction: 12, EfSearch: 8, RandomSeed: 0x564543_213UL);
    var baseIndex = new HnswIndex(dimension: 3, VectorMetric.SquaredEuclidean, options);
    baseIndex.Add(1001, [1.0f, 0.0f, 0.0f]);
    baseIndex.Add(1002, [0.0f, 1.0f, 0.0f]);
    baseIndex.Add(1003, [0.0f, 0.0f, 1.0f]);

    var mutable = new HnswMutableIndex(baseIndex);
    VectorMutationResult add = mutable.TryAdd(1004, [0.9f, 0.1f, 0.0f]);
    Require(add.Status == VectorMutationStatus.Committed, "Mutable HNSW TryAdd should commit a delta row.");
    VectorMutationResult delete = mutable.TryDelete(1002);
    Require(delete.Status == VectorMutationStatus.Committed, "Mutable HNSW TryDelete should tombstone a base row.");

    Span<SearchResult> results = stackalloc SearchResult[3];
    var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: results.Length);
    int written = mutable.Search([0.9f, 0.1f, 0.0f], results, workspace);
    Require(written == 3, "Mutable HNSW search should merge base and exact delta rows.");
    Require(results[0].Id == 1004, "Mutable HNSW exact delta row should be searchable.");
    Require(!ContainsId(results[..written], 1002), "Mutable HNSW tombstoned base row should be hidden.");

    string checkpointPath = Path.Combine(artifactRoot, "checkpoint");
    HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
    Require(checkpoint.Status == HnswMutableCheckpointStatus.Published, "Mutable HNSW checkpoint should publish after mutations.");
    Require(checkpoint.LiveVectorCount == 3, "Mutable HNSW checkpoint should report the live vector count.");

    HnswIndex openedCheckpoint = HnswIndex.OpenReadOnly(checkpointPath);
    HnswSearchWorkspace openedWorkspace = openedCheckpoint.CreateSearchWorkspace();
    written = openedCheckpoint.Search([0.9f, 0.1f, 0.0f], results, openedWorkspace);
    Require(written == 3 && results[0].Id == 1004, "Checkpoint-opened HNSW index should preserve rebuilt search behavior.");
    Require(!ContainsId(results[..written], 1002), "Checkpoint-opened HNSW index should preserve tombstone visibility.");
}

static async Task RunVectorDataIncludeVectorsSmoke()
{
    var store = new VecNetVectorStore();
    VectorStoreCollection<string, SmokeRecord> collection =
        store.GetCollection<string, SmokeRecord>("vec212-records", CreateDefinition());

    await collection.EnsureCollectionExistsAsync();
    await collection.UpsertAsync(CreateRecord("doc-a", [1f, 0f], "tenant-a", "public"));
    await collection.UpsertAsync(CreateRecord("doc-b", [2f, 0f], "tenant-a", "private"));
    await collection.UpsertAsync(CreateRecord("doc-c", [4f, 0f], "tenant-b", "public"));

    SmokeRecord? omittedGet = await collection.GetAsync(
        "doc-a",
        new RecordRetrievalOptions { IncludeVectors = false });
    SmokeRecord? includedGet = await collection.GetAsync(
        "doc-a",
        new RecordRetrievalOptions { IncludeVectors = true });

    Require(omittedGet is not null && omittedGet.Vector.IsEmpty, "VectorData Get IncludeVectors=false should omit vectors.");
    Require(includedGet is not null && includedGet.Vector.ToArray().SequenceEqual([1f, 0f]),
        "VectorData Get IncludeVectors=true should include vectors.");

    List<SmokeRecord> omittedFiltered = await ToListAsync(
        collection.GetAsync(
            record => record.Tenant == "tenant-a" && record.Category == "public",
            top: 1,
            new FilteredRecordRetrievalOptions<SmokeRecord> { IncludeVectors = false }));
    List<SmokeRecord> includedFiltered = await ToListAsync(
        collection.GetAsync(
            record => record.Tenant == "tenant-a" && record.Category == "public",
            top: 1,
            new FilteredRecordRetrievalOptions<SmokeRecord> { IncludeVectors = true }));

    Require(omittedFiltered.Count == 1 && omittedFiltered[0].Vector.IsEmpty,
        "VectorData filtered Get IncludeVectors=false should omit vectors.");
    Require(includedFiltered.Count == 1 && includedFiltered[0].Vector.ToArray().SequenceEqual([1f, 0f]),
        "VectorData filtered Get IncludeVectors=true should include vectors.");

    var omittedSearchOptions = new VectorSearchOptions<SmokeRecord>
    {
        IncludeVectors = false,
        Filter = record => record.Tenant == "tenant-a"
    };
    var includedSearchOptions = new VectorSearchOptions<SmokeRecord>
    {
        IncludeVectors = true,
        Filter = record => record.Tenant == "tenant-a"
    };

    List<VectorSearchResult<SmokeRecord>> omittedSearch = await Search(collection, [0f, 0f], top: 2, omittedSearchOptions);
    List<VectorSearchResult<SmokeRecord>> includedSearch = await Search(collection, [0f, 0f], top: 2, includedSearchOptions);

    Require(omittedSearch.Select(result => result.Record.Id).SequenceEqual(["doc-a", "doc-b"]),
        "VectorData filtered search should return exact-flat records.");
    Require(omittedSearch.All(result => result.Record.Vector.IsEmpty),
        "VectorData search IncludeVectors=false should omit vectors.");
    Require(includedSearch.Select(result => result.Record.Id).SequenceEqual(["doc-a", "doc-b"]),
        "VectorData IncludeVectors=true should preserve search ordering.");
    Require(includedSearch[0].Record.Vector.ToArray().SequenceEqual([1f, 0f]) &&
        includedSearch[1].Record.Vector.ToArray().SequenceEqual([2f, 0f]),
        "VectorData search IncludeVectors=true should include vectors.");
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

internal sealed class SmokeRecord
{
    public string Id { get; init; } = string.Empty;

    public ReadOnlyMemory<float> Vector { get; init; }

    public string Tenant { get; init; } = string.Empty;

    public string Category { get; init; } = string.Empty;
}
