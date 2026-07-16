using Microsoft.Extensions.VectorData;
using VecNet.Integration.VectorData;

var store = new VecNetVectorStore();
VectorStoreCollection<string, SmokeRecord> collection =
    store.GetCollection<string, SmokeRecord>("smoke-records", CreateDefinition());

Require(!await collection.CollectionExistsAsync(), "Collection should not exist before creation.");
Require(!await store.CollectionExistsAsync("smoke-records"), "Store should report collection missing before creation.");

await collection.EnsureCollectionExistsAsync();
Require(await collection.CollectionExistsAsync(), "Collection should exist after creation.");
Require(await store.CollectionExistsAsync("smoke-records"), "Store should report collection after creation.");
Require(await ToListAsync(store.ListCollectionNamesAsync()) is ["smoke-records"], "Collection listing did not include the created collection.");

await collection.UpsertAsync(CreateRecord("a", [0f, 0f], "red"));
await collection.UpsertAsync(CreateRecord("b", [2f, 0f], "blue"));
await collection.UpsertAsync(CreateRecord("c", [4f, 0f], "blue"));

SmokeRecord? fetched = await collection.GetAsync("a");
Require(fetched?.Tag == "red", "Get should return the upserted record.");

await collection.UpsertAsync(CreateRecord("a", [1f, 0f], "red-updated"));
fetched = await collection.GetAsync("a");
Require(fetched?.Tag == "red-updated", "Replacement upsert should expose the current record.");

List<VectorSearchResult<SmokeRecord>> replacementResults = await Search(collection, [0f, 0f], top: 3);
Require(
    replacementResults.Select(result => result.Record.Id).SequenceEqual(["a", "b", "c"]),
    "Replacement vector should participate in search and old vector should be tombstoned.");
Require(replacementResults[0].Score == 1, "Euclidean-squared score projection should return squared L2 distance.");

var filteredOptions = new VectorSearchOptions<SmokeRecord>
{
    Filter = record => record.Tag == "blue",
    ScoreThreshold = 16
};
List<VectorSearchResult<SmokeRecord>> filteredResults = await Search(collection, [0f, 0f], top: 2, filteredOptions);
Require(
    filteredResults.Select(result => result.Record.Id).SequenceEqual(["b", "c"]),
    "Expression filter should restrict search through the adapter allowlist.");

await collection.DeleteAsync("b");
Require(await collection.GetAsync("b") is null, "Deleted record should not be returned by Get.");

List<VectorSearchResult<SmokeRecord>> afterDeleteResults = await Search(collection, [0f, 0f], top: 5);
Require(
    afterDeleteResults.Select(result => result.Record.Id).SequenceEqual(["a", "c"]),
    "Deleted vector should not remain searchable.");

ExpectThrows<NotSupportedException>(
    () => store.GetCollection<string, SmokeRecord>(
        "unsupported-hnsw",
        CreateDefinition(indexKind: Microsoft.Extensions.VectorData.IndexKind.Hnsw)),
    "HNSW VectorData collections must be rejected by the exact-flat adapter.");

await collection.EnsureCollectionDeletedAsync();
Require(!await collection.CollectionExistsAsync(), "Collection should not exist after delete.");
await AssertVectorStoreExceptionAsync(() => collection.GetAsync("a"), "Operations on deleted collection should fail.");

Console.WriteLine("VEC188_VECTOR_DATA_ADAPTER_SMOKE_PASSED");

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

static async Task AssertVectorStoreExceptionAsync(Func<Task> action, string message)
{
    try
    {
        await action();
    }
    catch (VectorStoreException)
    {
        return;
    }

    throw new InvalidOperationException(message);
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
