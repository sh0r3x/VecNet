using Microsoft.Extensions.VectorData;
using VecNet.Integration.VectorData;

namespace VecNet.Integration.VectorData.Tests;

public sealed class VecNetVectorStoreCollectionIndependentTests
{
    [Fact]
    public async Task BatchUpsertInvalidReplacementPreservesCurrentMappingsForProcessedAndFailedKeys()
    {
        VectorStoreCollection<string, IndependentRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [10, 0], "a-old"));
        await collection.UpsertAsync(CreateRecord("b", [0, 10], "b-old"));

        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await collection.UpsertAsync(
            [
                CreateRecord("a", [0, 0], "a-new"),
                CreateRecord("b", [float.PositiveInfinity, 0], "b-invalid"),
                CreateRecord("c", [1, 0], "c-not-processed")
            ]));

        Assert.Equal("a-new", (await collection.GetAsync("a"))?.Tag);
        Assert.Equal("b-old", (await collection.GetAsync("b"))?.Tag);
        Assert.Null(await collection.GetAsync("c"));

        List<VectorSearchResult<IndependentRecord>> results = await Search(collection, [0, 0], top: 10);
        Assert.Equal(["a", "b"], results.Select(result => result.Record.Id));
        Assert.Equal([0d, 100d], results.Select(result => result.Score.GetValueOrDefault()));
    }

    [Fact]
    public async Task DeleteAfterReplacementRemovesOnlyCurrentInternalMapping()
    {
        VectorStoreCollection<string, IndependentRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [5, 0], "old"));
        await collection.UpsertAsync(CreateRecord("a", [1, 0], "current"));
        await collection.UpsertAsync(CreateRecord("b", [2, 0], "other"));

        await collection.DeleteAsync("a");

        Assert.Null(await collection.GetAsync("a"));
        List<VectorSearchResult<IndependentRecord>> results = await Search(collection, [0, 0], top: 10);
        VectorSearchResult<IndependentRecord> result = Assert.Single(results);
        Assert.Equal("b", result.Record.Id);
        Assert.Equal(4, result.Score);
    }

    [Fact]
    public async Task FilterSkipAndScoreThresholdInteractAfterAllowlistProjection()
    {
        VectorStoreCollection<string, IndependentRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("red-near", [0, 0], "red"));
        await collection.UpsertAsync(CreateRecord("blue-one", [1, 0], "blue"));
        await collection.UpsertAsync(CreateRecord("blue-two", [2, 0], "blue"));
        await collection.UpsertAsync(CreateRecord("blue-three", [3, 0], "blue"));
        await collection.UpsertAsync(CreateRecord("blue-four", [4, 0], "blue"));

        var options = new VectorSearchOptions<IndependentRecord>
        {
            Filter = record => record.Tag == "blue",
            ScoreThreshold = 9,
            Skip = 1
        };

        List<VectorSearchResult<IndependentRecord>> results = await Search(collection, [0, 0], top: 2, options);

        Assert.Equal(["blue-two", "blue-three"], results.Select(result => result.Record.Id));
        Assert.Equal([4d, 9d], results.Select(result => result.Score.GetValueOrDefault()));
    }

    [Fact]
    public async Task SupportedScoreProjectionBoundaryCasesUseInclusiveThresholds()
    {
        VectorStoreCollection<string, IndependentRecord> squared = CreateCollection();
        await squared.EnsureCollectionExistsAsync();
        await squared.UpsertAsync(CreateRecord("exact", [0, 0], "x"));
        await squared.UpsertAsync(CreateRecord("nonzero", [1, 0], "x"));

        List<VectorSearchResult<IndependentRecord>> squaredResults = await Search(
            squared,
            [0, 0],
            top: 2,
            new VectorSearchOptions<IndependentRecord> { ScoreThreshold = 0 });

        VectorSearchResult<IndependentRecord> squaredResult = Assert.Single(squaredResults);
        Assert.Equal("exact", squaredResult.Record.Id);
        Assert.Equal(0, squaredResult.Score);

        VectorStoreCollection<string, IndependentRecord> cosineDistance =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance));
        await cosineDistance.EnsureCollectionExistsAsync();
        await cosineDistance.UpsertAsync(CreateRecord("scaled-same", [10, 0], "x"));
        await cosineDistance.UpsertAsync(CreateRecord("orthogonal", [0, 1], "x"));

        List<VectorSearchResult<IndependentRecord>> cosineDistanceResults = await Search(
            cosineDistance,
            [1, 0],
            top: 2,
            new VectorSearchOptions<IndependentRecord> { ScoreThreshold = 0 });

        VectorSearchResult<IndependentRecord> cosineDistanceResult = Assert.Single(cosineDistanceResults);
        Assert.Equal("scaled-same", cosineDistanceResult.Record.Id);
        Assert.Equal(0, cosineDistanceResult.Score.GetValueOrDefault(), precision: 6);

        VectorStoreCollection<string, IndependentRecord> dotProduct =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity));
        await dotProduct.EnsureCollectionExistsAsync();
        await dotProduct.UpsertAsync(CreateRecord("positive", [2, 0], "x"));
        await dotProduct.UpsertAsync(CreateRecord("zero", [0, 5], "x"));
        await dotProduct.UpsertAsync(CreateRecord("negative", [-2, 0], "x"));

        List<VectorSearchResult<IndependentRecord>> dotResults = await Search(
            dotProduct,
            [1, 0],
            top: 3,
            new VectorSearchOptions<IndependentRecord> { ScoreThreshold = 0 });

        Assert.Equal(["positive", "zero"], dotResults.Select(result => result.Record.Id));
        Assert.Equal([2d, 0d], dotResults.Select(result => result.Score.GetValueOrDefault()));
    }

    [Fact]
    public async Task PreCanceledBatchOperationsAndSearchDoNotMutateCollection()
    {
        VectorStoreCollection<string, IndependentRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "original"));

        using var canceled = new CancellationTokenSource();
        canceled.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            collection.UpsertAsync([CreateRecord("b", [1, 0], "new")], canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            collection.DeleteAsync(["a"], canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
            await ToListAsync(collection.SearchAsync(new float[] { 0, 0 }, top: 1, cancellationToken: canceled.Token)));

        Assert.Equal("original", (await collection.GetAsync("a"))?.Tag);
        List<VectorSearchResult<IndependentRecord>> results = await Search(collection, [0, 0], top: 10);
        VectorSearchResult<IndependentRecord> result = Assert.Single(results);
        Assert.Equal("a", result.Record.Id);
        Assert.Equal(0, result.Score);
    }

    [Fact]
    public async Task UnsupportedVectorSearchPropertySelectorFailsClearly()
    {
        VectorStoreCollection<string, IndependentRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "red"));

        var options = new VectorSearchOptions<IndependentRecord>
        {
            VectorProperty = record => record.Tag
        };

        NotSupportedException exception = await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await Search(collection, [0, 0], top: 1, options));

        Assert.Contains("one vector property", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CollectionDeleteAndRecreateClearsOldKeyMappings()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, IndependentRecord> collection =
            store.GetCollection<string, IndependentRecord>("records", CreateDefinition());
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("old", [0, 0], "before-delete"));

        await collection.EnsureCollectionDeletedAsync();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("new", [1, 0], "after-recreate"));

        Assert.Null(await collection.GetAsync("old"));
        Assert.Equal("new", (await collection.GetAsync("new"))?.Id);

        List<VectorSearchResult<IndependentRecord>> results = await Search(collection, [0, 0], top: 10);
        VectorSearchResult<IndependentRecord> result = Assert.Single(results);
        Assert.Equal("new", result.Record.Id);
        Assert.Equal(1, result.Score);
    }

    private static VectorStoreCollection<string, IndependentRecord> CreateCollection(
        VectorStoreCollectionDefinition? definition = null)
    {
        var store = new VecNetVectorStore();
        return store.GetCollection<string, IndependentRecord>("records", definition ?? CreateDefinition());
    }

    private static VectorStoreCollectionDefinition CreateDefinition(
        string distanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance)
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(IndependentRecord.Id), typeof(string)),
                new VectorStoreVectorProperty(
                    nameof(IndependentRecord.Vector),
                    typeof(ReadOnlyMemory<float>),
                    dimensions: 2)
                {
                    IndexKind = Microsoft.Extensions.VectorData.IndexKind.Flat,
                    DistanceFunction = distanceFunction
                },
                new VectorStoreDataProperty(nameof(IndependentRecord.Tag), typeof(string))
            ]
        };
    }

    private static IndependentRecord CreateRecord(string id, float[] vector, string tag) =>
        new() { Id = id, Vector = vector, Tag = tag };

    private static Task<List<VectorSearchResult<IndependentRecord>>> Search(
        VectorStoreCollection<string, IndependentRecord> collection,
        float[] query,
        int top,
        VectorSearchOptions<IndependentRecord>? options = null) =>
        ToListAsync(collection.SearchAsync(query, top, options));

    private static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
    {
        var results = new List<T>();
        await foreach (T item in source)
        {
            results.Add(item);
        }

        return results;
    }

    private sealed class IndependentRecord
    {
        public string Id { get; init; } = string.Empty;

        public ReadOnlyMemory<float> Vector { get; init; }

        public string Tag { get; init; } = string.Empty;
    }
}
