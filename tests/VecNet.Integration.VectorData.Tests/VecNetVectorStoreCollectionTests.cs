using Microsoft.Extensions.VectorData;
using VecNet.Integration.VectorData;

namespace VecNet.Integration.VectorData.Tests;

public sealed class VecNetVectorStoreCollectionTests
{
    [Fact]
    public async Task CollectionLifecycleCreatesListsAndDeletesAdapterState()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, TestRecord> collection =
            store.GetCollection<string, TestRecord>("records", CreateDefinition());

        Assert.False(await collection.CollectionExistsAsync());
        Assert.False(await store.CollectionExistsAsync("records"));

        await collection.EnsureCollectionExistsAsync();

        Assert.True(await collection.CollectionExistsAsync());
        Assert.True(await store.CollectionExistsAsync("records"));
        Assert.Equal(["records"], await ToListAsync(store.ListCollectionNamesAsync()));

        await collection.UpsertAsync(CreateRecord("a", [1, 2], "red"));
        Assert.NotNull(await collection.GetAsync("a"));

        await store.EnsureCollectionDeletedAsync("records");

        Assert.False(await collection.CollectionExistsAsync());
        await Assert.ThrowsAsync<VectorStoreException>(() => collection.GetAsync("a"));
    }

    [Fact]
    public async Task UpsertCreateAndUpdateUsesCurrentRecordAndReplacementVector()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();

        await collection.UpsertAsync(CreateRecord("a", [10, 0], "old"));
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "new"));
        await collection.UpsertAsync(CreateRecord("b", [4, 0], "other"));

        TestRecord? record = await collection.GetAsync("a");
        Assert.NotNull(record);
        Assert.Equal("new", record.Tag);

        List<VectorSearchResult<TestRecord>> results = await Search(collection, [0, 0], top: 2);

        Assert.Equal(["a", "b"], results.Select(result => result.Record.Id));
        Assert.Equal(0, results[0].Score);
        Assert.Equal(16, results[1].Score);
    }

    [Fact]
    public async Task FailedReplacementUpsertKeepsPreviousCurrentRecordAndVectorSearchable()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();

        await collection.UpsertAsync(CreateRecord("a", [0, 0], "original"));

        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await collection.UpsertAsync(CreateRecord("a", [float.NaN, 0], "invalid")));

        TestRecord? record = await collection.GetAsync("a");
        Assert.NotNull(record);
        Assert.Equal("original", record.Tag);

        List<VectorSearchResult<TestRecord>> preservedResults = await Search(collection, [0, 0], top: 5);
        VectorSearchResult<TestRecord> preservedResult = Assert.Single(preservedResults);
        Assert.Equal("a", preservedResult.Record.Id);
        Assert.Equal(0, preservedResult.Score);

        await collection.UpsertAsync(CreateRecord("b", [2, 0], "other"));
        List<VectorSearchResult<TestRecord>> allResults = await Search(collection, [0, 0], top: 5);

        Assert.Equal(["a", "b"], allResults.Select(result => result.Record.Id));
        Assert.Equal(2, allResults.Count);
    }

    [Fact]
    public async Task DeleteRemovesCurrentMappingAndNotFoundDeleteSucceeds()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "red"));
        await collection.UpsertAsync(CreateRecord("b", [1, 0], "blue"));

        await collection.DeleteAsync("a");
        await collection.DeleteAsync("missing");

        Assert.Null(await collection.GetAsync("a"));
        Assert.Equal("b", Assert.Single(await ToListAsync(collection.GetAsync(["a", "b"]))).Id);

        List<VectorSearchResult<TestRecord>> results = await Search(collection, [0, 0], top: 5);
        Assert.Equal("b", Assert.Single(results).Record.Id);
    }

    [Fact]
    public async Task SearchSupportsTopSkipUnderfillThresholdAndFilterAllowlist()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "red"));
        await collection.UpsertAsync(CreateRecord("b", [1, 0], "blue"));
        await collection.UpsertAsync(CreateRecord("c", [2, 0], "blue"));

        var options = new VectorSearchOptions<TestRecord>
        {
            Skip = 1,
            ScoreThreshold = 4,
            Filter = record => record.Tag == "blue"
        };

        List<VectorSearchResult<TestRecord>> results = await Search(collection, [0, 0], top: 5, options);

        Assert.Equal("c", Assert.Single(results).Record.Id);
        Assert.Equal(4, results[0].Score);
    }

    [Fact]
    public async Task EuclideanDistanceProjectsSquareRootScoreAndDistanceThreshold()
    {
        VectorStoreCollection<string, TestRecord> collection =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.EuclideanDistance));
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [3, 4], "far"));
        await collection.UpsertAsync(CreateRecord("b", [0, 2], "near"));

        List<VectorSearchResult<TestRecord>> results = await Search(
            collection,
            [0, 0],
            top: 2,
            new VectorSearchOptions<TestRecord> { ScoreThreshold = 2 });

        VectorSearchResult<TestRecord> result = Assert.Single(results);
        Assert.Equal("b", result.Record.Id);
        Assert.Equal(2, result.Score);
    }

    [Fact]
    public async Task CosineDistanceAndSimilarityProjectScores()
    {
        VectorStoreCollection<string, TestRecord> distanceCollection =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance));
        await distanceCollection.EnsureCollectionExistsAsync();
        await distanceCollection.UpsertAsync(CreateRecord("same", [1, 0], "x"));
        await distanceCollection.UpsertAsync(CreateRecord("orthogonal", [0, 1], "x"));

        List<VectorSearchResult<TestRecord>> distanceResults = await Search(distanceCollection, [1, 0], top: 2);
        Assert.Equal(["same", "orthogonal"], distanceResults.Select(result => result.Record.Id));
        Assert.Equal(0, distanceResults[0].Score.GetValueOrDefault(), precision: 6);
        Assert.Equal(1, distanceResults[1].Score.GetValueOrDefault(), precision: 6);

        VectorStoreCollection<string, TestRecord> similarityCollection =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity));
        await similarityCollection.EnsureCollectionExistsAsync();
        await similarityCollection.UpsertAsync(CreateRecord("same", [1, 0], "x"));
        await similarityCollection.UpsertAsync(CreateRecord("orthogonal", [0, 1], "x"));

        List<VectorSearchResult<TestRecord>> similarityResults = await Search(
            similarityCollection,
            [1, 0],
            top: 2,
            new VectorSearchOptions<TestRecord> { ScoreThreshold = 0.5 });

        VectorSearchResult<TestRecord> similarityResult = Assert.Single(similarityResults);
        Assert.Equal("same", similarityResult.Record.Id);
        Assert.Equal(1, similarityResult.Score.GetValueOrDefault(), precision: 6);
    }

    [Fact]
    public async Task DotProductSimilarityProjectsScores()
    {
        VectorStoreCollection<string, TestRecord> collection =
            CreateCollection(CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity));
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("large", [3, 0], "x"));
        await collection.UpsertAsync(CreateRecord("small", [1, 0], "x"));

        List<VectorSearchResult<TestRecord>> results = await Search(
            collection,
            [2, 0],
            top: 2,
            new VectorSearchOptions<TestRecord> { ScoreThreshold = 3 });

        VectorSearchResult<TestRecord> result = Assert.Single(results);
        Assert.Equal("large", result.Record.Id);
        Assert.Equal(6, result.Score);
    }

    [Fact]
    public async Task AttributeSchemaSupportsFloatArrayVectors()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, AttributeRecord> collection =
            store.GetCollection<string, AttributeRecord>("attributes");
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new AttributeRecord { Id = "a", Vector = [0, 0], Tag = "red" });
        await collection.UpsertAsync(new AttributeRecord { Id = "b", Vector = [2, 0], Tag = "blue" });

        List<VectorSearchResult<AttributeRecord>> results = await ToListAsync(
            collection.SearchAsync(new float[] { 0, 0 }, top: 1));

        Assert.Equal("a", Assert.Single(results).Record.Id);
    }

    [Fact]
    public async Task UnsupportedFeaturesThrowClearExceptions()
    {
        var store = new VecNetVectorStore();

        Assert.Throws<NotSupportedException>(() =>
            store.GetCollection<string, TestRecord>(
                "hnsw",
                CreateDefinition(indexKind: Microsoft.Extensions.VectorData.IndexKind.Hnsw)));

        Assert.Throws<NotSupportedException>(() =>
            store.GetCollection<string, TestRecord>(
                "negative-dot",
                CreateDefinition(Microsoft.Extensions.VectorData.DistanceFunction.NegativeDotProductSimilarity)));

        Assert.Throws<NotSupportedException>(() =>
            store.GetCollection<string, MultiVectorRecord>("multi", CreateMultiVectorDefinition()));

        await Assert.ThrowsAsync<NotSupportedException>(async () =>
        {
            VectorStoreCollection<string, TestRecord> collection = CreateCollection();
            await collection.EnsureCollectionExistsAsync();
            await collection.UpsertAsync(CreateRecord("a", [0, 0], "red"));
            await ToListAsync(collection.SearchAsync("embedding text", top: 1));
        });

        Assert.Throws<NotSupportedException>(() =>
            store.GetDynamicCollection("dynamic", CreateDefinition()));
    }

    [Fact]
    public async Task CancellationIsObservedBeforeOperationsAndDuringEnumeration()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [0, 0], "red"));

        using var beforeOperation = new CancellationTokenSource();
        beforeOperation.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            collection.UpsertAsync(CreateRecord("b", [1, 0], "blue"), beforeOperation.Token));

        using var duringEnumeration = new CancellationTokenSource();
        IAsyncEnumerable<VectorSearchResult<TestRecord>> results =
            collection.SearchAsync(new float[] { 0, 0 }, top: 1, cancellationToken: duringEnumeration.Token);
        duringEnumeration.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (VectorSearchResult<TestRecord> _ in results)
            {
            }
        });
    }

    [Fact]
    public void CoreProjectRemainsFreeOfPackageReferencesAndVectorDataDependency()
    {
        string root = FindRepositoryRoot();
        string coreProject = File.ReadAllText(Path.Combine(root, "src", "VecNet", "VecNet.csproj"));

        Assert.DoesNotContain("<PackageReference", coreProject, StringComparison.Ordinal);
        Assert.DoesNotContain("VectorData", coreProject, StringComparison.Ordinal);
    }

    private static VectorStoreCollection<string, TestRecord> CreateCollection(
        VectorStoreCollectionDefinition? definition = null)
    {
        var store = new VecNetVectorStore();
        return store.GetCollection<string, TestRecord>("records", definition ?? CreateDefinition());
    }

    private static VectorStoreCollectionDefinition CreateDefinition(
        string distanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance,
        string indexKind = Microsoft.Extensions.VectorData.IndexKind.Flat)
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(TestRecord.Id), typeof(string)),
                new VectorStoreVectorProperty(
                    nameof(TestRecord.Vector),
                    typeof(ReadOnlyMemory<float>),
                    dimensions: 2)
                {
                    IndexKind = indexKind,
                    DistanceFunction = distanceFunction
                },
                new VectorStoreDataProperty(nameof(TestRecord.Tag), typeof(string))
            ]
        };
    }

    private static VectorStoreCollectionDefinition CreateMultiVectorDefinition()
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(MultiVectorRecord.Id), typeof(string)),
                new VectorStoreVectorProperty(
                    nameof(MultiVectorRecord.FirstVector),
                    typeof(ReadOnlyMemory<float>),
                    dimensions: 2),
                new VectorStoreVectorProperty(
                    nameof(MultiVectorRecord.SecondVector),
                    typeof(ReadOnlyMemory<float>),
                    dimensions: 2)
            ]
        };
    }

    private static TestRecord CreateRecord(string id, float[] vector, string tag) =>
        new() { Id = id, Vector = vector, Tag = tag };

    private static Task<List<VectorSearchResult<TestRecord>>> Search(
        VectorStoreCollection<string, TestRecord> collection,
        float[] query,
        int top,
        VectorSearchOptions<TestRecord>? options = null) =>
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

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "VecNet.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Repository root was not found.");
    }

    private sealed class TestRecord
    {
        public string Id { get; init; } = string.Empty;

        public ReadOnlyMemory<float> Vector { get; init; }

        public string Tag { get; init; } = string.Empty;
    }

    private sealed class AttributeRecord
    {
        [VectorStoreKey]
        public string Id { get; init; } = string.Empty;

        [VectorStoreVector(2, IndexKind = Microsoft.Extensions.VectorData.IndexKind.Flat,
            DistanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance)]
        public float[] Vector { get; init; } = [];

        [VectorStoreData]
        public string Tag { get; init; } = string.Empty;
    }

    private sealed class MultiVectorRecord
    {
        public string Id { get; init; } = string.Empty;

        public ReadOnlyMemory<float> FirstVector { get; init; }

        public ReadOnlyMemory<float> SecondVector { get; init; }
    }
}
