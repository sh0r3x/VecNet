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
    public async Task SingleAndBatchGetHonorIncludeVectorsForReadOnlyMemoryRecords()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [1, 2], "red"));
        await collection.UpsertAsync(CreateRecord("b", [3, 4], "blue"));

        TestRecord? nullOptions = await collection.GetAsync("a");
        TestRecord? defaultOptions = await collection.GetAsync("a", new RecordRetrievalOptions());
        TestRecord? falseOptions = await collection.GetAsync(
            "a",
            new RecordRetrievalOptions { IncludeVectors = false });
        TestRecord? trueOptions = await collection.GetAsync(
            "a",
            new RecordRetrievalOptions { IncludeVectors = true });

        Assert.NotNull(nullOptions);
        Assert.NotNull(defaultOptions);
        Assert.NotNull(falseOptions);
        Assert.NotNull(trueOptions);
        Assert.True(nullOptions.Vector.IsEmpty);
        Assert.True(defaultOptions.Vector.IsEmpty);
        Assert.True(falseOptions.Vector.IsEmpty);
        Assert.Equal([1f, 2f], trueOptions.Vector.ToArray());

        List<TestRecord> nullBatch = await ToListAsync(collection.GetAsync(["a", "b"]));
        List<TestRecord> defaultBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions()));
        List<TestRecord> omittedBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions { IncludeVectors = false }));
        List<TestRecord> includedBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions { IncludeVectors = true }));

        Assert.All(nullBatch, record => Assert.True(record.Vector.IsEmpty));
        Assert.All(defaultBatch, record => Assert.True(record.Vector.IsEmpty));
        Assert.All(omittedBatch, record => Assert.True(record.Vector.IsEmpty));
        Assert.Equal([1f, 2f], includedBatch[0].Vector.ToArray());
        Assert.Equal([3f, 4f], includedBatch[1].Vector.ToArray());
    }

    [Fact]
    public async Task FilteredGetHonorsIncludeVectorsForReadOnlyMemoryRecords()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [1, 2], "red"));
        await collection.UpsertAsync(CreateRecord("b", [3, 4], "blue"));

        List<TestRecord> nullOptions = await ToListAsync(collection.GetAsync(record => record.Tag == "red", top: 1));
        List<TestRecord> defaultOptions = await ToListAsync(
            collection.GetAsync(record => record.Tag == "red", top: 1, new FilteredRecordRetrievalOptions<TestRecord>()));
        List<TestRecord> falseOptions = await ToListAsync(
            collection.GetAsync(
                record => record.Tag == "red",
                top: 1,
                new FilteredRecordRetrievalOptions<TestRecord> { IncludeVectors = false }));
        List<TestRecord> trueOptions = await ToListAsync(
            collection.GetAsync(
                record => record.Tag == "red",
                top: 1,
                new FilteredRecordRetrievalOptions<TestRecord> { IncludeVectors = true }));

        Assert.True(Assert.Single(nullOptions).Vector.IsEmpty);
        Assert.True(Assert.Single(defaultOptions).Vector.IsEmpty);
        Assert.True(Assert.Single(falseOptions).Vector.IsEmpty);
        Assert.Equal([1f, 2f], Assert.Single(trueOptions).Vector.ToArray());
    }

    [Fact]
    public async Task SearchHonorsIncludeVectorsForReadOnlyMemoryRecords()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(CreateRecord("a", [1, 0], "red"));
        await collection.UpsertAsync(CreateRecord("b", [2, 0], "blue"));

        List<VectorSearchResult<TestRecord>> nullOptions = await Search(collection, [0, 0], top: 1);
        List<VectorSearchResult<TestRecord>> defaultOptions = await Search(
            collection,
            [0, 0],
            top: 1,
            new VectorSearchOptions<TestRecord>());
        List<VectorSearchResult<TestRecord>> falseOptions = await Search(
            collection,
            [0, 0],
            top: 1,
            new VectorSearchOptions<TestRecord> { IncludeVectors = false });
        List<VectorSearchResult<TestRecord>> trueOptions = await Search(
            collection,
            [0, 0],
            top: 1,
            new VectorSearchOptions<TestRecord> { IncludeVectors = true });

        Assert.True(Assert.Single(nullOptions).Record.Vector.IsEmpty);
        Assert.True(Assert.Single(defaultOptions).Record.Vector.IsEmpty);
        Assert.True(Assert.Single(falseOptions).Record.Vector.IsEmpty);
        Assert.Equal([1f, 0f], Assert.Single(trueOptions).Record.Vector.ToArray());
    }

    [Fact]
    public async Task GetAndSearchHonorIncludeVectorsForFloatArrayRecords()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, AttributeRecord> collection =
            store.GetCollection<string, AttributeRecord>("attributes");
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new AttributeRecord { Id = "a", Vector = [1, 0], Tag = "red" });
        await collection.UpsertAsync(new AttributeRecord { Id = "b", Vector = [2, 0], Tag = "blue" });

        AttributeRecord? nullOptionsGet = await collection.GetAsync("a");
        AttributeRecord? defaultOptionsGet = await collection.GetAsync("a", new RecordRetrievalOptions());
        AttributeRecord? omittedGet = await collection.GetAsync(
            "a",
            new RecordRetrievalOptions { IncludeVectors = false });
        AttributeRecord? includedGet = await collection.GetAsync(
            "a",
            new RecordRetrievalOptions { IncludeVectors = true });
        List<AttributeRecord> nullOptionsBatch = await ToListAsync(collection.GetAsync(["a", "b"]));
        List<AttributeRecord> defaultOptionsBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions()));
        List<AttributeRecord> omittedBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions { IncludeVectors = false }));
        List<AttributeRecord> includedBatch = await ToListAsync(
            collection.GetAsync(["a", "b"], new RecordRetrievalOptions { IncludeVectors = true }));
        List<AttributeRecord> nullOptionsFiltered = await ToListAsync(
            collection.GetAsync(record => record.Tag == "red", top: 1));
        List<AttributeRecord> defaultOptionsFiltered = await ToListAsync(
            collection.GetAsync(
                record => record.Tag == "red",
                top: 1,
                new FilteredRecordRetrievalOptions<AttributeRecord>()));
        List<AttributeRecord> omittedFiltered = await ToListAsync(
            collection.GetAsync(
                record => record.Tag == "red",
                top: 1,
                new FilteredRecordRetrievalOptions<AttributeRecord> { IncludeVectors = false }));
        List<AttributeRecord> includedFiltered = await ToListAsync(
            collection.GetAsync(
                record => record.Tag == "red",
                top: 1,
                new FilteredRecordRetrievalOptions<AttributeRecord> { IncludeVectors = true }));
        List<VectorSearchResult<AttributeRecord>> nullOptionsSearch = await ToListAsync(
            collection.SearchAsync(new float[] { 0, 0 }, top: 1));
        List<VectorSearchResult<AttributeRecord>> defaultOptionsSearch = await ToListAsync(
            collection.SearchAsync(
                new float[] { 0, 0 },
                top: 1,
                new VectorSearchOptions<AttributeRecord>()));
        List<VectorSearchResult<AttributeRecord>> omittedSearch = await ToListAsync(
            collection.SearchAsync(
                new float[] { 0, 0 },
                top: 1,
                new VectorSearchOptions<AttributeRecord> { IncludeVectors = false }));
        List<VectorSearchResult<AttributeRecord>> includedSearch = await ToListAsync(
            collection.SearchAsync(
                new float[] { 0, 0 },
                top: 1,
                new VectorSearchOptions<AttributeRecord> { IncludeVectors = true }));

        Assert.NotNull(nullOptionsGet);
        Assert.NotNull(defaultOptionsGet);
        Assert.NotNull(omittedGet);
        Assert.NotNull(includedGet);
        Assert.Null(nullOptionsGet.Vector);
        Assert.Null(defaultOptionsGet.Vector);
        Assert.Null(omittedGet.Vector);
        Assert.Equal([1f, 0f], includedGet.Vector);
        Assert.All(nullOptionsBatch, record => Assert.Null(record.Vector));
        Assert.All(defaultOptionsBatch, record => Assert.Null(record.Vector));
        Assert.All(omittedBatch, record => Assert.Null(record.Vector));
        Assert.Equal([1f, 0f], includedBatch[0].Vector);
        Assert.Equal([2f, 0f], includedBatch[1].Vector);
        Assert.Null(Assert.Single(nullOptionsFiltered).Vector);
        Assert.Null(Assert.Single(defaultOptionsFiltered).Vector);
        Assert.Null(Assert.Single(omittedFiltered).Vector);
        Assert.Equal([1f, 0f], Assert.Single(includedFiltered).Vector);
        Assert.Null(Assert.Single(nullOptionsSearch).Record.Vector);
        Assert.Null(Assert.Single(defaultOptionsSearch).Record.Vector);
        Assert.Null(Assert.Single(omittedSearch).Record.Vector);
        Assert.Equal([1f, 0f], Assert.Single(includedSearch).Record.Vector);
    }

    [Fact]
    public async Task UnsupportedProjectionShapesFailOnlyWhenVectorsAreOmitted()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, ConstructorOnlyRecord> collection =
            store.GetCollection<string, ConstructorOnlyRecord>("constructor-only");
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new ConstructorOnlyRecord("a", [1, 0], "red"));

        ConstructorOnlyRecord? includedGet = await collection.GetAsync(
            "a",
            new RecordRetrievalOptions { IncludeVectors = true });
        List<VectorSearchResult<ConstructorOnlyRecord>> includedSearch = await ToListAsync(
            collection.SearchAsync(
                new float[] { 0, 0 },
                top: 1,
                new VectorSearchOptions<ConstructorOnlyRecord> { IncludeVectors = true }));

        Assert.NotNull(includedGet);
        Assert.Equal([1f, 0f], includedGet.Vector);
        Assert.Equal([1f, 0f], Assert.Single(includedSearch).Record.Vector);

        NotSupportedException defaultGet = await Assert.ThrowsAsync<NotSupportedException>(() =>
            collection.GetAsync("a"));
        NotSupportedException omittedGet = await Assert.ThrowsAsync<NotSupportedException>(() =>
            collection.GetAsync("a", new RecordRetrievalOptions { IncludeVectors = false }));
        NotSupportedException omittedSearch = await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await ToListAsync(
                collection.SearchAsync(
                    new float[] { 0, 0 },
                    top: 1,
                    new VectorSearchOptions<ConstructorOnlyRecord> { IncludeVectors = false })));

        Assert.Contains("cannot omit vectors", defaultGet.Message, StringComparison.Ordinal);
        Assert.Contains("cannot omit vectors", omittedGet.Message, StringComparison.Ordinal);
        Assert.Contains("cannot omit vectors", omittedSearch.Message, StringComparison.Ordinal);
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

    private sealed class ConstructorOnlyRecord(string id, float[] vector, string tag)
    {
        [VectorStoreKey]
        public string Id { get; } = id;

        [VectorStoreVector(2, IndexKind = Microsoft.Extensions.VectorData.IndexKind.Flat,
            DistanceFunction = Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance)]
        public float[] Vector { get; } = vector;

        [VectorStoreData]
        public string Tag { get; } = tag;
    }

    private sealed class MultiVectorRecord
    {
        public string Id { get; init; } = string.Empty;

        public ReadOnlyMemory<float> FirstVector { get; init; }

        public ReadOnlyMemory<float> SecondVector { get; init; }
    }
}
