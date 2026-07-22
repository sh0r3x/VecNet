using Microsoft.Extensions.VectorData;
using VecNet.Integration.VectorData;

namespace VecNet.Integration.VectorData.Tests;

public sealed class VecNetVectorStoreConformanceStyleTests
{
    [Fact]
    public async Task MissingRecordBehaviorOmitsUnknownKeysAndDeletedRecords()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 1 });
        await collection.UpsertAsync(new TestRecord { Id = "b", Vector = [1, 0], Tag = "blue", Sort = 2 });
        await collection.DeleteAsync("a");

        Assert.Null(await collection.GetAsync("missing"));
        Assert.Null(await collection.GetAsync("a"));
        Assert.Equal(["b"], (await ToListAsync(collection.GetAsync(["missing", "a", "b"]))).Select(record => record.Id));
        Assert.Equal(["b"], (await Search(collection, [0, 0], top: 10)).Select(result => result.Record.Id));
    }

    [Fact]
    public async Task BatchUpsertCreatesUpdatesAndCommitsSequentiallyBeforeInvalidItem()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new TestRecord { Id = "a", Vector = [5, 0], Tag = "old", Sort = 1 });

        await Assert.ThrowsAsync<ArgumentException>(() =>
            collection.UpsertAsync(
            [
                new TestRecord { Id = "a", Vector = [0, 0], Tag = "updated", Sort = 2 },
                new TestRecord { Id = "b", Vector = [1, 0], Tag = "created", Sort = 3 },
                new TestRecord { Id = "c", Vector = [float.NaN, 0], Tag = "invalid", Sort = 4 },
                new TestRecord { Id = "d", Vector = [2, 0], Tag = "not-processed", Sort = 5 }
            ]));

        Assert.Equal("updated", (await collection.GetAsync("a"))?.Tag);
        Assert.Equal("created", (await collection.GetAsync("b"))?.Tag);
        Assert.Null(await collection.GetAsync("c"));
        Assert.Null(await collection.GetAsync("d"));
        Assert.Equal(["a", "b"], (await Search(collection, [0, 0], top: 10)).Select(result => result.Record.Id));
    }

    [Fact]
    public async Task BatchDeleteIsIdempotentAndToleratesMissingAndDuplicateKeys()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(
        [
            new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 1 },
            new TestRecord { Id = "b", Vector = [1, 0], Tag = "blue", Sort = 2 },
            new TestRecord { Id = "c", Vector = [2, 0], Tag = "green", Sort = 3 }
        ]);

        await collection.DeleteAsync(["a", "missing", "a", "c"]);
        await collection.DeleteAsync(["a", "c"]);

        Assert.Equal(["b"], (await ToListAsync(collection.GetAsync(["a", "b", "c"]))).Select(record => record.Id));
        Assert.Equal(["b"], (await Search(collection, [0, 0], top: 10)).Select(result => result.Record.Id));
    }

    [Fact]
    public async Task CollectionLifecycleListsNamesAndSharesSameTypedHandles()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, TestRecord> alpha =
            store.GetCollection<string, TestRecord>("alpha", CreateDefinition<string, TestRecord>());
        VectorStoreCollection<string, TestRecord> beta =
            store.GetCollection<string, TestRecord>("beta", CreateDefinition<string, TestRecord>());
        VectorStoreCollection<string, TestRecord> alphaAgain =
            store.GetCollection<string, TestRecord>("alpha", CreateDefinition<string, TestRecord>());

        await alpha.EnsureCollectionExistsAsync();
        await alpha.EnsureCollectionExistsAsync();
        await beta.EnsureCollectionExistsAsync();
        await alpha.UpsertAsync(new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 1 });

        Assert.Equal("a", (await alphaAgain.GetAsync("a"))?.Id);
        Assert.Equal(["alpha", "beta"], await ToListAsync(store.ListCollectionNamesAsync()));

        await store.EnsureCollectionDeletedAsync("alpha");
        await store.EnsureCollectionDeletedAsync("alpha");
        Assert.False(await alpha.CollectionExistsAsync());
        Assert.Equal(["beta"], await ToListAsync(store.ListCollectionNamesAsync()));

        await alpha.EnsureCollectionExistsAsync();
        Assert.Null(await alpha.GetAsync("a"));
        await Assert.ThrowsAsync<VectorStoreException>(() =>
            Task.FromResult(store.GetCollection<Guid, GuidKeyRecord>("alpha", CreateDefinition<Guid, GuidKeyRecord>())));
    }

    [Theory]
    [MemberData(nameof(SupportedKeys))]
    public async Task SupportedKeyTypesRoundTrip<TKey>(TKey key)
        where TKey : notnull
    {
        VectorStoreCollection<TKey, KeyRecord<TKey>> collection = CreateCollection<TKey, KeyRecord<TKey>>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new KeyRecord<TKey> { Id = key, Vector = [0, 0], Tag = "value" });

        KeyRecord<TKey>? record = await collection.GetAsync(
            key,
            new RecordRetrievalOptions { IncludeVectors = true });

        Assert.NotNull(record);
        Assert.Equal(key, record.Id);
        Assert.Equal([0f, 0f], record.Vector.ToArray());
    }

    [Fact]
    public async Task NullAndInvalidInputsThrowClearExceptions()
    {
        var store = new VecNetVectorStore();
        Assert.Throws<ArgumentException>(() => store.GetCollection<string, TestRecord>(" ", CreateDefinition<string, TestRecord>()));
        await Assert.ThrowsAsync<ArgumentException>(() => store.CollectionExistsAsync(""));

        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 1 });

        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.UpsertAsync(record: null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.UpsertAsync(records: null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.DeleteAsync(keys: null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.GetAsync(keys: null!).ToListAsync().AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.GetAsync(filter: null!, top: 1).ToListAsync().AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => collection.SearchAsync<float[]>(null!, top: 1).ToListAsync().AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => Task.FromResult(collection.GetService(null!)));

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => collection.GetAsync(record => true, top: -1).ToListAsync().AsTask());
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => collection.SearchAsync(new float[] { 0, 0 }, top: -1).ToListAsync().AsTask());
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => collection.SearchAsync(
            new float[] { 0, 0 },
            top: 1,
            new VectorSearchOptions<TestRecord> { Skip = -1 }).ToListAsync().AsTask());
        Assert.Empty(await ToListAsync(collection.SearchAsync(new float[] { 0, 0 }, top: 0)));

        await Assert.ThrowsAsync<ArgumentException>(() => collection.UpsertAsync(new TestRecord { Id = "bad", Vector = [0], Tag = "bad" }));
        await Assert.ThrowsAsync<ArgumentException>(() => collection.UpsertAsync(new TestRecord { Id = "bad", Vector = [float.PositiveInfinity, 0], Tag = "bad" }));
        await Assert.ThrowsAsync<ArgumentException>(() => collection.SearchAsync(new float[] { 0 }, top: 1).ToListAsync().AsTask());
        await Assert.ThrowsAsync<NotSupportedException>(() => collection.SearchAsync(
            new float[] { 0, 0 },
            top: 1,
            new VectorSearchOptions<TestRecord> { VectorProperty = record => record.Tag }).ToListAsync().AsTask());

        VectorStoreCollection<string, NullableVectorRecord> nullableVector =
            store.GetCollection<string, NullableVectorRecord>("nullable-vector", CreateNullableVectorDefinition());
        await nullableVector.EnsureCollectionExistsAsync();
        await Assert.ThrowsAsync<VectorStoreException>(() =>
            nullableVector.UpsertAsync(new NullableVectorRecord { Id = "null-vector", Vector = null }));

        VectorStoreCollection<string, NullableKeyRecord> nullableKey =
            store.GetCollection<string, NullableKeyRecord>("nullable-key", CreateNullableKeyDefinition());
        await nullableKey.EnsureCollectionExistsAsync();
        await Assert.ThrowsAsync<VectorStoreException>(() =>
            nullableKey.UpsertAsync(new NullableKeyRecord { Id = null, Vector = [0, 0] }));

        Assert.Throws<NotSupportedException>(() =>
            store.GetCollection<string, TestRecord>("auto-key", CreateDefinition<string, TestRecord>(autoGeneratedKey: true)));
    }

    [Fact]
    public async Task SupportedFiltersAndOrderByUseInMemoryRecordSemantics()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(
        [
            new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 2 },
            new TestRecord { Id = "b", Vector = [1, 0], Tag = "blue", Sort = 1 },
            new TestRecord { Id = "c", Vector = [2, 0], Tag = "blue", Sort = 3 }
        ]);

        string captured = "blue";
        List<TestRecord> insertionOrder = await ToListAsync(collection.GetAsync(record => record.Sort > 0, top: 3));
        List<TestRecord> ordered = await ToListAsync(collection.GetAsync(
            record => record.Tag == captured && record.Sort >= 1,
            top: 2,
            new FilteredRecordRetrievalOptions<TestRecord>
            {
                Skip = 0,
                OrderBy = order => order.Descending(record => record.Sort)
            }));

        Assert.Equal(["a", "b", "c"], insertionOrder.Select(record => record.Id));
        Assert.Equal(["c", "b"], ordered.Select(record => record.Id));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public async Task InvalidOrderBySelectorIsRejectedForAnyFilteredCardinality(int matchingRecordCount)
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new TestRecord { Id = "other", Vector = [9, 0], Tag = "other", Sort = 9 });

        for (int i = 0; i < matchingRecordCount; i++)
        {
            await collection.UpsertAsync(new TestRecord
            {
                Id = $"match-{i}",
                Vector = [i, 0],
                Tag = "match",
                Sort = i
            });
        }

        NotSupportedException exception = await Assert.ThrowsAsync<NotSupportedException>(() =>
            collection.GetAsync(
                record => record.Tag == "match",
                top: 10,
                new FilteredRecordRetrievalOptions<TestRecord>
                {
                    OrderBy = order => order.Ascending(record => record.Sort + 1)
                }).ToListAsync().AsTask());

        Assert.Contains("OrderBy", exception.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("VectorSearchOptions.VectorProperty", exception.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(DistanceFunction.EuclideanSquaredDistance, 1.0, "near")]
    [InlineData(DistanceFunction.EuclideanDistance, 1.0, "near")]
    [InlineData(DistanceFunction.CosineDistance, 0.5, "same")]
    [InlineData(DistanceFunction.CosineSimilarity, 0.5, "same")]
    [InlineData(DistanceFunction.DotProductSimilarity, 1.0, "large")]
    public async Task ScoreAndThresholdDirectionMatchesDistanceFunction(
        string distanceFunction,
        double threshold,
        string expectedFirst)
    {
        VectorStoreCollection<string, TestRecord> collection =
            CreateCollection<string, TestRecord>(CreateDefinition<string, TestRecord>(distanceFunction: distanceFunction));
        await collection.EnsureCollectionExistsAsync();

        if (distanceFunction is DistanceFunction.CosineDistance or DistanceFunction.CosineSimilarity)
        {
            await collection.UpsertAsync(new TestRecord { Id = "same", Vector = [1, 0], Tag = "x" });
            await collection.UpsertAsync(new TestRecord { Id = "orthogonal", Vector = [0, 1], Tag = "x" });
        }
        else if (distanceFunction == DistanceFunction.DotProductSimilarity)
        {
            await collection.UpsertAsync(new TestRecord { Id = "large", Vector = [2, 0], Tag = "x" });
            await collection.UpsertAsync(new TestRecord { Id = "small", Vector = [0.25f, 0], Tag = "x" });
        }
        else
        {
            await collection.UpsertAsync(new TestRecord { Id = "near", Vector = [1, 0], Tag = "x" });
            await collection.UpsertAsync(new TestRecord { Id = "far", Vector = [2, 0], Tag = "x" });
        }

        float[] query = distanceFunction is DistanceFunction.CosineDistance or DistanceFunction.CosineSimilarity or DistanceFunction.DotProductSimilarity
            ? [1, 0]
            : [0, 0];

        List<VectorSearchResult<TestRecord>> results = await Search(
            collection,
            query,
            top: 2,
            new VectorSearchOptions<TestRecord> { ScoreThreshold = threshold });

        Assert.Equal(expectedFirst, Assert.Single(results).Record.Id);
    }

    [Fact]
    public async Task CancellationIsObservedAcrossStoreCollectionAndBatchOperations()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, TestRecord> collection =
            store.GetCollection<string, TestRecord>("records", CreateDefinition<string, TestRecord>());

        using var canceled = new CancellationTokenSource();
        canceled.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.EnsureCollectionExistsAsync(canceled.Token));
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(new TestRecord { Id = "a", Vector = [0, 0], Tag = "red", Sort = 1 });

        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.GetAsync("a", cancellationToken: canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.GetAsync(["a"], cancellationToken: canceled.Token).ToListAsync().AsTask());
        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.GetAsync(record => true, top: 1, cancellationToken: canceled.Token).ToListAsync().AsTask());
        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.DeleteAsync("a", canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(() => store.CollectionExistsAsync("records", canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(() => store.EnsureCollectionDeletedAsync("records", canceled.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(() => store.ListCollectionNamesAsync(canceled.Token).ToListAsync().AsTask());

        var source = new CancelAfterFirstRecordEnumerable(canceledAfterFirst: new CancellationTokenSource());
        await Assert.ThrowsAsync<OperationCanceledException>(() => collection.UpsertAsync(source.Records, source.CanceledAfterFirst.Token));
        Assert.Equal("first", (await collection.GetAsync("first"))?.Id);
        Assert.Null(await collection.GetAsync("second"));
    }

    [Fact]
    public async Task BatchGetAndSearchAsyncEnumerationUseDocumentedOrdering()
    {
        VectorStoreCollection<string, TestRecord> collection = CreateCollection<string, TestRecord>();
        await collection.EnsureCollectionExistsAsync();
        await collection.UpsertAsync(
        [
            new TestRecord { Id = "a", Vector = [2, 0], Tag = "x", Sort = 1 },
            new TestRecord { Id = "b", Vector = [0, 0], Tag = "x", Sort = 2 },
            new TestRecord { Id = "c", Vector = [1, 0], Tag = "x", Sort = 3 }
        ]);

        Assert.Equal(["c", "a", "b"], (await ToListAsync(collection.GetAsync(["c", "missing", "a", "b"]))).Select(record => record.Id));
        Assert.Equal(["b", "c", "a"], (await Search(collection, [0, 0], top: 3)).Select(result => result.Record.Id));
    }

    [Fact]
    public async Task ServiceLookupReturnsAdapterMetadataAndHonorsServiceKey()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, TestRecord> collection =
            store.GetCollection<string, TestRecord>("records", CreateDefinition<string, TestRecord>());

        Assert.Same(store, store.GetService(typeof(VecNetVectorStore)));
        Assert.Null(store.GetService(typeof(VecNetVectorStore), serviceKey: "keyed"));
        var storeMetadata = Assert.IsType<VectorStoreMetadata>(store.GetService(typeof(VectorStoreMetadata)));
        Assert.Equal("vecnet", storeMetadata.VectorStoreSystemName);

        Assert.Same(collection, collection.GetService(collection.GetType()));
        Assert.Null(collection.GetService(collection.GetType(), serviceKey: "keyed"));
        var collectionMetadata = Assert.IsType<VectorStoreCollectionMetadata>(
            collection.GetService(typeof(VectorStoreCollectionMetadata)));
        Assert.Equal("vecnet", collectionMetadata.VectorStoreSystemName);
        Assert.Equal("records", collectionMetadata.CollectionName);
    }

    [Fact]
    public async Task ReadOnlyOperationsMayOverlapOnExistingCollection()
    {
        var store = new VecNetVectorStore();
        VectorStoreCollection<string, TestRecord> collection =
            store.GetCollection<string, TestRecord>("records", CreateDefinition<string, TestRecord>());
        await collection.EnsureCollectionExistsAsync();

        for (int i = 0; i < 20; i++)
        {
            await collection.UpsertAsync(new TestRecord { Id = i.ToString(), Vector = [i, 0], Tag = "x", Sort = i });
        }

        Task[] readers = Enumerable.Range(0, 8)
            .Select(async _ =>
            {
                Assert.True(await collection.CollectionExistsAsync());
                Assert.Contains("records", await ToListAsync(store.ListCollectionNamesAsync()));
                Assert.NotNull(await collection.GetAsync("0"));
                Assert.NotEmpty(await Search(collection, [0, 0], top: 5));
            })
            .ToArray();

        await Task.WhenAll(readers);
    }

    public static TheoryData<object> SupportedKeys =>
    [
        "string-key",
        Guid.Parse("00000000-0000-0000-0000-000000000123"),
        123,
        123L,
        123UL
    ];

    private static VectorStoreCollection<TKey, TRecord> CreateCollection<TKey, TRecord>(
        VectorStoreCollectionDefinition? definition = null)
        where TKey : notnull
        where TRecord : class
    {
        var store = new VecNetVectorStore();
        return store.GetCollection<TKey, TRecord>("records", definition ?? CreateDefinition<TKey, TRecord>());
    }

    private static VectorStoreCollectionDefinition CreateDefinition<TKey, TRecord>(
        string distanceFunction = DistanceFunction.EuclideanSquaredDistance,
        bool autoGeneratedKey = false)
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(TKey)) { IsAutoGenerated = autoGeneratedKey },
                new VectorStoreVectorProperty("Vector", typeof(float[]), dimensions: 2)
                {
                    IndexKind = IndexKind.Flat,
                    DistanceFunction = distanceFunction
                },
                new VectorStoreDataProperty("Tag", typeof(string)),
                new VectorStoreDataProperty("Sort", typeof(int))
            ]
        };
    }

    private static VectorStoreCollectionDefinition CreateNullableVectorDefinition()
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(NullableVectorRecord.Id), typeof(string)),
                new VectorStoreVectorProperty(nameof(NullableVectorRecord.Vector), typeof(float[]), dimensions: 2)
            ]
        };
    }

    private static VectorStoreCollectionDefinition CreateNullableKeyDefinition()
    {
        return new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(NullableKeyRecord.Id), typeof(string)),
                new VectorStoreVectorProperty(nameof(NullableKeyRecord.Vector), typeof(float[]), dimensions: 2)
            ]
        };
    }

    private static Task<List<VectorSearchResult<TRecord>>> Search<TKey, TRecord>(
        VectorStoreCollection<TKey, TRecord> collection,
        float[] query,
        int top,
        VectorSearchOptions<TRecord>? options = null)
        where TKey : notnull
        where TRecord : class =>
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

    private sealed class CancelAfterFirstRecordEnumerable
    {
        public CancelAfterFirstRecordEnumerable(CancellationTokenSource canceledAfterFirst)
        {
            CanceledAfterFirst = canceledAfterFirst;
        }

        public CancellationTokenSource CanceledAfterFirst { get; }

        public IEnumerable<TestRecord> Records
        {
            get
            {
                yield return new TestRecord { Id = "first", Vector = [1, 0], Tag = "x", Sort = 1 };
                CanceledAfterFirst.Cancel();
                yield return new TestRecord { Id = "second", Vector = [2, 0], Tag = "x", Sort = 2 };
            }
        }
    }

    private sealed class TestRecord
    {
        public string Id { get; init; } = string.Empty;

        public float[] Vector { get; init; } = [];

        public string Tag { get; init; } = string.Empty;

        public int Sort { get; init; }
    }

    private sealed class KeyRecord<TKey>
    {
        public TKey Id { get; init; } = default!;

        public float[] Vector { get; init; } = [];

        public string Tag { get; init; } = string.Empty;
    }

    private sealed class GuidKeyRecord
    {
        public Guid Id { get; init; }

        public float[] Vector { get; init; } = [];
    }

    private sealed class NullableVectorRecord
    {
        public string Id { get; init; } = string.Empty;

        public float[]? Vector { get; init; }
    }

    private sealed class NullableKeyRecord
    {
        public string? Id { get; init; }

        public float[] Vector { get; init; } = [];
    }
}
