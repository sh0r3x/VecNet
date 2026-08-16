namespace VecNet.Tests;

public sealed class Vec300HnswMutableTombstoneAdaptiveRetryIndependentTests
{
    [Fact]
    public void SquaredEuclideanAdaptiveRetryFindsLiveBaseRowsHiddenBehindTombstonedFirstPass()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [
                new(101, [0.00f, 0f]),
                new(102, [0.05f, 0f]),
                new(103, [0.10f, 0f]),
                new(201, [1.00f, 0f]),
                new(202, [1.25f, 0f]),
                new(203, [1.50f, 0f])
            ],
            efSearch: 3);
        DeleteCommitted(mutable, 101, 102, 103);

        SearchResult[] tight = FilledResults(3, 900);
        int tightWritten = mutable.Search(
            [0f, 0f],
            tight,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 3),
            efSearch: 3);

        SearchResult[] retried = FilledResults(3, 800);
        int retriedWritten = mutable.Search(
            [0f, 0f],
            retried,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 6),
            efSearch: 3);

        Assert.True(tightWritten < retriedWritten, $"Expected retry to improve over tight count {tightWritten}.");
        Assert.Equal(3, retriedWritten);
        Assert.Equal([201UL, 202UL, 203UL], retried[..retriedWritten].Select(static result => result.Id));
        AssertNoIds(retried[..retriedWritten], 101, 102, 103);
    }

    [Fact]
    public void CosineAdaptiveRetryFindsLiveBaseRowsHiddenBehindTombstonedFirstPass()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.Cosine,
            [
                new(11, [1.00f, 0.00f, 0f]),
                new(12, [0.99f, 0.05f, 0f]),
                new(13, [0.98f, 0.10f, 0f]),
                new(21, [0.70f, 0.70f, 0f]),
                new(22, [0.45f, 0.89f, 0f]),
                new(23, [0.10f, 0.99f, 0f])
            ],
            efSearch: 3);
        DeleteCommitted(mutable, 11, 12, 13);

        SearchResult[] tight = FilledResults(3, 700);
        int tightWritten = mutable.Search(
            [1f, 0f, 0f],
            tight,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 3),
            efSearch: 3);

        SearchResult[] retried = FilledResults(3, 600);
        int retriedWritten = mutable.Search(
            [1f, 0f, 0f],
            retried,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 6),
            efSearch: 3);

        Assert.True(tightWritten < retriedWritten, $"Expected cosine retry to improve over tight count {tightWritten}.");
        Assert.Equal(3, retriedWritten);
        AssertNoIds(retried[..retriedWritten], 11, 12, 13);
        Assert.All(retried[..retriedWritten], static result => Assert.True(float.IsFinite(result.Distance)));
    }

    [Fact]
    public void BroadAllowlistAdaptiveRetrySuppressesTombstonedUnknownAndDuplicateIds()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [
                new(10, [0f]),
                new(20, [0.1f]),
                new(30, [0.2f]),
                new(40, [1f]),
                new(50, [2f]),
                new(60, [3f]),
                new(70, [4f])
            ],
            efSearch: 3);
        DeleteCommitted(mutable, 10, 20, 30);

        SearchResult[] results = FilledResults(3, 500);
        HnswMutableSearchWorkspace workspace = mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 7);
        int written = mutable.Search(
            [0f],
            [999, 10, 40, 40, 20, 50, 30, 60, 70, 60],
            results,
            workspace,
            efSearch: 3);

        Assert.Equal(3, written);
        Assert.Equal(7, workspace.MaxEfSearch);
        Assert.Equal([40UL, 50UL, 60UL], results[..written].Select(static result => result.Id));
        AssertNoIds(results[..written], 999, 10, 20, 30);
        Assert.Equal(written, results[..written].Select(static result => result.Id).Distinct().Count());
    }

    [Fact]
    public void AdaptiveRetryIsCappedByCallerWorkspaceWidth()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [
                new(10, [0f]),
                new(20, [0.1f]),
                new(30, [0.2f]),
                new(40, [1f]),
                new(50, [2f]),
                new(60, [3f])
            ],
            efSearch: 3);
        DeleteCommitted(mutable, 10, 20, 30);

        SearchResult[] capacityFour = FilledResults(3, 400);
        int capacityFourWritten = mutable.Search(
            [0f],
            capacityFour,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 4),
            efSearch: 3);

        SearchResult[] capacitySix = FilledResults(3, 300);
        int capacitySixWritten = mutable.Search(
            [0f],
            capacitySix,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 6),
            efSearch: 3);

        Assert.Equal(1, capacityFourWritten);
        Assert.Equal(40UL, capacityFour[0].Id);
        Assert.Equal(new SearchResult(401, 401), capacityFour[1]);
        Assert.Equal(3, capacitySixWritten);
        Assert.True(capacityFourWritten < capacitySixWritten);
    }

    [Fact]
    public void AdaptiveRetryDoesNotInventResultsWhenVisibleLiveRowsAreFewerThanRequested()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [
                new(1, [0f, 0f]),
                new(2, [0.1f, 0f]),
                new(3, [0.2f, 0f]),
                new(4, [1f, 0f])
            ],
            efSearch: 3);
        DeleteCommitted(mutable, 1, 2, 3);

        SearchResult[] results = FilledResults(3, 200);
        int written = mutable.Search(
            [0f, 0f],
            results,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 4),
            efSearch: 3);

        Assert.Equal(1, written);
        Assert.Equal(4UL, results[0].Id);
        Assert.Equal([new SearchResult(201, 201), new SearchResult(202, 202)], results[1..]);
    }

    [Fact]
    public void StaleWorkspaceAfterMutationAndCheckpointIsRejectedBeforeDestinationWrites()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [new(10, [0f]), new(20, [1f]), new(30, [2f])],
            efSearch: 3);
        SearchResult[] destination = FilledResults(2, 100);
        SearchResult[] original = destination.ToArray();

        HnswMutableSearchWorkspace staleAfterAdd = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3);
        AssertCommitted(mutable.TryAdd(15, [0.5f]));
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], destination, staleAfterAdd, efSearch: 2));
        Assert.Equal(original, destination);

        HnswMutableSearchWorkspace staleAfterDelete = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3);
        AssertCommitted(mutable.TryDelete(20));
        Assert.Throws<InvalidOperationException>(
            () => mutable.Search([0f], [10, 15, 30], destination, staleAfterDelete, efSearch: 2));
        Assert.Equal(original, destination);

        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswMutableIndex checkpointMutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [new(100, [0f]), new(200, [1f]), new(300, [2f])],
            efSearch: 3);
        AssertCommitted(checkpointMutable.TryAdd(150, [0.5f]));
        HnswMutableSearchWorkspace staleAfterCheckpoint =
            checkpointMutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3);

        HnswMutableCheckpointResult checkpointResult = checkpointMutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Throws<InvalidOperationException>(
            () => checkpointMutable.Search([0f], destination, staleAfterCheckpoint, efSearch: 2));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void PublicPerSearchEfSearchValidationAndInnerProductAdmissionRemainUnchanged()
    {
        HnswMutableIndex mutable = CreateMutable(
            VectorMetric.SquaredEuclidean,
            [new(10, [0f]), new(20, [1f])],
            efSearch: 2);
        SearchResult[] destination = FilledResults(2, 50);
        SearchResult[] original = destination.ToArray();

        ArgumentOutOfRangeException efSearchTooLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination.AsSpan(0, 1), mutable.CreateSearchWorkspace(1, 1), efSearch: 0));
        Assert.Equal("efSearch", efSearchTooLow.ParamName);

        ArgumentOutOfRangeException efSearchTooHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], [10], destination.AsSpan(0, 1), mutable.CreateSearchWorkspace(1, 1), efSearch: 4097));
        Assert.Equal("efSearch", efSearchTooHigh.ParamName);

        ArgumentOutOfRangeException resultCountExceedsEfSearch = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination, mutable.CreateSearchWorkspace(2, 1), efSearch: 1));
        Assert.Equal("results", resultCountExceedsEfSearch.ParamName);
        Assert.Equal(original, destination);

        Assert.Throws<ArgumentException>(
            () => mutable.Search([0f], destination, mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 2), efSearch: 2));
        Assert.Equal(original, destination);

        Assert.Equal(VectorMetric.InnerProduct, new HnswIndex(2, VectorMetric.InnerProduct).Metric);
        Assert.Equal(VectorMetric.InnerProduct, new HnswIndex(2, VectorMetric.InnerProduct, HnswIndexOptions.Default).Metric);
    }

    private static HnswMutableIndex CreateMutable(
        VectorMetric metric,
        IEnumerable<Row> rows,
        int efSearch)
    {
        Row[] materialized = rows.ToArray();
        int dimension = materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            metric,
            new HnswIndexOptions(M: 2, EfConstruction: 12, EfSearch: efSearch, RandomSeed: 0x300UL),
            () => 0);

        foreach (Row row in materialized)
        {
            index.Add(row.Id, row.Vector);
        }

        return new HnswMutableIndex(index);
    }

    private static SearchResult[] FilledResults(int count, ulong firstId) =>
        Enumerable.Range(0, count)
            .Select(offset => new SearchResult(firstId + (ulong)offset, (float)(firstId + (ulong)offset)))
            .ToArray();

    private static void DeleteCommitted(HnswMutableIndex index, params ulong[] ids)
    {
        foreach (ulong id in ids)
        {
            AssertCommitted(index.TryDelete(id));
        }
    }

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

    private static void AssertNoIds(ReadOnlySpan<SearchResult> results, params ulong[] forbiddenIds)
    {
        foreach (SearchResult result in results)
        {
            Assert.DoesNotContain(result.Id, forbiddenIds);
        }
    }

    private sealed record Row(ulong Id, float[] Vector);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() =>
            new(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-VEC300-" + Guid.NewGuid().ToString("N")));

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
    }
}
