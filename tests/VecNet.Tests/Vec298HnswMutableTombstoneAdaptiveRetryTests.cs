namespace VecNet.Tests;

public sealed class Vec298HnswMutableTombstoneAdaptiveRetryTests
{
    [Fact]
    public void SquaredEuclideanMutableSearch_RetriesWithinCallerWorkspaceAfterBaseTombstoneUnderfill()
    {
        HnswMutableIndex mutable = CreateSquaredMutable(
            [(10UL, 0f), (20UL, 1f), (30UL, 2f), (40UL, 3f)],
            efSearch: 2);
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(20));

        SearchResult[] tightResults = [new(901, 901), new(902, 902)];
        int tightWritten = mutable.Search(
            [0f],
            tightResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2),
            efSearch: 2);

        SearchResult[] widerResults = [new(801, 801), new(802, 802)];
        int widerWritten = mutable.Search(
            [0f],
            widerResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 4),
            efSearch: 2);

        Assert.True(tightWritten < 2, $"Tight first-width search unexpectedly filled {tightWritten} results.");
        Assert.Equal(2, widerWritten);
        Assert.Equal([30UL, 40UL], widerResults.Select(static result => result.Id));
    }

    [Fact]
    public void CosineMutableSearch_RetriesWithinCallerWorkspaceAfterBaseTombstoneUnderfill()
    {
        HnswMutableIndex mutable = CreateCosineMutable(DefaultCosineRows(), efSearch: 2);
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(20));

        SearchResult[] tightResults = [new(901, 901), new(902, 902)];
        int tightWritten = mutable.Search(
            [1f, 0f],
            tightResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2),
            efSearch: 2);

        SearchResult[] widerResults = [new(801, 801), new(802, 802)];
        int widerWritten = mutable.Search(
            [1f, 0f],
            widerResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 4),
            efSearch: 2);

        Assert.True(tightWritten < 2, $"Tight first-width search unexpectedly filled {tightWritten} results.");
        Assert.Equal(2, widerWritten);
        Assert.DoesNotContain(widerResults[..widerWritten], static result => result.Id is 10 or 20);
        Assert.All(widerResults[..widerWritten], static result => Assert.True(float.IsFinite(result.Distance)));
    }

    [Fact]
    public void BroadAllowlistMutableSearch_RetriesAndSuppressesTombstonedUnknownAndDuplicateIds()
    {
        HnswMutableIndex mutable = CreateSquaredMutable(
            [(10UL, 0f), (20UL, 1f), (30UL, 2f), (40UL, 3f), (50UL, 4f)],
            efSearch: 2);
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(20));

        SearchResult[] results = [new(701, 701), new(702, 702)];
        int written = mutable.Search(
            [0f],
            [999UL, 10UL, 30UL, 30UL, 20UL, 40UL, 50UL],
            results,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 5),
            efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([30UL, 40UL], results.Select(static result => result.Id));
        Assert.DoesNotContain(results[..written], static result => result.Id is 10 or 20 or 999);
        Assert.Equal(written, results[..written].Select(static result => result.Id).Distinct().Count());
    }

    [Fact]
    public void MutableSearch_DoesNotChangeVisibleResultsWhenFirstPassAlreadyFills()
    {
        HnswMutableIndex mutable = CreateSquaredMutable(
            [(10UL, 0f), (20UL, 1f), (30UL, 2f), (40UL, 3f), (50UL, 100f)],
            efSearch: 2);
        AssertCommitted(mutable.TryDelete(50));

        SearchResult[] tightResults = new SearchResult[2];
        int tightWritten = mutable.Search(
            [0f],
            tightResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2),
            efSearch: 2);

        SearchResult[] widerResults = new SearchResult[2];
        int widerWritten = mutable.Search(
            [0f],
            widerResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 4),
            efSearch: 2);

        Assert.Equal(2, tightWritten);
        Assert.Equal(tightWritten, widerWritten);
        Assert.Equal(tightResults, widerResults);
    }

    [Fact]
    public void MutableSearch_DoesNotRetryBeyondWorkspaceCapacityAndKeepsUndersizedValidation()
    {
        HnswMutableIndex mutable = CreateSquaredMutable(
            [(10UL, 0f), (20UL, 1f), (30UL, 2f), (40UL, 3f)],
            efSearch: 2);
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(20));

        SearchResult[] capacityBoundResults = [new(601, 601), new(602, 602)];
        int capacityBoundWritten = mutable.Search(
            [0f],
            capacityBoundResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2),
            efSearch: 2);

        Assert.True(capacityBoundWritten < 2);

        ArgumentOutOfRangeException tooManyResults = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search(
                [0f],
                new SearchResult[3],
                mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 4),
                efSearch: 2));
        Assert.Equal("results", tooManyResults.ParamName);

        Assert.Throws<ArgumentException>(
            () => mutable.Search(
                [0f],
                new SearchResult[2],
                mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 1),
                efSearch: 2));
    }

    [Fact]
    public void MutableSearch_CannotInventResultsWhenVisibleLiveRowsAreFewerThanRequested()
    {
        HnswMutableIndex mutable = CreateSquaredMutable(
            [(10UL, 0f), (20UL, 1f), (30UL, 2f)],
            efSearch: 2);
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(20));

        SearchResult[] results = [new(501, 501), new(502, 502)];
        int written = mutable.Search(
            [0f],
            results,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3),
            efSearch: 2);

        Assert.Equal(1, written);
        Assert.Equal(30UL, results[0].Id);
        Assert.Equal(new SearchResult(502, 502), results[1]);
    }

    [Fact]
    public void MutableSearch_StillRejectsStaleWorkspaceAfterAddDeleteAndPublishedCheckpointBeforeWrites()
    {
        HnswMutableIndex mutable = CreateSquaredMutable([(10UL, 0f), (20UL, 1f)], efSearch: 2);
        SearchResult[] destination = [new(401, 401), new(402, 402)];
        SearchResult[] original = destination.ToArray();

        HnswMutableSearchWorkspace staleAfterAdd = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);
        AssertCommitted(mutable.TryAdd(15, [0.5f]));
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], destination, staleAfterAdd, efSearch: 2));
        Assert.Equal(original, destination);

        HnswMutableSearchWorkspace staleAfterDelete = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);
        AssertCommitted(mutable.TryDelete(20));
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], [10UL, 15UL], destination, staleAfterDelete, efSearch: 2));
        Assert.Equal(original, destination);

        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswMutableIndex checkpointMutable = CreateSquaredMutable([(100UL, 0f), (200UL, 1f)], efSearch: 2);
        AssertCommitted(checkpointMutable.TryAdd(150, [0.5f]));
        HnswMutableSearchWorkspace staleAfterCheckpoint = checkpointMutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);

        HnswMutableCheckpointResult checkpointResult = checkpointMutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Throws<InvalidOperationException>(() => checkpointMutable.Search([0f], destination, staleAfterCheckpoint, efSearch: 2));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void HnswInnerProductRejectionRemainsUnchanged()
    {
        Assert.Throws<NotSupportedException>(() => new HnswIndex(2, VectorMetric.InnerProduct));
        Assert.Throws<NotSupportedException>(
            () => new HnswIndex(2, VectorMetric.InnerProduct, new HnswIndexOptions(2, 8, 2, 0x298UL)));
    }

    private static HnswMutableIndex CreateSquaredMutable(
        IEnumerable<(ulong Id, float Value)> rows,
        int efSearch)
    {
        (ulong Id, float Value)[] materialized = rows.ToArray();
        var index = new HnswIndex(
            dimension: 1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(M: 2, EfConstruction: 8, EfSearch: efSearch, RandomSeed: 0x298UL),
            () => 0);

        foreach ((ulong id, float value) in materialized)
        {
            index.Add(id, [value]);
        }

        return new HnswMutableIndex(index);
    }

    private static HnswMutableIndex CreateCosineMutable(
        IEnumerable<(ulong Id, float[] Vector)> rows,
        int efSearch)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        var index = new HnswIndex(
            dimension: 2,
            VectorMetric.Cosine,
            new HnswIndexOptions(M: 2, EfConstruction: 8, EfSearch: efSearch, RandomSeed: 0x298UL),
            () => 0);

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return new HnswMutableIndex(index);
    }

    private static (ulong Id, float[] Vector)[] DefaultCosineRows() =>
        [
            (10UL, [1f, 0f]),
            (20UL, [0.99f, 0.1f]),
            (30UL, [0.5f, 0.8660254f]),
            (40UL, [0f, 1f])
        ];

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-VEC298-" + Guid.NewGuid().ToString("N"));
            return new TempIndexDirectory(path);
        }

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
