namespace VecNet.Tests;

public sealed class Vec292HnswPerSearchEfSearchTests
{
    [Fact]
    public void ImmutableUnfiltered_PerSearchEfSearchCanExceedConfiguredDefault()
    {
        HnswIndex index = CreateHnsw([(10UL, 0f), (20UL, 1f), (30UL, 2f)], efSearch: 1);
        SearchResult[] results = [new(999, 999), new(998, 998)];

        ArgumentOutOfRangeException defaultException = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], results, index.CreateSearchWorkspace()));
        Assert.Equal("results", defaultException.ParamName);

        int written = index.Search([0f], results, index.CreateSearchWorkspace(maxEfSearch: 2), efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results.Select(static result => result.Id));
    }

    [Fact]
    public void ImmutableAllowlist_PerSearchEfSearchControlsExactFallbackAndEmissionSuppression()
    {
        HnswIndex index = CreateHnsw([(10UL, 0f), (20UL, 10f), (30UL, 20f)], efSearch: 1);

        SearchResult[] exactResults = [new(999, 999), new(998, 998)];
        int exactWritten = index.Search(
            [0f],
            [20UL, 999UL, 20UL, 30UL],
            exactResults,
            index.CreateSearchWorkspace(maxEfSearch: 2),
            efSearch: 2);

        Assert.Equal(2, exactWritten);
        Assert.Equal([20UL, 30UL], exactResults.Select(static result => result.Id));

        SearchResult[] suppressedResults = [new(777, 777)];
        int suppressedWritten = index.Search(
            [0f],
            [20UL, 30UL],
            suppressedResults,
            index.CreateSearchWorkspace(maxEfSearch: 1),
            efSearch: 1);

        Assert.Equal(0, suppressedWritten);
        Assert.Equal(new SearchResult(777, 777), suppressedResults[0]);
    }

    [Fact]
    public void ImmutableDefaultOverloadsRemainEquivalentToConfiguredEfSearch()
    {
        HnswIndex index = CreateHnsw([(10UL, 0f), (20UL, 1f), (30UL, 2f), (40UL, 3f)], efSearch: 3);
        SearchResult[] defaultResults = new SearchResult[2];
        SearchResult[] explicitResults = new SearchResult[2];

        int defaultWritten = index.Search([0f], defaultResults, index.CreateSearchWorkspace());
        int explicitWritten = index.Search([0f], explicitResults, index.CreateSearchWorkspace(3), efSearch: index.Options.EfSearch);

        Assert.Equal(defaultWritten, explicitWritten);
        Assert.Equal(defaultResults, explicitResults);

        SearchResult[] defaultAllowedResults = new SearchResult[2];
        SearchResult[] explicitAllowedResults = new SearchResult[2];
        int defaultAllowedWritten = index.Search([0f], [10UL, 20UL, 30UL], defaultAllowedResults, index.CreateSearchWorkspace());
        int explicitAllowedWritten = index.Search(
            [0f],
            [10UL, 20UL, 30UL],
            explicitAllowedResults,
            index.CreateSearchWorkspace(3),
            efSearch: index.Options.EfSearch);

        Assert.Equal(defaultAllowedWritten, explicitAllowedWritten);
        Assert.Equal(defaultAllowedResults, explicitAllowedResults);
    }

    [Fact]
    public void ImmutablePerSearchEfSearch_ValidatesWidthResultsAndWorkspace()
    {
        HnswIndex index = CreateHnsw([(10UL, 0f), (20UL, 1f)], efSearch: 2);

        ArgumentOutOfRangeException factoryLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.CreateSearchWorkspace(0));
        Assert.Equal("maxEfSearch", factoryLow.ParamName);

        ArgumentOutOfRangeException factoryHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.CreateSearchWorkspace(4097));
        Assert.Equal("maxEfSearch", factoryHigh.ParamName);

        ArgumentOutOfRangeException efLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], new SearchResult[1], index.CreateSearchWorkspace(1), efSearch: 0));
        Assert.Equal("efSearch", efLow.ParamName);

        ArgumentOutOfRangeException efHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], new SearchResult[1], index.CreateSearchWorkspace(1), efSearch: 4097));
        Assert.Equal("efSearch", efHigh.ParamName);

        ArgumentOutOfRangeException resultsTooLarge = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], new SearchResult[2], index.CreateSearchWorkspace(1), efSearch: 1));
        Assert.Equal("results", resultsTooLarge.ParamName);

        Assert.Throws<ArgumentException>(
            () => index.Search([0f], new SearchResult[1], new HnswSearchWorkspace(maxElements: 1, maxEf: 2), efSearch: 2));
        Assert.Throws<ArgumentException>(
            () => index.Search([0f], new SearchResult[1], new HnswSearchWorkspace(maxElements: index.Count, maxEf: 1), efSearch: 2));
    }

    [Fact]
    public void MutableFactories_SizeWorkspaceFromCurrentGenerationAndRequestedWidth()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f), (30UL, 2f)], efSearch: 1);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(25, [1.5f]).Status);

        HnswMutableSearchWorkspace configured = mutable.CreateSearchWorkspace(maxResults: 2);
        Assert.Equal(mutable.Generation, configured.Generation);
        Assert.Equal(mutable.BasePhysicalVectorCount, configured.MaxBaseElements);
        Assert.Equal(mutable.Options.EfSearch, configured.MaxEfSearch);
        Assert.Equal(Math.Min(mutable.BasePhysicalVectorCount, mutable.Options.EfSearch), configured.MaxBaseCandidates);
        Assert.Equal(2, configured.MaxDeltaCandidates);
        Assert.Equal(mutable.DeltaPhysicalVectorCount, configured.MaxDeltaFilterElements);

        HnswMutableSearchWorkspace explicitWidth = mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 2);
        Assert.Equal(mutable.Generation, explicitWidth.Generation);
        Assert.Equal(2, explicitWidth.MaxEfSearch);
        Assert.Equal(Math.Min(mutable.BasePhysicalVectorCount, 2), explicitWidth.MaxBaseCandidates);
        Assert.Equal(3, explicitWidth.MaxDeltaCandidates);
        Assert.Equal(mutable.DeltaPhysicalVectorCount, explicitWidth.MaxDeltaFilterElements);

        ArgumentOutOfRangeException low = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 0));
        Assert.Equal("maxEfSearch", low.ParamName);

        ArgumentOutOfRangeException high = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 4097));
        Assert.Equal("maxEfSearch", high.ParamName);
    }

    [Fact]
    public void MutableUnfiltered_PerSearchEfSearchCanExceedConfiguredDefault()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f)], efSearch: 1);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.25f]).Status);
        SearchResult[] results = [new(999, 999), new(998, 998)];

        ArgumentOutOfRangeException defaultException = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], results, mutable.CreateSearchWorkspace(maxResults: 2)));
        Assert.Equal("results", defaultException.ParamName);

        int written = mutable.Search([0f], results, mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2), efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 15UL], results.Select(static result => result.Id));
    }

    [Fact]
    public void MutableAllowlist_PerSearchEfSearchUsesLiveKnownAllowedCountAcrossBaseDeltaAndTombstones()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f), (30UL, 10f)], efSearch: 1);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(25, [2f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(25).Status);

        SearchResult[] results = [new(999, 999), new(998, 998), new(997, 997)];
        int written = mutable.Search(
            [0f],
            [10UL, 15UL, 20UL, 25UL, 30UL, 999UL, 15UL],
            results,
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 3),
            efSearch: 3);

        Assert.Equal(3, written);
        Assert.Equal([15UL, 20UL, 30UL], results.Select(static result => result.Id));
    }

    [Fact]
    public void MutableDefaultOverloadsRemainEquivalentToConfiguredEfSearch()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f), (30UL, 2f)], efSearch: 3);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);

        SearchResult[] defaultResults = new SearchResult[2];
        SearchResult[] explicitResults = new SearchResult[2];
        int defaultWritten = mutable.Search([0f], defaultResults, mutable.CreateSearchWorkspace(maxResults: 2));
        int explicitWritten = mutable.Search(
            [0f],
            explicitResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3),
            efSearch: mutable.Options.EfSearch);

        Assert.Equal(defaultWritten, explicitWritten);
        Assert.Equal(defaultResults, explicitResults);

        SearchResult[] defaultAllowedResults = new SearchResult[2];
        SearchResult[] explicitAllowedResults = new SearchResult[2];
        int defaultAllowedWritten = mutable.Search([0f], [10UL, 15UL, 20UL], defaultAllowedResults, mutable.CreateSearchWorkspace(maxResults: 2));
        int explicitAllowedWritten = mutable.Search(
            [0f],
            [10UL, 15UL, 20UL],
            explicitAllowedResults,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 3),
            efSearch: mutable.Options.EfSearch);

        Assert.Equal(defaultAllowedWritten, explicitAllowedWritten);
        Assert.Equal(defaultAllowedResults, explicitAllowedResults);
    }

    [Fact]
    public void MutablePerSearchEfSearch_ValidatesWidthResultsWorkspaceAndStaleGenerationBeforeWrites()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f)], efSearch: 2);

        ArgumentOutOfRangeException efLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], new SearchResult[1], mutable.CreateSearchWorkspace(maxResults: 1), efSearch: 0));
        Assert.Equal("efSearch", efLow.ParamName);

        ArgumentOutOfRangeException efHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], new SearchResult[1], mutable.CreateSearchWorkspace(maxResults: 1), efSearch: 4097));
        Assert.Equal("efSearch", efHigh.ParamName);

        ArgumentOutOfRangeException resultsTooLarge = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], new SearchResult[2], mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 1), efSearch: 1));
        Assert.Equal("results", resultsTooLarge.ParamName);

        Assert.Throws<ArgumentException>(
            () => mutable.Search([0f], new SearchResult[1], mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 1), efSearch: 2));

        SearchResult[] destination = [new(701, -701), new(702, -702)];
        SearchResult[] original = destination.ToArray();
        HnswMutableSearchWorkspace staleAfterAdd = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], destination, staleAfterAdd, efSearch: 2));
        Assert.Equal(original, destination);

        HnswMutableSearchWorkspace staleAfterDelete = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], [10UL, 15UL], destination, staleAfterDelete, efSearch: 2));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void MutablePerSearchEfSearch_RejectsWorkspaceAfterPublishedCheckpointBeforeWrites()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswMutableIndex mutable = CreateMutable([(10UL, 0f), (20UL, 1f)], efSearch: 2);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);

        HnswMutableSearchWorkspace stale = mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2);
        SearchResult[] destination = [new(701, -701), new(702, -702)];

        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], destination, stale, efSearch: 2));
        Assert.Equal([new SearchResult(701, -701), new SearchResult(702, -702)], destination);
    }

    [Fact]
    public void HnswInnerProductRejectionRemainsUnchanged()
    {
        Assert.Throws<NotSupportedException>(() => new HnswIndex(1, VectorMetric.InnerProduct));
        HnswIndex baseIndex = CreateHnsw([(10UL, 0f)], efSearch: 2);
        Assert.Equal(VectorMetric.SquaredEuclidean, new HnswMutableIndex(baseIndex).Metric);
    }

    private static HnswMutableIndex CreateMutable(IEnumerable<(ulong Id, float Value)> rows, int efSearch) =>
        new(CreateHnsw(rows, efSearch));

    private static HnswIndex CreateHnsw(IEnumerable<(ulong Id, float Value)> rows, int efSearch)
    {
        var index = new HnswIndex(
            dimension: 1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(M: 2, EfConstruction: 8, EfSearch: efSearch, RandomSeed: 0x292UL),
            () => 0);

        foreach ((ulong id, float value) in rows)
        {
            index.Add(id, [value]);
        }

        return index;
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-VEC292-" + Guid.NewGuid().ToString("N"));
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
