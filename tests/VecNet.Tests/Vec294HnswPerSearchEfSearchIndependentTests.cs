namespace VecNet.Tests;

public sealed class Vec294HnswPerSearchEfSearchIndependentTests
{
    [Fact]
    public void ImmutableUnfiltered_ExplicitEfSearchUsesPerCallWidthAndLeavesConfiguredDefaultUnchanged()
    {
        HnswIndex index = CreateHnsw([(10UL, [0f, 0f]), (20UL, [1f, 0f]), (30UL, [2f, 0f])], efSearch: 1);
        SearchResult[] destination = [new(901, 901), new(902, 902)];

        ArgumentOutOfRangeException defaultWidthException = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f, 0f], destination, index.CreateSearchWorkspace()));
        Assert.Equal("results", defaultWidthException.ParamName);
        Assert.Equal([new SearchResult(901, 901), new SearchResult(902, 902)], destination);

        HnswSearchWorkspace explicitWorkspace = index.CreateSearchWorkspace(maxEfSearch: 2);
        int written = index.Search([0f, 0f], destination, explicitWorkspace, efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], destination.Select(static result => result.Id));
        Assert.Equal(1, index.Options.EfSearch);
        Assert.Equal(index.Count, explicitWorkspace.MaxElements);
        Assert.Equal(2, explicitWorkspace.MaxEf);
    }

    [Fact]
    public void ImmutableAllowlist_PerSearchWidthSelectsExactFallbackAndBroadEmissionSuppression()
    {
        HnswIndex index = CreateHnsw(
            [(10UL, [0f]), (20UL, [10f]), (30UL, [20f]), (40UL, [30f])],
            efSearch: 1);

        SearchResult[] exactDestination = [new(801, 801), new(802, 802), new(803, 803)];
        int exactWritten = index.Search(
            [0f],
            [999UL, 30UL, 20UL, 30UL],
            exactDestination.AsSpan(0, 2),
            index.CreateSearchWorkspace(maxEfSearch: 2),
            efSearch: 2);

        Assert.Equal(2, exactWritten);
        Assert.Equal(
            [new SearchResult(20, 100f), new SearchResult(30, 400f)],
            exactDestination[..exactWritten]);
        Assert.Equal(new SearchResult(803, 803), exactDestination[2]);

        SearchResult[] unknownOnlyDestination = [new(701, 701)];
        int unknownOnlyWritten = index.Search(
            [0f],
            [999UL, 999UL],
            unknownOnlyDestination,
            index.CreateSearchWorkspace(maxEfSearch: 1),
            efSearch: 1);
        Assert.Equal(0, unknownOnlyWritten);
        Assert.Equal(new SearchResult(701, 701), unknownOnlyDestination[0]);

        SearchResult[] suppressedDestination = [new(601, 601)];
        int suppressedWritten = index.Search(
            [0f],
            [20UL, 30UL],
            suppressedDestination,
            index.CreateSearchWorkspace(maxEfSearch: 1),
            efSearch: 1);

        Assert.Equal(0, suppressedWritten);
        Assert.Equal(new SearchResult(601, 601), suppressedDestination[0]);
    }

    [Fact]
    public void ImmutableDefaultOverloadsMatchExplicitConfiguredEfSearch()
    {
        HnswIndex index = CreateHnsw(
            [(10UL, [0f]), (20UL, [1f]), (30UL, [2f]), (40UL, [3f])],
            efSearch: 3);

        SearchResult[] defaultUnfiltered = new SearchResult[2];
        SearchResult[] explicitUnfiltered = new SearchResult[2];
        int defaultUnfilteredWritten = index.Search([0f], defaultUnfiltered, index.CreateSearchWorkspace());
        int explicitUnfilteredWritten = index.Search(
            [0f],
            explicitUnfiltered,
            index.CreateSearchWorkspace(maxEfSearch: index.Options.EfSearch),
            efSearch: index.Options.EfSearch);

        Assert.Equal(defaultUnfilteredWritten, explicitUnfilteredWritten);
        Assert.Equal(defaultUnfiltered, explicitUnfiltered);

        SearchResult[] defaultAllowed = new SearchResult[2];
        SearchResult[] explicitAllowed = new SearchResult[2];
        int defaultAllowedWritten = index.Search([0f], [10UL, 20UL, 40UL], defaultAllowed, index.CreateSearchWorkspace());
        int explicitAllowedWritten = index.Search(
            [0f],
            [10UL, 20UL, 40UL],
            explicitAllowed,
            index.CreateSearchWorkspace(maxEfSearch: index.Options.EfSearch),
            efSearch: index.Options.EfSearch);

        Assert.Equal(defaultAllowedWritten, explicitAllowedWritten);
        Assert.Equal(defaultAllowed, explicitAllowed);
    }

    [Fact]
    public void ImmutablePerSearchEfSearch_RejectsInvalidValuesResultsAndUndersizedWorkspacesBeforeWriting()
    {
        HnswIndex index = CreateHnsw([(10UL, [0f]), (20UL, [1f])], efSearch: 2);
        SearchResult[] destination = [new(501, 501), new(502, 502)];
        SearchResult[] original = destination.ToArray();

        ArgumentOutOfRangeException maxEfLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.CreateSearchWorkspace(maxEfSearch: 0));
        Assert.Equal("maxEfSearch", maxEfLow.ParamName);

        ArgumentOutOfRangeException maxEfHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.CreateSearchWorkspace(maxEfSearch: 4097));
        Assert.Equal("maxEfSearch", maxEfHigh.ParamName);

        ArgumentOutOfRangeException efLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], destination.AsSpan(0, 1), index.CreateSearchWorkspace(maxEfSearch: 1), efSearch: 0));
        Assert.Equal("efSearch", efLow.ParamName);

        ArgumentOutOfRangeException efHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], destination.AsSpan(0, 1), index.CreateSearchWorkspace(maxEfSearch: 1), efSearch: 4097));
        Assert.Equal("efSearch", efHigh.ParamName);

        ArgumentOutOfRangeException resultsTooWide = Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f], destination, index.CreateSearchWorkspace(maxEfSearch: 1), efSearch: 1));
        Assert.Equal("results", resultsTooWide.ParamName);
        Assert.Equal(original, destination);

        Assert.Throws<ArgumentException>(
            () => index.Search([0f], destination.AsSpan(0, 1), new HnswSearchWorkspace(maxElements: 1, maxEf: 2), efSearch: 2));
        Assert.Equal(original, destination);

        Assert.Throws<ArgumentException>(
            () => index.Search([0f], destination.AsSpan(0, 1), new HnswSearchWorkspace(maxElements: index.Count, maxEf: 1), efSearch: 2));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void MutableFactories_SizeCurrentShapeForConfiguredAndExplicitWidths()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, [0f]), (20UL, [1f]), (30UL, [2f])], efSearch: 3);
        AssertCommitted(mutable.TryAdd(15, [0.5f]));
        AssertCommitted(mutable.TryAdd(25, [1.5f]));
        AssertCommitted(mutable.TryDelete(10));

        HnswMutableSearchWorkspace configured = mutable.CreateSearchWorkspace(maxResults: 2);
        Assert.Equal(mutable.Generation, configured.Generation);
        Assert.Equal(mutable.BasePhysicalVectorCount, configured.MaxBaseElements);
        Assert.Equal(mutable.Options.EfSearch, configured.MaxEfSearch);
        Assert.Equal(Math.Min(mutable.BasePhysicalVectorCount, mutable.Options.EfSearch), configured.MaxBaseCandidates);
        Assert.Equal(2, configured.MaxDeltaCandidates);
        Assert.Equal(mutable.DeltaPhysicalVectorCount, configured.MaxDeltaFilterElements);

        HnswMutableSearchWorkspace explicitWidth = mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 2);
        Assert.Equal(mutable.Generation, explicitWidth.Generation);
        Assert.Equal(mutable.BasePhysicalVectorCount, explicitWidth.MaxBaseElements);
        Assert.Equal(2, explicitWidth.MaxEfSearch);
        Assert.Equal(Math.Min(mutable.BasePhysicalVectorCount, 2), explicitWidth.MaxBaseCandidates);
        Assert.Equal(1, explicitWidth.MaxDeltaCandidates);
        Assert.Equal(mutable.DeltaPhysicalVectorCount, explicitWidth.MaxDeltaFilterElements);
    }

    [Fact]
    public void MutableUnfiltered_ExplicitEfSearchControlsBaseWidthWhileDeltaSearchRemainsResultBounded()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, [0f]), (20UL, [1f])], efSearch: 1);
        AssertCommitted(mutable.TryAdd(15, [0.25f]));
        SearchResult[] destination = [new(401, 401), new(402, 402)];

        ArgumentOutOfRangeException defaultWidthException = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination, mutable.CreateSearchWorkspace(maxResults: 2)));
        Assert.Equal("results", defaultWidthException.ParamName);
        Assert.Equal([new SearchResult(401, 401), new SearchResult(402, 402)], destination);

        int written = mutable.Search(
            [0f],
            destination,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 2),
            efSearch: 2);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 15UL], destination.Select(static result => result.Id));
    }

    [Fact]
    public void MutableAllowlist_PerSearchWidthCountsOnlyLiveUniqueAllowedIdsAcrossBaseDeltaAndTombstones()
    {
        HnswMutableIndex mutable = CreateMutable(
            [(10UL, [0f]), (20UL, [1f]), (30UL, [3f])],
            efSearch: 1);
        AssertCommitted(mutable.TryAdd(15, [0.5f]));
        AssertCommitted(mutable.TryAdd(25, [2f]));
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(25));

        SearchResult[] exactDestination = [new(301, 301), new(302, 302), new(303, 303), new(304, 304)];
        int exactWritten = mutable.Search(
            [0f],
            [10UL, 15UL, 20UL, 25UL, 30UL, 999UL, 15UL],
            exactDestination.AsSpan(0, 3),
            mutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 3),
            efSearch: 3);

        Assert.Equal(3, exactWritten);
        Assert.Equal(
            [new SearchResult(15, 0.25f), new SearchResult(20, 1f), new SearchResult(30, 9f)],
            exactDestination[..exactWritten]);
        Assert.Equal(new SearchResult(304, 304), exactDestination[3]);

        SearchResult[] suppressedDestination = [new(201, 201)];
        int suppressedWritten = mutable.Search(
            [0f],
            [20UL, 30UL],
            suppressedDestination,
            mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 1),
            efSearch: 1);

        Assert.Equal(0, suppressedWritten);
        Assert.Equal(new SearchResult(201, 201), suppressedDestination[0]);
    }

    [Fact]
    public void MutableDefaultOverloadsMatchExplicitConfiguredEfSearch()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, [0f]), (20UL, [1f]), (30UL, [2f])], efSearch: 3);
        AssertCommitted(mutable.TryAdd(15, [0.5f]));

        SearchResult[] defaultUnfiltered = new SearchResult[2];
        SearchResult[] explicitUnfiltered = new SearchResult[2];
        int defaultUnfilteredWritten = mutable.Search([0f], defaultUnfiltered, mutable.CreateSearchWorkspace(maxResults: 2));
        int explicitUnfilteredWritten = mutable.Search(
            [0f],
            explicitUnfiltered,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: mutable.Options.EfSearch),
            efSearch: mutable.Options.EfSearch);

        Assert.Equal(defaultUnfilteredWritten, explicitUnfilteredWritten);
        Assert.Equal(defaultUnfiltered, explicitUnfiltered);

        SearchResult[] defaultAllowed = new SearchResult[2];
        SearchResult[] explicitAllowed = new SearchResult[2];
        int defaultAllowedWritten = mutable.Search(
            [0f],
            [10UL, 15UL, 20UL],
            defaultAllowed,
            mutable.CreateSearchWorkspace(maxResults: 2));
        int explicitAllowedWritten = mutable.Search(
            [0f],
            [10UL, 15UL, 20UL],
            explicitAllowed,
            mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: mutable.Options.EfSearch),
            efSearch: mutable.Options.EfSearch);

        Assert.Equal(defaultAllowedWritten, explicitAllowedWritten);
        Assert.Equal(defaultAllowed, explicitAllowed);
    }

    [Fact]
    public void MutablePerSearchEfSearch_RejectsInvalidValuesResultsAndUndersizedWorkspacesBeforeWriting()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, [0f]), (20UL, [1f])], efSearch: 2);
        SearchResult[] destination = [new(101, 101), new(102, 102)];
        SearchResult[] original = destination.ToArray();

        ArgumentOutOfRangeException maxResultsNegative = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.CreateSearchWorkspace(maxResults: -1));
        Assert.Equal("maxResults", maxResultsNegative.ParamName);

        ArgumentOutOfRangeException maxEfLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 0));
        Assert.Equal("maxEfSearch", maxEfLow.ParamName);

        ArgumentOutOfRangeException maxEfHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 4097));
        Assert.Equal("maxEfSearch", maxEfHigh.ParamName);

        ArgumentOutOfRangeException efLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination.AsSpan(0, 1), mutable.CreateSearchWorkspace(1, 1), efSearch: 0));
        Assert.Equal("efSearch", efLow.ParamName);

        ArgumentOutOfRangeException efHigh = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination.AsSpan(0, 1), mutable.CreateSearchWorkspace(1, 1), efSearch: 4097));
        Assert.Equal("efSearch", efHigh.ParamName);

        ArgumentOutOfRangeException resultsTooWide = Assert.Throws<ArgumentOutOfRangeException>(
            () => mutable.Search([0f], destination, mutable.CreateSearchWorkspace(maxResults: 2, maxEfSearch: 1), efSearch: 1));
        Assert.Equal("results", resultsTooWide.ParamName);
        Assert.Equal(original, destination);

        Assert.Throws<ArgumentException>(
            () => mutable.Search([0f], destination, mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 2), efSearch: 2));
        Assert.Equal(original, destination);

        Assert.Throws<ArgumentException>(
            () => mutable.Search([0f], destination.AsSpan(0, 1), mutable.CreateSearchWorkspace(maxResults: 1, maxEfSearch: 1), efSearch: 2));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void CompositePerSearchEfSearch_RejectsUndersizedDeltaFilterWorkspaceBeforeWriting()
    {
        HnswBasePlusExactDeltaIndex composite = new(CreateHnsw([(10UL, [0f]), (20UL, [1f])], efSearch: 2));
        AssertCommitted(composite.TryAdd(15, [0.5f]));
        SearchResult[] destination = [new(91, 91)];

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            [10UL, 15UL],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                maxEfSearch: 2,
                maxBaseCandidates: Math.Min(composite.BasePhysicalVectorCount, 2),
                maxDeltaCandidates: 1,
                maxDeltaFilterElements: 0),
            efSearch: 2));

        Assert.Equal(new SearchResult(91, 91), destination[0]);
    }

    [Fact]
    public void MutablePerSearchEfSearch_RejectsStaleWorkspaceAfterCommittedAddDeleteAndPublishedCheckpoint()
    {
        HnswMutableIndex mutable = CreateMutable([(10UL, [0f]), (20UL, [1f])], efSearch: 2);
        SearchResult[] destination = [new(71, 71), new(72, 72)];
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
        HnswMutableIndex checkpointMutable = CreateMutable([(100UL, [0f]), (200UL, [1f])], efSearch: 2);
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
        NotSupportedException defaultException = Assert.Throws<NotSupportedException>(
            () => new HnswIndex(2, VectorMetric.InnerProduct));
        NotSupportedException explicitException = Assert.Throws<NotSupportedException>(
            () => new HnswIndex(2, VectorMetric.InnerProduct, HnswIndexOptions.Default));

        Assert.Contains("inner product", defaultException.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("inner product", explicitException.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static HnswMutableIndex CreateMutable(IEnumerable<(ulong Id, float[] Vector)> rows, int efSearch) =>
        new(CreateHnsw(rows, efSearch));

    private static HnswIndex CreateHnsw(IEnumerable<(ulong Id, float[] Vector)> rows, int efSearch)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(M: 2, EfConstruction: 8, EfSearch: efSearch, RandomSeed: 0x294UL),
            () => 0);

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

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
                "VecNet-VEC294-" + Guid.NewGuid().ToString("N"));
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
