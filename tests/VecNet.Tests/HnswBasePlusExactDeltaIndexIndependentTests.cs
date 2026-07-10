namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaIndexIndependentTests
{
    [Fact]
    public void FailedStatusMutationsReturnCurrentSnapshotWithoutGenerationMovement()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 2f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        AssertCommitted(composite.TryAdd(30, [1f]), generation: 1, liveCount: 3, deltaLiveCount: 1, tombstoneCount: 0);
        AssertCommitted(composite.TryDelete(20), generation: 2, liveCount: 2, deltaLiveCount: 1, tombstoneCount: 1);

        VectorMutationResult duplicateBase = composite.TryAdd(10, [10f]);
        VectorMutationResult duplicateDelta = composite.TryAdd(30, [30f]);
        VectorMutationResult duplicateReserved = composite.TryAdd(20, [20f]);
        VectorMutationResult unknownDelete = composite.TryDelete(999);
        VectorMutationResult alreadyDeleted = composite.TryDelete(20);

        AssertSnapshot(duplicateBase, VectorMutationStatus.DuplicateId);
        AssertSnapshot(duplicateDelta, VectorMutationStatus.DuplicateId);
        AssertSnapshot(duplicateReserved, VectorMutationStatus.DuplicateId);
        AssertSnapshot(unknownDelete, VectorMutationStatus.UnknownId);
        AssertSnapshot(alreadyDeleted, VectorMutationStatus.AlreadyDeleted);
        Assert.Equal(2, composite.Generation);
        Assert.Equal(2, composite.LiveVectorCount);
        Assert.Equal(1, composite.DeltaLiveVectorCount);
        Assert.Equal(1, composite.TombstoneCount);
        Assert.Equal(1, composite.DeletedReservedIdCount);

        void AssertSnapshot(VectorMutationResult result, VectorMutationStatus status)
        {
            Assert.Equal(status, result.Status);
            Assert.Equal(2, result.Generation);
            Assert.Equal(2, result.LiveVectorCount);
            Assert.Equal(1, result.DeltaVectorCount);
            Assert.Equal(1, result.TombstoneCount);
        }
    }

    [Fact]
    public void TombstonedDeltaRowsRemainPhysicalButDoNotAffectSearchOrLiveCounts()
    {
        HnswIndex baseIndex = CreateBaseIndex([(100UL, new[] { 100f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        AssertCommitted(composite.TryAdd(1, [0f]), generation: 1, liveCount: 2, deltaLiveCount: 1, tombstoneCount: 0);
        AssertCommitted(composite.TryAdd(2, [0.25f]), generation: 2, liveCount: 3, deltaLiveCount: 2, tombstoneCount: 0);
        AssertCommitted(composite.TryAdd(3, [0.5f]), generation: 3, liveCount: 4, deltaLiveCount: 3, tombstoneCount: 0);
        AssertCommitted(composite.TryDelete(1), generation: 4, liveCount: 3, deltaLiveCount: 2, tombstoneCount: 1);

        SearchResult[] results = [new(999, 999), new(999, 999), new(999, 999)];
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: 3));

        Assert.Equal(3, composite.DeltaPhysicalVectorCount);
        Assert.Equal(2, composite.DeltaLiveVectorCount);
        Assert.Equal(1, composite.DeltaTombstoneCount);
        Assert.Equal(3, written);
        Assert.Equal(
            [new SearchResult(2, 0.0625f), new SearchResult(3, 0.25f), new SearchResult(100, 10000f)],
            results);
    }

    [Fact]
    public void SearchMergesBaseTombstonesAndDeltaCandidatesInOrderForLargerTopK()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [
                (10UL, new[] { 0f }),
                (20UL, new[] { 1f }),
                (30UL, new[] { 2f }),
                (40UL, new[] { 3f }),
                (50UL, new[] { 4f }),
                (60UL, new[] { 5f })
            ]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        AssertCommitted(composite.TryAdd(15, [0.5f]), generation: 1, liveCount: 7, deltaLiveCount: 1, tombstoneCount: 0);
        AssertCommitted(composite.TryAdd(35, [2.5f]), generation: 2, liveCount: 8, deltaLiveCount: 2, tombstoneCount: 0);
        AssertCommitted(composite.TryAdd(55, [4.5f]), generation: 3, liveCount: 9, deltaLiveCount: 3, tombstoneCount: 0);
        AssertCommitted(composite.TryDelete(20), generation: 4, liveCount: 8, deltaLiveCount: 3, tombstoneCount: 1);
        AssertCommitted(composite.TryDelete(40), generation: 5, liveCount: 7, deltaLiveCount: 3, tombstoneCount: 2);

        SearchResult[] results = Enumerable.Repeat(new SearchResult(999, 999), 6).ToArray();
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: results.Length));

        Assert.Equal(6, written);
        Assert.Equal(
            [
                new SearchResult(10, 0f),
                new SearchResult(15, 0.25f),
                new SearchResult(30, 4f),
                new SearchResult(35, 6.25f),
                new SearchResult(50, 16f),
                new SearchResult(55, 20.25f)
            ],
            results);
    }

    [Fact]
    public void EmptyDestinationValidatesQueryAndWorkspaceButWritesNothing()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        AssertCommitted(composite.TryAdd(20, [1f]), generation: 1, liveCount: 2, deltaLiveCount: 1, tombstoneCount: 0);

        int written = composite.Search([0f], Span<SearchResult>.Empty, CreateWorkspace(composite, topK: 0));

        Assert.Equal(0, written);
        Assert.Equal(1, composite.Generation);
        Assert.Throws<ArgumentException>(() => composite.Search([float.NaN], Span<SearchResult>.Empty, CreateWorkspace(composite, topK: 0)));
    }

    [Fact]
    public void WorkspaceEfUndersizingAndInvalidQueryLeaveDestinationUntouched()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 1f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        AssertCommitted(composite.TryAdd(5, [0.5f]), generation: 1, liveCount: 3, deltaLiveCount: 1, tombstoneCount: 0);
        SearchResult[] destination = [new(111, 111), new(222, 222)];

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch - 1,
                Math.Min(composite.BasePhysicalVectorCount, composite.Options.EfSearch),
                destination.Length)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f, 1f],
            destination,
            CreateWorkspace(composite, topK: destination.Length)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);
    }

    [Fact]
    public void EqualDistanceBaseAndDeltaCandidatesTieByExternalId()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(30UL, new[] { -1f }), (5UL, new[] { 2f }), (50UL, new[] { 4f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        AssertCommitted(composite.TryAdd(20, [1f]), generation: 1, liveCount: 4, deltaLiveCount: 1, tombstoneCount: 0);
        AssertCommitted(composite.TryAdd(40, [-1f]), generation: 2, liveCount: 5, deltaLiveCount: 2, tombstoneCount: 0);

        SearchResult[] results = new SearchResult[4];
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: 4));

        Assert.Equal(4, written);
        Assert.Equal(
            [new SearchResult(20, 1f), new SearchResult(30, 1f), new SearchResult(40, 1f), new SearchResult(5, 4f)],
            results);
    }

    [Fact]
    public void BaseMutationAfterConstructionIsRejectedBeforeDestinationWrite()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 2f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        SearchResult[] destination = [new(111, 111)];

        baseIndex.Add(30, [1f]);

        Assert.Throws<InvalidOperationException>(() => composite.Search(
            [0f],
            destination,
            CreateWorkspace(composite, topK: destination.Length)));
        Assert.Equal(new SearchResult(111, 111), destination[0]);
        Assert.Equal(0, composite.Generation);
        Assert.Equal(2, composite.LiveVectorCount);
    }

    [Fact]
    public void InvalidAddVectorsThrowWithoutReservingIdOrChangingState()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f, 0f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        Assert.Throws<ArgumentException>(() => composite.TryAdd(20, [1f]));
        Assert.Throws<ArgumentException>(() => composite.TryAdd(20, [1f, float.NegativeInfinity]));

        Assert.Equal(0, composite.Generation);
        Assert.Equal(1, composite.LiveVectorCount);
        Assert.Equal(0, composite.DeltaPhysicalVectorCount);
        Assert.Equal(0, composite.DeletedReservedIdCount);
        AssertCommitted(composite.TryAdd(20, [1f, 1f]), generation: 1, liveCount: 2, deltaLiveCount: 1, tombstoneCount: 0);
    }

    [Fact]
    public void ReadOnlyCompositeReportsCurrentCountsAndDoesNotChangeOverlayState()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 1f })]);
        var readOnly = new HnswBasePlusExactDeltaIndex(baseIndex, isReadOnly: true);

        VectorMutationResult add = readOnly.TryAdd(30, [2f]);
        VectorMutationResult delete = readOnly.TryDelete(10);

        AssertReadOnlySnapshot(add);
        AssertReadOnlySnapshot(delete);
        Assert.Equal(0, readOnly.Generation);
        Assert.Equal(2, readOnly.LiveVectorCount);
        Assert.Equal(0, readOnly.DeltaPhysicalVectorCount);
        Assert.Equal(0, readOnly.TombstoneCount);
        Assert.Equal(0, readOnly.DeletedReservedIdCount);

        void AssertReadOnlySnapshot(VectorMutationResult result)
        {
            Assert.Equal(VectorMutationStatus.ReadOnly, result.Status);
            Assert.Equal(0, result.Generation);
            Assert.Equal(2, result.LiveVectorCount);
            Assert.Equal(0, result.DeltaVectorCount);
            Assert.Equal(0, result.TombstoneCount);
        }
    }

    private static HnswIndex CreateBaseIndex(IEnumerable<(ulong Id, float[] Vector)> rows)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(8, 32, 32, 0x123_123UL),
            () => 0);

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK);

    private static void AssertCommitted(
        VectorMutationResult result,
        long generation,
        int liveCount,
        int deltaLiveCount,
        int tombstoneCount)
    {
        Assert.Equal(VectorMutationStatus.Committed, result.Status);
        Assert.Equal(generation, result.Generation);
        Assert.Equal(liveCount, result.LiveVectorCount);
        Assert.Equal(deltaLiveCount, result.DeltaVectorCount);
        Assert.Equal(tombstoneCount, result.TombstoneCount);
    }
}
