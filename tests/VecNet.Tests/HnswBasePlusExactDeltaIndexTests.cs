namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaIndexTests
{
    [Fact]
    public void Constructor_CapturesImmutableBaseCountsAndStartsWithEmptyOverlay()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 2f })]);

        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        Assert.Equal(1, composite.Dimension);
        Assert.Equal(VectorMetric.SquaredEuclidean, composite.Metric);
        Assert.Equal(2, composite.BasePhysicalVectorCount);
        Assert.Equal(2, composite.BaseLiveVectorCount);
        Assert.Equal(0, composite.DeltaPhysicalVectorCount);
        Assert.Equal(0, composite.DeltaLiveVectorCount);
        Assert.Equal(0, composite.TombstoneCount);
        Assert.Equal(0, composite.BaseTombstoneCount);
        Assert.Equal(0, composite.DeltaTombstoneCount);
        Assert.Equal(2, composite.LiveVectorCount);
        Assert.Equal(0, composite.DeletedReservedIdCount);
        Assert.Equal(0, composite.Generation);
    }

    [Fact]
    public void TryAddAndTryDelete_ReturnStatusesCountsAndGenerationWithoutPublicHnswMutation()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 1f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        VectorMutationResult added = composite.TryAdd(20, [0.5f]);

        Assert.Equal(VectorMutationStatus.Committed, added.Status);
        Assert.Equal(1, added.Generation);
        Assert.Equal(2, added.LiveVectorCount);
        Assert.Equal(1, added.DeltaVectorCount);
        Assert.Equal(0, added.TombstoneCount);
        Assert.Equal(1, composite.DeltaPhysicalVectorCount);
        Assert.Equal(1, baseIndex.Count);

        VectorMutationResult duplicateBase = composite.TryAdd(10, [2f]);
        VectorMutationResult duplicateDelta = composite.TryAdd(20, [2f]);

        Assert.Equal(VectorMutationStatus.DuplicateId, duplicateBase.Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, duplicateDelta.Status);
        Assert.Equal(1, composite.Generation);

        VectorMutationResult deletedBase = composite.TryDelete(10);
        VectorMutationResult deletedDelta = composite.TryDelete(20);

        Assert.Equal(VectorMutationStatus.Committed, deletedBase.Status);
        Assert.Equal(VectorMutationStatus.Committed, deletedDelta.Status);
        Assert.Equal(3, composite.Generation);
        Assert.Equal(0, composite.LiveVectorCount);
        Assert.Equal(1, composite.BaseTombstoneCount);
        Assert.Equal(1, composite.DeltaTombstoneCount);
        Assert.Equal(2, composite.DeletedReservedIdCount);

        Assert.Equal(VectorMutationStatus.AlreadyDeleted, composite.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, composite.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.UnknownId, composite.TryDelete(999).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(10, [3f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [3f]).Status);
        Assert.Equal(3, composite.Generation);
    }

    [Fact]
    public void ReadOnlyCompositeRejectsOverlayMutationsByStatus()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 1f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex, isReadOnly: true);

        Assert.Equal(VectorMutationStatus.ReadOnly, composite.TryAdd(20, [2f]).Status);
        Assert.Equal(VectorMutationStatus.ReadOnly, composite.TryDelete(10).Status);
        Assert.Equal(0, composite.Generation);
        Assert.Equal(1, composite.LiveVectorCount);
    }

    [Fact]
    public void Search_MergesBaseAndDeltaBySquaredL2ThenExternalId()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 1f }), (40UL, new[] { 4f })],
            new HnswIndexOptions(2, 8, 8, 0x123UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(20, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(5, [-1f]).Status);

        SearchResult[] results = [new(999, 999), new(999, 999), new(999, 999), new(999, 999)];
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: results.Length));

        Assert.Equal(4, written);
        Assert.Equal(
            [new SearchResult(20, 0.25f), new SearchResult(5, 1f), new SearchResult(10, 1f), new SearchResult(40, 16f)],
            results);
    }

    [Fact]
    public void Search_SuppressesBaseAndDeltaTombstonesAndKeepsDeletedIdsReserved()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })],
            new HnswIndexOptions(2, 8, 8, 0x124UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(5, [0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(5).Status);

        var results = new SearchResult[3];
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: 3));

        Assert.Equal(1, written);
        Assert.Equal(new SearchResult(20, 4f), results[0]);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(10, [10f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(5, [5f]).Status);
    }

    [Fact]
    public void Search_CanUnderfillWhenBaseOverfetchIsExhaustedByTombstones()
    {
        var options = new HnswIndexOptions(2, 4, 1, 0x125UL);
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 10f })],
            options);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(10).Status);

        SearchResult[] results = [new(999, 999)];
        int written = composite.Search([0f], results, CreateWorkspace(composite, topK: 1));

        Assert.Equal(0, written);
        Assert.Equal(new SearchResult(999, 999), results[0]);
        Assert.Equal(1, composite.LiveVectorCount);
    }

    [Fact]
    public void Search_ValidatesWorkspaceBeforeWritingAndCanBeRetried()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 1f })],
            new HnswIndexOptions(2, 8, 4, 0x126UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(5, [0.5f]).Status);

        SearchResult[] destination = [new(111, 111), new(222, 222)];

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(1, composite.Options.EfSearch, 2, 2)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch,
                1,
                2)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch,
                Math.Min(composite.BasePhysicalVectorCount, composite.Options.EfSearch),
                1)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);

        int written = composite.Search([0f], destination, CreateWorkspace(composite, topK: 2));

        Assert.Equal(2, written);
        Assert.Equal([10UL, 5UL], destination.Select(static result => result.Id));
    }

    [Fact]
    public void Search_RejectsRequestedCountLargerThanEfSearch()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 1f })],
            new HnswIndexOptions(2, 8, 1, 0x127UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        Assert.Throws<ArgumentOutOfRangeException>(() => composite.Search(
            [0f],
            new SearchResult[2],
            CreateWorkspace(composite, topK: 2)));
    }

    [Fact]
    public void OverlayMutationsDoNotChangeBaseGraphOrBaseSearch()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 1f }), (30UL, new[] { 2f }), (40UL, new[] { 3f })],
            new HnswIndexOptions(2, 8, 8, 0x128UL));
        string beforeGraph = CreateGraphSnapshot(baseIndex);
        SearchResult[] baseBefore = Search(baseIndex, [0f], topK: 4);

        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(5, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(10).Status);

        Assert.Equal(4, baseIndex.Count);
        Assert.Equal(beforeGraph, CreateGraphSnapshot(baseIndex));
        Assert.Equal(baseBefore, Search(baseIndex, [0f], topK: 4));
    }

    [Fact]
    public void SearchRejectsBaseChangedAfterCompositeConstruction()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        baseIndex.Add(20, [1f]);

        Assert.Throws<InvalidOperationException>(() => composite.Search(
            [0f],
            new SearchResult[1],
            CreateWorkspace(composite, topK: 1)));
    }

    [Fact]
    public void InvalidVectorsFailWithoutAdvancingGeneration()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f, 0f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        Assert.Throws<ArgumentException>(() => composite.TryAdd(20, [1f]));
        Assert.Throws<ArgumentException>(() => composite.TryAdd(20, [float.NaN, 0f]));
        Assert.Throws<ArgumentException>(() => composite.Search(
            [float.PositiveInfinity, 0f],
            new SearchResult[1],
            CreateWorkspace(composite, topK: 1)));

        Assert.Equal(0, composite.Generation);
        Assert.Equal(1, composite.LiveVectorCount);
    }

    private static HnswIndex CreateBaseIndex(
        IEnumerable<(ulong Id, float[] Vector)> rows,
        HnswIndexOptions? options = null)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            options ?? new HnswIndexOptions(2, 8, 8, 0x122UL),
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

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static string CreateGraphSnapshot(HnswIndex index)
    {
        var parts = new List<string>
        {
            $"count={index.Count};entry={index.EntryPoint};max={index.MaxLayer}"
        };

        for (int ordinal = 0; ordinal < index.Count; ordinal++)
        {
            parts.Add($"level[{ordinal}]={index.DebugGetLevel(ordinal)}");
        }

        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                parts.Add($"l{layer}n{ordinal}={string.Join(",", GetNeighbors(index, layer, ordinal))}");
            }
        }

        return string.Join("|", parts);
    }

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }
}
