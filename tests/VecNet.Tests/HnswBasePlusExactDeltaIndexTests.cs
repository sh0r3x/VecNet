namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaIndexTests
{
    [Fact]
    public void Checkpoint_PublishesLiveViewValidatesOutputAndClearsOverlay()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [
                (10UL, new[] { 0f }),
                (20UL, new[] { 1f }),
                (30UL, new[] { 2f }),
                (40UL, new[] { 3f })
            ],
            new HnswIndexOptions(2, 8, 8, 0x1310UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(25, [1.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(35, [2.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(25).Status);

        ulong[] expectedLiveIds = [10, 30, 40, 15, 35];
        SearchResult[] preResults = SearchComposite(composite, [0f], topK: 8);
        AssertReturnedIdsAreLiveAndFinite(preResults, expectedLiveIds);
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(result.Generation, composite.Generation);
        Assert.Equal(5, result.RebuiltBaseVectorCount);
        Assert.Equal(5, result.LiveVectorCount);
        Assert.Equal(5, result.BasePhysicalVectorCount);
        Assert.Equal(5, result.BaseLiveVectorCount);
        Assert.Equal(0, result.DeltaPhysicalVectorCount);
        Assert.Equal(0, result.DeltaLiveVectorCount);
        Assert.Equal(0, result.BaseTombstoneCount);
        Assert.Equal(0, result.DeltaTombstoneCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(2, result.FoldedDeltaVectorCount);
        Assert.Equal(1, result.FoldedBaseTombstoneCount);
        Assert.Equal(1, result.FoldedDeltaTombstoneCount);
        Assert.Equal(5, composite.BasePhysicalVectorCount);
        Assert.Equal(0, composite.DeltaPhysicalVectorCount);
        Assert.Equal(0, composite.TombstoneCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [10f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(25, [10f]).Status);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(expectedLiveIds, opened.InternalIds.ToArray());
        Assert.Throws<InvalidOperationException>(() => opened.Add(999, [9f]));

        SearchResult[] postResults = SearchComposite(composite, [0f], topK: 8);
        SearchResult[] openedResults = SearchHnsw(opened, [0f], topK: 8);
        Assert.Equal(postResults, openedResults);
        AssertReturnedIdsAreLiveAndFinite(postResults, expectedLiveIds);
    }

    [Fact]
    public void CheckpointWithDiagnostics_PublishedMeasuresAllPhasesAndMatchesCheckpointResultContract()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [
                (10UL, new[] { 0f, 0f }),
                (20UL, new[] { 1f, 1f }),
                (30UL, new[] { 2f, 2f })
            ],
            new HnswIndexOptions(2, 8, 8, 0x1330UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f, 0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointDiagnosticResult measured = composite.CheckpointWithDiagnostics(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, measured.Result.Status);
        Assert.Equal(beforeGeneration + 1, measured.Result.Generation);
        Assert.Equal(measured.Result.Generation, composite.Generation);
        Assert.Equal(3, measured.Result.RebuiltBaseVectorCount);
        Assert.Equal(3, measured.Result.LiveVectorCount);
        Assert.Equal(0, measured.Result.DeltaPhysicalVectorCount);
        Assert.Equal(0, measured.Result.TombstoneCount);
        Assert.Equal(1, measured.Result.FoldedDeltaVectorCount);
        Assert.Equal(1, measured.Result.FoldedBaseTombstoneCount);
        Assert.Equal(0, measured.Result.FoldedDeltaTombstoneCount);
        AssertMeasured(measured.Diagnostics.LiveSnapshot);
        AssertMeasured(measured.Diagnostics.RebuildBuild);
        AssertMeasured(measured.Diagnostics.Save);
        AssertMeasured(measured.Diagnostics.OpenValidation);
        AssertMeasured(measured.Diagnostics.Publication);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal([10UL, 30UL, 15UL], opened.InternalIds.ToArray());
        Assert.Equal(SearchComposite(composite, [0f, 0f], topK: 3), SearchHnsw(opened, [0f, 0f], topK: 3));
    }

    [Fact]
    public void Checkpoint_NoChangesWritesNoOutputAndDoesNotAdvanceGeneration()
    {
        using TempIndexDirectory missing = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })],
            new HnswIndexOptions(2, 8, 8, 0x1311UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(missing.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, result.Status);
        Assert.Equal(beforeGeneration, result.Generation);
        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.Equal(2, result.RebuiltBaseVectorCount);
        Assert.Equal(2, result.LiveVectorCount);
        Assert.Equal(2, result.BasePhysicalVectorCount);
        Assert.Equal(2, result.BaseLiveVectorCount);
        Assert.Equal(0, result.DeltaPhysicalVectorCount);
        Assert.Equal(0, result.DeltaLiveVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(0, result.DeletedReservedIdCount);
        Assert.False(Directory.Exists(missing.Path));
        Assert.Equal([10UL, 20UL], SearchComposite(composite, [0f], topK: 2).Select(static result => result.Id));
    }

    [Fact]
    public void CheckpointWithDiagnostics_NoChangesMarksAllPhasesNotExecuted()
    {
        using TempIndexDirectory missing = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })],
            new HnswIndexOptions(2, 8, 8, 0x1331UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointDiagnosticResult measured = composite.CheckpointWithDiagnostics(missing.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, measured.Result.Status);
        Assert.Equal(beforeGeneration, measured.Result.Generation);
        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.False(Directory.Exists(missing.Path));
        AssertNotExecuted(measured.Diagnostics.LiveSnapshot);
        AssertNotExecuted(measured.Diagnostics.RebuildBuild);
        AssertNotExecuted(measured.Diagnostics.Save);
        AssertNotExecuted(measured.Diagnostics.OpenValidation);
        AssertNotExecuted(measured.Diagnostics.Publication);
    }

    [Fact]
    public void Checkpoint_FailedTargetValidationLeavesOldCompositeSearchableAndUnchanged()
    {
        using TempIndexDirectory nonEmpty = TempIndexDirectory.Create();
        File.WriteAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt"), "keep");
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f }), (30UL, new[] { 4f })],
            new HnswIndexOptions(2, 8, 8, 0x1312UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        SearchResult[] expected = SearchComposite(composite, [0f], topK: 4);
        long beforeGeneration = composite.Generation;

        Assert.Throws<IOException>(() => composite.Checkpoint(nonEmpty.Path));

        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.Equal(3, composite.BasePhysicalVectorCount);
        Assert.Equal(1, composite.DeltaPhysicalVectorCount);
        Assert.Equal(1, composite.TombstoneCount);
        Assert.Equal(expected, SearchComposite(composite, [0f], topK: 4));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt")));
        Assert.False(File.Exists(Path.Combine(nonEmpty.Path, HnswIndexStorage.ManifestFileName)));
    }

    [Fact]
    public void CheckpointWithDiagnostics_FailedTargetValidationLeavesOldCompositeSearchableAndUnchanged()
    {
        using TempIndexDirectory nonEmpty = TempIndexDirectory.Create();
        File.WriteAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt"), "keep");
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f }), (30UL, new[] { 4f })],
            new HnswIndexOptions(2, 8, 8, 0x1332UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        SearchResult[] expected = SearchComposite(composite, [0f], topK: 4);
        long beforeGeneration = composite.Generation;

        Assert.Throws<IOException>(() => composite.CheckpointWithDiagnostics(nonEmpty.Path));

        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.Equal(3, composite.BasePhysicalVectorCount);
        Assert.Equal(1, composite.DeltaPhysicalVectorCount);
        Assert.Equal(1, composite.TombstoneCount);
        Assert.Equal(expected, SearchComposite(composite, [0f], topK: 4));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt")));
        Assert.False(File.Exists(Path.Combine(nonEmpty.Path, HnswIndexStorage.ManifestFileName)));
    }

    [Fact]
    public void Checkpoint_AllDeletedPublishesEmptyOutputAndRetainsReservations()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 1f })],
            new HnswIndexOptions(2, 8, 8, 0x1313UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(0, result.RebuiltBaseVectorCount);
        Assert.Equal(0, result.LiveVectorCount);
        Assert.Equal(0, result.BasePhysicalVectorCount);
        Assert.Equal(0, result.BaseLiveVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(2, result.FoldedBaseTombstoneCount);
        Assert.Equal([], SearchComposite(composite, [0f], topK: 4));

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(0, opened.Count);
        Assert.Equal([], SearchHnsw(opened, [0f], topK: 4));
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(10, [3f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [4f]).Status);
    }

    [Fact]
    public void Checkpoint_PublishedGenerationRejectsPreviouslyUsedWorkspaceBeforeWrite()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })],
            new HnswIndexOptions(2, 8, 8, 0x1314UL));
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [1f]).Status);
        var staleWorkspace = CreateWorkspace(composite, topK: 2);
        _ = composite.Search([0f], new SearchResult[2], staleWorkspace);

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);

        SearchResult[] destination = [new(111, 111), new(222, 222)];
        Assert.Throws<InvalidOperationException>(() => composite.Search([0f], destination, staleWorkspace));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);
        Assert.Equal(2, composite.Search([0f], destination, CreateWorkspace(composite, topK: 2)));
    }

    [Fact]
    public void Checkpoint_ReadOnlyCompositeThrowsWithoutWritingOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f })]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex, isReadOnly: true);

        Assert.Throws<InvalidOperationException>(() => composite.Checkpoint(checkpoint.Path));

        Assert.False(Directory.Exists(checkpoint.Path));
        Assert.Equal(0, composite.Generation);
    }

    [Fact]
    public void CheckpointResultAndStatusSurfaceStayInternalAndNarrow()
    {
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointStatus).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnosticResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseStatus).IsNotPublic);
        Assert.Equal(
            ["NoChanges", "Published"],
            Enum.GetNames<HnswBasePlusExactDeltaCheckpointStatus>().Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(
            ["Failed", "Measured", "NotExecuted"],
            Enum.GetNames<HnswBasePlusExactDeltaCheckpointPhaseStatus>().Order(StringComparer.Ordinal).ToArray());
    }

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

    private static SearchResult[] SearchComposite(HnswBasePlusExactDeltaIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, CreateWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static void AssertReturnedIdsAreLiveAndFinite(SearchResult[] results, IReadOnlyCollection<ulong> liveIds)
    {
        var live = new HashSet<ulong>(liveIds);
        foreach (SearchResult result in results)
        {
            Assert.True(live.Contains(result.Id), $"Unexpected ID {result.Id}.");
            Assert.True(float.IsFinite(result.Distance), $"Distance for ID {result.Id} was not finite.");
        }
    }

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        Assert.Equal(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured, diagnostics.Status);
        Assert.True(diagnostics.ElapsedTicks >= 0);
        Assert.True(diagnostics.ManagedAllocatedBytes >= 0);
    }

    private static void AssertNotExecuted(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        Assert.Equal(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted, diagnostics.Status);
        Assert.Equal(0, diagnostics.ElapsedTicks);
        Assert.Equal(0, diagnostics.ManagedAllocatedBytes);
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

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path)
        {
            Path = path;
        }

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = CreatePath();
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public static TempIndexDirectory CreateMissing() => new(CreatePath());

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

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswCompositeCheckpointTests-" + Guid.NewGuid().ToString("N"));
    }
}
