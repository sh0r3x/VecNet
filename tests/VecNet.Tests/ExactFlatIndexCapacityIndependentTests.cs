namespace VecNet.Tests;

public sealed class ExactFlatIndexCapacityIndependentTests
{
    [Fact]
    public void Vec168_CapacityStatesPreserveCountsGenerationAndTieOrdering()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, 3);
        var emptyResults = new SearchResult[4];

        Assert.Equal(3, index.Capacity);
        AssertCountState(index, physical: 0, live: 0, tombstones: 0, generation: 0);
        Assert.Equal(0, index.Search([0f, 0f], emptyResults));

        index.Add(30, [3f, 0f]);
        index.Add(10, [1f, 0f]);
        Assert.Equal(3, index.Capacity);
        AssertCountState(index, physical: 2, live: 2, tombstones: 0, generation: 2);
        AssertSearchIds(index, [0f, 0f], topK: 8, [10, 30]);

        index.Add(20, [-1f, 0f]);
        Assert.Equal(3, index.Capacity);
        AssertCountState(index, physical: 3, live: 3, tombstones: 0, generation: 3);
        AssertSearchIds(index, [0f, 0f], topK: 8, [10, 20, 30]);

        index.Add(5, [2f, 0f]);
        Assert.True(index.Capacity >= 4);
        AssertCountState(index, physical: 4, live: 4, tombstones: 0, generation: 4);
        AssertSearchIds(index, [0f, 0f], topK: 8, [10, 20, 5, 30]);
    }

    [Fact]
    public void Vec168_EnsureCapacityKeepsReusableCandidatesGenerationBoundAndWorkspacePhysicalCountBound()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, 4);
        index.Add(100, [10f]);
        index.Add(200, [2f]);
        index.Add(300, [3f]);

        ulong[] scope = [300, 999, 100, 300];
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(scope);
        var candidateResults = new SearchResult[4];
        Assert.Equal([300UL, 100UL], SearchIds(index, [0f], candidates, candidateResults));

        var workspace = new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount);
        var rawResults = new SearchResult[4];
        Assert.Equal([300UL, 100UL], SearchIds(index, [0f], scope, rawResults, workspace));

        long generation = index.Generation;
        index.EnsureCapacity(64);

        Assert.Equal(64, index.Capacity);
        Assert.Equal(generation, index.Generation);
        Assert.Equal([300UL, 100UL], SearchIds(index, [0f], candidates, candidateResults));
        Assert.Equal([300UL, 100UL], SearchIds(index, [0f], scope, rawResults, workspace));

        index.Add(50, [0.5f]);

        Assert.Throws<InvalidOperationException>(() => index.Search([0f], candidates, candidateResults));
        Assert.Throws<ArgumentException>(() => index.Search([0f], scope, rawResults, workspace));
        Assert.Equal(
            [50UL, 300UL, 100UL],
            SearchIds(index, [0f], [50UL, 300UL, 100UL], rawResults, new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount)));
    }

    [Fact]
    public void Vec168_InvalidEnsureCapacityRequestsDoNotMutateTombstoneFilterOrCandidateState()
    {
        var index = new ExactFlatIndex(4, VectorMetric.SquaredEuclidean, 6);
        index.Add(10, [1f, 0f, 0f, 0f]);
        index.Add(20, [2f, 0f, 0f, 0f]);
        index.Add(30, [3f, 0f, 0f, 0f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);

        ulong[] scope = [20, 30, 10, 10, 999];
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(scope);
        var candidateResults = new SearchResult[4];
        Assert.Equal([10UL, 30UL], SearchIds(index, [0f, 0f, 0f, 0f], candidates, candidateResults));

        int capacity = index.Capacity;
        long generation = index.Generation;
        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity(-8));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity((int.MaxValue / index.Dimension) + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity((Array.MaxLength / index.Dimension) + 1));

        Assert.Equal(capacity, index.Capacity);
        AssertCountState(index, physical: 3, live: 2, tombstones: 1, generation);
        Assert.Equal(1, index.DeletedReservedIdCount);
        Assert.Equal([10UL, 30UL], SearchIds(index, [0f, 0f, 0f, 0f], candidates, candidateResults));
        Assert.Equal(
            [10UL, 30UL],
            SearchIds(
                index,
                [0f, 0f, 0f, 0f],
                scope,
                new SearchResult[4],
                new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount)));
    }

    [Fact]
    public void Vec168_SaveOpenEmptyPlannedIndexIsCompactAndReadOnlyEnsureCapacityAlwaysRejects()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(5, VectorMetric.InnerProduct, 12);

        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Equal(12, index.Capacity);
        Assert.Equal(0, opened.Capacity);
        AssertCountState(opened, physical: 0, live: 0, tombstones: 0, generation: 0);
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(0));
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(-1));
        Assert.Equal(0, opened.Search([1f, 0f, 0f, 0f, 0f], new SearchResult[3]));
    }

    [Fact]
    public void Vec168_CosinePlanningRejectsZeroVectorsWithoutReservingIdsAndNormalizesAfterGrowth()
    {
        var index = new ExactFlatIndex(3, VectorMetric.Cosine, 1);
        index.Add(1, [10f, 0f, 0f]);

        long generationBeforeInvalidAdd = index.Generation;
        Assert.Throws<ArgumentException>(() => index.Add(99, [0f, 0f, 0f]));
        Assert.Equal(generationBeforeInvalidAdd, index.Generation);
        Assert.Equal(1, index.PhysicalVectorCount);

        index.EnsureCapacity(3);
        index.Add(2, [0f, 5f, 0f]);
        index.Add(3, [0f, 0f, -7f]);
        index.Add(99, [0f, 2f, 0f]);
        index.Add(4, [0f, -2f, 0f]);

        Assert.True(index.Capacity >= 5);
        AssertCountState(index, physical: 5, live: 5, tombstones: 0, generation: 5);

        SearchResult[] results = Search(index, [0f, 10f, 0f], topK: 5);
        Assert.Equal([2UL, 99UL, 1UL, 3UL, 4UL], results.Select(static result => result.Id).ToArray());
        Assert.Equal(0f, results[0].Distance);
        Assert.Equal(0f, results[1].Distance);
        Assert.Equal(1f, results[2].Distance);
        Assert.Equal(1f, results[3].Distance);
        Assert.Equal(2f, results[4].Distance);
    }

    [Fact]
    public void Vec168_CheckpointCompactionPreservesDeletedIdReservationAndAllowsLaterCapacityPlanning()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, 10);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        index.Add(30, [3f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(40, [4f]).Status);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(3, index.Capacity);
        AssertCountState(index, physical: 3, live: 3, tombstones: 0, generation: 6);
        Assert.Equal(1, index.DeletedReservedIdCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(20, [2f]).Status);

        index.EnsureCapacity(8);
        Assert.Equal(8, index.Capacity);
        AssertCountState(index, physical: 3, live: 3, tombstones: 0, generation: 6);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(50, [0.5f]).Status);
        AssertSearchIds(index, [0f], topK: 8, [50, 30, 40, 10]);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(3, opened.Capacity);
        AssertSearchIds(opened, [0f], topK: 8, [30, 40, 10]);
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(8));
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static ulong[] SearchIds(ExactFlatIndex index, float[] query, int topK) =>
        Search(index, query, topK).Select(static result => result.Id).ToArray();

    private static ulong[] SearchIds(
        ExactFlatIndex index,
        float[] query,
        ExactFlatCandidateSet candidates,
        SearchResult[] results)
    {
        int written = index.Search(query, candidates, results);
        return results[..written].Select(static result => result.Id).ToArray();
    }

    private static ulong[] SearchIds(
        ExactFlatIndex index,
        float[] query,
        ulong[] allowlist,
        SearchResult[] results,
        ExactFlatSearchFilterWorkspace workspace)
    {
        int written = index.Search(query, allowlist, results, workspace);
        return results[..written].Select(static result => result.Id).ToArray();
    }

    private static void AssertSearchIds(ExactFlatIndex index, float[] query, int topK, ulong[] expected) =>
        Assert.Equal(expected, SearchIds(index, query, topK));

    private static void AssertCountState(
        ExactFlatIndex index,
        int physical,
        int live,
        int tombstones,
        long generation)
    {
        Assert.Equal(physical, index.VectorCount);
        Assert.Equal(physical, index.PhysicalVectorCount);
        Assert.Equal(live, index.LiveVectorCount);
        Assert.Equal(tombstones, index.TombstoneCount);
        Assert.Equal(generation, index.Generation);
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() => new(CreatePath());

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
    }
}
