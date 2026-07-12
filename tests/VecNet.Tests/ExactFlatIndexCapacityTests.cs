namespace VecNet.Tests;

public sealed class ExactFlatIndexCapacityTests
{
    [Fact]
    public void ConstructorCapacity_ValidatesNegativeOverflowAndContiguousArrayLimit()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, -1));

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ExactFlatIndex((int.MaxValue / 2) + 1, VectorMetric.SquaredEuclidean, 2));

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, Array.MaxLength + 1));
    }

    [Fact]
    public void ConstructorCapacity_PreallocatesWithoutChangingCountsGenerationOrSearch()
    {
        var planned = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, 8);

        Assert.Equal(8, planned.Capacity);
        Assert.Equal(0, planned.PhysicalVectorCount);
        Assert.Equal(0, planned.LiveVectorCount);
        Assert.Equal(0, planned.Generation);
        Assert.Equal(0, planned.Search([0f, 0f], new SearchResult[4]));

        planned.Add(20, [2f, 0f]);
        planned.Add(10, [1f, 0f]);

        var results = new SearchResult[2];
        int written = planned.Search([0f, 0f], results);

        Assert.Equal(8, planned.Capacity);
        Assert.Equal(2, planned.PhysicalVectorCount);
        Assert.Equal(2, planned.LiveVectorCount);
        Assert.Equal(2, planned.Generation);
        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results[..written].Select(static result => result.Id));

        var unplanned = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        Assert.Equal(0, unplanned.Capacity);
        unplanned.Add(1, [0f, 0f]);
        Assert.Equal(4, unplanned.Capacity);
    }

    [Fact]
    public void EnsureCapacity_ValidatesAndPreallocatesWithoutChangingLogicalState()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        long generation = index.Generation;
        int count = index.PhysicalVectorCount;

        index.EnsureCapacity(12);

        Assert.Equal(12, index.Capacity);
        Assert.Equal(count, index.PhysicalVectorCount);
        Assert.Equal(count, index.LiveVectorCount);
        Assert.Equal(generation, index.Generation);
        Assert.Equal([10UL], SearchIds(index, [0f, 0f], topK: 4));

        index.EnsureCapacity(3);
        Assert.Equal(12, index.Capacity);
        Assert.Equal(generation, index.Generation);

        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ExactFlatIndex((int.MaxValue / 2) + 1, VectorMetric.SquaredEuclidean).EnsureCapacity(2));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ExactFlatIndex(1, VectorMetric.SquaredEuclidean).EnsureCapacity(Array.MaxLength + 1));

        Assert.Equal(12, index.Capacity);
        Assert.Equal(count, index.PhysicalVectorCount);
        Assert.Equal(generation, index.Generation);
    }

    [Fact]
    public void AddBeyondPlannedCapacity_GrowsAndPreservesRowsTombstonesAndIdMappings()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, 2);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);

        index.Add(30, [3f]);
        index.Add(40, [4f]);
        index.Add(50, [5f]);

        Assert.True(index.Capacity >= 5);
        Assert.Equal(5, index.PhysicalVectorCount);
        Assert.Equal(4, index.LiveVectorCount);
        Assert.Equal(1, index.TombstoneCount);
        Assert.Throws<ArgumentException>(() => index.Add(20, [200f]));
        Assert.Throws<ArgumentException>(() => index.Add(10, [100f]));
        Assert.Equal([30UL, 40UL, 50UL, 20UL], SearchIds(index, [0f], topK: 8));
    }

    [Fact]
    public void DuplicateDeleteRawAllowlistAndCandidateSetBehaviorSurvivePreallocation()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, 16);
        index.Add(10, [10f]);
        index.Add(20, [2f]);
        index.Add(30, [3f]);
        index.Add(40, [4f]);

        Assert.Throws<ArgumentException>(() => index.Add(20, [20f]));
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(30).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(30, [30f]).Status);

        ulong[] allowlist = [40, 30, 20, 20, 999, 10];
        var raw = new SearchResult[4];
        int rawWritten = index.Search(
            [0f],
            allowlist,
            raw,
            new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));

        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowlist);
        var candidate = new SearchResult[4];
        int candidateWritten = index.Search([0f], candidates, candidate);

        Assert.Equal(3, candidates.Count);
        Assert.Equal(3, rawWritten);
        Assert.Equal(raw[..rawWritten], candidate[..candidateWritten]);
        Assert.Equal([20UL, 40UL, 10UL], raw[..rawWritten].Select(static result => result.Id));
        Assert.Equal(16, index.Capacity);
    }

    [Fact]
    public void SaveOpenAndCheckpoint_CompactMutableOvercapacityToLiveRows()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, 20);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [3f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);

        Assert.Equal(20, index.Capacity);
        Assert.Equal([30UL, 10UL], SearchIds(index, [0f], topK: 4));

        index.Save(saved.Path);
        ExactFlatIndex openedSaved = ExactFlatIndex.OpenReadOnly(saved.Path);

        Assert.Equal(2, openedSaved.Capacity);
        Assert.Equal(2, openedSaved.PhysicalVectorCount);
        Assert.Equal(2, openedSaved.LiveVectorCount);
        Assert.Equal([30UL, 10UL], SearchIds(openedSaved, [0f], topK: 4));

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);
        ExactFlatIndex openedCheckpoint = ExactFlatIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(2, index.Capacity);
        Assert.Equal(2, index.PhysicalVectorCount);
        Assert.Equal(2, index.LiveVectorCount);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(1, index.DeletedReservedIdCount);
        Assert.Equal(2, openedCheckpoint.Capacity);
        Assert.Equal([30UL, 10UL], SearchIds(index, [0f], topK: 4));
        Assert.Equal([30UL, 10UL], SearchIds(openedCheckpoint, [0f], topK: 4));
    }

    [Fact]
    public void ReadOnlyEnsureCapacity_RejectsExplicitlyWithoutChangingCompactView()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, 8);
        index.Add(10, [1f, 0f]);
        index.Add(20, [0f, 1f]);
        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Equal(2, opened.Capacity);
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(2));
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(10));
        Assert.Equal(2, opened.Capacity);
        Assert.Equal(2, opened.PhysicalVectorCount);
        Assert.Equal([10UL, 20UL], SearchIds(opened, [0f, 0f], topK: 4));
    }

    [Fact]
    public void CapacityPlanning_PreservesCosineNormalization()
    {
        var index = new ExactFlatIndex(2, VectorMetric.Cosine, 4);
        index.Add(1, [0f, 5f]);
        index.Add(2, [2f, 0f]);

        var results = new SearchResult[2];
        int written = index.Search([0f, 2f], results);

        Assert.Equal(4, index.Capacity);
        Assert.Equal(2, written);
        Assert.Equal(1UL, results[0].Id);
        Assert.Equal(0f, results[0].Distance);
        Assert.Equal(2UL, results[1].Id);
        Assert.Equal(1f, results[1].Distance);
    }

    [Fact]
    public void CapacityPlannedPublicExactSearch_RemainsAllocationFreeAfterWarmup()
    {
        var index = new ExactFlatIndex(8, VectorMetric.SquaredEuclidean, 512);
        for (int row = 0; row < 512; row++)
        {
            var vector = new float[8];
            vector[0] = row % 29;
            vector[1] = row / 29;
            index.Add((ulong)(10_000 + row), vector);
        }

        float[] query = [5f, 3f, 0f, 0f, 0f, 0f, 0f, 0f];
        var results = new SearchResult[100];

        ExactFlatAllocationSmoke.AssertUnfilteredSearchDoesNotAllocateAfterWarmup(
            index,
            query,
            results,
            expectedWritten: 100);
    }

    private static ulong[] SearchIds(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written].Select(static result => result.Id).ToArray();
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
