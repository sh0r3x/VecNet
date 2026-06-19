namespace VecNet.Tests;

public sealed class ExactFlatIndexCheckpointTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Checkpoint_PublishesCompactLiveViewAndValidatedDurableOutput(VectorMetric metric)
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, metric);
        index.Add(10, CreateVector(metric, 10));
        index.Add(20, CreateVector(metric, 20));
        index.Add(30, CreateVector(metric, 30));
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(40, CreateVector(metric, 40)).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(50, CreateVector(metric, 50)).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(50).Status);

        float[] query = CreateQuery(metric);
        SearchResult[] expectedUnfiltered = SearchAll(index, query, topK: 10);
        SearchResult[] expectedRaw = SearchRaw(index, query, [50, 40, 20, 30, 10, 40, 999], topK: 10);
        ExactFlatCandidateSet oldCandidates = index.CreateCandidateSet([10, 20, 30, 40, 50, 999]);
        SearchResult[] expectedCandidate = SearchCandidates(index, query, oldCandidates, topK: 10);
        long beforeGeneration = index.Generation;
        Assert.Equal(5, index.VectorCount);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(result.Generation, index.Generation);
        Assert.Equal(3, result.PhysicalVectorCount);
        Assert.Equal(3, result.LiveVectorCount);
        Assert.Equal(3, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(1, result.FoldedDeltaVectorCount);
        Assert.Equal(2, result.FoldedTombstoneCount);
        Assert.Equal(3, index.VectorCount);

        Assert.Equal(expectedUnfiltered, SearchAll(index, query, topK: 10));
        Assert.Equal(expectedRaw, SearchRaw(index, query, [50, 40, 20, 30, 10, 40, 999], topK: 10));
        ExactFlatCandidateSet newCandidates = index.CreateCandidateSet([10, 20, 30, 40, 50, 999]);
        Assert.Equal(expectedCandidate, SearchCandidates(index, query, newCandidates, topK: 10));

        var staleDestination = new[] { new SearchResult(123, 456) };
        Assert.Throws<InvalidOperationException>(() => index.Search(query, oldCandidates, staleDestination));
        Assert.Equal(new SearchResult(123, 456), staleDestination[0]);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(3, opened.VectorCount);
        Assert.Equal(expectedUnfiltered, SearchAll(opened, query, topK: 10));
        Assert.Equal(expectedRaw, SearchRaw(opened, query, [50, 40, 20, 30, 10, 40, 999], topK: 10));

        VectorMutationResult reservedBase = index.TryAdd(20, CreateVector(metric, 20));
        VectorMutationResult reservedDelta = index.TryAdd(50, CreateVector(metric, 50));
        Assert.Equal(VectorMutationStatus.DuplicateId, reservedBase.Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, reservedDelta.Status);
        Assert.Equal(0, reservedBase.TombstoneCount);
        Assert.Equal(0, reservedDelta.TombstoneCount);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(20).Status);
    }

    [Fact]
    public void Checkpoint_NoChangesDoesNotWriteAdvanceGenerationOrStaleCandidateSets()
    {
        using TempIndexDirectory missing = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        index.Add(20, [0f, 1f]);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet([10, 20]);
        long beforeGeneration = index.Generation;

        ExactFlatCheckpointResult result = index.Checkpoint(missing.Path);

        Assert.Equal(ExactFlatCheckpointStatus.NoChanges, result.Status);
        Assert.Equal(beforeGeneration, result.Generation);
        Assert.Equal(beforeGeneration, index.Generation);
        Assert.Equal(2, result.PhysicalVectorCount);
        Assert.Equal(2, result.LiveVectorCount);
        Assert.Equal(2, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(0, result.DeletedReservedIdCount);
        Assert.Equal(0, result.FoldedDeltaVectorCount);
        Assert.Equal(0, result.FoldedTombstoneCount);
        Assert.False(Directory.Exists(missing.Path));

        var results = new SearchResult[2];
        int written = index.Search([0f, 0f], candidates, results);
        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results[..written].Select(static item => item.Id));
    }

    [Fact]
    public void Checkpoint_FailedTargetValidationLeavesOldStateAndCandidateSetsUsable()
    {
        using TempIndexDirectory nonEmpty = TempIndexDirectory.Create();
        File.WriteAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt"), "keep");
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        index.Add(20, [2f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet([10, 20, 30]);
        SearchResult[] expected = SearchAll(index, [0f], topK: 4);
        long beforeGeneration = index.Generation;

        Assert.Throws<IOException>(() => index.Checkpoint(nonEmpty.Path));

        Assert.Equal(beforeGeneration, index.Generation);
        Assert.Equal(3, index.VectorCount);
        Assert.Equal(expected, SearchAll(index, [0f], topK: 4));
        Assert.Equal(expected, SearchCandidates(index, [0f], candidates, topK: 4));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt")));
        Assert.False(File.Exists(Path.Combine(nonEmpty.Path, ExactFlatIndexStorage.ManifestFileName)));
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(40, [4f]).Status);
    }

    [Fact]
    public void Checkpoint_ReadOnlyIndexThrowsWithoutWritingOutput()
    {
        using TempIndexDirectory source = TempIndexDirectory.Create();
        using TempIndexDirectory output = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        index.Save(source.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(source.Path);

        Assert.Throws<InvalidOperationException>(() => opened.Checkpoint(output.Path));

        Assert.False(Directory.Exists(output.Path));
    }

    [Fact]
    public void Checkpoint_AllDeletedIndexPublishesEmptyOutputAndRetainsReservations()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        index.Add(20, [2f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        long beforeGeneration = index.Generation;

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(0, result.PhysicalVectorCount);
        Assert.Equal(0, result.LiveVectorCount);
        Assert.Equal(0, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(0, result.FoldedDeltaVectorCount);
        Assert.Equal(2, result.FoldedTombstoneCount);
        Assert.Equal(0, index.VectorCount);
        Assert.Equal([], SearchAll(index, [0f], topK: 4));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(0, opened.VectorCount);
        Assert.Equal([], SearchAll(opened, [0f], topK: 4));
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(10, [3f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(20, [4f]).Status);
    }

    private static SearchResult[] SearchAll(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] SearchRaw(ExactFlatIndex index, float[] query, ulong[] scope, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, scope, results, new ExactFlatSearchFilterWorkspace(index.VectorCount));
        return results[..written];
    }

    private static SearchResult[] SearchCandidates(
        ExactFlatIndex index,
        float[] query,
        ExactFlatCandidateSet candidates,
        int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, candidates, results);
        return results[..written];
    }

    private static float[] CreateQuery(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? [1f, 0.25f, 0.5f] : [0.25f, -0.5f, 1f];

    private static float[] CreateVector(VectorMetric metric, ulong id) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => id switch
            {
                10 => [1f, 0f, 0f],
                20 => [0f, 1f, 0f],
                30 => [0f, 0f, 1f],
                40 => [1f, 1f, 0f],
                50 => [1f, 0f, 1f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            VectorMetric.InnerProduct => id switch
            {
                10 => [1f, 0f, 0f],
                20 => [0f, 2f, 0f],
                30 => [0f, 0f, 3f],
                40 => [2f, 1f, 0f],
                50 => [2f, 0f, 1f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            VectorMetric.Cosine => id switch
            {
                10 => [1f, 0f, 0f],
                20 => [0f, 1f, 0f],
                30 => [0f, 0f, 1f],
                40 => [1f, 1f, 0f],
                50 => [1f, 0f, 1f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

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
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
    }
}
