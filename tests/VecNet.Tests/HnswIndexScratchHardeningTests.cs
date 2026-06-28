namespace VecNet.Tests;

public sealed class HnswIndexScratchHardeningTests
{
    [Fact]
    public void CandidateQueue_PopsNearestByDistanceThenExternalIdThenOrdinal()
    {
        ulong[] ids = [50, 10, 10, 30, 20];
        int[] ordinals = new int[ids.Length];
        float[] distances = new float[ids.Length];
        int count = 0;

        HnswPriorityQueues.PushNearest(ordinals, distances, ids, ref count, 0, 2f);
        HnswPriorityQueues.PushNearest(ordinals, distances, ids, ref count, 3, 1f);
        HnswPriorityQueues.PushNearest(ordinals, distances, ids, ref count, 4, 1f);
        HnswPriorityQueues.PushNearest(ordinals, distances, ids, ref count, 2, 1f);
        HnswPriorityQueues.PushNearest(ordinals, distances, ids, ref count, 1, 1f);

        var popped = new HnswQueueItem[ids.Length];
        for (int i = 0; i < popped.Length; i++)
        {
            popped[i] = HnswPriorityQueues.PopNearest(ordinals, distances, ids, ref count);
        }

        Assert.Equal([1, 2, 4, 3, 0], popped.Select(static item => item.Ordinal));
        Assert.Equal(0, count);
    }

    [Fact]
    public void AcceptedResultQueue_PrunesDeterministicWorstCandidate()
    {
        ulong[] ids = [50, 10, 40, 30, 20];
        int[] ordinals = new int[ids.Length];
        float[] distances = new float[ids.Length];
        int count = 0;

        Assert.True(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 0, 5f));
        Assert.True(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 1, 1f));
        Assert.True(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 2, 3f));
        Assert.True(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 3, 3f));
        Assert.False(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 0, 5f));
        HnswQueueItem preReplacementWorst = HnswPriorityQueues.PeekWorst(ordinals, distances, count);
        Assert.Equal(2, preReplacementWorst.Ordinal);
        Assert.Equal(3f, preReplacementWorst.Distance);

        Assert.True(HnswPriorityQueues.AddBoundedNearest(ordinals, distances, ids, ref count, 3, 4, 0.5f));

        Assert.Equal(3, count);
        HnswQueueItem worst = HnswPriorityQueues.PeekWorst(ordinals, distances, count);
        Assert.Equal(3, worst.Ordinal);
        Assert.Equal(3f, worst.Distance);
        Assert.Equal([4, 1, 3], SnapshotAsNearest(ordinals, distances, ids, count));
    }

    [Fact]
    public void Add_ReusesBuildSearchWorkspaceWithinCapacityWindow()
    {
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 8, 0x1030UL), () => 0);

        index.Add(10, [0f, 0f]);
        Assert.Null(index.DebugBuildSearchWorkspace);

        index.Add(20, [1f, 0f]);
        HnswSearchWorkspace? firstWorkspace = index.DebugBuildSearchWorkspace;
        Assert.NotNull(firstWorkspace);
        Assert.Equal(4, firstWorkspace!.MaxElements);

        index.Add(30, [2f, 0f]);
        index.Add(40, [3f, 0f]);

        Assert.Same(firstWorkspace, index.DebugBuildSearchWorkspace);
        Assert.True(firstWorkspace.CurrentVisitMark > 0);
    }

    [Fact]
    public void Add_LevelProviderFailureDoesNotConsumeIdOrCorruptPriorSearch()
    {
        int[] levels = [0, -1, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 4, 4, 0x1031UL),
            () => levels[nextLevel++]);

        index.Add(10, [0f]);

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(() => index.Add(20, [1f]));
        Assert.Contains("negative level", exception.Message, StringComparison.Ordinal);
        Assert.Equal(1, index.Count);

        var results = new SearchResult[1];
        Assert.Equal(1, index.Search([0f], results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch)));
        Assert.Equal(new SearchResult(10, 0f), results[0]);

        index.Add(20, [1f]);
        Assert.Equal(2, index.Count);
    }

    [Fact]
    public void Search_UndersizedWorkspaceFailsBeforeDestinationMutationAndCanBeRetried()
    {
        var options = new HnswIndexOptions(4, 16, 8, 0x1032UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, () => 0);
        for (int i = 0; i < 6; i++)
        {
            index.Add((ulong)(100 + i), [i, 0f]);
        }

        SearchResult[] results = [new(1, 1f), new(2, 2f)];

        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count - 1, options.EfSearch)));
        Assert.Equal([new SearchResult(1, 1f), new SearchResult(2, 2f)], results);

        int written = index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(2, written);
        Assert.Equal([100UL, 101UL], results.Select(static item => item.Id));
    }

    [Fact]
    public void Search_DeterministicEqualDistanceTiesStayOrderedByExternalId()
    {
        var options = new HnswIndexOptions(8, 16, 16, 0x1033UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, () => 0);

        index.Add(40, [1f, 0f]);
        index.Add(10, [0f, 1f]);
        index.Add(30, [-1f, 0f]);
        index.Add(20, [0f, -1f]);

        var results = new SearchResult[4];
        int written = index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(4, written);
        Assert.Equal([10UL, 20UL, 30UL, 40UL], results.Select(static item => item.Id));
        Assert.All(results, static item => Assert.Equal(1f, item.Distance));
    }

    [Fact]
    public void SaveOpenReadOnly_PreservesHardenedSearchResults()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(8, 32, 32, 0x1034UL);
        var source = new HnswIndex(5, VectorMetric.SquaredEuclidean, options);
        var random = new Random(0x1034);

        for (int i = 0; i < 40; i++)
        {
            source.Add((ulong)(1_000 + i), CreateVector(random, 5, i % 4));
        }

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        for (int i = 0; i < 8; i++)
        {
            float[] query = CreateVector(random, 5, i % 4);
            SearchResult[] sourceResults = Search(source, query, 7);
            SearchResult[] openedResults = Search(opened, query, 7);

            Assert.Equal(sourceResults, openedResults);
        }
    }

    private static int[] SnapshotAsNearest(int[] ordinals, float[] distances, ulong[] ids, int count)
    {
        int[] copyOrdinals = new int[count];
        float[] copyDistances = new float[count];
        int copyCount = 0;
        for (int i = 0; i < count; i++)
        {
            HnswPriorityQueues.PushNearest(copyOrdinals, copyDistances, ids, ref copyCount, ordinals[i], distances[i]);
        }

        var drained = new int[count];
        for (int i = 0; i < drained.Length; i++)
        {
            drained[i] = HnswPriorityQueues.PopNearest(copyOrdinals, copyDistances, ids, ref copyCount).Ordinal;
        }

        return drained;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static float[] CreateVector(Random random, int dimension, int cluster)
    {
        var vector = new float[dimension];
        float center = cluster * 5f;
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = center + i + ((random.NextSingle() - 0.5f) * 0.1f);
        }

        return vector;
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
