namespace VecNet.Tests;

public sealed class HnswIndexScratchHardeningIndependentTests
{
    [Fact]
    public void PriorityQueues_MatchReferenceOrderingAndBoundedPruningAcrossAdversarialTies()
    {
        ulong[] ids =
        [
            44, 11, 11, 33, 22, 22, 55, 33,
            11, 44, 66, 22, 55, 33, 66, 11
        ];
        float[] sourceDistances =
        [
            3f, 1f, 1f, 2f, 2f, 1f, 4f, 2f,
            1f, 3f, 0f, 2f, 0f, 2f, 4f, 1f
        ];
        int[] insertionOrder = [14, 0, 5, 2, 10, 9, 1, 7, 12, 4, 8, 3, 15, 6, 11, 13];

        int[] nearestOrdinals = new int[ids.Length];
        float[] nearestDistances = new float[ids.Length];
        int nearestCount = 0;
        foreach (int ordinal in insertionOrder)
        {
            HnswPriorityQueues.PushNearest(
                nearestOrdinals,
                nearestDistances,
                ids,
                ref nearestCount,
                ordinal,
                sourceDistances[ordinal]);
        }

        int[] expectedNearest = insertionOrder
            .OrderBy(ordinal => sourceDistances[ordinal])
            .ThenBy(ordinal => ids[ordinal])
            .ThenBy(static ordinal => ordinal)
            .ToArray();

        var popped = new int[ids.Length];
        for (int i = 0; i < popped.Length; i++)
        {
            popped[i] = HnswPriorityQueues.PopNearest(
                nearestOrdinals,
                nearestDistances,
                ids,
                ref nearestCount).Ordinal;
        }

        Assert.Equal(expectedNearest, popped);
        Assert.Equal(0, nearestCount);

        const int maxResults = 5;
        int[] boundedOrdinals = new int[maxResults];
        float[] boundedDistances = new float[maxResults];
        int boundedCount = 0;
        var seen = new List<int>(ids.Length);

        foreach (int ordinal in insertionOrder)
        {
            bool accepted = HnswPriorityQueues.AddBoundedNearest(
                boundedOrdinals,
                boundedDistances,
                ids,
                ref boundedCount,
                maxResults,
                ordinal,
                sourceDistances[ordinal]);

            int[] expectedPrefix = seen
                .Append(ordinal)
                .OrderBy(candidate => sourceDistances[candidate])
                .ThenBy(candidate => ids[candidate])
                .ThenBy(static candidate => candidate)
                .Take(maxResults)
                .ToArray();
            bool expectedAccepted = seen.Count < maxResults || expectedPrefix.Contains(ordinal);

            seen.Add(ordinal);

            Assert.Equal(expectedAccepted, accepted);
            Assert.Equal(expectedPrefix, DrainAsNearest(boundedOrdinals, boundedDistances, ids, boundedCount));
        }
    }

    [Fact]
    public void Search_ReturnsSortedPartialResultsAfterHeapTraversalWithEqualDistances()
    {
        var options = new HnswIndexOptions(8, 16, 4, 0x1035UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, () => 0);

        foreach (ulong id in new ulong[] { 900, 100, 700, 300, 500, 200 })
        {
            index.Add(id, [1f, 0f]);
        }

        var results = new SearchResult[3];
        int written = index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(3, written);
        Assert.Equal(
            [new SearchResult(100, 1f), new SearchResult(200, 1f), new SearchResult(300, 1f)],
            results);
    }

    [Fact]
    public void Add_ReusesBuildScratchUntilCapacityGrowthAndThenReplacesIt()
    {
        var options = new HnswIndexOptions(4, 8, 8, 0x1036UL);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options, () => 0);

        index.Add(10, [0f]);
        index.Add(20, [1f]);
        HnswSearchWorkspace? firstScratch = index.DebugBuildSearchWorkspace;

        Assert.NotNull(firstScratch);
        Assert.Equal(4, firstScratch!.MaxElements);

        index.Add(30, [2f]);
        index.Add(40, [3f]);
        Assert.Same(firstScratch, index.DebugBuildSearchWorkspace);

        index.Add(50, [4f]);
        HnswSearchWorkspace? secondScratch = index.DebugBuildSearchWorkspace;

        Assert.NotNull(secondScratch);
        Assert.NotSame(firstScratch, secondScratch);
        Assert.Equal(8, secondScratch!.MaxElements);

        index.Add(60, [5f]);
        index.Add(70, [6f]);
        index.Add(80, [7f]);
        Assert.Same(secondScratch, index.DebugBuildSearchWorkspace);

        index.Add(90, [8f]);

        Assert.NotSame(secondScratch, index.DebugBuildSearchWorkspace);
        Assert.Equal(16, index.DebugBuildSearchWorkspace!.MaxElements);
    }

    [Fact]
    public void Add_LevelProviderFailureAfterScratchCreationPreservesPublishedRowsAndScratch()
    {
        int[] levels = [0, 0, -1, 0];
        int nextLevel = 0;
        var options = new HnswIndexOptions(2, 4, 4, 0x1037UL);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options, () => levels[nextLevel++]);

        index.Add(10, [0f]);
        index.Add(20, [2f]);
        HnswSearchWorkspace? scratchBeforeFailure = index.DebugBuildSearchWorkspace;

        Assert.NotNull(scratchBeforeFailure);
        Assert.Throws<InvalidOperationException>(() => index.Add(30, [1f]));
        Assert.Equal(2, index.Count);
        Assert.Same(scratchBeforeFailure, index.DebugBuildSearchWorkspace);
        Assert.Equal([10UL, 20UL], SearchIds(index, [0f], 2));

        index.Add(30, [1f]);

        Assert.Equal(3, index.Count);
        Assert.Equal([10UL, 30UL, 20UL], SearchIds(index, [0f], 3));
    }

    [Fact]
    public void Search_CallerOwnedWorkspaceCanBeReusedButDoesNotAutoGrowAfterIndexGrowth()
    {
        var options = new HnswIndexOptions(4, 16, 8, 0x1038UL);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options, () => 0);
        for (int i = 0; i < 4; i++)
        {
            index.Add((ulong)(100 + i), [i]);
        }

        var workspace = new HnswSearchWorkspace(index.Count, options.EfSearch);
        SearchResult[] results = [new(999, 999f), new(998, 998f)];

        int firstWritten = index.Search([1.1f], results, workspace);
        int markAfterSuccess = workspace.CurrentVisitMark;

        Assert.Equal(2, firstWritten);
        Assert.Equal([101UL, 102UL], results.Select(static result => result.Id));
        Assert.True(markAfterSuccess > 0);

        index.Add(200, [0.05f]);
        SearchResult[] sentinelResults = [new(777, 777f), new(778, 778f)];

        Assert.Throws<ArgumentException>(() => index.Search([0f], sentinelResults, workspace));
        Assert.Equal([new SearchResult(777, 777f), new SearchResult(778, 778f)], sentinelResults);
        Assert.Equal(markAfterSuccess, workspace.CurrentVisitMark);

        var grownWorkspace = new HnswSearchWorkspace(index.Count, options.EfSearch);
        int secondWritten = index.Search([0f], sentinelResults, grownWorkspace);

        Assert.Equal(2, secondWritten);
        Assert.Equal([100UL, 200UL], sentinelResults.Select(static result => result.Id));
    }

    [Fact]
    public void OpenReadOnly_SearchParityHoldsWhenCallerReusesWorkspaceAcrossQueries()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(8, 32, 12, 0x1039UL);
        var source = new HnswIndex(4, VectorMetric.SquaredEuclidean, options);
        var random = new Random(0x1039);

        for (int i = 0; i < 36; i++)
        {
            source.Add((ulong)(10_000 + i * 13), CreateVector(random, 4, i % 6));
        }

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        var sourceWorkspace = new HnswSearchWorkspace(source.Count, options.EfSearch);
        var openedWorkspace = new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch);
        var sourceResults = new SearchResult[5];
        var openedResults = new SearchResult[5];
        int previousSourceMark = 0;
        int previousOpenedMark = 0;

        for (int q = 0; q < 9; q++)
        {
            float[] query = CreateVector(random, 4, q % 6);
            int sourceWritten = source.Search(query, sourceResults, sourceWorkspace);
            int openedWritten = opened.Search(query, openedResults, openedWorkspace);

            Assert.Equal(sourceWritten, openedWritten);
            Assert.Equal(sourceResults[..sourceWritten], openedResults[..openedWritten]);
            int sourceDelta = sourceWorkspace.CurrentVisitMark - previousSourceMark;
            int openedDelta = openedWorkspace.CurrentVisitMark - previousOpenedMark;

            Assert.True(sourceDelta > 0);
            Assert.Equal(sourceDelta, openedDelta);

            previousSourceMark = sourceWorkspace.CurrentVisitMark;
            previousOpenedMark = openedWorkspace.CurrentVisitMark;
        }
    }

    private static int[] DrainAsNearest(int[] ordinals, float[] distances, ulong[] ids, int count)
    {
        int[] copyOrdinals = new int[count];
        float[] copyDistances = new float[count];
        int copyCount = 0;
        for (int i = 0; i < count; i++)
        {
            HnswPriorityQueues.PushNearest(
                copyOrdinals,
                copyDistances,
                ids,
                ref copyCount,
                ordinals[i],
                distances[i]);
        }

        var drained = new int[count];
        for (int i = 0; i < drained.Length; i++)
        {
            drained[i] = HnswPriorityQueues.PopNearest(copyOrdinals, copyDistances, ids, ref copyCount).Ordinal;
        }

        return drained;
    }

    private static ulong[] SearchIds(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written].Select(static result => result.Id).ToArray();
    }

    private static float[] CreateVector(Random random, int dimension, int cluster)
    {
        var vector = new float[dimension];
        float center = (cluster - 2) * 4.5f;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = center + (i * 0.25f) + ((random.NextSingle() - 0.5f) * 0.15f);
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
