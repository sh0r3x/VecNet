namespace VecNet.Tests;

public sealed class HnswIndexTests
{
    [Fact]
    public void Constructor_ValidatesDimensionMetricAndOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(0, VectorMetric.SquaredEuclidean, HnswIndexOptions.Default));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, (VectorMetric)99, HnswIndexOptions.Default));
        Assert.Throws<NotSupportedException>(
            () => new HnswIndex(2, VectorMetric.InnerProduct, HnswIndexOptions.Default));
        Assert.Throws<NotSupportedException>(
            () => new HnswIndex(2, VectorMetric.Cosine, HnswIndexOptions.Default));

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(1, 10, 10, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(65, 100, 10, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(8, 7, 10, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(8, 4097, 10, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(8, 10, 0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(8, 10, 4097, 1)));
    }

    [Fact]
    public void Add_ValidatesRowsRejectsDuplicatesAndPreservesPriorGraphAfterFailedAdd()
    {
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 8, 11));
        index.Add(10, [0f, 0f]);
        index.Add(20, [2f, 0f]);

        Assert.Throws<ArgumentException>(() => index.Add(30, [1f]));
        Assert.Throws<ArgumentException>(() => index.Add(30, [float.NaN, 0f]));
        Assert.Throws<ArgumentException>(() => index.Add(20, [9f, 9f]));

        index.Add(30, [1f, 0f]);

        var results = new SearchResult[3];
        int written = index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, 8));

        Assert.Equal(3, written);
        Assert.Equal([10UL, 30UL, 20UL], results.Select(static result => result.Id));
        AssertGraphInvariants(index);
    }

    [Fact]
    public void Add_UsesCopiedRowMajorStorageAndOpaqueExternalIds()
    {
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 8, 22));
        float[] vector = [1f, 0f];

        index.Add(999, vector);
        vector[0] = 100f;

        var results = new SearchResult[1];
        int written = index.Search([1f, 0f], results, new HnswSearchWorkspace(1, 8));

        Assert.Equal(1, written);
        Assert.Equal(new SearchResult(999, 0f), results[0]);
        Assert.Equal(0, index.EntryPoint);
    }

    [Fact]
    public void FixedSeedInsertion_IsDeterministicForLevelsGraphAndResults()
    {
        var options = new HnswIndexOptions(4, 12, 12, 0x5EED_035UL);
        var left = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);
        var right = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);

        for (int i = 0; i < 24; i++)
        {
            float[] vector = [(i % 5) - 2f, i / 5f, (i % 3) * 0.25f];
            left.Add((ulong)(1000 + i), vector);
            right.Add((ulong)(1000 + i), vector);
        }

        Assert.Equal(CreateGraphSnapshot(left), CreateGraphSnapshot(right));

        var leftResults = new SearchResult[6];
        var rightResults = new SearchResult[6];
        float[] query = [0.25f, 2.5f, 0.5f];
        Assert.Equal(6, left.Search(query, leftResults, new HnswSearchWorkspace(left.Count, 12)));
        Assert.Equal(6, right.Search(query, rightResults, new HnswSearchWorkspace(right.Count, 12)));
        Assert.Equal(leftResults, rightResults);
    }

    [Fact]
    public void ProductionLevelGeneration_MatchesPinnedSplitMix64Sequence()
    {
        var options = new HnswIndexOptions(4, 12, 12, 0x5EED_035UL);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options);
        int[] expectedLevels = [0, 2, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0];

        for (int i = 0; i < expectedLevels.Length; i++)
        {
            index.Add((ulong)i, [i * 10f]);
        }

        Assert.Equal(expectedLevels, Enumerable.Range(0, expectedLevels.Length).Select(index.DebugGetLevel));
    }

    [Fact]
    public void ForcedLevels_PinEntryPointAndLayerMembershipInTinyGraph()
    {
        int[] levels = [0, 2, 1, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 4, 4, 1),
            () => levels[nextLevel++]);

        index.Add(10, [0f]);
        Assert.Equal(0, index.EntryPoint);
        Assert.Equal(0, index.MaxLayer);

        index.Add(20, [10f]);
        Assert.Equal(1, index.EntryPoint);
        Assert.Equal(2, index.MaxLayer);

        index.Add(30, [5f]);
        index.Add(40, [1f]);

        Assert.Equal(levels, Enumerable.Range(0, index.Count).Select(index.DebugGetLevel));
        AssertGraphInvariants(index);
        Assert.Empty(GetNeighbors(index, 2, 0));
        Assert.Empty(GetNeighbors(index, 2, 2));
    }

    [Fact]
    public void HeuristicNeighborSelection_PrunesRedundantCandidatesAndKeepsDiverseBridge()
    {
        int[] levels = [0, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 4, 4, 1),
            () => levels[nextLevel++]);

        index.Add(10, [0f, 0f]);
        index.Add(20, [1f, 0f]);
        index.Add(30, [2f, 0f]);
        index.Add(40, [-2f, 0f]);

        Span<int> selected = stackalloc int[3];
        int selectedCount = index.DebugSelectNeighbors(0, [1, 2, 3], selected, 0);

        Assert.Equal(2, selectedCount);
        Assert.Equal([1, 3], selected[..selectedCount].ToArray());
    }

    [Fact]
    public void HeuristicNeighborSelection_UsesStrictEqualDistancePruning()
    {
        int[] levels = [0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 4, 4, 1),
            () => levels[nextLevel++]);

        index.Add(10, [0f, 0f]);
        index.Add(20, [0.5f, 0.8660254f]);
        index.Add(30, [1f, 0f]);

        Span<int> selected = stackalloc int[2];
        int selectedCount = index.DebugSelectNeighbors(0, [1, 2], selected, 0);

        Assert.Equal(1, selectedCount);
        Assert.Equal(1, selected[0]);
    }

    [Fact]
    public void PruningFullNeighborLists_RespectsLayerDegreeLimits()
    {
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, new HnswIndexOptions(2, 4, 4, 1));
        for (int i = 0; i < 20; i++)
        {
            index.Add((ulong)i, [i]);
        }

        AssertGraphInvariants(index);
    }

    [Fact]
    public void Search_ValidatesQueryWorkspaceAndEfBeforeTraversal()
    {
        var empty = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 2, 1));

        Assert.Throws<ArgumentException>(
            () => empty.Search([float.NaN, 0f], Span<SearchResult>.Empty, new HnswSearchWorkspace(0, 2)));
        Assert.Throws<ArgumentException>(
            () => empty.Search([0f], Span<SearchResult>.Empty, new HnswSearchWorkspace(0, 2)));
        Assert.Throws<ArgumentException>(
            () => empty.Search([0f, 0f], Span<SearchResult>.Empty, new HnswSearchWorkspace(0, 1)));
        Assert.Equal(0, empty.Search([0f, 0f], Span<SearchResult>.Empty, new HnswSearchWorkspace(0, 2)));

        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 2, 1));
        index.Add(1, [0f, 0f]);
        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], new SearchResult[1], new HnswSearchWorkspace(0, 2)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f, 0f], new SearchResult[3], new HnswSearchWorkspace(1, 2)));
    }

    [Fact]
    public void Search_WithSmallEf_CanHaveMorePendingCandidatesThanEfWithoutExceedingWorkspace()
    {
        int[] levels = [0, 0, 0, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(4, 8, 1, 1),
            () => levels[nextLevel++]);

        index.Add(100, [100f]);
        index.Add(50, [50f]);
        index.Add(1, [0f]);
        index.Add(2, [1f]);
        index.Add(3, [2f]);
        index.Add(4, [3f]);

        Assert.True(GetNeighbors(index, 0, 1).Length > 1);

        var results = new SearchResult[1];
        int written = index.Search([0f], results, new HnswSearchWorkspace(index.Count, 1));

        Assert.Equal(1, written);
        Assert.Equal(1UL, results[0].Id);
        Assert.Equal(0f, results[0].Distance);
    }

    [Fact]
    public void Search_HandlesEmptyDestinationEmptyIndexAndKGreaterThanCount()
    {
        var options = new HnswIndexOptions(4, 8, 8, 1);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options);

        Assert.Equal(0, index.Search([0f], new SearchResult[2], new HnswSearchWorkspace(0, 8)));

        index.Add(20, [2f]);
        index.Add(10, [1f]);

        Assert.Equal(0, index.Search([0f], Span<SearchResult>.Empty, new HnswSearchWorkspace(index.Count, 8)));

        var results = new SearchResult[4];
        int written = index.Search([0f], results, new HnswSearchWorkspace(index.Count, 8));

        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void Search_WithEfSearchAtLeastCount_MatchesExactTruthForSmallSafelySeparatedCase()
    {
        var options = new HnswIndexOptions(4, 16, 16, 0x3501UL);
        var hnsw = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);
        var exact = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);

        for (int i = 0; i < 12; i++)
        {
            float[] vector = [i * 3f, (i % 3) * 5f, (i % 2) * 7f];
            hnsw.Add((ulong)(100 + i), vector);
            exact.Add((ulong)(100 + i), vector);
        }

        float[] query = [10f, 4f, 6f];
        var expected = new SearchResult[12];
        var actual = new SearchResult[12];

        Assert.Equal(12, exact.Search(query, expected));
        Assert.Equal(12, hnsw.Search(query, actual, new HnswSearchWorkspace(hnsw.Count, 16)));
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Search_FixedSeedRecallSmoke_ComparesAgainstExactTruth()
    {
        const int count = 160;
        const int dimension = 32;
        const int k = 10;
        var options = new HnswIndexOptions(16, 120, 120, 0x5EED_350UL);
        var hnsw = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);
        var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var random = new Random(0x350);

        for (int i = 0; i < count; i++)
        {
            float[] vector = CreateClusteredVector(random, dimension, i % 8);
            hnsw.Add((ulong)(10_000 + i), vector);
            exact.Add((ulong)(10_000 + i), vector);
        }

        double recallSum = 0;
        for (int q = 0; q < 8; q++)
        {
            float[] query = CreateClusteredVector(random, dimension, q % 8);
            var exactResults = new SearchResult[k];
            var hnswResults = new SearchResult[k];
            exact.Search(query, exactResults);
            hnsw.Search(query, hnswResults, new HnswSearchWorkspace(hnsw.Count, options.EfSearch));

            HashSet<ulong> exactIds = exactResults.Select(static result => result.Id).ToHashSet();
            recallSum += hnswResults.Count(result => exactIds.Contains(result.Id)) / (double)k;
            Assert.True(hnswResults.Zip(hnswResults.Skip(1), IsSortedPair).All(static sorted => sorted));
        }

        Assert.True(recallSum / 8 >= 0.80, "Fixed-seed recall smoke should preserve broad approximate quality.");
    }

    [Fact]
    public void Search_SupportsParallelReadOnlyQueriesAfterBuildCompletionWithIndependentWorkspaces()
    {
        const int count = 96;
        const int dimension = 8;
        var options = new HnswIndexOptions(8, 40, 40, 0xCAFEUL);
        var index = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);

        for (int i = 0; i < count; i++)
        {
            index.Add((ulong)i, Enumerable.Range(0, dimension).Select(j => (float)((i + j) % 11)).ToArray());
        }

        Parallel.For(
            0,
            32,
            queryIndex =>
            {
                float[] query = Enumerable.Range(0, dimension)
                    .Select(j => (float)(((queryIndex * 3) + j) % 11))
                    .ToArray();
                var results = new SearchResult[5];
                var workspace = new HnswSearchWorkspace(index.Count, options.EfSearch);

                int written = index.Search(query, results, workspace);

                Assert.Equal(5, written);
                Assert.True(results.Zip(results.Skip(1), IsSortedPair).All(static sorted => sorted));
            });
    }

    private static bool IsSortedPair(SearchResult left, SearchResult right) =>
        left.Distance < right.Distance ||
        (left.Distance == right.Distance && left.Id <= right.Id);

    private static float[] CreateClusteredVector(Random random, int dimension, int cluster)
    {
        var vector = new float[dimension];
        float center = cluster * 10f;
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = center + (random.NextSingle() - 0.5f);
        }

        return vector;
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

    private static void AssertGraphInvariants(HnswIndex index)
    {
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            int degreeLimit = layer == 0 ? index.Options.M * 2 : index.Options.M;
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                int[] neighbors = GetNeighbors(index, layer, ordinal);
                if (index.DebugGetLevel(ordinal) < layer)
                {
                    Assert.Empty(neighbors);
                    continue;
                }

                Assert.InRange(neighbors.Length, 0, degreeLimit);
                Assert.DoesNotContain(ordinal, neighbors);
                Assert.Equal(neighbors.Length, neighbors.Distinct().Count());
                Assert.All(neighbors, neighbor =>
                {
                    Assert.InRange(neighbor, 0, index.Count - 1);
                    Assert.True(index.DebugGetLevel(neighbor) >= layer);
                });
            }
        }
    }

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }
}
