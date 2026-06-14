using System.Numerics;

namespace VecNet.Tests;

public sealed class HnswIndexIndependentTests
{
    [Fact]
    public void Search_WithLargeEfMatchesExactTruthAcrossDuplicateAndZeroVectors()
    {
        const int dimension = 5;
        var options = new HnswIndexOptions(32, 64, 64, 0x3500_0001UL);
        var hnsw = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options, () => 0);
        var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        (ulong Id, float[] Vector)[] rows =
        [
            (91, [0f, 0f, 0f, 0f, 0f]),
            (13, [1f, -1f, 2f, -2f, 0.5f]),
            (7, [1f, -1f, 2f, -2f, 0.5f]),
            (42, [-4f, 0f, 1f, 3f, -2f]),
            (3, [8f, 8f, 8f, 8f, 8f]),
            (ulong.MaxValue, [-1f, -1f, -1f, -1f, -1f]),
            (100, [0f, 0f, 0f, 0f, 0f]),
            (55, [2f, 4f, 6f, 8f, 10f]),
            (21, [-3f, 5f, -7f, 11f, -13f])
        ];

        foreach ((ulong id, float[] vector) in rows)
        {
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        float[][] queries =
        [
            [0f, 0f, 0f, 0f, 0f],
            [1f, -1f, 2f, -2f, 0.5f],
            [3f, 3f, 3f, 3f, 3f]
        ];

        foreach (float[] query in queries)
        {
            var expected = new SearchResult[rows.Length];
            var actual = new SearchResult[rows.Length + 3];

            int exactWritten = exact.Search(query, expected);
            int hnswWritten = hnsw.Search(query, actual, new HnswSearchWorkspace(hnsw.Count, options.EfSearch));

            Assert.Equal(exactWritten, hnswWritten);
            Assert.Equal(expected, actual[..hnswWritten]);
            AssertGraphInvariants(hnsw);
        }
    }

    [Fact]
    public void Search_FixedSeedRecallSmokeUsesExactTruthForMultipleDimensionsAndEfValues()
    {
        int[] dimensions = [3, Vector<float>.Count + 1, 32];
        foreach (int dimension in dimensions)
        {
            double lowEfRecall = MeasureAverageRecall(dimension, efSearch: 10, seed: 0x3500_1000 + dimension);
            double highEfRecall = MeasureAverageRecall(dimension, efSearch: 80, seed: 0x3500_1000 + dimension);

            Assert.True(lowEfRecall >= 0.55, $"Low-ef recall smoke failed for dimension {dimension}: {lowEfRecall}.");
            Assert.True(highEfRecall >= 0.90, $"High-ef recall smoke failed for dimension {dimension}: {highEfRecall}.");
            Assert.True(highEfRecall >= lowEfRecall, $"Higher efSearch should not reduce fixed-smoke recall for dimension {dimension}.");
        }
    }

    [Fact]
    public void InsertionOrderVariationCanChangeGraphWhileMaintainingSearchInvariantsAndRecall()
    {
        const int dimension = 6;
        const int count = 48;
        const int k = 6;
        var options = new HnswIndexOptions(8, 32, 32, 0x3500_2000UL);
        (ulong Id, float[] Vector)[] rows = CreateDeterministicRows(count, dimension, 0x3500_2001);
        var forward = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);
        var reverse = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);
        var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        foreach ((ulong id, float[] vector) in rows)
        {
            forward.Add(id, vector);
            exact.Add(id, vector);
        }

        foreach ((ulong id, float[] vector) in rows.Reverse())
        {
            reverse.Add(id, vector);
        }

        Assert.NotEqual(CreateGraphSnapshot(forward), CreateGraphSnapshot(reverse));
        AssertGraphInvariants(forward);
        AssertGraphInvariants(reverse);

        double forwardRecall = 0;
        double reverseRecall = 0;
        for (int q = 0; q < 8; q++)
        {
            float[] query = CreateClusteredVector(new Random(0x3500_2100 + q), dimension, q % 6);
            SearchResult[] truth = Search(exact, query, k);
            forwardRecall += RecallAtK(Search(forward, query, k, options.EfSearch), truth);
            reverseRecall += RecallAtK(Search(reverse, query, k, options.EfSearch), truth);
        }

        Assert.True(forwardRecall / 8 >= 0.75, $"Forward insertion recall was {forwardRecall / 8}.");
        Assert.True(reverseRecall / 8 >= 0.75, $"Reverse insertion recall was {reverseRecall / 8}.");
    }

    [Fact]
    public void Search_RejectsUndersizedWorkspaceWithoutChangingDestination()
    {
        var options = new HnswIndexOptions(4, 16, 8, 0x3500_3000UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options);
        for (int i = 0; i < 6; i++)
        {
            index.Add((ulong)i, [i, -i]);
        }

        SearchResult[] results = [new(777, 777f), new(888, 888f)];

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count - 1, options.EfSearch)));
        Assert.Equal([777UL, 888UL], results.Select(static result => result.Id));
        Assert.Equal([777f, 888f], results.Select(static result => result.Distance));

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch - 1)));
        Assert.Equal([777UL, 888UL], results.Select(static result => result.Id));
        Assert.Equal([777f, 888f], results.Select(static result => result.Distance));
    }

    [Fact]
    public void Search_UsesExactIdTieOrderingForDuplicateVectorsAndDuplicateAddPreservesOriginal()
    {
        var options = new HnswIndexOptions(8, 16, 16, 0x3500_4000UL);
        var hnsw = new HnswIndex(3, VectorMetric.SquaredEuclidean, options, () => 0);
        var exact = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);

        (ulong Id, float[] Vector)[] rows =
        [
            (40, [1f, 2f, 3f]),
            (10, [1f, 2f, 3f]),
            (30, [1f, 2f, 3f]),
            (20, [-1f, -2f, -3f])
        ];

        foreach ((ulong id, float[] vector) in rows)
        {
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        Assert.Throws<ArgumentException>(() => hnsw.Add(10, [99f, 99f, 99f]));

        SearchResult[] expected = Search(exact, [1f, 2f, 3f], 4);
        SearchResult[] actual = Search(hnsw, [1f, 2f, 3f], 4, options.EfSearch);

        Assert.Equal(expected, actual);
        Assert.Equal([10UL, 30UL, 40UL, 20UL], actual.Select(static result => result.Id));
    }

    [Fact]
    public void Search_AllowsCandidateCapacityEqualToCountWhenEfSearchIsOne()
    {
        int[] levels = Enumerable.Repeat(0, 40).ToArray();
        int nextLevel = 0;
        var options = new HnswIndexOptions(16, 32, 1, 0x3500_5000UL);
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options, () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(10_000 + i), [i]);
        }

        var results = new SearchResult[1];
        int written = index.Search([19.1f], results, new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(1, written);
        Assert.True(results[0].Id is 10019UL or 10020UL);
        AssertGraphInvariants(index);
    }

    [Fact]
    public void Search_ParallelReadOnlyStressMatchesExactTruthWithIndependentWorkspaces()
    {
        const int dimension = 7;
        const int count = 64;
        const int k = 8;
        var options = new HnswIndexOptions(32, 64, 64, 0x3500_6000UL);
        var hnsw = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options, () => 0);
        var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        (ulong Id, float[] Vector)[] rows = CreateDeterministicRows(count, dimension, 0x3500_6001);

        foreach ((ulong id, float[] vector) in rows)
        {
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        Parallel.For(
            0,
            48,
            queryIndex =>
            {
                float[] query = CreateClusteredVector(new Random(0x3500_6100 + queryIndex), dimension, queryIndex % 8);

                SearchResult[] expected = Search(exact, query, k);
                SearchResult[] actual = Search(hnsw, query, k, options.EfSearch);

                Assert.Equal(expected, actual);
            });
    }

    private static double MeasureAverageRecall(int dimension, int efSearch, int seed)
    {
        const int count = 192;
        const int k = 10;
        var options = new HnswIndexOptions(12, 80, efSearch, (ulong)seed);
        var hnsw = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options);
        var exact = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var random = new Random(seed);

        for (int i = 0; i < count; i++)
        {
            float[] vector = CreateClusteredVector(random, dimension, i % 12);
            ulong id = (ulong)(1_000_000 + (i * 37));
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        double recall = 0;
        for (int q = 0; q < 12; q++)
        {
            float[] query = CreateClusteredVector(random, dimension, q % 12);
            SearchResult[] exactResults = Search(exact, query, k);
            SearchResult[] hnswResults = Search(hnsw, query, k, efSearch);

            recall += RecallAtK(hnswResults, exactResults);
            AssertSorted(hnswResults);
            AssertReturnedDistancesMatchExact(hnswResults, exact, query, count);
        }

        return recall / 12;
    }

    private static (ulong Id, float[] Vector)[] CreateDeterministicRows(int count, int dimension, int seed)
    {
        var random = new Random(seed);
        var rows = new (ulong Id, float[] Vector)[count];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = ((ulong)(50_000 + i * 19), CreateClusteredVector(random, dimension, i % 8));
        }

        return rows;
    }

    private static float[] CreateClusteredVector(Random random, int dimension, int cluster)
    {
        var vector = new float[dimension];
        float center = cluster * 7.5f;
        for (int i = 0; i < vector.Length; i++)
        {
            float axisOffset = (i % 5) - 2f;
            vector[i] = center + axisOffset + ((random.NextSingle() - 0.5f) * 0.75f);
        }

        return vector;
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int k)
    {
        var results = new SearchResult[k];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int k, int efSearch)
    {
        var results = new SearchResult[k];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, efSearch));
        return results[..written];
    }

    private static double RecallAtK(SearchResult[] actual, SearchResult[] expected)
    {
        HashSet<ulong> truth = expected.Select(static result => result.Id).ToHashSet();
        return actual.Count(result => truth.Contains(result.Id)) / (double)expected.Length;
    }

    private static void AssertReturnedDistancesMatchExact(SearchResult[] actual, ExactFlatIndex exact, float[] query, int count)
    {
        var allExact = new SearchResult[count];
        int written = exact.Search(query, allExact);
        Dictionary<ulong, float> distanceById = allExact[..written].ToDictionary(static result => result.Id, static result => result.Distance);

        foreach (SearchResult result in actual)
        {
            Assert.True(distanceById.TryGetValue(result.Id, out float expectedDistance));
            Assert.Equal(expectedDistance, result.Distance);
        }
    }

    private static void AssertSorted(SearchResult[] results)
    {
        for (int i = 1; i < results.Length; i++)
        {
            Assert.True(
                results[i - 1].Distance < results[i].Distance ||
                (results[i - 1].Distance == results[i].Distance && results[i - 1].Id <= results[i].Id),
                $"Results were not sorted at index {i - 1}.");
        }
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
