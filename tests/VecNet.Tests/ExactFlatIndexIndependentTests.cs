namespace VecNet.Tests;

public sealed class ExactFlatIndexIndependentTests
{
    private const int RandomSeed = 0x5EED_002;

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, 1, 3)]
    [InlineData(VectorMetric.SquaredEuclidean, 7, 29)]
    [InlineData(VectorMetric.InnerProduct, 3, 3)]
    [InlineData(VectorMetric.InnerProduct, 7, 29)]
    [InlineData(VectorMetric.Cosine, 1, 3)]
    [InlineData(VectorMetric.Cosine, 7, 29)]
    public void Search_MatchesIndependentFixedSeedBruteForceTruth(
        VectorMetric metric,
        int dimension,
        int k)
    {
        var random = new Random(RandomSeed + (int)metric * 101 + dimension * 17 + k);
        var entries = new List<(ulong Id, float[] Vector)>();
        var index = new ExactFlatIndex(dimension, metric);

        for (int i = 0; i < 17; i++)
        {
            float[] vector = CreateNonZeroRandomVector(random, dimension);
            ulong id = (ulong)((i * 43 + 29) % 101);
            entries.Add((id, vector));
            index.Add(id, vector);
        }

        float[] query = CreateNonZeroRandomVector(random, dimension);
        SearchResult[] expected = GetBruteForceResults(entries, query, metric, k);
        var actual = new SearchResult[k];

        int written = index.Search(query, actual);

        Assert.Equal(expected.Length, written);
        for (int i = 0; i < written; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            Assert.InRange(
                MathF.Abs(expected[i].Distance - actual[i].Distance),
                0f,
                metric == VectorMetric.Cosine ? 2e-6f : 1e-5f);
        }
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Search_UsesExternalIdForDuplicateAndEquidistantRowsAcrossGrowthBoundaries(
        VectorMetric metric)
    {
        ulong[] ids = [99, 3, ulong.MaxValue, 42, 7, 1, 88, 2, 9];
        var index = new ExactFlatIndex(3, metric);

        for (int i = 0; i < ids.Length; i++)
        {
            index.Add(ids[i], CreateTiedVector(metric, i));
        }

        var results = new SearchResult[3];
        int written = index.Search(CreateTiedQuery(metric), results);

        Assert.Equal(3, written);
        Assert.Equal([1UL, 2UL, 3UL], results.Select(static result => result.Id));
        Assert.Equal(results[0].Distance, results[1].Distance);
        Assert.Equal(results[1].Distance, results[2].Distance);
    }

    [Fact]
    public void CosineSearch_AcceptsFiniteVectorsAcrossExtremeNonZeroScales()
    {
        var entries = new List<(ulong Id, float[] Vector)>
        {
            (ulong.MaxValue, [float.Epsilon, 0f, 0f]),
            (4, [float.MaxValue, float.MaxValue, 0f]),
            (2, [-float.Epsilon, 0f, 0f])
        };
        var query = new[] { float.MaxValue, 0f, 0f };
        var index = new ExactFlatIndex(3, VectorMetric.Cosine);
        foreach ((ulong id, float[] vector) in entries)
        {
            index.Add(id, vector);
        }

        SearchResult[] expected = GetBruteForceResults(entries, query, VectorMetric.Cosine, 3);
        var actual = new SearchResult[3];
        int written = index.Search(query, actual);

        Assert.Equal(3, written);
        Assert.Equal(expected.Select(static result => result.Id), actual.Select(static result => result.Id));
        for (int i = 0; i < written; i++)
        {
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, 2e-6f);
        }
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    public void NonCosineSearch_AcceptsWholeZeroStoredVectorsAndQueries(VectorMetric metric)
    {
        var entries = new List<(ulong Id, float[] Vector)>
        {
            (8, [0f, 0f, 0f]),
            (3, [1f, -1f, 2f])
        };
        var query = new[] { 0f, 0f, 0f };
        var index = new ExactFlatIndex(3, metric);
        foreach ((ulong id, float[] vector) in entries)
        {
            index.Add(id, vector);
        }

        SearchResult[] expected = GetBruteForceResults(entries, query, metric, 2);
        var actual = new SearchResult[2];
        int written = index.Search(query, actual);

        Assert.Equal(2, written);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Search_RejectsNonFiniteQueriesBeforeEmptyIndexOrEmptyDestinationEarlyReturn(
        VectorMetric metric)
    {
        var index = new ExactFlatIndex(3, metric);

        Assert.Throws<ArgumentException>(
            () => index.Search([1f, float.NaN, 3f], Span<SearchResult>.Empty));
    }

    [Fact]
    public void CosineSearch_RejectsZeroQueryBeforeEmptyIndexOrEmptyDestinationEarlyReturn()
    {
        var index = new ExactFlatIndex(3, VectorMetric.Cosine);

        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f, 0f], Span<SearchResult>.Empty));
    }

    [Fact]
    public void Add_InvalidRowAtGrowthBoundaryDoesNotConsumeIdentifierOrCorruptPriorRows()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        for (ulong id = 1; id <= 4; id++)
        {
            index.Add(id, [(float)id]);
        }

        Assert.Throws<ArgumentException>(() => index.Add(5, [float.PositiveInfinity]));
        index.Add(5, [5f]);

        var results = new SearchResult[6];
        int written = index.Search([0f], results);

        Assert.Equal(5, written);
        Assert.Equal([1UL, 2UL, 3UL, 4UL, 5UL], results[..written].Select(static result => result.Id));
    }

    private static float[] CreateNonZeroRandomVector(Random random, int dimension)
    {
        var vector = new float[dimension];
        bool containsNonZero = false;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.Next(-20, 21) / 4f;
            containsNonZero |= vector[i] != 0f;
        }

        if (!containsNonZero)
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static float[] CreateTiedVector(VectorMetric metric, int ordinal)
    {
        bool even = ordinal % 2 == 0;
        return metric switch
        {
            VectorMetric.SquaredEuclidean => even ? [1f, 0f, 0f] : [-1f, 0f, 0f],
            VectorMetric.InnerProduct => even ? [2f, 1f, 0f] : [2f, -1f, 0f],
            VectorMetric.Cosine => even ? [0f, 1f, 0f] : [0f, -1f, 0f],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };
    }

    private static float[] CreateTiedQuery(VectorMetric metric) =>
        metric == VectorMetric.SquaredEuclidean ? [0f, 0f, 0f] : [1f, 0f, 0f];

    private static SearchResult[] GetBruteForceResults(
        IEnumerable<(ulong Id, float[] Vector)> entries,
        float[] query,
        VectorMetric metric,
        int k) =>
        entries
            .Select(entry => new SearchResult(entry.Id, CalculateReferenceDistance(query, entry.Vector, metric)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(k)
            .ToArray();

    private static float CalculateReferenceDistance(float[] query, float[] vector, VectorMetric metric)
    {
        return metric switch
        {
            VectorMetric.SquaredEuclidean => (float)query
                .Zip(vector, static (left, right) => Math.Pow(left - right, 2))
                .Sum(),
            VectorMetric.InnerProduct => (float)-query
                .Zip(vector, static (left, right) => (double)left * right)
                .Sum(),
            VectorMetric.Cosine => CalculateReferenceCosineDistance(query, vector),
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };
    }

    private static float CalculateReferenceCosineDistance(float[] query, float[] vector)
    {
        double dotProduct = query.Zip(vector, static (left, right) => (double)left * right).Sum();
        double queryNorm = Math.Sqrt(query.Sum(static component => (double)component * component));
        double vectorNorm = Math.Sqrt(vector.Sum(static component => (double)component * component));

        return (float)(1 - dotProduct / (queryNorm * vectorNorm));
    }
}
