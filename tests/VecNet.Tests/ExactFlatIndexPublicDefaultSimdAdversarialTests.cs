using System.Numerics;

namespace VecNet.Tests;

public sealed class ExactFlatIndexPublicDefaultSimdAdversarialTests
{
    private const int RandomSeed = 0x7EC_007;

    [Fact]
    public void PublicDefaultSquaredL2_StaysWithinD026ToleranceAcrossAbsoluteAndRelativeBranches()
    {
        int dimension = Vector<float>.Count * 3 + 5;
        double crossoverDistance = 2e-4 * 16_777_216.0 / (8.0 * dimension);

        AssertPublicDefaultWithinToleranceAtTargetDistance(dimension, crossoverDistance * 0.75, expectFloorBranch: true);
        AssertPublicDefaultWithinToleranceAtTargetDistance(dimension, crossoverDistance * 1.25, expectFloorBranch: false);
    }

    [Fact]
    public void PublicDefaultSquaredL2_ReturnsPositiveInfinityAndNotNaNWhenReferenceOverflowsFloat()
    {
        int dimension = Vector<float>.Count + 3;
        var scalar = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        float[] hugePositive = new float[dimension];
        hugePositive[0] = float.MaxValue;
        float[] hugeNegative = new float[dimension];
        hugeNegative[^1] = -float.MaxValue;

        scalar.Add(40, hugePositive);
        scalar.Add(10, hugeNegative);
        publicDefault.Add(40, hugePositive);
        publicDefault.Add(10, hugeNegative);

        SearchResult[] scalarResults = SearchAll(scalar, new float[dimension], 2);
        SearchResult[] publicResults = SearchAll(publicDefault, new float[dimension], 2);

        Assert.All(scalarResults, static result => Assert.Equal(float.PositiveInfinity, result.Distance));
        Assert.All(publicResults, static result => Assert.Equal(float.PositiveInfinity, result.Distance));
        Assert.DoesNotContain(publicResults, static result => float.IsNaN(result.Distance));
        Assert.Equal([10UL, 40UL], publicResults.Select(static result => result.Id));
    }

    [Fact]
    public void PublicDefaultSquaredL2_MatchesScalarReferenceForSafelySeparatedFixedSeedPartialTopK()
    {
        int dimension = Vector<float>.Count * 5 + 3;
        const int vectorCount = 31;
        const int topK = 7;
        var random = new Random(RandomSeed);
        float[] query = CreateRandomVector(random, dimension);
        var scalar = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        foreach ((ulong id, float[] vector) in CreateSafelySeparatedCorpus(random, query, vectorCount))
        {
            scalar.Add(id, vector);
            publicDefault.Add(id, vector);
        }

        SearchResult[] scalarResults = SearchAll(scalar, query, topK);
        SearchResult[] publicResults = SearchAll(publicDefault, query, topK);

        AssertSafelySeparated(scalarResults, dimension);
        Assert.Equal(scalarResults.Select(static result => result.Id), publicResults.Select(static result => result.Id));
        for (int i = 0; i < topK; i++)
        {
            AssertWithinD026Tolerance(scalarResults[i].Distance, publicResults[i].Distance, dimension);
        }
    }

    [Fact]
    public void PublicDefaultSquaredL2_NearTieUsesExecutingDistanceNotToleranceBucketOrId()
    {
        int dimension = Vector<float>.Count;
        var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var vectorMode = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        float[] lowerExecutingDistanceWithHigherId = CreateLargeLeadingVector(dimension, smallLaneValue: 1f);
        float[] higherExecutingDistanceWithLowerId = new float[dimension];
        higherExecutingDistanceWithLowerId[0] = 4096f;
        higherExecutingDistanceWithLowerId[1] = 2f;

        publicDefault.Add(100, lowerExecutingDistanceWithHigherId);
        publicDefault.Add(1, higherExecutingDistanceWithLowerId);
        vectorMode.Add(100, lowerExecutingDistanceWithHigherId);
        vectorMode.Add(1, higherExecutingDistanceWithLowerId);

        SearchResult[] publicResults = SearchAll(publicDefault, new float[dimension], 2);
        SearchResult[] vectorResults = SearchAll(vectorMode, new float[dimension], 2);

        Assert.Equal(vectorResults, publicResults);
        Assert.Equal([100UL, 1UL], publicResults.Select(static result => result.Id));
        Assert.True(publicResults[0].Distance < publicResults[1].Distance);

        float combinedTolerance =
            CalculateD026Tolerance(dimension, publicResults[0].Distance) +
            CalculateD026Tolerance(dimension, publicResults[1].Distance);
        Assert.True(publicResults[1].Distance - publicResults[0].Distance <= combinedTolerance);
    }

    [Theory]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void PublicDefaultChange_DoesNotAlterInnerProductOrCosineResults(VectorMetric metric)
    {
        int dimension = Vector<float>.Count + 3;
        var random = new Random(RandomSeed + (int)metric * 4099);
        var publicIndex = new ExactFlatIndex(dimension, metric);
        var scalarReference = new ExactFlatIndex(dimension, metric, ExactFlatIndexDistanceMode.ScalarDouble);

        for (int row = 0; row < 13; row++)
        {
            float[] vector = CreateNonZeroRandomVector(random, dimension);
            ulong id = (ulong)(500 + row * 17);
            publicIndex.Add(id, vector);
            scalarReference.Add(id, vector);
        }

        float[] query = CreateNonZeroRandomVector(random, dimension);
        SearchResult[] publicResults = SearchAll(publicIndex, query, 5);
        SearchResult[] scalarResults = SearchAll(scalarReference, query, 5);

        Assert.Equal(scalarResults.Select(static result => result.Id), publicResults.Select(static result => result.Id));
        for (int i = 0; i < publicResults.Length; i++)
        {
            Assert.Equal(scalarResults[i].Distance, publicResults[i].Distance);
        }
    }

    [Fact]
    public void PublicDefaultSquaredL2_ValidationStillRejectsDuplicateIdsAfterFailedGrowthBoundaryInsertion()
    {
        var index = new ExactFlatIndex(Vector<float>.Count + 1, VectorMetric.SquaredEuclidean);
        for (ulong id = 1; id <= 4; id++)
        {
            index.Add(id, CreateConstantVector(index.Dimension, (float)id));
        }

        Assert.Throws<ArgumentException>(() => index.Add(5, CreateVectorWith(index.Dimension, float.NaN)));
        index.Add(5, CreateConstantVector(index.Dimension, 5f));

        Assert.Throws<ArgumentException>(() => index.Add(5, CreateConstantVector(index.Dimension, 6f)));
        SearchResult[] results = SearchAll(index, new float[index.Dimension], 5);
        Assert.Equal([1UL, 2UL, 3UL, 4UL, 5UL], results.Select(static result => result.Id));
    }

    private static void AssertPublicDefaultWithinToleranceAtTargetDistance(
        int dimension,
        double targetDistance,
        bool expectFloorBranch)
    {
        float offset = (float)Math.Sqrt(targetDistance / dimension);
        float[] vector = CreateConstantVector(dimension, offset);
        var scalar = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        scalar.Add(10, vector);
        publicDefault.Add(10, vector);

        SearchResult scalarResult = SearchAll(scalar, new float[dimension], 1)[0];
        SearchResult publicResult = SearchAll(publicDefault, new float[dimension], 1)[0];

        Assert.Equal(expectFloorBranch, UsesAbsoluteFloorTolerance(dimension, scalarResult.Distance));
        AssertWithinD026Tolerance(scalarResult.Distance, publicResult.Distance, dimension);
    }

    private static IEnumerable<(ulong Id, float[] Vector)> CreateSafelySeparatedCorpus(
        Random random,
        float[] query,
        int count)
    {
        int[] insertionOrder = Enumerable.Range(0, count).OrderBy(_ => random.Next()).ToArray();
        foreach (int rank in insertionOrder)
        {
            float amplitude = 0.5f + rank * 2.75f;
            float[] vector = new float[query.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                float laneScale = 0.75f + (i % 11) * 0.125f;
                float sign = ((rank + i) & 1) == 0 ? 1f : -1f;
                vector[i] = query[i] + sign * amplitude * laneScale;
            }

            yield return ((ulong)(10_000 + rank * 131), vector);
        }
    }

    private static void AssertSafelySeparated(ReadOnlySpan<SearchResult> results, int dimension)
    {
        for (int i = 0; i < results.Length - 1; i++)
        {
            float gap = results[i + 1].Distance - results[i].Distance;
            float combinedTolerance =
                CalculateD026Tolerance(dimension, results[i].Distance) +
                CalculateD026Tolerance(dimension, results[i + 1].Distance);
            Assert.True(gap > combinedTolerance * 4f);
        }
    }

    private static SearchResult[] SearchAll(ExactFlatIndex index, float[] query, int capacity)
    {
        var results = new SearchResult[capacity];
        int written = index.Search(query, results);
        Assert.Equal(capacity, written);
        return results;
    }

    private static float[] CreateRandomVector(Random random, int dimension)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (float)(random.NextDouble() * 4.0 - 2.0);
        }

        return vector;
    }

    private static float[] CreateNonZeroRandomVector(Random random, int dimension)
    {
        float[] vector = CreateRandomVector(random, dimension);
        bool hasNonZero = vector.Any(static value => value != 0f);
        if (!hasNonZero)
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static float[] CreateLargeLeadingVector(int dimension, float smallLaneValue)
    {
        var vector = new float[dimension];
        vector[0] = 4096f;
        for (int i = 1; i < vector.Length; i++)
        {
            vector[i] = smallLaneValue;
        }

        return vector;
    }

    private static float[] CreateConstantVector(int dimension, float value)
    {
        var vector = new float[dimension];
        Array.Fill(vector, value);
        return vector;
    }

    private static float[] CreateVectorWith(int dimension, float value)
    {
        var vector = new float[dimension];
        vector[0] = value;
        return vector;
    }

    private static void AssertWithinD026Tolerance(float scalarReference, float optimized, int dimension)
    {
        Assert.True(float.IsFinite(scalarReference), "This assertion is only for finite scalar-reference cases.");
        Assert.True(float.IsFinite(optimized), "Optimized distance must be finite for finite reference cases.");
        Assert.True(optimized >= 0f, "Optimized squared-L2 distance must be non-negative.");
        Assert.InRange(
            MathF.Abs(optimized - scalarReference),
            0f,
            CalculateD026Tolerance(dimension, scalarReference));
    }

    private static bool UsesAbsoluteFloorTolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return relative < 2e-4;
    }

    private static float CalculateD026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }
}
