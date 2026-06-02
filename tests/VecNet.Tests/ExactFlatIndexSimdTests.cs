using System.Reflection;
using System.Numerics;

namespace VecNet.Tests;

public sealed class ExactFlatIndexSimdTests
{
    private const int RandomSeed = 0x5EED_005;

    [Fact]
    public void PublicDefaultSquaredL2_MatchesScalarReferenceWithinD026ToleranceForAdmissionCoverage()
    {
        foreach (int dimension in GetAdmissionCoverageDimensions())
        {
            var random = new Random(RandomSeed + dimension);
            float[][] queries = CreateAdmissionQueries(random, dimension);
            float[][] vectors = CreateAdmissionVectors(random, dimension);

            foreach (float[] query in queries)
            {
                foreach (float[] vector in vectors)
                {
                    var scalar = new ExactFlatIndex(
                        dimension,
                        VectorMetric.SquaredEuclidean,
                        ExactFlatIndexDistanceMode.ScalarDouble);
                    var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

                    scalar.Add(10, vector);
                    publicDefault.Add(10, vector);

                    var scalarResults = new SearchResult[1];
                    var publicResults = new SearchResult[1];
                    Assert.Equal(1, scalar.Search(query, scalarResults));
                    Assert.Equal(1, publicDefault.Search(query, publicResults));

                    Assert.Equal(10UL, publicResults[0].Id);
                    AssertWithinD026Tolerance(scalarResults[0].Distance, publicResults[0].Distance, dimension);
                }
            }
        }
    }

    [Fact]
    public void PublicDefaultSquaredL2_PreservesFullTopKOrderingWhenScalarDistancesAreSafelySeparated()
    {
        int dimension = Vector<float>.Count + 3;
        var scalar = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var publicDefault = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        AddSeparatedVector(scalar, publicDefault, 40, dimension, 12f);
        AddSeparatedVector(scalar, publicDefault, 10, dimension, 1f);
        AddSeparatedVector(scalar, publicDefault, 30, dimension, 8f);
        AddSeparatedVector(scalar, publicDefault, 20, dimension, 4f);
        AddSeparatedVector(scalar, publicDefault, 50, dimension, 18f);

        float[] query = new float[dimension];
        var scalarResults = new SearchResult[5];
        var publicResults = new SearchResult[5];

        Assert.Equal(5, scalar.Search(query, scalarResults));
        Assert.Equal(5, publicDefault.Search(query, publicResults));
        AssertScalarGapsAreSafelySeparated(scalarResults, dimension);
        Assert.Equal(scalarResults.Select(static result => result.Id), publicResults.Select(static result => result.Id));
        Assert.Equal([10UL, 20UL, 30UL, 40UL, 50UL], publicResults.Select(static result => result.Id));
    }

    [Fact]
    public void PublicDefaultSquaredL2_OrdersExactlyEqualExecutingKernelDistancesByAscendingExternalId()
    {
        int dimension = Vector<float>.Count + 3;
        var index = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        index.Add(30, CreateSingleValueVector(dimension, 1f));
        index.Add(10, CreateSingleValueVector(dimension, -1f));
        index.Add(20, CreateSingleValueVector(dimension, 1f));

        var results = new SearchResult[3];
        int written = index.Search(new float[dimension], results);

        Assert.Equal(3, written);
        Assert.Equal([10UL, 20UL, 30UL], results.Select(static result => result.Id));
        Assert.Equal(results[0].Distance, results[1].Distance);
        Assert.Equal(results[1].Distance, results[2].Distance);
    }

    [Fact]
    public void PublicConstructor_SelectsOptimizedModeOnlyForSquaredEuclidean()
    {
        Assert.Equal(
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2,
            GetDistanceMode(new ExactFlatIndex(3, VectorMetric.SquaredEuclidean)));
        Assert.Equal(
            ExactFlatIndexDistanceMode.ScalarDouble,
            GetDistanceMode(new ExactFlatIndex(3, VectorMetric.InnerProduct)));
        Assert.Equal(
            ExactFlatIndexDistanceMode.ScalarDouble,
            GetDistanceMode(new ExactFlatIndex(3, VectorMetric.Cosine)));
    }

    [Fact]
    public void PublicDefaultSquaredL2_DocumentsAllowedNearTieDivergenceFromScalarReference()
    {
        int dimension = Vector<float>.Count;
        var publicIndex = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var scalarIndex = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var vectorIndex = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        float[] manySmallLaneTerms = CreateLargeLeadingValueVector(dimension, smallLaneValue: 1f);
        float[] oneLargerLaneTerm = new float[dimension];
        oneLargerLaneTerm[0] = 4096f;
        oneLargerLaneTerm[1] = 2f;

        publicIndex.Add(10, manySmallLaneTerms);
        publicIndex.Add(20, oneLargerLaneTerm);
        scalarIndex.Add(10, manySmallLaneTerms);
        scalarIndex.Add(20, oneLargerLaneTerm);
        vectorIndex.Add(10, manySmallLaneTerms);
        vectorIndex.Add(20, oneLargerLaneTerm);

        var publicResults = new SearchResult[2];
        var scalarResults = new SearchResult[2];
        var vectorResults = new SearchResult[2];
        float[] query = new float[dimension];

        publicIndex.Search(query, publicResults);
        scalarIndex.Search(query, scalarResults);
        vectorIndex.Search(query, vectorResults);

        Assert.Equal([20UL, 10UL], scalarResults.Select(static result => result.Id));
        Assert.Equal([10UL, 20UL], publicResults.Select(static result => result.Id));
        Assert.Equal(vectorResults, publicResults);
        Assert.True(publicResults[0].Distance <= publicResults[1].Distance);

        float combinedTolerance =
            CalculateD026Tolerance(dimension, scalarResults[0].Distance) +
            CalculateD026Tolerance(dimension, scalarResults[1].Distance);
        Assert.True(
            MathF.Abs(scalarResults[0].Distance - scalarResults[1].Distance) <= combinedTolerance,
            "The crafted divergence must remain inside the D-026 near-tie tolerance envelope.");
    }

    [Fact]
    public void VectorFloatSquaredL2Mode_IsAvailableOnlyForSquaredEuclidean()
    {
        Assert.Throws<ArgumentException>(
            () => new ExactFlatIndex(3, VectorMetric.InnerProduct, ExactFlatIndexDistanceMode.VectorFloatSquaredL2));
        Assert.Throws<ArgumentException>(
            () => new ExactFlatIndex(3, VectorMetric.Cosine, ExactFlatIndexDistanceMode.VectorFloatSquaredL2));
    }

    private static int[] GetAdmissionCoverageDimensions()
    {
        int width = Vector<float>.Count;
        return new[]
            {
                1,
                2,
                3,
                Math.Max(1, width - 1),
                width,
                width + 1,
                width * 2,
                96,
                128,
                384,
                386,
                768
            }
            .Distinct()
            .ToArray();
    }

    private static float[][] CreateAdmissionQueries(Random random, int dimension) =>
    [
        new float[dimension],
        CreateRandomVector(random, dimension),
        CreateSmallOffsetVector(dimension)
    ];

    private static float[][] CreateAdmissionVectors(Random random, int dimension)
    {
        float[] duplicateVector = CreateRandomVector(random, dimension);
        return
        [
            new float[dimension],
            duplicateVector,
            duplicateVector.ToArray(),
            CreateAlternatingVector(dimension, 0.001f),
            CreateAlternatingVector(dimension, 2.5f),
            CreateRandomVector(random, dimension),
            CreateClusteredVector(dimension, 0.125f),
            CreateTailLaneVector(dimension)
        ];
    }

    private static float[] CreateRandomVector(Random random, int dimension)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.Next(-32, 33) / 8f;
        }

        return vector;
    }

    private static float[] CreateSmallOffsetVector(int dimension)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (i % 3 - 1) * 0.0005f;
        }

        return vector;
    }

    private static float[] CreateAlternatingVector(int dimension, float magnitude)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = i % 2 == 0 ? magnitude : -magnitude;
        }

        return vector;
    }

    private static float[] CreateClusteredVector(int dimension, float center)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = center + (i % 5 - 2) * 0.00025f;
        }

        return vector;
    }

    private static float[] CreateTailLaneVector(int dimension)
    {
        var vector = new float[dimension];
        vector[0] = -3.5f;
        vector[^1] = 4.25f;
        return vector;
    }

    private static void AddSeparatedVector(
        ExactFlatIndex scalar,
        ExactFlatIndex publicDefault,
        ulong id,
        int dimension,
        float firstComponent)
    {
        float[] values = CreateSingleValueVector(dimension, firstComponent);
        scalar.Add(id, values);
        publicDefault.Add(id, values);
    }

    private static float[] CreateSingleValueVector(int dimension, float firstComponent)
    {
        var vector = new float[dimension];
        vector[0] = firstComponent;
        return vector;
    }

    private static float[] CreateLargeLeadingValueVector(int dimension, float smallLaneValue)
    {
        var vector = new float[dimension];
        vector[0] = 4096f;
        for (int i = 1; i < vector.Length; i++)
        {
            vector[i] = smallLaneValue;
        }

        return vector;
    }

    private static void AssertWithinD026Tolerance(float scalarReference, float optimized, int dimension)
    {
        Assert.True(float.IsFinite(optimized), "Optimized distance must be finite for finite reference cases.");
        Assert.True(optimized >= 0f, "Optimized squared-L2 distance must be non-negative.");
        Assert.InRange(
            MathF.Abs(optimized - scalarReference),
            0f,
            CalculateD026Tolerance(dimension, scalarReference));
    }

    private static void AssertScalarGapsAreSafelySeparated(ReadOnlySpan<SearchResult> scalarResults, int dimension)
    {
        for (int i = 0; i < scalarResults.Length - 1; i++)
        {
            float gap = scalarResults[i + 1].Distance - scalarResults[i].Distance;
            float combinedTolerance =
                CalculateD026Tolerance(dimension, scalarResults[i].Distance) +
                CalculateD026Tolerance(dimension, scalarResults[i + 1].Distance);
            Assert.True(gap > combinedTolerance * 4f);
        }
    }

    private static float CalculateD026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static ExactFlatIndexDistanceMode GetDistanceMode(ExactFlatIndex index)
    {
        FieldInfo field = typeof(ExactFlatIndex).GetField(
            "_distanceMode",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        return (ExactFlatIndexDistanceMode)field.GetValue(index)!;
    }
}
