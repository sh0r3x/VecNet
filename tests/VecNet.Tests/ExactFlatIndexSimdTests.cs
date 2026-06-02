using System.Numerics;

namespace VecNet.Tests;

public sealed class ExactFlatIndexSimdTests
{
    private const int RandomSeed = 0x5EED_005;
    private const float DistanceTolerance = 2e-4f;

    [Fact]
    public void VectorFloatSquaredL2_MatchesScalarReferenceForVectorWidthAndTailDimensions()
    {
        foreach (int dimension in GetVectorWidthCoverageDimensions())
        {
            var random = new Random(RandomSeed + dimension);
            for (int trial = 0; trial < 32; trial++)
            {
                float[] query = CreateRandomVector(random, dimension);
                float[] vector = CreateRandomVector(random, dimension);

                var index = new ExactFlatIndex(
                    dimension,
                    VectorMetric.SquaredEuclidean,
                    ExactFlatIndexDistanceMode.VectorFloatSquaredL2);
                index.Add(10, vector);

                var results = new SearchResult[1];
                int written = index.Search(query, results);

                Assert.Equal(1, written);
                Assert.Equal(10UL, results[0].Id);
                Assert.InRange(
                    MathF.Abs(results[0].Distance - CalculateScalarSquaredL2(query, vector)),
                    0f,
                    DistanceTolerance);
            }
        }
    }

    [Fact]
    public void VectorFloatSquaredL2_PreservesTopKOrderingWhenDistancesAreSafelySeparated()
    {
        int dimension = Vector<float>.Count + 3;
        var scalar = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var vector = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        AddSeparatedVector(scalar, vector, 40, dimension, 12f);
        AddSeparatedVector(scalar, vector, 10, dimension, 1f);
        AddSeparatedVector(scalar, vector, 30, dimension, 8f);
        AddSeparatedVector(scalar, vector, 20, dimension, 4f);
        AddSeparatedVector(scalar, vector, 50, dimension, 18f);

        float[] query = new float[dimension];
        var scalarResults = new SearchResult[3];
        var vectorResults = new SearchResult[3];

        Assert.Equal(3, scalar.Search(query, scalarResults));
        Assert.Equal(3, vector.Search(query, vectorResults));
        Assert.Equal(scalarResults.Select(static result => result.Id), vectorResults.Select(static result => result.Id));
        Assert.Equal([10UL, 20UL, 30UL], vectorResults.Select(static result => result.Id));
    }

    [Fact]
    public void VectorFloatSquaredL2_OrdersExactlyEqualDistancesByAscendingExternalId()
    {
        int dimension = Vector<float>.Count + 3;
        var index = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

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
    public void PublicConstructor_RemainsScalarDoubleSquaredEuclideanBehavior()
    {
        int dimension = Vector<float>.Count;
        float[] vector = CreateLargeLeadingValueVector(dimension);
        float[] query = new float[dimension];

        var publicIndex = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var scalarIndex = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        var vectorIndex = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        publicIndex.Add(1, vector);
        scalarIndex.Add(1, vector);
        vectorIndex.Add(1, vector);

        var publicResults = new SearchResult[1];
        var scalarResults = new SearchResult[1];
        var vectorResults = new SearchResult[1];

        publicIndex.Search(query, publicResults);
        scalarIndex.Search(query, scalarResults);
        vectorIndex.Search(query, vectorResults);

        float scalarReference = CalculateScalarSquaredL2(query, vector);
        Assert.Equal(scalarReference, publicResults[0].Distance);
        Assert.Equal(scalarResults[0].Distance, publicResults[0].Distance);
        Assert.NotEqual(vectorResults[0].Distance, publicResults[0].Distance);
    }

    [Fact]
    public void VectorFloatSquaredL2Mode_IsAvailableOnlyForSquaredEuclidean()
    {
        Assert.Throws<ArgumentException>(
            () => new ExactFlatIndex(3, VectorMetric.InnerProduct, ExactFlatIndexDistanceMode.VectorFloatSquaredL2));
        Assert.Throws<ArgumentException>(
            () => new ExactFlatIndex(3, VectorMetric.Cosine, ExactFlatIndexDistanceMode.VectorFloatSquaredL2));
    }

    private static int[] GetVectorWidthCoverageDimensions()
    {
        int width = Vector<float>.Count;
        return new[]
            {
                Math.Max(1, width - 1),
                width,
                width + 1,
                width + 3
            }
            .Distinct()
            .ToArray();
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

    private static void AddSeparatedVector(
        ExactFlatIndex scalar,
        ExactFlatIndex vector,
        ulong id,
        int dimension,
        float firstComponent)
    {
        float[] values = CreateSingleValueVector(dimension, firstComponent);
        scalar.Add(id, values);
        vector.Add(id, values);
    }

    private static float[] CreateSingleValueVector(int dimension, float firstComponent)
    {
        var vector = new float[dimension];
        vector[0] = firstComponent;
        return vector;
    }

    private static float[] CreateLargeLeadingValueVector(int dimension)
    {
        var vector = new float[dimension];
        vector[0] = 4096f;
        for (int i = 1; i < vector.Length; i++)
        {
            vector[i] = 1f;
        }

        return vector;
    }

    private static float CalculateScalarSquaredL2(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }
}
