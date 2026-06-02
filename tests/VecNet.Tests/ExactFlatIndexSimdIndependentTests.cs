using System.Numerics;

namespace VecNet.Tests;

public sealed class ExactFlatIndexSimdIndependentTests
{
    private const int RandomSeed = 0x51D_005;
    private const float DistanceTolerance = 2e-4f;

    [Fact]
    public void PublicConstructors_DoNotExposeDistanceModeSelection()
    {
        var publicConstructors = typeof(ExactFlatIndex).GetConstructors();

        var constructor = Assert.Single(publicConstructors);
        Assert.Equal(
            [typeof(int), typeof(VectorMetric)],
            constructor.GetParameters().Select(static parameter => parameter.ParameterType));
    }

    [Fact]
    public void VectorFloatSquaredL2_MatchesScalarDistancesForAwkwardDimensionsAndTails()
    {
        foreach (int dimension in GetAwkwardDimensions())
        {
            var random = new Random(RandomSeed + dimension * 997);
            var scalar = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
            var vector = new ExactFlatIndex(
                dimension,
                VectorMetric.SquaredEuclidean,
                ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

            List<(ulong Id, float[] Vector)> entries = [];
            for (int row = 0; row < 23; row++)
            {
                float[] values = CreateRandomVector(random, dimension);
                ulong id = (ulong)(1000 + row * 37);
                entries.Add((id, values));
                scalar.Add(id, values);
                vector.Add(id, values);
            }

            for (int trial = 0; trial < 12; trial++)
            {
                float[] query = CreateRandomVector(random, dimension);
                SearchResult[] scalarResults = SearchAll(scalar, query, entries.Count);
                SearchResult[] vectorResults = SearchAll(vector, query, entries.Count);

                Assert.Equal(scalarResults.Select(static result => result.Id), vectorResults.Select(static result => result.Id));
                for (int i = 0; i < scalarResults.Length; i++)
                {
                    Assert.InRange(
                        MathF.Abs(scalarResults[i].Distance - vectorResults[i].Distance),
                        0f,
                        DistanceTolerance);
                }
            }
        }
    }

    [Fact]
    public void VectorFloatSquaredL2_OrdersExactlyEqualTailAndVectorDistancesByExternalId()
    {
        int dimension = Vector<float>.Count * 2 + 5;
        var index = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        index.Add(50, CreateAlternatingVector(dimension, 1f));
        index.Add(10, CreateAlternatingVector(dimension, -1f));
        index.Add(30, CreateAlternatingVector(dimension, 1f));
        index.Add(20, CreateAlternatingVector(dimension, -1f));

        SearchResult[] results = SearchAll(index, new float[dimension], 4);

        Assert.Equal([10UL, 20UL, 30UL, 50UL], results.Select(static result => result.Id));
        Assert.All(results, result => Assert.Equal(results[0].Distance, result.Distance));
    }

    [Fact]
    public void VectorFloatSquaredL2_PreservesTopKWhenDistancesAreSafelySeparatedAcrossTail()
    {
        int dimension = Vector<float>.Count * 3 + 7;
        var index = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        AddDistanceControlledVector(index, 70, dimension, 12f, tailValue: 2f);
        AddDistanceControlledVector(index, 30, dimension, 4f, tailValue: 0.5f);
        AddDistanceControlledVector(index, 10, dimension, 1f, tailValue: 0.25f);
        AddDistanceControlledVector(index, 50, dimension, 8f, tailValue: 1f);
        AddDistanceControlledVector(index, 90, dimension, 20f, tailValue: 3f);

        SearchResult[] results = SearchAll(index, new float[dimension], 3);

        Assert.Equal([10UL, 30UL, 50UL], results.Select(static result => result.Id));
        Assert.True(results[0].Distance < results[1].Distance);
        Assert.True(results[1].Distance < results[2].Distance);
    }

    [Fact]
    public void NearTieOrdering_DocumentsScalarAndVectorModesCanDiverge()
    {
        int dimension = Vector<float>.Count;
        var scalar = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        var vector = new ExactFlatIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        float[] manySmallLaneTerms = CreateLargeLeadingVector(dimension, smallLaneValue: 1f);
        float[] oneLargerLaneTerm = new float[dimension];
        oneLargerLaneTerm[0] = 4096f;
        oneLargerLaneTerm[1] = 2f;

        scalar.Add(10, manySmallLaneTerms);
        scalar.Add(20, oneLargerLaneTerm);
        vector.Add(10, manySmallLaneTerms);
        vector.Add(20, oneLargerLaneTerm);

        float[] query = new float[dimension];
        SearchResult[] scalarResults = SearchAll(scalar, query, 2);
        SearchResult[] vectorResults = SearchAll(vector, query, 2);

        Assert.Equal([20UL, 10UL], scalarResults.Select(static result => result.Id));
        Assert.Equal([10UL, 20UL], vectorResults.Select(static result => result.Id));
        Assert.True(MathF.Abs(scalarResults[0].Distance - scalarResults[1].Distance) <= 8f);
    }

    [Fact]
    public void PublicConstructor_KeepsScalarOrderingForNearTieCase()
    {
        int dimension = Vector<float>.Count;
        var publicIndex = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);

        publicIndex.Add(10, CreateLargeLeadingVector(dimension, smallLaneValue: 1f));
        float[] oneLargerLaneTerm = new float[dimension];
        oneLargerLaneTerm[0] = 4096f;
        oneLargerLaneTerm[1] = 2f;
        publicIndex.Add(20, oneLargerLaneTerm);

        SearchResult[] results = SearchAll(publicIndex, new float[dimension], 2);

        Assert.Equal([20UL, 10UL], results.Select(static result => result.Id));
    }

    [Fact]
    public void VectorFloatSquaredL2_ValidationRemainsBeforeEmptyDestinationReturn()
    {
        var index = new ExactFlatIndex(
            Vector<float>.Count + 3,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);
        index.Add(1, CreateAlternatingVector(index.Dimension, 1f));

        Assert.Throws<ArgumentException>(
            () => index.Search(CreateVectorWith(index.Dimension, float.NaN), Span<SearchResult>.Empty));
        Assert.Throws<ArgumentException>(
            () => index.Search(new float[index.Dimension - 1], Span<SearchResult>.Empty));
    }

    [Fact]
    public void VectorFloatSquaredL2_AddRejectsInvalidVectorsWithoutConsumingIdentifier()
    {
        var index = new ExactFlatIndex(
            Vector<float>.Count + 1,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        Assert.Throws<ArgumentException>(() => index.Add(7, CreateVectorWith(index.Dimension, float.PositiveInfinity)));

        float[] valid = CreateAlternatingVector(index.Dimension, 0.5f);
        index.Add(7, valid);
        SearchResult[] results = SearchAll(index, valid, 1);

        Assert.Equal(new SearchResult(7, 0f), results[0]);
    }

    private static int[] GetAwkwardDimensions()
    {
        int width = Vector<float>.Count;
        return
        [
            1,
            2,
            3,
            Math.Max(1, width - 1),
            width,
            width + 1,
            width + 5,
            width * 2 - 1,
            width * 2,
            width * 2 + 3,
            width * 4 + 7
        ];
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
            vector[i] = (float)((random.NextDouble() * 6.0) - 3.0);
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

    private static void AddDistanceControlledVector(
        ExactFlatIndex index,
        ulong id,
        int dimension,
        float firstValue,
        float tailValue)
    {
        var vector = new float[dimension];
        vector[0] = firstValue;
        vector[^1] = tailValue;
        index.Add(id, vector);
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

    private static float[] CreateVectorWith(int dimension, float value)
    {
        var vector = new float[dimension];
        vector[0] = value;
        return vector;
    }
}
