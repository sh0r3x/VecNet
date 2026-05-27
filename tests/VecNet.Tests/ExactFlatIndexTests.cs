namespace VecNet.Tests;

public sealed class ExactFlatIndexTests
{
    [Fact]
    public void Search_ComputesSquaredEuclideanDistanceInNearestFirstOrder()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(20, [2f, 3f]);
        index.Add(10, [1f, 1f]);

        var results = new SearchResult[2];
        int written = index.Search([0f, 1f], results);

        Assert.Equal(2, written);
        Assert.Equal(new SearchResult(10, 1f), results[0]);
        Assert.Equal(new SearchResult(20, 8f), results[1]);
    }

    [Fact]
    public void Search_ComputesNegativeInnerProductDistance()
    {
        var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        index.Add(1, [2f, 3f]);
        index.Add(2, [-1f, 0f]);

        var results = new SearchResult[2];
        int written = index.Search([1f, 2f], results);

        Assert.Equal(2, written);
        Assert.Equal(new SearchResult(1, -8f), results[0]);
        Assert.Equal(new SearchResult(2, 1f), results[1]);
    }

    [Fact]
    public void Search_NormalizesCosineInputsAndComputesCosineDistance()
    {
        var index = new ExactFlatIndex(2, VectorMetric.Cosine);
        index.Add(1, [0f, 5f]);
        index.Add(2, [2f, 0f]);

        var results = new SearchResult[2];
        int written = index.Search([0f, 2f], results);

        Assert.Equal(2, written);
        Assert.Equal(1UL, results[0].Id);
        Assert.Equal(0f, results[0].Distance);
        Assert.Equal(2UL, results[1].Id);
        Assert.Equal(1f, results[1].Distance);
    }

    [Fact]
    public void Search_OrdersEqualDistancesByAscendingExternalId()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(30, [-1f]);
        index.Add(10, [1f]);
        index.Add(20, [-1f]);

        var results = new SearchResult[3];
        int written = index.Search([0f], results);

        Assert.Equal(3, written);
        Assert.Equal([10UL, 20UL, 30UL], results.Select(static result => result.Id));
    }

    [Fact]
    public void Search_HandlesEmptyIndexZeroCapacityAndPartialResultSets()
    {
        var empty = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        Assert.Equal(0, empty.Search([0f], new SearchResult[2]));

        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(2, [2f]);
        index.Add(1, [1f]);

        Assert.Equal(0, index.Search([0f], []));

        var moreCapacityThanItems = new SearchResult[4];
        int written = index.Search([0f], moreCapacityThanItems);

        Assert.Equal(2, written);
        Assert.Equal(1UL, moreCapacityThanItems[0].Id);
        Assert.Equal(2UL, moreCapacityThanItems[1].Id);
    }

    [Fact]
    public void Search_ReturnsBestItemsAfterStorageGrowthWhenBufferIsSmallerThanCount()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(50, [5f]);
        index.Add(10, [1f]);
        index.Add(30, [3f]);
        index.Add(20, [2f]);
        index.Add(40, [4f]);

        var results = new SearchResult[2];
        int written = index.Search([0f], results);

        Assert.Equal(2, written);
        Assert.Equal(new SearchResult(10, 1f), results[0]);
        Assert.Equal(new SearchResult(20, 4f), results[1]);
    }

    [Fact]
    public void Add_RejectsDuplicateIdentifiersWithoutReplacingExistingVector()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(7, [1f]);

        Assert.Throws<ArgumentException>(() => index.Add(7, [9f]));

        var results = new SearchResult[1];
        index.Search([1f], results);
        Assert.Equal(new SearchResult(7, 0f), results[0]);
    }

    [Fact]
    public void Constructor_RejectsInvalidDimensionAndMetric()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ExactFlatIndex(0, VectorMetric.SquaredEuclidean));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ExactFlatIndex(2, (VectorMetric)99));
    }

    [Fact]
    public void AddAndSearch_RejectMismatchedDimensions()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);

        Assert.Throws<ArgumentException>(() => index.Add(1, [1f]));
        Assert.Throws<ArgumentException>(() => index.Search([1f], new SearchResult[1]));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Add_RejectsNonFiniteComponents(float invalidComponent)
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);

        Assert.Throws<ArgumentException>(() => index.Add(1, [invalidComponent, 0f]));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Search_RejectsNonFiniteComponents(float invalidComponent)
    {
        var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        index.Add(1, [1f, 0f]);

        Assert.Throws<ArgumentException>(
            () => index.Search([invalidComponent, 0f], new SearchResult[1]));
    }

    [Fact]
    public void Cosine_RejectsZeroVectorsForInsertionAndSearch()
    {
        var index = new ExactFlatIndex(2, VectorMetric.Cosine);

        Assert.Throws<ArgumentException>(() => index.Add(1, [0f, 0f]));

        index.Add(1, [1f, 0f]);
        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], new SearchResult[1]));
    }
}
