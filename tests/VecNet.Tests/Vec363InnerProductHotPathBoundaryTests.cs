namespace VecNet.Tests;

public sealed class Vec363InnerProductHotPathBoundaryTests
{
    [Theory]
    [InlineData(31)]
    [InlineData(33)]
    [InlineData(128)]
    [InlineData(386)]
    [InlineData(768)]
    public void SharedPrimitive_MatchesScalarFiniteAndInfinityCategoriesWithoutComputedNaN(int dimension)
    {
        (float[] Left, float[] Right)[] cases =
        [
            (new float[dimension], new float[dimension]),
            (CreatePatternVector(dimension, 2), CreatePatternVector(dimension, 7)),
            (CreateSingleLaneVector(dimension, 2f), CreateSingleLaneVector(dimension, float.MaxValue)),
            (CreateSingleLaneVector(dimension, 2f), CreateSingleLaneVector(dimension, -float.MaxValue))
        ];

        foreach ((float[] left, float[] right) in cases)
        {
            float expected = ScalarNegativeDot(left, right);
            float actual = InnerProductDistancePrimitive.Distance(left, right);

            AssertSameCategory(expected, actual);
            Assert.False(float.IsNaN(actual), "Finite inner-product inputs must not produce computed NaN.");
            if (float.IsFinite(expected))
            {
                Assert.Equal(expected, actual);
            }
        }
    }

    [Theory]
    [InlineData(31)]
    [InlineData(128)]
    public void ExactFlatSearch_MatchesScalarBoundaryCategoriesAndTieOrdering(int dimension)
    {
        Row[] rows = CreateBoundaryRows(dimension);
        var index = new ExactFlatIndex(dimension, VectorMetric.InnerProduct);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        SearchResult[] actual = Search(index, CreateBoundaryQuery(dimension), rows.Length);
        SearchResult[] expected = ScalarTruth(rows, CreateBoundaryQuery(dimension), rows.Length);

        AssertResultsMatchScalar(expected, actual);
        Assert.Equal([5UL, 40UL, 20UL, 25UL, 30UL, 50UL], actual.Select(static result => result.Id));
    }

    [Fact]
    public void ImmutableAndOpenedReadOnlyHnswSearch_MatchScalarBoundaryCategories()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        int dimension = 33;
        Row[] rows = CreateBoundaryRows(dimension);
        HnswIndex source = CreateHnsw(rows, new HnswIndexOptions(8, 64, 64, 0x3630_1001UL));
        float[] query = CreateBoundaryQuery(dimension);
        SearchResult[] expected = ScalarTruth(rows, query, rows.Length);

        SearchResult[] sourceResults = Search(source, query, rows.Length, efSearch: 64);
        AssertResultsMatchScalar(expected, sourceResults);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        SearchResult[] openedResults = Search(opened, query, rows.Length, efSearch: 64);

        AssertResultsMatchScalar(expected, openedResults);
        AssertResultsMatchScalar(sourceResults, openedResults);
    }

    [Fact]
    public void MutableBasePlusExactDeltaSearchCheckpointAndOpenedReadOnly_MatchScalarBoundaryCategories()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        int dimension = 128;
        Row[] allRows = CreateBoundaryRows(dimension);
        Row[] baseRows = allRows.Where(static row => row.Id is 20 or 25 or 30 or 50).ToArray();
        Row[] liveRows = allRows.Where(static row => row.Id != 30).ToArray();
        float[] query = CreateBoundaryQuery(dimension);

        HnswMutableIndex mutable = new(CreateHnsw(baseRows, new HnswIndexOptions(8, 64, 64, 0x3630_2001UL)));
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(5, allRows.Single(static row => row.Id == 5).Vector).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(40, allRows.Single(static row => row.Id == 40).Vector).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(30).Status);

        SearchResult[] expected = ScalarTruth(liveRows, query, topK: liveRows.Length);
        SearchResult[] beforeCheckpoint = Search(mutable, query, liveRows.Length, efSearch: 64);
        AssertResultsMatchScalar(expected, beforeCheckpoint);

        SearchResult[] expectedAllowed = ScalarTruth(
            liveRows.Where(static row => row.Id is 5 or 20 or 30 or 50).ToArray(),
            query,
            topK: 4);
        SearchResult[] allowed = Search(mutable, query, [999, 5, 20, 30, 50, 5], topK: 4, efSearch: 64);
        AssertResultsMatchScalar(expectedAllowed, allowed);

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        AssertResultsMatchScalar(expected, Search(mutable, query, liveRows.Length, efSearch: 64));
        AssertResultsMatchScalar(expected, Search(opened, query, liveRows.Length, efSearch: 64));
    }

    private static HnswIndex CreateHnsw(Row[] rows, HnswIndexOptions options)
    {
        var index = new HnswIndex(rows[0].Vector.Length, VectorMetric.InnerProduct, options, () => 0);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] ScalarTruth(Row[] rows, float[] query, int topK) =>
        rows
            .Select(row => new SearchResult(row.Id, ScalarNegativeDot(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static void AssertResultsMatchScalar(SearchResult[] expected, SearchResult[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            AssertSameCategory(expected[i].Distance, actual[i].Distance);
            Assert.False(float.IsNaN(actual[i].Distance), $"Distance for ID {actual[i].Id} was computed NaN.");
            if (float.IsFinite(expected[i].Distance))
            {
                Assert.Equal(expected[i].Distance, actual[i].Distance);
            }
        }
    }

    private static void AssertSameCategory(float expected, float actual)
    {
        Assert.Equal(float.IsFinite(expected), float.IsFinite(actual));
        Assert.Equal(float.IsPositiveInfinity(expected), float.IsPositiveInfinity(actual));
        Assert.Equal(float.IsNegativeInfinity(expected), float.IsNegativeInfinity(actual));
        Assert.Equal(float.IsNaN(expected), float.IsNaN(actual));
    }

    private static Row[] CreateBoundaryRows(int dimension) =>
    [
        new(30, CreateBoundaryVector(dimension, 1f, 1f, 1f)),
        new(5, CreateSingleLaneVector(dimension, float.MaxValue)),
        new(25, new float[dimension]),
        new(50, CreateSingleLaneVector(dimension, -float.MaxValue)),
        new(20, CreateBoundaryVector(dimension, 1f, 1f, 1f)),
        new(40, CreateBoundaryVector(dimension, 5f, 1f, 0f))
    ];

    private static float[] CreateBoundaryQuery(int dimension) =>
        CreateBoundaryVector(dimension, 2f, -3f, 1f);

    private static float[] CreateBoundaryVector(int dimension, float first, float second, float third)
    {
        var vector = new float[dimension];
        vector[0] = first;
        if (dimension > 1)
        {
            vector[1] = second;
        }

        if (dimension > 2)
        {
            vector[2] = third;
        }

        return vector;
    }

    private static float[] CreateSingleLaneVector(int dimension, float value)
    {
        var vector = new float[dimension];
        vector[0] = value;
        return vector;
    }

    private static float[] CreatePatternVector(int dimension, int seed)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = ((i + seed) % 5) - 2;
        }

        return vector;
    }

    private static float ScalarNegativeDot(float[] query, float[] vector)
    {
        double dot = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
        }

        return (float)-dot;
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() =>
            new(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                "Vec363-" + Guid.NewGuid().ToString("N")));

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }

    private readonly record struct Row(ulong Id, float[] Vector);
}
