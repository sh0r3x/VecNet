using System.Numerics;

namespace VecNet.Tests;

public sealed class Vec365InnerProductNumericalBoundaryIndependentTests
{
    [Fact]
    public void SharedPrimitiveMatchesScalarOracleAcrossVectorWidthBoundariesAndInfinityCategories()
    {
        int[] dimensions =
        [
            1,
            Math.Max(1, Vector<float>.Count - 1),
            Vector<float>.Count,
            Vector<float>.Count + 1,
            (Vector<float>.Count * 2) + 3,
            65
        ];

        foreach (int dimension in dimensions.Distinct())
        {
            (float[] Left, float[] Right)[] cases =
            [
                (new float[dimension], new float[dimension]),
                (CreatePatternVector(dimension, 3), CreatePatternVector(dimension, 11)),
                (CreateSingleLaneVector(dimension, 2f, dimension - 1), CreateSingleLaneVector(dimension, float.MaxValue, dimension - 1)),
                (CreateSingleLaneVector(dimension, -2f, dimension / 2), CreateSingleLaneVector(dimension, float.MaxValue, dimension / 2))
            ];

            foreach ((float[] left, float[] right) in cases)
            {
                float expected = ScalarNegativeDot(left, right);
                float actual = InnerProductDistancePrimitive.Distance(left, right);

                AssertSameCategory(expected, actual);
                Assert.False(float.IsNaN(actual), $"Dimension {dimension} produced computed NaN for finite inputs.");
                if (float.IsFinite(expected))
                {
                    Assert.Equal(expected, actual);
                }
            }
        }
    }

    [Fact]
    public void ExactFlatInnerProductMatchesScalarOracleForAwkwardMixedSignTiesZerosAndInfinities()
    {
        const int dimension = 65;
        Row[] rows = CreateBoundaryRows(dimension);
        float[] query = CreateBoundaryQuery(dimension);
        var index = new ExactFlatIndex(dimension, VectorMetric.InnerProduct);

        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        SearchResult[] actual = Search(index, query, rows.Length);
        SearchResult[] expected = ScalarTruth(rows, query, rows.Length);

        AssertResultsMatchScalar(expected, actual);
        Assert.Equal([90UL, 20UL, 25UL, 50UL, 80UL, 40UL, 60UL, 10UL], actual.Select(static result => result.Id));
    }

    [Fact]
    public void ImmutableAndOpenedReadOnlyHnswInnerProductMatchScalarOracleForBoundaryRows()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        const int dimension = 65;
        Row[] rows = CreateBoundaryRows(dimension);
        float[] query = CreateBoundaryQuery(dimension);
        var options = new HnswIndexOptions(16, 96, 96, 0x3650_1001UL);
        HnswIndex source = CreateHnsw(rows, options);
        SearchResult[] expected = ScalarTruth(rows, query, rows.Length);

        SearchResult[] sourceResults = Search(source, query, rows.Length, efSearch: options.EfSearch);
        AssertResultsMatchScalar(expected, sourceResults);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        SearchResult[] openedResults = Search(opened, query, rows.Length, efSearch: options.EfSearch);

        Assert.Equal(VectorMetric.InnerProduct, opened.Metric);
        AssertResultsMatchScalar(expected, openedResults);
        AssertResultsMatchScalar(sourceResults, openedResults);
    }

    [Fact]
    public void MutableInnerProductMergesDeltaTombstonesAllowlistsCheckpointAndReopenWithBoundaryDistances()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        const int dimension = 65;
        float[] query = CreateBoundaryQuery(dimension);
        Row[] baseRows =
        [
            BoundaryRow(10, dimension, float.MaxValue),
            BoundaryRow(20, dimension, 4f),
            BoundaryRow(30, dimension, 0f),
            BoundaryRow(40, dimension, -float.MaxValue)
        ];
        HnswMutableIndex mutable = new(CreateHnsw(baseRows, new HnswIndexOptions(16, 96, 96, 0x3650_2001UL)));

        AssertCommitted(mutable.TryAdd(5, BoundaryVector(dimension, float.MaxValue)));
        AssertCommitted(mutable.TryAdd(25, BoundaryVector(dimension, 4f)));
        AssertCommitted(mutable.TryAdd(35, BoundaryVector(dimension, 0f)));
        AssertCommitted(mutable.TryAdd(15, BoundaryVector(dimension, 123f)));
        AssertCommitted(mutable.TryDelete(10));
        AssertCommitted(mutable.TryDelete(15));
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(10, BoundaryVector(dimension, 1f)).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(15, BoundaryVector(dimension, 1f)).Status);

        Row[] liveRows =
        [
            BoundaryRow(5, dimension, float.MaxValue),
            BoundaryRow(20, dimension, 4f),
            BoundaryRow(25, dimension, 4f),
            BoundaryRow(30, dimension, 0f),
            BoundaryRow(35, dimension, 0f),
            BoundaryRow(40, dimension, -float.MaxValue)
        ];

        SearchResult[] expected = ScalarTruth(liveRows, query, liveRows.Length);
        SearchResult[] beforeCheckpoint = Search(mutable, query, liveRows.Length, efSearch: 96);
        AssertResultsMatchScalar(expected, beforeCheckpoint);

        SearchResult[] expectedAllowed = ScalarTruth(
            liveRows.Where(static row => row.Id is 5 or 20 or 25 or 40).ToArray(),
            query,
            topK: 4);
        SearchResult[] allowed = Search(mutable, query, [10, 15, 5, 20, 25, 40, 5], topK: 4, efSearch: 96);
        AssertResultsMatchScalar(expectedAllowed, allowed);
        Assert.DoesNotContain(allowed, static result => result.Id is 10 or 15);

        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(10, BoundaryVector(dimension, 1f)).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(15, BoundaryVector(dimension, 1f)).Status);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.InnerProduct, opened.Metric);
        Assert.DoesNotContain(10UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(15UL, opened.InternalIds.ToArray());
        AssertResultsMatchScalar(expected, Search(mutable, query, liveRows.Length, efSearch: 96));
        AssertResultsMatchScalar(expected, Search(opened, query, liveRows.Length, efSearch: 96));
    }

    [Fact]
    public void SquaredL2AndCosineSmokeRemainRoutedToTheirMetricSpecificDistances()
    {
        var exactL2 = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        exactL2.Add(20, [1f, 2f, 3f]);
        exactL2.Add(10, [1f, -2f, 3f]);
        Assert.Equal(
            [new SearchResult(20, 0f), new SearchResult(10, 16f)],
            Search(exactL2, [1f, 2f, 3f], topK: 2));

        var hnswCosine = new HnswIndex(3, VectorMetric.Cosine, new HnswIndexOptions(4, 16, 16, 0x3650_3001UL), () => 0);
        hnswCosine.Add(30, [0f, 5f, 0f]);
        hnswCosine.Add(10, [2f, 0f, 0f]);

        SearchResult[] results = Search(hnswCosine, [0f, 3f, 0f], topK: 2, efSearch: 16);

        Assert.Equal(30UL, results[0].Id);
        Assert.Equal(0f, results[0].Distance, precision: 6);
        Assert.Equal(10UL, results[1].Id);
        Assert.Equal(1f, results[1].Distance, precision: 6);
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
        BoundaryRow(90, dimension, float.MaxValue),
        BoundaryRow(10, dimension, -float.MaxValue),
        BoundaryRow(20, dimension, 4f),
        BoundaryRow(25, dimension, 4f),
        BoundaryRow(40, dimension, 0f),
        new(50, CreatePatternVector(dimension, 17)),
        BoundaryRow(60, dimension, -3f),
        new(80, CreateMixedSignTailVector(dimension))
    ];

    private static Row BoundaryRow(ulong id, int dimension, float firstLane) =>
        new(id, BoundaryVector(dimension, firstLane));

    private static float[] BoundaryVector(int dimension, float firstLane)
    {
        var vector = new float[dimension];
        vector[0] = firstLane;
        if (dimension > 3 && float.IsFinite(firstLane) && firstLane != 0f)
        {
            vector[3] = -firstLane * 0.25f;
        }

        return vector;
    }

    private static float[] CreateBoundaryQuery(int dimension)
    {
        var query = new float[dimension];
        query[0] = 2f;
        if (dimension > 3)
        {
            query[3] = -8f;
        }

        if (dimension > 17)
        {
            query[17] = 0.5f;
        }

        if (dimension > 64)
        {
            query[64] = -0.25f;
        }

        return query;
    }

    private static float[] CreateMixedSignTailVector(int dimension)
    {
        var vector = new float[dimension];
        vector[0] = -3f;
        if (dimension > 17)
        {
            vector[17] = 8f;
        }

        if (dimension > 64)
        {
            vector[64] = -16f;
        }

        return vector;
    }

    private static float[] CreatePatternVector(int dimension, int seed)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (((i * 7) + seed) % 9 - 4) * 0.25f;
        }

        return vector;
    }

    private static float[] CreateSingleLaneVector(int dimension, float value, int lane)
    {
        var vector = new float[dimension];
        vector[lane] = value;
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

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

    private readonly record struct Row(ulong Id, float[] Vector);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() =>
            new(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                "Vec365-" + Guid.NewGuid().ToString("N")));

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
