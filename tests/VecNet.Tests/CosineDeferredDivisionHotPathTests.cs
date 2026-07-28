namespace VecNet.Tests;

public sealed class CosineDeferredDivisionHotPathTests
{
    [Theory]
    [InlineData(3, 12, 1)]
    [InlineData(31, 40, 10)]
    [InlineData(32, 40, 10)]
    [InlineData(33, 40, 10)]
    [InlineData(384, 128, 100)]
    public void ExactFlatCosineSearch_MatchesOldFormulaReferenceAcrossDimensionsAndTopK(
        int dimension,
        int rowCount,
        int topK)
    {
        (ulong Id, float[] Vector)[] rows = CreateRows(dimension, rowCount, idBase: 10_000);
        var index = new ExactFlatIndex(dimension, VectorMetric.Cosine);
        foreach ((ulong id, float[] vector) in rows)
        {
            index.Add(id, vector);
        }

        foreach (float[] query in CreateQueries(dimension))
        {
            SearchResult[] expected = OldFormulaTruth(rows, query, topK);
            SearchResult[] actual = Search(index, query, topK);

            AssertResultsEqual(expected, actual, tolerance: 2e-6f);
        }
    }

    [Fact]
    public void ExactFlatCosineSearch_PreservesOldFormulaNearTieOrdering()
    {
        (ulong Id, float[] Vector)[] rows =
        [
            (40, [1f, 0.0020f, 0.1f]),
            (10, [1f, 0.0010f, 0.1f]),
            (30, [1f, 0.0015f, 0.1f]),
            (20, [1f, 0.0012f, 0.1f]),
            (50, [-1f, 0f, 0.1f])
        ];
        float[] query = [1f, 0.0011f, 0.1f];
        var index = new ExactFlatIndex(3, VectorMetric.Cosine);
        foreach ((ulong id, float[] vector) in rows)
        {
            index.Add(id, vector);
        }

        SearchResult[] expected = OldFormulaTruth(rows, query, topK: rows.Length);
        SearchResult[] actual = Search(index, query, topK: rows.Length);

        Assert.Equal(expected.Select(static result => result.Id), actual.Select(static result => result.Id));
        AssertResultsEqual(expected, actual, tolerance: 2e-6f);
    }

    [Fact]
    public void ImmutableHnswCosineSearchAndOpen_MatchOldFormulaReferenceAtHighEfSearch()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        const int dimension = 96;
        (ulong Id, float[] Vector)[] rows = CreateRows(dimension, rowCount: 48, idBase: 20_000);
        var options = new HnswIndexOptions(64, 128, 128, 0x2680UL);
        HnswIndex source = CreateHnsw(rows, options);
        float[] query = CreateQuery(dimension, phase: 7);

        SearchResult[] expected = OldFormulaTruth(rows, query, topK: 10);
        SearchResult[] sourceResults = Search(source, query, topK: 10);
        AssertReturnedResultsAreSortedUniqueAndFinite(sourceResults);
        AssertResultsEqual(expected, sourceResults, tolerance: 2e-6f);

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        SearchResult[] openedResults = Search(opened, query, topK: 10);

        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        AssertResultsEqual(sourceResults, openedResults, tolerance: 0f);
        AssertResultsEqual(expected, openedResults, tolerance: 2e-6f);
    }

    [Fact]
    public void MutableHnswCosineSearchAllowlistCheckpointAndReopen_MatchOldFormulaReference()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(16, 64, 64, 0x2681UL);
        (ulong Id, float[] Vector)[] baseRows =
        [
            (10, [1f, 0f, 0.25f, 0.5f, 0.75f]),
            (20, [0.5f, 1f, 0.25f, 0.5f, 0.75f]),
            (30, [-1f, 0f, 0.25f, 0.5f, 0.75f]),
            (40, [0f, -1f, 0.25f, 0.5f, 0.75f])
        ];
        (ulong Id, float[] Vector)[] deltaRows =
        [
            (15, [1f, 1f, 0.25f, 0.5f, 0.75f]),
            (25, [0f, 2f, 0.25f, 0.5f, 0.75f]),
            (35, [-1f, 1f, 0.25f, 0.5f, 0.75f])
        ];
        var mutable = new HnswMutableIndex(CreateHnsw(baseRows, options));

        foreach ((ulong id, float[] vector) in deltaRows)
        {
            Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(id, vector).Status);
        }

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(35).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [9f, 9f, 9f, 9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(35, [9f, 9f, 9f, 9f, 9f]).Status);

        float[] query = [1f, 0.75f, 0.25f, 0.5f, 0.75f];
        (ulong Id, float[] Vector)[] liveRows =
        [
            baseRows[0],
            baseRows[2],
            baseRows[3],
            deltaRows[0],
            deltaRows[1]
        ];

        SearchResult[] expected = OldFormulaTruth(liveRows, query, topK: 5);
        SearchResult[] actual = Search(mutable, query, topK: 5);
        AssertReturnedResultsAreSortedUniqueAndFinite(actual);
        AssertResultsEqual(expected, actual, tolerance: 2e-6f);

        ulong[] allowlist = [999, 20, 35, 25, 15, 10, 15];
        SearchResult[] expectedAllowed = OldFormulaTruth([baseRows[0], deltaRows[0], deltaRows[1]], query, topK: 3);
        SearchResult[] actualAllowed = Search(mutable, query, allowlist, topK: 3);
        AssertResultsEqual(expectedAllowed, actualAllowed, tolerance: 2e-6f);

        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.DoesNotContain(20UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(35UL, opened.InternalIds.ToArray());
        AssertResultsEqual(expected, Search(mutable, query, topK: 5), tolerance: 2e-6f);
        AssertResultsEqual(expected, Search(opened, query, topK: 5), tolerance: 2e-6f);
    }

    [Fact]
    public void HnswInnerProduct_RemainsUnsupportedForImmutableAndMutableCosinePath()
    {
        Assert.Throws<NotSupportedException>(() => new HnswIndex(3, VectorMetric.InnerProduct));
        Assert.Throws<NotSupportedException>(
            () => new HnswIndex(3, VectorMetric.InnerProduct, HnswIndexOptions.Default));
    }

    private static HnswIndex CreateHnsw(
        (ulong Id, float[] Vector)[] rows,
        HnswIndexOptions options)
    {
        int dimension = rows.Length == 0 ? 1 : rows[0].Vector.Length;
        var index = new HnswIndex(dimension, VectorMetric.Cosine, options, () => 0);
        foreach ((ulong id, float[] vector) in rows)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] OldFormulaTruth(
        (ulong Id, float[] Vector)[] rows,
        float[] query,
        int topK)
    {
        double queryMagnitude = Magnitude(query);
        return rows
            .Select(row => new SearchResult(row.Id, OldFormulaDistance(query, queryMagnitude, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float OldFormulaDistance(float[] query, double queryMagnitude, float[] rawStored)
    {
        float[] normalizedStored = NormalizeAsStored(rawStored);
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += normalizedStored[i] * (query[i] / queryMagnitude);
        }

        return (float)(1 - dotProduct);
    }

    private static float[] NormalizeAsStored(float[] vector)
    {
        double magnitude = Magnitude(vector);
        var normalized = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = (float)(vector[i] / magnitude);
        }

        return normalized;
    }

    private static double Magnitude(float[] vector)
    {
        double squaredMagnitude = 0;
        foreach (float component in vector)
        {
            squaredMagnitude += (double)component * component;
        }

        return Math.Sqrt(squaredMagnitude);
    }

    private static (ulong Id, float[] Vector)[] CreateRows(int dimension, int rowCount, ulong idBase)
    {
        var rows = new (ulong Id, float[] Vector)[rowCount];
        for (int row = 0; row < rowCount; row++)
        {
            rows[row] = (idBase + (ulong)((row * 37) % (rowCount * 3)), CreateVector(dimension, row + 1));
        }

        return rows;
    }

    private static float[][] CreateQueries(int dimension) =>
    [
        CreateQuery(dimension, phase: 3),
        CreateQuery(dimension, phase: 9),
        CreateQuery(dimension, phase: 17)
    ];

    private static float[] CreateQuery(int dimension, int phase) => CreateVector(dimension, phase + 101);

    private static float[] CreateVector(int dimension, int phase)
    {
        var vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            int lane = (((i + 5) * (phase + 7)) % 23) - 11;
            vector[i] = (lane * (1f + ((phase % 5) * 0.125f))) + ((i % 7) * 0.03125f);
        }

        if (vector.All(static value => value == 0f))
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static void AssertResultsEqual(SearchResult[] expected, SearchResult[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float absoluteDelta = Math.Abs(expected[i].Distance - actual[i].Distance);
            Assert.True(
                absoluteDelta <= tolerance,
                $"Distance mismatch at result {i} for ID {expected[i].Id}: expected {expected[i].Distance}, actual {actual[i].Distance}, delta {absoluteDelta}.");
        }
    }

    private static void AssertReturnedResultsAreSortedUniqueAndFinite(SearchResult[] results)
    {
        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        for (int i = 0; i < results.Length; i++)
        {
            Assert.True(float.IsFinite(results[i].Distance), $"Distance for ID {results[i].Id} was not finite.");
            if (i == 0)
            {
                continue;
            }

            Assert.True(
                results[i - 1].Distance < results[i].Distance ||
                (results[i - 1].Distance == results[i].Distance && results[i - 1].Id <= results[i].Id),
                $"Results were not sorted at positions {i - 1} and {i}.");
        }
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = CreatePath();
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public static TempIndexDirectory CreateMissing() => new(CreatePath());

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
            else if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-CosineDeferredDivision-" + Guid.NewGuid().ToString("N"));
    }
}
