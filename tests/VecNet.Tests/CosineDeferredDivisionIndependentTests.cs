namespace VecNet.Tests;

public sealed class CosineDeferredDivisionIndependentTests
{
    [Fact]
    public void ExactFlatCosineTop100_MatchesIndependentOldFormulaReference()
    {
        const int dimension = 37;
        Row[] rows = CreateAngularRows(dimension, rowCount: 140, idBase: 70_000);
        var index = new ExactFlatIndex(dimension, VectorMetric.Cosine);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        float[] query = UnitAxisQuery(dimension);
        SearchResult[] expected = OldFormulaReference(rows, query, topK: 100);
        SearchResult[] actual = Search(index, query, topK: 100);

        AssertSameResults(expected, actual, tolerance: 2e-6f);
    }

    [Fact]
    public void ImmutableHnswCosineSearchAllowlistAndOpen_MatchIndependentOldFormulaReference()
    {
        using TempIndexDirectory saved = TempIndexDirectory.Create();
        Row[] rows =
        [
            new(81, [1f, 0f, 0.125f, 0.25f, 0.5f, 0.75f]),
            new(21, [0.9f, 0.3f, 0.125f, 0.25f, 0.5f, 0.75f]),
            new(61, [0.5f, 0.8f, 0.125f, 0.25f, 0.5f, 0.75f]),
            new(41, [0f, 1f, 0.125f, 0.25f, 0.5f, 0.75f]),
            new(101, [-0.4f, 0.8f, 0.125f, 0.25f, 0.5f, 0.75f]),
            new(11, [-1f, 0f, 0.125f, 0.25f, 0.5f, 0.75f])
        ];
        HnswIndex index = CreateCosineHnsw(rows, new HnswIndexOptions(16, 64, 64, 0x2700_0001UL));
        float[] query = [0.95f, 0.35f, 0.125f, 0.25f, 0.5f, 0.75f];

        SearchResult[] expected = OldFormulaReference(rows, query, topK: 4);
        SearchResult[] actual = Search(index, query, topK: 4);
        AssertSortedUniqueFinite(actual);
        AssertSameResults(expected, actual, tolerance: 2e-6f);

        ulong[] allowlist = [999, 41, 11, 21, 21];
        SearchResult[] expectedAllowed = OldFormulaReference(
            rows.Where(row => allowlist.Contains(row.Id)).ToArray(),
            query,
            topK: 3);
        SearchResult[] actualAllowed = Search(index, query, allowlist, topK: 3);
        AssertSameResults(expectedAllowed, actualAllowed, tolerance: 2e-6f);

        index.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);
        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        AssertSameResults(actual, Search(opened, query, topK: 4), tolerance: 0f);
        AssertSameResults(expectedAllowed, Search(opened, query, allowlist, topK: 3), tolerance: 2e-6f);
    }

    [Fact]
    public void MutableHnswCosineChurnAllowlistCheckpointAndReopen_MatchIndependentOldFormulaReference()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        Row[] baseRows =
        [
            new(10, [1f, 0f, 0f, 0.5f]),
            new(20, [0.7f, 0.7f, 0f, 0.5f]),
            new(30, [0f, 1f, 0f, 0.5f]),
            new(40, [-1f, 0f, 0f, 0.5f]),
            new(50, [0f, -1f, 0f, 0.5f])
        ];
        Row[] deltaRows =
        [
            new(15, [1f, 1f, 0.25f, 0.5f]),
            new(25, [0.2f, 1.1f, 0.25f, 0.5f]),
            new(35, [-0.4f, 1f, 0.25f, 0.5f])
        ];
        var mutable = new HnswMutableIndex(
            CreateCosineHnsw(baseRows, new HnswIndexOptions(16, 64, 64, 0x2700_0002UL)));

        foreach (Row row in deltaRows)
        {
            Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(row.Id, row.Vector).Status);
        }

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(35).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [9f, 9f, 9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(35, [9f, 9f, 9f, 9f]).Status);

        float[] query = [0.9f, 0.8f, 0.25f, 0.5f];
        Row[] liveRows = [baseRows[0], baseRows[2], baseRows[3], baseRows[4], deltaRows[0], deltaRows[1]];
        SearchResult[] expected = OldFormulaReference(liveRows, query, topK: 6);
        SearchResult[] actual = Search(mutable, query, topK: 6);
        AssertSortedUniqueFinite(actual);
        AssertSameResults(expected, actual, tolerance: 2e-6f);
        Assert.DoesNotContain(actual, static result => result.Id is 20 or 35);

        ulong[] allowlist = [35, 20, 25, 15, 10, 10, 999];
        SearchResult[] expectedAllowed = OldFormulaReference([baseRows[0], deltaRows[0], deltaRows[1]], query, topK: 3);
        AssertSameResults(expectedAllowed, Search(mutable, query, allowlist, topK: 3), tolerance: 2e-6f);

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaTombstoneCount);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.DoesNotContain(20UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(35UL, opened.InternalIds.ToArray());
        AssertSameResults(expected, Search(mutable, query, topK: 6), tolerance: 2e-6f);
        AssertSameResults(expected, Search(opened, query, topK: 6), tolerance: 2e-6f);
        AssertSameResults(expectedAllowed, Search(opened, query, allowlist, topK: 3), tolerance: 2e-6f);
    }

    [Fact]
    public void CosineZeroVectors_AreRejectedAcrossExactImmutableMutableAndOpenedSearch()
    {
        var exact = new ExactFlatIndex(3, VectorMetric.Cosine);
        Assert.Throws<ArgumentException>(() => exact.Add(1, [0f, 0f, 0f]));
        exact.Add(1, [1f, 0f, 0f]);
        Assert.Throws<ArgumentException>(() => exact.Search([0f, 0f, 0f], Span<SearchResult>.Empty));
        Assert.Throws<ArgumentException>(() => exact.Search(
            [0f, 0f, 0f],
            [1UL],
            Span<SearchResult>.Empty,
            exact.CreateSearchFilterWorkspace()));

        using TempIndexDirectory saved = TempIndexDirectory.Create();
        var hnsw = new HnswIndex(3, VectorMetric.Cosine, new HnswIndexOptions(4, 16, 16, 0x2700_0003UL));
        Assert.Throws<ArgumentException>(() => hnsw.Add(1, [0f, 0f, 0f]));
        hnsw.Add(1, [1f, 0f, 0f]);
        Assert.Throws<ArgumentException>(() => hnsw.Search([0f, 0f, 0f], Span<SearchResult>.Empty, hnsw.CreateSearchWorkspace()));
        Assert.Throws<ArgumentException>(() => hnsw.Search(
            [0f, 0f, 0f],
            [1UL],
            Span<SearchResult>.Empty,
            hnsw.CreateSearchWorkspace()));

        hnsw.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);
        Assert.Throws<ArgumentException>(() => opened.Search([0f, 0f, 0f], Span<SearchResult>.Empty, opened.CreateSearchWorkspace()));

        var mutable = new HnswMutableIndex(opened);
        Assert.Throws<ArgumentException>(() => mutable.TryAdd(2, [0f, 0f, 0f]));
        Assert.Throws<ArgumentException>(() => mutable.Search(
            [0f, 0f, 0f],
            Span<SearchResult>.Empty,
            new HnswMutableSearchWorkspace(mutable, maxResults: 0)));
    }

    [Fact]
    public void HnswInnerProductConstructors_AreAdmitted()
    {
        Assert.Equal(VectorMetric.InnerProduct, new HnswIndex(4, VectorMetric.InnerProduct).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            new HnswIndex(4, VectorMetric.InnerProduct, new HnswIndexOptions(4, 16, 16, 0x2700_0004UL)).Metric);
    }

    private static Row[] CreateAngularRows(int dimension, int rowCount, ulong idBase)
    {
        var rows = new Row[rowCount];
        for (int i = 0; i < rows.Length; i++)
        {
            double angle = i * 0.011;
            float scale = 0.5f + (i % 9);
            float[] vector = new float[dimension];
            vector[0] = (float)(Math.Cos(angle) * scale);
            vector[1] = (float)(Math.Sin(angle) * scale);
            for (int j = 2; j < dimension; j++)
            {
                vector[j] = scale * ((((i + 3) * (j + 5)) % 13) - 6) * 0.0001f;
            }

            rows[i] = new Row(idBase + (ulong)((i * 7_919) % 100_003), vector);
        }

        return rows;
    }

    private static float[] UnitAxisQuery(int dimension)
    {
        var query = new float[dimension];
        query[0] = 1f;
        for (int i = 2; i < dimension; i++)
        {
            query[i] = ((i % 5) - 2) * 0.0002f;
        }

        return query;
    }

    private static HnswIndex CreateCosineHnsw(Row[] rows, HnswIndexOptions options)
    {
        var index = new HnswIndex(rows[0].Vector.Length, VectorMetric.Cosine, options, () => 0);
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

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace());
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

    private static SearchResult[] OldFormulaReference(Row[] rows, float[] query, int topK)
    {
        double queryMagnitude = Magnitude(query);
        return rows
            .Select(row => new SearchResult(row.Id, OldFormulaDistance(query, queryMagnitude, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float OldFormulaDistance(float[] query, double queryMagnitude, float[] stored)
    {
        double storedMagnitude = Magnitude(stored);
        double dot = 0;
        for (int i = 0; i < query.Length; i++)
        {
            float storedComponentAsPersisted = (float)(stored[i] / storedMagnitude);
            dot += storedComponentAsPersisted * (query[i] / queryMagnitude);
        }

        return (float)(1 - dot);
    }

    private static double Magnitude(float[] vector)
    {
        double sum = 0;
        foreach (float component in vector)
        {
            sum += (double)component * component;
        }

        return Math.Sqrt(sum);
    }

    private static void AssertSameResults(SearchResult[] expected, SearchResult[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float delta = Math.Abs(expected[i].Distance - actual[i].Distance);
            Assert.True(
                delta <= tolerance,
                $"Distance mismatch at rank {i}: ID {expected[i].Id}, expected {expected[i].Distance}, actual {actual[i].Distance}, delta {delta}.");
        }
    }

    private static void AssertSortedUniqueFinite(SearchResult[] results)
    {
        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        for (int i = 0; i < results.Length; i++)
        {
            Assert.True(float.IsFinite(results[i].Distance), $"Distance for ID {results[i].Id} was not finite.");
            if (i > 0)
            {
                Assert.True(
                    results[i - 1].Distance < results[i].Distance ||
                    (results[i - 1].Distance == results[i].Distance && results[i - 1].Id <= results[i].Id),
                    $"Results were not sorted at ranks {i - 1} and {i}.");
            }
        }
    }

    private sealed record Row(ulong Id, float[] Vector);

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
                "VecNet-CosineDeferredDivisionIndependent-" + Guid.NewGuid().ToString("N"));
    }
}
