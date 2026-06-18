namespace VecNet.Tests;

public sealed class ExactFlatIndexMutationTests
{
    [Fact]
    public void TryAdd_CommitsNewIdAndReturnsNewGeneration()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        long before = index.Generation;

        VectorMutationResult result = index.TryAdd(20, [0f, 1f]);

        Assert.Equal(VectorMutationStatus.Committed, result.Status);
        Assert.Equal(before + 1, result.Generation);
        Assert.Equal(result.Generation, index.Generation);
        Assert.Equal(2, result.VectorCount);
        Assert.Equal(1, result.DeltaCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal([10UL, 20UL], SearchIds(index, [0f, 0f], topK: 4));
    }

    [Fact]
    public void Add_StillThrowsForDuplicateInvalidReservedAndReadOnlyInserts()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);

        Assert.Throws<ArgumentException>(() => index.Add(10, [2f, 0f]));
        Assert.Throws<ArgumentException>(() => index.Add(20, [1f]));
        Assert.Throws<ArgumentException>(() => index.Add(20, [float.NaN, 0f]));

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        Assert.Throws<ArgumentException>(() => index.Add(10, [3f, 0f]));

        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);
        Assert.Throws<InvalidOperationException>(() => opened.Add(20, [2f, 0f]));
    }

    [Fact]
    public void DuplicateAndReservedTryAddDoNotAdvanceGeneration()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        long beforeDuplicate = index.Generation;

        VectorMutationResult duplicate = index.TryAdd(10, [2f, 0f]);

        Assert.Equal(VectorMutationStatus.DuplicateId, duplicate.Status);
        Assert.Equal(beforeDuplicate, duplicate.Generation);
        Assert.Equal(beforeDuplicate, index.Generation);
        Assert.Equal(1, duplicate.VectorCount);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        long beforeReserved = index.Generation;

        VectorMutationResult reserved = index.TryAdd(10, [3f, 0f]);

        Assert.Equal(VectorMutationStatus.DuplicateId, reserved.Status);
        Assert.Equal(beforeReserved, reserved.Generation);
        Assert.Equal(beforeReserved, index.Generation);
        Assert.Equal(0, reserved.VectorCount);
        Assert.Equal(1, reserved.TombstoneCount);
    }

    [Fact]
    public void InvalidTryAddLeavesStateUnchanged()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        long generation = index.Generation;

        Assert.Throws<ArgumentException>(() => index.TryAdd(20, [1f]));
        Assert.Throws<ArgumentException>(() => index.TryAdd(20, [float.PositiveInfinity, 0f]));

        Assert.Equal(generation, index.Generation);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [0f, 1f]).Status);
        Assert.Equal([10UL, 20UL], SearchIds(index, [0f, 0f], topK: 4));
    }

    [Fact]
    public void TryDelete_TombstonesVisibleBaseIdAndAdvancesGeneration()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        index.Add(20, [2f]);
        long before = index.Generation;

        VectorMutationResult result = index.TryDelete(10);

        Assert.Equal(VectorMutationStatus.Committed, result.Status);
        Assert.Equal(before + 1, result.Generation);
        Assert.Equal(1, result.VectorCount);
        Assert.Equal(0, result.DeltaCount);
        Assert.Equal(1, result.TombstoneCount);
        Assert.Equal([20UL], SearchIds(index, [0f], topK: 4));
    }

    [Fact]
    public void TryDelete_TombstonesVisibleDeltaIdAndAdvancesGeneration()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [0.5f]).Status);
        long before = index.Generation;

        VectorMutationResult result = index.TryDelete(20);

        Assert.Equal(VectorMutationStatus.Committed, result.Status);
        Assert.Equal(before + 1, result.Generation);
        Assert.Equal(1, result.VectorCount);
        Assert.Equal(0, result.DeltaCount);
        Assert.Equal(1, result.TombstoneCount);
        Assert.Equal([10UL], SearchIds(index, [0f], topK: 4));
    }

    [Fact]
    public void UnknownAndRepeatedDeleteDoNotAdvanceGeneration()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f]);
        long beforeUnknown = index.Generation;

        VectorMutationResult unknown = index.TryDelete(999);

        Assert.Equal(VectorMutationStatus.UnknownId, unknown.Status);
        Assert.Equal(beforeUnknown, unknown.Generation);
        Assert.Equal(beforeUnknown, index.Generation);
        Assert.Equal(0, unknown.TombstoneCount);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        long beforeRepeated = index.Generation;

        VectorMutationResult repeated = index.TryDelete(10);

        Assert.Equal(VectorMutationStatus.AlreadyDeleted, repeated.Status);
        Assert.Equal(beforeRepeated, repeated.Generation);
        Assert.Equal(beforeRepeated, index.Generation);
        Assert.Equal(1, repeated.TombstoneCount);
    }

    [Fact]
    public void Search_GloballyMergesLiveRowsAndExcludesTombstones()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(100, [100f]);
        index.Add(10, [10f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(1, [1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(5, [5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);

        var results = new SearchResult[4];
        int written = index.Search([0f], results);

        Assert.Equal(3, written);
        Assert.Equal([1UL, 5UL, 100UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void RawAllowlistSearchIncludesLiveDeltaCoalescesDuplicatesAndExcludesTombstones()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(30, [30f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var results = new SearchResult[5];

        int written = index.Search([0f], [999, 10, 20, 20, 30, 888, 10], results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([20UL, 30UL], results[..written].Select(static result => result.Id));
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void CandidateSetCreationAfterMutationsUsesCurrentLiveGeneration(VectorMetric metric)
    {
        (ExactFlatIndex Index, List<Row> Rows, float[] Query) fixture = CreateMetricFixture(metric);
        Assert.Equal(VectorMutationStatus.Committed, fixture.Index.TryAdd(40, CreateVector(metric, 40)).Status);
        fixture.Rows.Add(new Row(40, CreateVector(metric, 40)));
        Assert.Equal(VectorMutationStatus.Committed, fixture.Index.TryDelete(20).Status);
        fixture.Rows.RemoveAll(static row => row.Id == 20);

        ExactFlatCandidateSet candidates = fixture.Index.CreateCandidateSet([20, 40, 40, 999, 10]);
        var actual = new SearchResult[4];
        int written = fixture.Index.Search(fixture.Query, candidates, actual);
        SearchResult[] expected = BruteForce(fixture.Rows, fixture.Query, metric, [40, 40, 999, 10], topK: 4);

        Assert.Equal(2, candidates.Count);
        AssertResultsEqual(expected, actual.AsSpan(0, written), metric);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void CandidateSetSearchMatchesBruteForceTruthForCurrentGeneration(VectorMetric metric)
    {
        (ExactFlatIndex Index, List<Row> Rows, float[] Query) fixture = CreateMetricFixture(metric);
        Assert.Equal(VectorMutationStatus.Committed, fixture.Index.TryAdd(40, CreateVector(metric, 40)).Status);
        fixture.Rows.Add(new Row(40, CreateVector(metric, 40)));
        Assert.Equal(VectorMutationStatus.Committed, fixture.Index.TryDelete(10).Status);
        fixture.Rows.RemoveAll(static row => row.Id == 10);

        ulong[] scope = [40, 30, 30, 10, 999, 20];
        ExactFlatCandidateSet candidates = fixture.Index.CreateCandidateSet(scope);
        var actual = new SearchResult[10];
        int written = fixture.Index.Search(fixture.Query, candidates, actual);
        SearchResult[] expected = BruteForce(fixture.Rows, fixture.Query, metric, scope, topK: 10);

        AssertResultsEqual(expected, actual.AsSpan(0, written), metric);
    }

    [Fact]
    public void CandidateSetsCreatedBeforeCommittedInsertOrDeleteFailStaleBeforeWritingResults()
    {
        var insertIndex = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        insertIndex.Add(10, [1f]);
        ExactFlatCandidateSet insertCandidates = insertIndex.CreateCandidateSet([10]);
        var insertDestination = new[] { new SearchResult(123, 456) };

        Assert.Equal(VectorMutationStatus.Committed, insertIndex.TryAdd(20, [2f]).Status);
        Assert.Throws<InvalidOperationException>(() => insertIndex.Search([0f], insertCandidates, insertDestination));
        Assert.Equal(new SearchResult(123, 456), insertDestination[0]);

        var deleteIndex = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        deleteIndex.Add(10, [1f]);
        deleteIndex.Add(20, [2f]);
        ExactFlatCandidateSet deleteCandidates = deleteIndex.CreateCandidateSet([10, 20]);
        var deleteDestination = new[] { new SearchResult(789, 101) };

        Assert.Equal(VectorMutationStatus.Committed, deleteIndex.TryDelete(10).Status);
        Assert.Throws<InvalidOperationException>(() => deleteIndex.Search([0f], deleteCandidates, deleteDestination));
        Assert.Equal(new SearchResult(789, 101), deleteDestination[0]);
    }

    [Fact]
    public void OpenedReadOnlyIndexesRejectMutationApisAndPreserveFilteringBehavior()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [1f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        index.Save(temp.Path);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Throws<InvalidOperationException>(() => opened.Add(40, [0f]));
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryAdd(40, [0f]).Status);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryDelete(20).Status);

        var raw = new SearchResult[4];
        int rawWritten = opened.Search(
            [0f],
            [10, 20, 30, 30, 999],
            raw,
            new ExactFlatSearchFilterWorkspace(opened.VectorCount));
        Assert.Equal([30UL, 20UL], raw[..rawWritten].Select(static result => result.Id));

        ExactFlatCandidateSet candidates = opened.CreateCandidateSet([10, 20, 30, 999, 30]);
        var candidateResults = new SearchResult[4];
        int candidateWritten = opened.Search([0f], candidates, candidateResults);
        Assert.Equal(2, candidates.Count);
        Assert.Equal(raw[..rawWritten], candidateResults[..candidateWritten]);
    }

    [Fact]
    public void EmptyAllDeletedZeroKHighKAndEqualDistanceTiesRemainDeterministic()
    {
        var empty = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        Assert.Equal(0, empty.Search([0f], new SearchResult[5]));

        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(30, [-1f]);
        index.Add(10, [1f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [-1f]).Status);

        Assert.Equal(0, index.Search([0f], []));

        var highK = new SearchResult[8];
        int highKWritten = index.Search([0f], highK);
        Assert.Equal(3, highKWritten);
        Assert.Equal([10UL, 20UL, 30UL], highK[..highKWritten].Select(static result => result.Id));

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(10).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(30).Status);

        var allDeleted = new SearchResult[8];
        Assert.Equal(0, index.Search([0f], allDeleted));
        Assert.Equal(0, index.CreateCandidateSet([10, 20, 30]).Count);
        Assert.Equal(0, index.Search([0f], [10, 20, 30], allDeleted, new ExactFlatSearchFilterWorkspace(index.VectorCount)));
    }

    private static ulong[] SearchIds(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written].Select(static result => result.Id).ToArray();
    }

    private static (ExactFlatIndex Index, List<Row> Rows, float[] Query) CreateMetricFixture(VectorMetric metric)
    {
        var index = new ExactFlatIndex(2, metric);
        var rows = new List<Row>
        {
            new(10, CreateVector(metric, 10)),
            new(20, CreateVector(metric, 20)),
            new(30, CreateVector(metric, 30))
        };

        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return (index, rows, metric == VectorMetric.Cosine ? [1f, 0.5f] : [0f, 0f]);
    }

    private static float[] CreateVector(VectorMetric metric, ulong id) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => id switch
            {
                10 => [1f, 0f],
                20 => [0f, 2f],
                30 => [3f, 0f],
                40 => [0f, 0.5f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            VectorMetric.InnerProduct => id switch
            {
                10 => [1f, 0f],
                20 => [0f, 2f],
                30 => [3f, 1f],
                40 => [2f, 3f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            VectorMetric.Cosine => id switch
            {
                10 => [1f, 0f],
                20 => [0f, 1f],
                30 => [1f, 1f],
                40 => [1f, 2f],
                _ => throw new ArgumentOutOfRangeException(nameof(id))
            },
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static SearchResult[] BruteForce(
        IEnumerable<Row> rows,
        float[] query,
        VectorMetric metric,
        IEnumerable<ulong> scope,
        int topK)
    {
        HashSet<ulong> allowed = scope.ToHashSet();
        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, Distance(query, row.Vector, metric)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float Distance(float[] query, float[] vector, VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
            VectorMetric.InnerProduct => InnerProduct(query, vector),
            VectorMetric.Cosine => Cosine(query, vector),
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float SquaredEuclidean(float[] query, float[] vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static float InnerProduct(float[] query, float[] vector)
    {
        double dot = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
        }

        return (float)-dot;
    }

    private static float Cosine(float[] query, float[] vector)
    {
        double dot = 0;
        double queryMagnitudeSquared = 0;
        double vectorMagnitudeSquared = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
            queryMagnitudeSquared += (double)query[i] * query[i];
            vectorMagnitudeSquared += (double)vector[i] * vector[i];
        }

        return (float)(1 - dot / (Math.Sqrt(queryMagnitudeSquared) * Math.Sqrt(vectorMagnitudeSquared)));
    }

    private static void AssertResultsEqual(SearchResult[] expected, ReadOnlySpan<SearchResult> actual, VectorMetric metric)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float tolerance = metric == VectorMetric.SquaredEuclidean ? 2e-4f : 2e-5f;
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, tolerance);
        }
    }

    private sealed record Row(ulong Id, float[] Vector);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
