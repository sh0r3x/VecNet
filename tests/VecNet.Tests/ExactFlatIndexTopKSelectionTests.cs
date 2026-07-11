namespace VecNet.Tests;

public sealed class ExactFlatIndexTopKSelectionTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(256)]
    public void Search_UsesCanonicalOrderingForDuplicateDistancesAcrossTopKThresholds(int topK)
    {
        Row[] rows = CreateDuplicateDistanceRows(rowCount: 384);
        var index = CreateIndex(rows);
        var results = CreateSentinelResults(topK);

        int written = index.Search([0f], results);

        SearchResult[] expected = Expected(rows, topK);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void Search_PreservesExactTieOrderingAtHeapThreshold()
    {
        var rows = new Row[64];
        for (int i = 0; i < rows.Length; i++)
        {
            float value = (i & 1) == 0 ? 1f : -1f;
            rows[i] = new Row((ulong)(10_000 - (i * 97)), value);
        }

        var index = CreateIndex(rows);
        var results = new SearchResult[10];

        int written = index.Search([0f], results);

        Assert.Equal(10, written);
        Assert.Equal(
            rows.Select(static row => row.Id).Order().Take(10),
            results.Select(static result => result.Id));
        Assert.All(results, static result => Assert.Equal(1f, result.Distance));
    }

    [Fact]
    public void SearchSurfaces_ReturnSameCanonicalHeapResultsForUnfilteredAllowlistAndCandidateSet()
    {
        Row[] rows = CreateDuplicateDistanceRows(rowCount: 256);
        var index = CreateIndex(rows);
        ulong[] allowedIds = rows
            .Where(static row => row.Id % 3 != 0)
            .Select(static row => row.Id)
            .Reverse()
            .Concat(rows.Where(static row => row.Id % 17 == 0).Select(static row => row.Id))
            .Concat([ulong.MaxValue, ulong.MaxValue - 1])
            .ToArray();
        SearchResult[] expected = Expected(rows.Where(row => allowedIds.Contains(row.Id)), topK: 100);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowedIds);
        var rawResults = CreateSentinelResults(100);
        var candidateResults = CreateSentinelResults(100);
        var allCandidateResults = CreateSentinelResults(100);
        ExactFlatCandidateSet allCandidates = index.CreateCandidateSet(rows.Select(static row => row.Id).ToArray());

        int rawWritten = index.Search([0f], allowedIds, rawResults, workspace);
        int candidateWritten = index.Search([0f], candidates, candidateResults);
        int allCandidateWritten = index.Search([0f], allCandidates, allCandidateResults);

        Assert.Equal(expected.Length, rawWritten);
        Assert.Equal(expected, rawResults[..rawWritten]);
        Assert.Equal(expected.Length, candidateWritten);
        Assert.Equal(expected, candidateResults[..candidateWritten]);
        Assert.Equal(Expected(rows, 100), allCandidateResults[..allCandidateWritten]);
    }

    [Fact]
    public void HeapSelection_PreservesUnderfilledBuffersAndUnwrittenSentinels()
    {
        Row[] rows =
        [
            new(30, 3f),
            new(10, 1f),
            new(20, -1f)
        ];
        var index = CreateIndex(rows);
        ulong[] allowedIds = [20, 999, 10, 20];
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowedIds);
        var unfiltered = CreateSentinelResults(10);
        var raw = CreateSentinelResults(10);
        var candidate = CreateSentinelResults(10);

        int unfilteredWritten = index.Search([0f], unfiltered);
        int rawWritten = index.Search([0f], allowedIds, raw, new ExactFlatSearchFilterWorkspace(index.VectorCount));
        int candidateWritten = index.Search([0f], candidates, candidate);

        Assert.Equal(Expected(rows, 10), unfiltered[..unfilteredWritten]);
        Assert.Equal(Expected(rows.Where(static row => row.Id is 10 or 20), 10), raw[..rawWritten]);
        Assert.Equal(raw[..rawWritten], candidate[..candidateWritten]);
        AssertSentinelsPreserved(unfiltered, unfilteredWritten);
        AssertSentinelsPreserved(raw, rawWritten);
        AssertSentinelsPreserved(candidate, candidateWritten);
    }

    [Fact]
    public void HeapSelection_PreservesMutationAndTombstoneVisibilityAcrossExactSurfaces()
    {
        Row[] baseRows = CreateDuplicateDistanceRows(rowCount: 48);
        var index = CreateIndex(baseRows);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(baseRows[0].Id).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(baseRows[7].Id).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(42, [0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(84, [-0.25f]).Status);

        Row[] liveRows = baseRows
            .Where(row => row.Id != baseRows[0].Id && row.Id != baseRows[7].Id)
            .Concat([new Row(42, 0.25f), new Row(84, -0.25f)])
            .ToArray();
        ulong[] allowedIds = liveRows.Select(static row => row.Id).Reverse().Concat([baseRows[0].Id, baseRows[7].Id]).ToArray();
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowedIds);
        var unfiltered = new SearchResult[10];
        var raw = new SearchResult[10];
        var candidate = new SearchResult[10];

        int unfilteredWritten = index.Search([0f], unfiltered);
        int rawWritten = index.Search([0f], allowedIds, raw, new ExactFlatSearchFilterWorkspace(index.VectorCount));
        int candidateWritten = index.Search([0f], candidates, candidate);

        SearchResult[] expected = Expected(liveRows, 10);
        Assert.Equal(expected, unfiltered[..unfilteredWritten]);
        Assert.Equal(expected, raw[..rawWritten]);
        Assert.Equal(expected, candidate[..candidateWritten]);
        Assert.DoesNotContain(unfiltered[..unfilteredWritten], result => result.Id == baseRows[0].Id || result.Id == baseRows[7].Id);
    }

    [Fact]
    public void Search_DoesNotAllocateWhenHeapResultBufferIsReusedAfterWarmup()
    {
        Row[] rows = CreateDuplicateDistanceRows(rowCount: 256);
        var index = CreateIndex(rows);
        var results = new SearchResult[100];

        Assert.Equal(100, index.Search([0f], results));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 128; i++)
        {
            int written = index.Search([0f], results);
            if (written != 100)
            {
                throw new InvalidOperationException("Unexpected exact search result count during allocation measurement.");
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    private static ExactFlatIndex CreateIndex(IEnumerable<Row> rows)
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        foreach (Row row in rows)
        {
            index.Add(row.Id, [row.Value]);
        }

        return index;
    }

    private static Row[] CreateDuplicateDistanceRows(int rowCount)
    {
        var rows = new Row[rowCount];
        for (int i = 0; i < rows.Length; i++)
        {
            int bucket = i % 37;
            float value = (i & 1) == 0 ? bucket : -bucket;
            ulong id = (ulong)(1_000_000 - (i * 7_919 % 999_983));
            rows[i] = new Row(id, value);
        }

        return rows;
    }

    private static SearchResult[] Expected(IEnumerable<Row> rows, int topK) =>
        rows
            .Select(static row => new SearchResult(row.Id, row.Value * row.Value))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static SearchResult[] CreateSentinelResults(int length)
    {
        var results = new SearchResult[length];
        for (int i = 0; i < results.Length; i++)
        {
            results[i] = new SearchResult(ulong.MaxValue - (ulong)i, -12345f - i);
        }

        return results;
    }

    private static void AssertSentinelsPreserved(SearchResult[] results, int written)
    {
        for (int i = written; i < results.Length; i++)
        {
            Assert.Equal(new SearchResult(ulong.MaxValue - (ulong)i, -12345f - i), results[i]);
        }
    }

    private readonly record struct Row(ulong Id, float Value);
}
