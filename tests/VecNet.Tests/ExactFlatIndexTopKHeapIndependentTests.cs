namespace VecNet.Tests;

public sealed class ExactFlatIndexTopKHeapIndependentTests
{
    [Fact]
    public void HeapPath_ReplacesFilledHeapWithLateBetterCandidatesAfterRejectingFarTail()
    {
        var rows = new List<Row>();

        for (int i = 0; i < 10; i++)
        {
            rows.Add(new Row((ulong)(2_000 + i), [20f + i, 0f]));
        }

        for (int i = 0; i < 96; i++)
        {
            rows.Add(new Row((ulong)(50_000 + i), [200f + i, 0f]));
        }

        Row[] lateBetterRows =
        [
            new(910, [3f, 0f]),
            new(410, [1f, 0f]),
            new(300, [-1f, 0f]),
            new(120, [0f, 0f]),
            new(700, [2f, 0f]),
            new(610, [2f, 0f]),
            new(500, [4f, 0f]),
            new(800, [5f, 0f]),
            new(220, [3f, 0f]),
            new(130, [6f, 0f]),
            new(640, [7f, 0f]),
            new(150, [8f, 0f])
        ];
        rows.AddRange(lateBetterRows);

        ExactFlatIndex index = BuildIndex(rows);
        var actual = new SearchResult[10];

        int written = index.Search([0f, 0f], actual);

        SearchResult[] expected = ReferenceTopK(rows, topK: 10);
        Assert.Equal(10, written);
        Assert.Equal(expected, actual);
        Assert.DoesNotContain(actual, result => result.Id >= 50_000);
        Assert.Contains(actual, result => result.Id == 120);
        Assert.Contains(actual, result => result.Id == 130);
    }

    [Fact]
    public void HeapPath_RawAllowlistUnderfillPreservesSentinelsAndOrdersDistanceTiesById()
    {
        Row[] rows =
        [
            new(800, [0f, 1f]),
            new(44, [0f, 2f]),
            new(77, [-1f, 0f]),
            new(11, [1f, 0f]),
            new(99, [9f, 0f]),
            new(500, [0f, -1f])
        ];
        ExactFlatIndex index = BuildIndex(rows);
        ulong[] allowlist = [999_001, 800, 44, 800, 77, 999_002, 11, 500, 500];
        var actual = SentinelResults(length: 12);

        int written = index.Search(
            [0f, 0f],
            allowlist,
            actual,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        SearchResult[] expected = ReferenceTopK(rows.Where(row => allowlist.Contains(row.Id)), topK: 12);
        Assert.Equal(5, written);
        Assert.Equal(expected, actual[..written]);
        Assert.Equal([11UL, 77UL, 500UL, 800UL, 44UL], actual[..written].Select(static result => result.Id));
        AssertUnwrittenSentinels(actual, written);
    }

    [Fact]
    public void HeapPath_CandidateSetUsesCanonicalOrderAfterStorageLateWinners()
    {
        var rows = new List<Row>();
        for (int i = 0; i < 18; i++)
        {
            rows.Add(new Row((ulong)(10_000 + i), [30f + i, 1f]));
        }

        rows.AddRange(
        [
            new Row(900, [0f, 3f]),
            new Row(40, [0f, -1f]),
            new Row(30, [1f, 0f]),
            new Row(20, [-1f, 0f]),
            new Row(10, [0f, 1f]),
            new Row(700, [0f, 2f]),
            new Row(600, [2f, 0f]),
            new Row(500, [-2f, 0f]),
            new Row(400, [3f, 0f]),
            new Row(300, [-3f, 0f])
        ]);

        ExactFlatIndex index = BuildIndex(rows);
        ulong[] requestedIds = rows
            .Select(static row => row.Id)
            .Concat([ulong.MaxValue, 40UL, 30UL, 20UL])
            .Reverse()
            .ToArray();
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(requestedIds);
        var actual = new SearchResult[10];

        int written = index.Search([0f, 0f], candidates, actual);

        SearchResult[] expected = ReferenceTopK(rows, topK: 10);
        Assert.Equal(10, written);
        Assert.Equal(expected, actual);
        Assert.Equal([10UL, 20UL, 30UL, 40UL], actual[..4].Select(static result => result.Id));
    }

    private static ExactFlatIndex BuildIndex(IEnumerable<Row> rows)
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static SearchResult[] ReferenceTopK(IEnumerable<Row> rows, int topK) =>
        rows
            .GroupBy(static row => row.Id)
            .Select(static group => group.First())
            .Select(static row => new SearchResult(row.Id, DistanceSquared(row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static float DistanceSquared(float[] vector)
    {
        float sum = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        return sum;
    }

    private static SearchResult[] SentinelResults(int length)
    {
        var results = new SearchResult[length];
        for (int i = 0; i < results.Length; i++)
        {
            results[i] = new SearchResult(ulong.MaxValue - (ulong)i, -10_000f - i);
        }

        return results;
    }

    private static void AssertUnwrittenSentinels(SearchResult[] results, int written)
    {
        for (int i = written; i < results.Length; i++)
        {
            Assert.Equal(new SearchResult(ulong.MaxValue - (ulong)i, -10_000f - i), results[i]);
        }
    }

    private sealed record Row(ulong Id, float[] Vector);
}
