namespace VecNet.Tests;

public sealed class HnswIndexConcurrentSearchTests
{
    private const int Count = 384;
    private const int Dimension = 17;
    private const int ParallelIterations = 768;

    [Fact]
    public void Search_OnBuiltIndexMatchesSingleThreadBaselinesUnderReadOnlyConcurrency()
    {
        HnswIndex index = CreateDeterministicIndex();
        float[][] queries = CreateQueries();
        Dictionary<(int QueryIndex, int TopK), SearchResult[]> expected = CreateBaselines(index, queries);

        Assert.True(index.MaxLayer >= 3);

        AssertConcurrentSearchMatchesBaselines(index, queries, expected);
        AssertConcurrentSearchMatchesBaselines(index, queries, expected);
    }

    [Fact]
    public void Search_OnOpenedReadOnlySnapshotMatchesSingleThreadBaselinesUnderReadOnlyConcurrency()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex source = CreateDeterministicIndex();
        float[][] queries = CreateQueries();

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        Dictionary<(int QueryIndex, int TopK), SearchResult[]> expected = CreateBaselines(opened, queries);

        Assert.Equal(source.Count, opened.Count);
        Assert.Equal(source.Options, opened.Options);
        Assert.True(opened.MaxLayer >= 3);

        AssertConcurrentSearchMatchesBaselines(opened, queries, expected);
        AssertConcurrentSearchMatchesBaselines(opened, queries, expected);
    }

    private static void AssertConcurrentSearchMatchesBaselines(
        HnswIndex index,
        float[][] queries,
        IReadOnlyDictionary<(int QueryIndex, int TopK), SearchResult[]> expected)
    {
        int[] topKValues = [3, 17];
        Parallel.For(0, ParallelIterations, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            int topK = topKValues[(iteration / queries.Length) % topKValues.Length];
            SearchResult[] baseline = expected[(queryIndex, topK)];
            float[] query = queries[queryIndex].ToArray();
            var results = Enumerable.Repeat(new SearchResult(ulong.MaxValue, -1f), topK + 2).ToArray();
            var workspace = new HnswSearchWorkspace(index.Count, index.Options.EfSearch);

            int written = index.Search(query, results.AsSpan(0, topK), workspace);

            Assert.Equal(baseline.Length, written);
            Assert.Equal(baseline, results.AsSpan(0, written).ToArray());
            Assert.Equal(new SearchResult(ulong.MaxValue, -1f), results[topK]);
            Assert.Equal(new SearchResult(ulong.MaxValue, -1f), results[topK + 1]);
            Assert.True(workspace.CurrentVisitMark > 0);
        });
    }

    private static Dictionary<(int QueryIndex, int TopK), SearchResult[]> CreateBaselines(
        HnswIndex index,
        float[][] queries)
    {
        var expected = new Dictionary<(int QueryIndex, int TopK), SearchResult[]>();
        foreach (int topK in new[] { 3, 17 })
        {
            for (int queryIndex = 0; queryIndex < queries.Length; queryIndex++)
            {
                SearchResult[] first = Search(index, queries[queryIndex], topK);
                SearchResult[] second = Search(index, queries[queryIndex], topK);

                Assert.Equal(topK, first.Length);
                Assert.Equal(first, second);
                AssertSorted(first);
                expected.Add((queryIndex, topK), first);
            }
        }

        return expected;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static HnswIndex CreateDeterministicIndex()
    {
        int nextOrdinal = 0;
        var options = new HnswIndexOptions(8, 96, 96, 0x7111_0001UL);
        var index = new HnswIndex(
            Dimension,
            VectorMetric.SquaredEuclidean,
            options,
            () => DeterministicLevel(nextOrdinal++));

        for (int ordinal = 0; ordinal < Count; ordinal++)
        {
            index.Add((ulong)(50_000 + ordinal * 37), CreateVector(ordinal));
        }

        Assert.Equal(Count, index.Count);
        return index;
    }

    private static int DeterministicLevel(int ordinal)
    {
        if (ordinal == 0 || ordinal % 97 == 0)
        {
            return 3;
        }

        if (ordinal % 31 == 0)
        {
            return 2;
        }

        return ordinal % 7 == 0 ? 1 : 0;
    }

    private static float[][] CreateQueries() =>
    [
        CreateQuery(0, 0x7111_1000),
        CreateQuery(3, 0x7111_1001),
        CreateQuery(8, 0x7111_1002),
        CreateQuery(13, 0x7111_1003),
        CreateQuery(21, 0x7111_1004),
        CreateQuery(34, 0x7111_1005),
        CreateQuery(55, 0x7111_1006),
        CreateQuery(89, 0x7111_1007),
        CreateQuery(144, 0x7111_1008),
        CreateQuery(233, 0x7111_1009),
        CreateQuery(377, 0x7111_1010),
        CreateQuery(610, 0x7111_1011)
    ];

    private static float[] CreateQuery(int anchor, int seed)
    {
        float[] vector = CreateVector(anchor % Count);
        var random = new Random(seed);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] += ((random.NextSingle() - 0.5f) * 0.35f) + ((i % 4) - 1.5f) * 0.05f;
        }

        return vector;
    }

    private static float[] CreateVector(int ordinal)
    {
        var vector = new float[Dimension];
        int cluster = ordinal % 16;
        int band = ordinal / 16;
        for (int i = 0; i < vector.Length; i++)
        {
            int lane = ((ordinal * 13) + (i * 17)) % 29;
            float clusterCenter = (cluster - 7.5f) * 5.75f;
            float bandOffset = (band % 11) * 0.41f;
            vector[i] = clusterCenter + bandOffset + (lane - 14) * 0.19f + ((i % 5) - 2) * 0.07f;
        }

        return vector;
    }

    private static void AssertSorted(SearchResult[] results)
    {
        for (int i = 1; i < results.Length; i++)
        {
            SearchResult previous = results[i - 1];
            SearchResult current = results[i];
            Assert.True(
                previous.Distance < current.Distance ||
                (previous.Distance == current.Distance && previous.Id <= current.Id));
        }
    }

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
