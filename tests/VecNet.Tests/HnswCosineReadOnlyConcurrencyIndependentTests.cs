namespace VecNet.Tests;

public sealed class HnswCosineReadOnlyConcurrencyIndependentTests
{
    private const int Count = 512;
    private const int Dimension = 23;
    private const int ParallelIterations = 1_024;
    private const int ConcurrentWaveCount = 4;

    [Fact]
    public void OpenedReadOnlyCosineUnfilteredSearch_MatchesSequentialBaselineUnderConcurrencyAndForcedGc()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateDeterministicCosineHnsw();
        source.Save(saved.Path);

        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);
        float[][] queries = CreateQueries();
        int[] topKValues = [1, 7, 19];
        Dictionary<SearchCase, SearchResult[]> baselines = CreateBaselines(opened, queries, topKValues);

        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.Equal(source.Count, opened.Count);
        Assert.True(opened.MaxLayer >= 3);

        AssertConcurrentSearchMatchesBaselines(
            opened,
            queries,
            topKValues,
            allowlists: null,
            baselines);
    }

    [Fact]
    public void OpenedReadOnlyCosineAllowlistSearch_MatchesSequentialBaselineUnderConcurrencyAndForcedGc()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateDeterministicCosineHnsw();
        source.Save(saved.Path);

        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);
        float[][] queries = CreateQueries();
        ulong[][] allowlists = CreateAllowlists();
        int[] topKValues = [3, 11, 23];
        Dictionary<SearchCase, SearchResult[]> baselines = CreateBaselines(opened, queries, topKValues, allowlists);

        Assert.Equal(VectorMetric.Cosine, opened.Metric);
        Assert.Equal(source.Count, opened.Count);
        Assert.All(allowlists, allowlist => Assert.True(allowlist.Distinct().Count() > opened.Options.EfSearch));

        AssertConcurrentSearchMatchesBaselines(
            opened,
            queries,
            topKValues,
            allowlists,
            baselines);
    }

    private static void AssertConcurrentSearchMatchesBaselines(
        HnswIndex index,
        float[][] queries,
        int[] topKValues,
        ulong[][]? allowlists,
        IReadOnlyDictionary<SearchCase, SearchResult[]> baselines)
    {
        for (int wave = 0; wave < ConcurrentWaveCount; wave++)
        {
            Parallel.For(0, ParallelIterations, iteration =>
            {
                int queryIndex = iteration % queries.Length;
                int topK = topKValues[(iteration / queries.Length) % topKValues.Length];
                int allowlistIndex = allowlists is null
                    ? SearchCase.NoAllowlist
                    : (iteration / (queries.Length * topKValues.Length)) % allowlists.Length;
                SearchCase searchCase = new(queryIndex, topK, allowlistIndex);
                SearchResult[] baseline = baselines[searchCase];
                SearchResult[] destination = Enumerable.Repeat(new SearchResult(ulong.MaxValue, float.NaN), topK + 2).ToArray();
                HnswSearchWorkspace workspace = index.CreateSearchWorkspace();
                float[] query = queries[queryIndex].ToArray();

                int written;
                if (allowlists is null)
                {
                    written = index.Search(query, destination.AsSpan(0, topK), workspace);
                }
                else
                {
                    ulong[] allowlist = allowlists[allowlistIndex].ToArray();
                    written = index.Search(query, allowlist, destination.AsSpan(0, topK), workspace);
                }

                SearchResult[] actual = destination.AsSpan(0, written).ToArray();

                Assert.Equal(baseline.Length, written);
                AssertResultIntegrity(actual);
                Assert.Equal(baseline, actual);
                Assert.Equal(new SearchResult(ulong.MaxValue, float.NaN), destination[topK]);
                Assert.Equal(new SearchResult(ulong.MaxValue, float.NaN), destination[topK + 1]);
                Assert.True(workspace.CurrentVisitMark > 0);
            });

            ForceFullCollectionStress();
        }
    }

    private static Dictionary<SearchCase, SearchResult[]> CreateBaselines(
        HnswIndex index,
        float[][] queries,
        int[] topKValues,
        ulong[][]? allowlists = null)
    {
        var baselines = new Dictionary<SearchCase, SearchResult[]>();
        int allowlistCount = allowlists?.Length ?? 1;

        for (int queryIndex = 0; queryIndex < queries.Length; queryIndex++)
        {
            foreach (int topK in topKValues)
            {
                for (int allowlistIndex = 0; allowlistIndex < allowlistCount; allowlistIndex++)
                {
                    int keyAllowlistIndex = allowlists is null ? SearchCase.NoAllowlist : allowlistIndex;
                    SearchResult[] first = allowlists is null
                        ? Search(index, queries[queryIndex], topK)
                        : Search(index, queries[queryIndex], allowlists[allowlistIndex], topK);
                    SearchResult[] second = allowlists is null
                        ? Search(index, queries[queryIndex], topK)
                        : Search(index, queries[queryIndex], allowlists[allowlistIndex], topK);

                    AssertResultIntegrity(first);
                    Assert.Equal(first, second);
                    baselines.Add(new SearchCase(queryIndex, topK, keyAllowlistIndex), first);
                }
            }
        }

        return baselines;
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

    private static HnswIndex CreateDeterministicCosineHnsw()
    {
        int nextOrdinal = 0;
        var options = new HnswIndexOptions(8, 128, 128, 0x2740_0001UL);
        var index = new HnswIndex(
            Dimension,
            VectorMetric.Cosine,
            options,
            () => DeterministicLevel(nextOrdinal++));

        for (int ordinal = 0; ordinal < Count; ordinal++)
        {
            index.Add((ulong)(90_000 + ordinal * 53), CreateVector(ordinal));
        }

        Assert.Equal(Count, index.Count);
        return index;
    }

    private static int DeterministicLevel(int ordinal)
    {
        if (ordinal == 0 || ordinal % 149 == 0)
        {
            return 4;
        }

        if (ordinal % 67 == 0)
        {
            return 3;
        }

        if (ordinal % 23 == 0)
        {
            return 2;
        }

        return ordinal % 6 == 0 ? 1 : 0;
    }

    private static float[][] CreateQueries() =>
    [
        CreateQuery(0, 0x2740_1000),
        CreateQuery(5, 0x2740_1001),
        CreateQuery(13, 0x2740_1002),
        CreateQuery(29, 0x2740_1003),
        CreateQuery(47, 0x2740_1004),
        CreateQuery(89, 0x2740_1005),
        CreateQuery(131, 0x2740_1006),
        CreateQuery(211, 0x2740_1007),
        CreateQuery(337, 0x2740_1008),
        CreateQuery(431, 0x2740_1009)
    ];

    private static ulong[][] CreateAllowlists()
    {
        ulong[][] allowlists = new ulong[4][];
        for (int set = 0; set < allowlists.Length; set++)
        {
            var ids = new List<ulong> { 1, 2, (ulong)(90_000 + set * 53), (ulong)(90_000 + set * 53) };
            int stride = set switch
            {
                0 => 2,
                1 => 3,
                2 => 3,
                _ => 2
            };

            for (int ordinal = set; ordinal < Count; ordinal += stride)
            {
                ids.Add((ulong)(90_000 + ordinal * 53));
            }

            ids.Add(ulong.MaxValue - (ulong)set);
            allowlists[set] = ids.ToArray();
        }

        return allowlists;
    }

    private static float[] CreateQuery(int anchor, int seed)
    {
        float[] query = CreateVector(anchor % Count);
        var random = new Random(seed);
        for (int i = 0; i < query.Length; i++)
        {
            query[i] += ((random.NextSingle() - 0.5f) * 0.55f) + ((i % 7) - 3) * 0.03125f;
        }

        if (query.All(static value => value == 0f))
        {
            query[0] = 1f;
        }

        return query;
    }

    private static float[] CreateVector(int ordinal)
    {
        var vector = new float[Dimension];
        int cluster = ordinal % 19;
        int band = ordinal / 19;
        for (int i = 0; i < vector.Length; i++)
        {
            int lane = ((ordinal * 17) + (i * 31) + (cluster * 7)) % 37;
            float clusterCenter = (cluster - 9f) * 0.73f;
            float bandOffset = ((band % 13) - 6) * 0.08125f;
            vector[i] = clusterCenter + bandOffset + (lane - 18) * 0.044f + ((i % 5) - 2) * 0.017f;
        }

        vector[ordinal % Dimension] += ordinal % 2 == 0 ? 0.5f : -0.5f;
        if (vector.All(static value => value == 0f))
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static void AssertResultIntegrity(SearchResult[] results)
    {
        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        for (int i = 0; i < results.Length; i++)
        {
            Assert.True(float.IsFinite(results[i].Distance), $"Distance for {results[i].Id} was not finite.");
            if (i == 0)
            {
                continue;
            }

            SearchResult previous = results[i - 1];
            SearchResult current = results[i];
            Assert.True(
                previous.Distance < current.Distance ||
                (previous.Distance == current.Distance && previous.Id <= current.Id),
                $"Results were not sorted at position {i}: {previous} then {current}.");
        }
    }

    private static void ForceFullCollectionStress()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
        }
    }

    private readonly record struct SearchCase(int QueryIndex, int TopK, int AllowlistIndex)
    {
        public const int NoAllowlist = -1;
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() =>
            new(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswCosineReadOnlyConcurrency-" + Guid.NewGuid().ToString("N")));

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
    }
}
