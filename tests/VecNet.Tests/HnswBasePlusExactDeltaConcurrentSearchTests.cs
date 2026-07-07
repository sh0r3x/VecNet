namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaConcurrentSearchTests
{
    private const int BaseCount = 256;
    private const int DeltaCount = 48;
    private const int Dimension = 23;
    private const int ParallelIterations = 768;

    private static readonly int[] TopKValues = [1, 17];

    [Fact]
    public void Search_MatchesStableBaselinesForCompositeCheckpointAndOpenedHnswUnderReadOnlyConcurrency()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswBasePlusExactDeltaIndex composite = CreateMutatedComposite();
        float[][] queries = CreateQueries();

        Assert.True(composite.BaseTombstoneCount > 0);
        Assert.True(composite.DeltaTombstoneCount > 0);
        Assert.True(composite.DeltaLiveVectorCount > 0);
        Assert.True(composite.LiveVectorCount > BaseCount / 2);

        Dictionary<(int QueryIndex, int TopK), SearchResult[]> sourceBaselines =
            CreateCompositeBaselines(composite, queries);
        AssertConcurrentCompositeSearchMatchesBaselines(composite, queries, sourceBaselines);

        HnswBasePlusExactDeltaCheckpointResult checkpointResult = composite.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(0, composite.DeltaPhysicalVectorCount);
        Assert.Equal(0, composite.TombstoneCount);

        Dictionary<(int QueryIndex, int TopK), SearchResult[]> rebuiltBaselines =
            CreateCompositeBaselines(composite, queries);
        AssertConcurrentCompositeSearchMatchesBaselines(composite, queries, rebuiltBaselines);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Dictionary<(int QueryIndex, int TopK), SearchResult[]> openedBaselines =
            CreateHnswBaselines(opened, queries);
        AssertBaselineParity(rebuiltBaselines, openedBaselines);
        AssertConcurrentHnswSearchMatchesBaselines(opened, queries, openedBaselines);
    }

    private static void AssertConcurrentCompositeSearchMatchesBaselines(
        HnswBasePlusExactDeltaIndex index,
        float[][] queries,
        IReadOnlyDictionary<(int QueryIndex, int TopK), SearchResult[]> expected)
    {
        Parallel.For(0, ParallelIterations, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            int topK = TopKValues[(iteration / queries.Length) % TopKValues.Length];
            SearchResult[] baseline = expected[(queryIndex, topK)];
            float[] query = queries[queryIndex].ToArray();
            SearchResult sentinel = new(ulong.MaxValue - (ulong)iteration, -1000f - iteration);
            SearchResult[] results = Enumerable.Repeat(sentinel, topK + 2).ToArray();
            HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(index, topK);

            int written = index.Search(query, results.AsSpan(0, topK), workspace);

            AssertMatchesBaseline(baseline, results, written);
            Assert.Equal(sentinel, results[topK]);
            Assert.Equal(sentinel, results[topK + 1]);
            Assert.Equal(index.Generation, workspace.ObservedGeneration);
        });
    }

    private static void AssertConcurrentHnswSearchMatchesBaselines(
        HnswIndex index,
        float[][] queries,
        IReadOnlyDictionary<(int QueryIndex, int TopK), SearchResult[]> expected)
    {
        Parallel.For(0, ParallelIterations, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            int topK = TopKValues[(iteration / queries.Length) % TopKValues.Length];
            SearchResult[] baseline = expected[(queryIndex, topK)];
            float[] query = queries[queryIndex].ToArray();
            SearchResult sentinel = new(ulong.MaxValue - (ulong)iteration, -2000f - iteration);
            SearchResult[] results = Enumerable.Repeat(sentinel, topK + 2).ToArray();
            var workspace = new HnswSearchWorkspace(index.Count, index.Options.EfSearch);

            int written = index.Search(query, results.AsSpan(0, topK), workspace);

            AssertMatchesBaseline(baseline, results, written);
            Assert.Equal(sentinel, results[topK]);
            Assert.Equal(sentinel, results[topK + 1]);
            Assert.True(workspace.CurrentVisitMark > 0);
        });
    }

    private static Dictionary<(int QueryIndex, int TopK), SearchResult[]> CreateCompositeBaselines(
        HnswBasePlusExactDeltaIndex index,
        float[][] queries)
    {
        var expected = new Dictionary<(int QueryIndex, int TopK), SearchResult[]>();
        foreach (int topK in TopKValues)
        {
            for (int queryIndex = 0; queryIndex < queries.Length; queryIndex++)
            {
                SearchResult[] first = SearchComposite(index, queries[queryIndex], topK);
                SearchResult[] second = SearchComposite(index, queries[queryIndex], topK);

                Assert.Equal(first, second);
                Assert.True(first.Length > 0);
                Assert.True(first.Length <= topK);
                AssertSorted(first);
                expected.Add((queryIndex, topK), first);
            }
        }

        return expected;
    }

    private static Dictionary<(int QueryIndex, int TopK), SearchResult[]> CreateHnswBaselines(
        HnswIndex index,
        float[][] queries)
    {
        var expected = new Dictionary<(int QueryIndex, int TopK), SearchResult[]>();
        foreach (int topK in TopKValues)
        {
            for (int queryIndex = 0; queryIndex < queries.Length; queryIndex++)
            {
                SearchResult[] first = SearchHnsw(index, queries[queryIndex], topK);
                SearchResult[] second = SearchHnsw(index, queries[queryIndex], topK);

                Assert.Equal(first, second);
                Assert.True(first.Length > 0);
                Assert.True(first.Length <= topK);
                AssertSorted(first);
                expected.Add((queryIndex, topK), first);
            }
        }

        return expected;
    }

    private static void AssertBaselineParity(
        IReadOnlyDictionary<(int QueryIndex, int TopK), SearchResult[]> expected,
        IReadOnlyDictionary<(int QueryIndex, int TopK), SearchResult[]> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        foreach (((int queryIndex, int topK), SearchResult[] expectedResults) in expected)
        {
            SearchResult[] actualResults = actual[(queryIndex, topK)];
            Assert.Equal(expectedResults, actualResults);
        }
    }

    private static void AssertMatchesBaseline(SearchResult[] baseline, SearchResult[] results, int written)
    {
        Assert.Equal(baseline.Length, written);
        for (int i = 0; i < written; i++)
        {
            Assert.Equal(baseline[i].Id, results[i].Id);
            Assert.Equal(baseline[i].Distance, results[i].Distance);
        }
    }

    private static SearchResult[] SearchComposite(HnswBasePlusExactDeltaIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, CreateCompositeWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static HnswBasePlusExactDeltaIndex CreateMutatedComposite()
    {
        HnswIndex baseIndex = CreateBaseIndex();
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);

        for (int ordinal = 0; ordinal < DeltaCount; ordinal++)
        {
            Assert.Equal(
                VectorMutationStatus.Committed,
                composite.TryAdd(DeltaId(ordinal), CreateDeltaVector(ordinal)).Status);
        }

        foreach (int ordinal in new[] { 0, 7, 19, 43, 88, 127, 191, 233 })
        {
            Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(BaseId(ordinal)).Status);
        }

        foreach (int ordinal in new[] { 1, 6, 17, 29, 41 })
        {
            Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(DeltaId(ordinal)).Status);
        }

        return composite;
    }

    private static HnswIndex CreateBaseIndex()
    {
        int nextOrdinal = 0;
        var options = new HnswIndexOptions(8, 96, 96, 0x1440_0001UL);
        var index = new HnswIndex(
            Dimension,
            VectorMetric.SquaredEuclidean,
            options,
            () => DeterministicLevel(nextOrdinal++));

        for (int ordinal = 0; ordinal < BaseCount; ordinal++)
        {
            index.Add(BaseId(ordinal), CreateBaseVector(ordinal));
        }

        Assert.Equal(BaseCount, index.Count);
        Assert.True(index.MaxLayer >= 3);
        return index;
    }

    private static int DeterministicLevel(int ordinal)
    {
        if (ordinal == 0 || ordinal % 79 == 0)
        {
            return 3;
        }

        if (ordinal % 29 == 0)
        {
            return 2;
        }

        return ordinal % 6 == 0 ? 1 : 0;
    }

    private static float[][] CreateQueries() =>
    [
        CreateQueryFromBase(3, 0x1440_1001),
        CreateQueryFromBase(16, 0x1440_1002),
        CreateQueryFromBase(64, 0x1440_1003),
        CreateQueryFromBase(121, 0x1440_1004),
        CreateQueryFromBase(214, 0x1440_1005),
        CreateQueryFromDelta(4, 0x1440_2001),
        CreateQueryFromDelta(22, 0x1440_2002),
        CreateQueryFromDelta(39, 0x1440_2003)
    ];

    private static float[] CreateQueryFromBase(int ordinal, int seed) =>
        Perturb(CreateBaseVector(ordinal), seed);

    private static float[] CreateQueryFromDelta(int ordinal, int seed) =>
        Perturb(CreateDeltaVector(ordinal), seed);

    private static float[] Perturb(float[] vector, int seed)
    {
        var random = new Random(seed);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] += ((random.NextSingle() - 0.5f) * 0.27f) + ((i % 5) - 2) * 0.031f;
        }

        return vector;
    }

    private static float[] CreateBaseVector(int ordinal)
    {
        var vector = new float[Dimension];
        int cluster = ordinal % 16;
        int band = ordinal / 16;
        for (int i = 0; i < vector.Length; i++)
        {
            int lane = ((ordinal * 11) + (i * 17)) % 37;
            vector[i] =
                ((cluster - 7.5f) * 4.25f) +
                ((band % 13) * 0.53f) +
                ((lane - 18) * 0.11f) +
                (((i % 7) - 3) * 0.047f);
        }

        return vector;
    }

    private static float[] CreateDeltaVector(int ordinal)
    {
        int anchor = (ordinal * 5 + 3) % BaseCount;
        float[] vector = CreateBaseVector(anchor);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] += ((ordinal % 9) - 4) * 0.067f + ((i % 4) - 1.5f) * 0.023f;
        }

        return vector;
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK);

    private static ulong BaseId(int ordinal) => 10_000UL + (ulong)ordinal * 17UL;

    private static ulong DeltaId(int ordinal) => 900_000UL + (ulong)ordinal * 19UL;

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

        public static TempIndexDirectory CreateMissing() => new(
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswCompositeConcurrentTests-" + Guid.NewGuid().ToString("N")));

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
