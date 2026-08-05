using Xunit.Abstractions;

namespace VecNet.Tests;

public sealed class Vec306CosineDistanceRangeAuditTests
{
    private const float RangeTolerance = 1e-6f;

    private readonly ITestOutputHelper _output;

    public Vec306CosineDistanceRangeAuditTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ExactFlatCosineSearch_ReturnedDistancesStayWithinTinyRangeTolerance()
    {
        RangeStats stats = RangeStats.Empty("exact-flat");

        foreach (int dimension in RepresentativeDimensions())
        {
            AuditSet set = CreateAuditSet(dimension);
            var index = new ExactFlatIndex(dimension, VectorMetric.Cosine);
            foreach (AuditRow row in set.Rows)
            {
                index.Add(row.Id, row.Vector);
            }

            SearchResult[] results = new SearchResult[set.Rows.Length];
            int written = index.Search(set.Query, results);

            Assert.Equal(set.Rows.Length, written);
            stats = stats.Include(dimension, results.AsSpan(0, written));
        }

        _output.WriteLine(stats.ToString());
        AssertWithinTinyCosineRange(stats);
    }

    [Fact]
    public void ImmutableAndOpenedHnswCosineSearch_ReturnedDistancesStayWithinTinyRangeTolerance()
    {
        using TempIndexDirectory saved = TempIndexDirectory.Create();
        RangeStats immutableStats = RangeStats.Empty("immutable-hnsw");
        RangeStats openedStats = RangeStats.Empty("opened-read-only-hnsw");

        foreach (int dimension in RepresentativeDimensions())
        {
            AuditSet set = CreateAuditSet(dimension);
            HnswIndex index = CreateHnsw(set, seed: 0x3060_1000UL + (ulong)dimension);

            SearchResult[] immutableResults = Search(index, set.Query, set.Rows.Length);
            Assert.Equal(set.Rows.Length, immutableResults.Length);
            immutableStats = immutableStats.Include(dimension, immutableResults);

            string path = Path.Combine(saved.Path, "d" + dimension);
            Directory.CreateDirectory(path);
            index.Save(path);
            HnswIndex opened = HnswIndex.OpenReadOnly(path);

            SearchResult[] openedResults = Search(opened, set.Query, set.Rows.Length);
            Assert.Equal(set.Rows.Length, openedResults.Length);
            openedStats = openedStats.Include(dimension, openedResults);
            Assert.Equal(immutableResults, openedResults);
        }

        _output.WriteLine(immutableStats.ToString());
        _output.WriteLine(openedStats.ToString());
        AssertWithinTinyCosineRange(immutableStats);
        AssertWithinTinyCosineRange(openedStats);
    }

    [Fact]
    public void MutableHnswCosineSearch_BasePlusExactDeltaReturnedDistancesStayWithinTinyRangeTolerance()
    {
        RangeStats stats = RangeStats.Empty("mutable-hnsw-base-plus-exact-delta");

        foreach (int dimension in RepresentativeDimensions())
        {
            AuditSet set = CreateAuditSet(dimension);
            AuditRow[] baseRows = set.Rows.Where((_, index) => index % 2 == 0).ToArray();
            AuditRow[] deltaRows = set.Rows.Where((_, index) => index % 2 != 0).ToArray();
            HnswIndex baseIndex = CreateHnsw(
                new AuditSet(set.Query, baseRows),
                seed: 0x3060_2000UL + (ulong)dimension);
            var mutable = new HnswMutableIndex(baseIndex);

            foreach (AuditRow row in deltaRows)
            {
                Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(row.Id, row.Vector).Status);
            }

            SearchResult[] results = new SearchResult[set.Rows.Length];
            int written = mutable.Search(
                set.Query,
                results,
                mutable.CreateSearchWorkspace(maxResults: set.Rows.Length, maxEfSearch: 64),
                efSearch: 64);

            Assert.Equal(set.Rows.Length, written);
            Assert.Contains(results.AsSpan(0, written).ToArray(), static result => result.Id == 10);
            Assert.Contains(results.AsSpan(0, written).ToArray(), static result => result.Id == 20);
            stats = stats.Include(dimension, results.AsSpan(0, written));
        }

        _output.WriteLine(stats.ToString());
        AssertWithinTinyCosineRange(stats);
    }

    private static void AssertWithinTinyCosineRange(RangeStats stats)
    {
        Assert.True(
            stats.MinimumDistance >= -RangeTolerance,
            $"{stats.Surface} minimum distance {stats.MinimumDistance:R} was below tolerance.");
        Assert.True(
            stats.MaximumDistance <= 2f + RangeTolerance,
            $"{stats.Surface} maximum distance {stats.MaximumDistance:R} was above tolerance.");
    }

    private static HnswIndex CreateHnsw(AuditSet set, ulong seed)
    {
        var index = new HnswIndex(
            set.Query.Length,
            VectorMetric.Cosine,
            new HnswIndexOptions(M: 8, EfConstruction: 64, EfSearch: 64, RandomSeed: seed),
            () => 0);

        foreach (AuditRow row in set.Rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(maxEfSearch: 64), efSearch: 64);
        return results[..written];
    }

    private static AuditSet CreateAuditSet(int dimension)
    {
        float[] query = CreateUnitVector(dimension, seed: 0x3060_0000 + dimension);
        float[] near = CreateNearVector(query, epsilon: 1e-4);
        float[] nearOpposite = Negate(near);
        var rows = new List<AuditRow>
        {
            new(10, query, "identical"),
            new(20, Negate(query), "opposite"),
            new(30, near, "near-identical"),
            new(40, nearOpposite, "near-opposite")
        };

        for (int i = 0; i < 10; i++)
        {
            rows.Add(new AuditRow(
                100UL + (ulong)i,
                CreateUnitVector(dimension, seed: 0x3060_3000 + dimension * 257 + i),
                "generated-normalized"));
        }

        return new AuditSet(query, rows.ToArray());
    }

    private static int[] RepresentativeDimensions() => [2, 3, 32, 33, 384];

    private static float[] CreateUnitVector(int dimension, int seed)
    {
        var random = new Random(seed);
        var vector = new float[dimension];
        double squaredMagnitude = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            double sample = random.NextDouble() * 2 - 1;
            if (i % 5 == 0)
            {
                sample *= 0.03125;
            }

            vector[i] = (float)sample;
            squaredMagnitude += sample * sample;
        }

        if (squaredMagnitude == 0)
        {
            vector[0] = 1f;
            return vector;
        }

        double magnitude = Math.Sqrt(squaredMagnitude);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (float)(vector[i] / magnitude);
        }

        return vector;
    }

    private static float[] CreateNearVector(float[] query, double epsilon)
    {
        float[] tangent = CreatePerpendicularUnitVector(query);
        var near = new float[query.Length];
        for (int i = 0; i < near.Length; i++)
        {
            near[i] = (float)(query[i] + epsilon * tangent[i]);
        }

        return Normalize(near);
    }

    private static float[] CreatePerpendicularUnitVector(float[] query)
    {
        var basis = new float[query.Length];
        int selected = Math.Abs(query[0]) < 0.9 ? 0 : 1;
        basis[selected] = 1f;

        double projection = 0;
        for (int i = 0; i < query.Length; i++)
        {
            projection += (double)basis[i] * query[i];
        }

        for (int i = 0; i < basis.Length; i++)
        {
            basis[i] -= (float)(projection * query[i]);
        }

        return Normalize(basis);
    }

    private static float[] Normalize(float[] vector)
    {
        double squaredMagnitude = 0;
        foreach (float component in vector)
        {
            squaredMagnitude += (double)component * component;
        }

        double magnitude = Math.Sqrt(squaredMagnitude);
        var normalized = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = (float)(vector[i] / magnitude);
        }

        return normalized;
    }

    private static float[] Negate(float[] vector)
    {
        var negated = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            negated[i] = -vector[i];
        }

        return negated;
    }

    private readonly record struct AuditSet(float[] Query, AuditRow[] Rows);

    private readonly record struct AuditRow(ulong Id, float[] Vector, string Kind);

    private readonly record struct RangeStats(
        string Surface,
        int DistanceCount,
        float MinimumDistance,
        float MaximumDistance,
        float MaximumBelowZero,
        float MaximumAboveTwo,
        string MinimumCase,
        string MaximumCase)
    {
        public static RangeStats Empty(string surface) =>
            new(surface, 0, float.PositiveInfinity, float.NegativeInfinity, 0f, 0f, "", "");

        public RangeStats Include(int dimension, ReadOnlySpan<SearchResult> results)
        {
            RangeStats stats = this;
            for (int i = 0; i < results.Length; i++)
            {
                SearchResult result = results[i];
                Assert.True(float.IsFinite(result.Distance), $"Non-finite {Surface} distance for ID {result.Id}.");

                string label = $"dimension={dimension}, id={result.Id}, rank={i}, distance={result.Distance:R}";
                float belowZero = result.Distance < 0 ? -result.Distance : 0f;
                float aboveTwo = result.Distance > 2f ? result.Distance - 2f : 0f;
                stats = new RangeStats(
                    Surface,
                    stats.DistanceCount + 1,
                    result.Distance < stats.MinimumDistance ? result.Distance : stats.MinimumDistance,
                    result.Distance > stats.MaximumDistance ? result.Distance : stats.MaximumDistance,
                    belowZero > stats.MaximumBelowZero ? belowZero : stats.MaximumBelowZero,
                    aboveTwo > stats.MaximumAboveTwo ? aboveTwo : stats.MaximumAboveTwo,
                    result.Distance < stats.MinimumDistance ? label : stats.MinimumCase,
                    result.Distance > stats.MaximumDistance ? label : stats.MaximumCase);
            }

            return stats;
        }

        public override string ToString() =>
            $"{Surface}: count={DistanceCount}, min={MinimumDistance:R} ({MinimumCase}), max={MaximumDistance:R} ({MaximumCase}), " +
            $"maxBelowZero={MaximumBelowZero:R}, maxAboveTwo={MaximumAboveTwo:R}";
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-VEC306-" + Guid.NewGuid().ToString("N"));
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
