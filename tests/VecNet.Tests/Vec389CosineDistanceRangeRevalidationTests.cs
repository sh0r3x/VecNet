using Xunit.Abstractions;

namespace VecNet.Tests;

public sealed class Vec389CosineDistanceRangeRevalidationTests
{
    private const float PolicyTolerance = 1e-6f;

    private readonly ITestOutputHelper _output;

    public Vec389CosineDistanceRangeRevalidationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void PublicCosineSearchSurfaces_ReturnFiniteSortedDistancesWithinTinyRoundoffRange()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var bySurface = new Dictionary<string, RangeStats>(StringComparer.Ordinal);
        RangeStats aggregate = RangeStats.Empty("all-public-cosine-surfaces");

        foreach (AuditSet set in CreateAuditSets())
        {
            aggregate = AuditExactFlat(set, bySurface, aggregate);
            aggregate = AuditImmutableAndOpenedHnsw(set, temp.Path, bySurface, aggregate);
            aggregate = AuditMutableHnsw(set, temp.Path, bySurface, aggregate);
        }

        string[] requiredSurfaces =
        [
            "exact-flat-search",
            "exact-flat-allowlist-search",
            "exact-flat-candidate-set-search",
            "immutable-hnsw-search",
            "immutable-hnsw-per-search-ef",
            "immutable-hnsw-allowlist-search",
            "opened-read-only-hnsw-search",
            "opened-read-only-hnsw-per-search-ef",
            "opened-read-only-hnsw-allowlist-search",
            "mutable-hnsw-base-plus-exact-delta-search",
            "mutable-hnsw-tombstone-search",
            "mutable-hnsw-allowlist-search",
            "mutable-hnsw-after-checkpoint-search",
            "reopened-checkpoint-hnsw-search",
            "reopened-checkpoint-hnsw-allowlist-search"
        ];

        foreach (string surface in requiredSurfaces)
        {
            Assert.True(bySurface.ContainsKey(surface), $"Missing cosine range observations for {surface}.");
            AssertWithinToleratedRange(bySurface[surface]);
            _output.WriteLine(bySurface[surface].ToString());
        }

        AssertWithinToleratedRange(aggregate);
        _output.WriteLine(aggregate.ToString());
    }

    private static RangeStats AuditExactFlat(
        AuditSet set,
        Dictionary<string, RangeStats> bySurface,
        RangeStats aggregate)
    {
        ExactFlatIndex index = CreateExactFlat(set.Rows, set.Dimension);
        Dictionary<ulong, string> kinds = RowKinds(set.Rows);

        SearchResult[] unfiltered = Search(index, set.Query, topK: set.Rows.Length);
        aggregate = Include("exact-flat-search", set, kinds, unfiltered, bySurface, aggregate);

        ulong[] allowlist = AllowlistWithSpecialsAndDuplicates(set);
        SearchResult[] allowed = Search(index, set.Query, allowlist, topK: 8);
        aggregate = Include("exact-flat-allowlist-search", set, kinds, allowed, bySurface, aggregate);

        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowlist);
        SearchResult[] candidateSet = Search(index, set.Query, candidates, topK: 8);
        aggregate = Include("exact-flat-candidate-set-search", set, kinds, candidateSet, bySurface, aggregate);

        Assert.Equal(allowed, candidateSet);
        return aggregate;
    }

    private static RangeStats AuditImmutableAndOpenedHnsw(
        AuditSet set,
        string tempRoot,
        Dictionary<string, RangeStats> bySurface,
        RangeStats aggregate)
    {
        HnswIndex index = CreateHnsw(
            set.Rows,
            set.Dimension,
            new HnswIndexOptions(M: 8, EfConstruction: 96, EfSearch: 64, RandomSeed: 0x3890_1000UL + (ulong)set.Dimension));
        Dictionary<ulong, string> kinds = RowKinds(set.Rows);

        SearchResult[] immutable = Search(index, set.Query, topK: set.Rows.Length);
        aggregate = Include("immutable-hnsw-search", set, kinds, immutable, bySurface, aggregate);

        SearchResult[] immutablePerSearchEf = Search(index, set.Query, topK: set.Rows.Length, efSearch: 96);
        aggregate = Include("immutable-hnsw-per-search-ef", set, kinds, immutablePerSearchEf, bySurface, aggregate);

        ulong[] allowlist = AllowlistWithSpecialsAndDuplicates(set);
        SearchResult[] immutableAllowed = Search(index, set.Query, allowlist, topK: 8, efSearch: 96);
        aggregate = Include("immutable-hnsw-allowlist-search", set, kinds, immutableAllowed, bySurface, aggregate);

        string savePath = Path.Combine(tempRoot, "hnsw-d" + set.Dimension.ToString("D3"));
        Directory.CreateDirectory(savePath);
        index.Save(savePath);
        HnswIndex opened = HnswIndex.OpenReadOnly(savePath);

        SearchResult[] openedDefault = Search(opened, set.Query, topK: set.Rows.Length);
        aggregate = Include("opened-read-only-hnsw-search", set, kinds, openedDefault, bySurface, aggregate);

        SearchResult[] openedPerSearchEf = Search(opened, set.Query, topK: set.Rows.Length, efSearch: 96);
        aggregate = Include("opened-read-only-hnsw-per-search-ef", set, kinds, openedPerSearchEf, bySurface, aggregate);

        SearchResult[] openedAllowed = Search(opened, set.Query, allowlist, topK: 8, efSearch: 96);
        aggregate = Include("opened-read-only-hnsw-allowlist-search", set, kinds, openedAllowed, bySurface, aggregate);

        Assert.Equal(immutable, openedDefault);
        Assert.Equal(immutablePerSearchEf, openedPerSearchEf);
        Assert.Equal(immutableAllowed, openedAllowed);
        return aggregate;
    }

    private static RangeStats AuditMutableHnsw(
        AuditSet set,
        string tempRoot,
        Dictionary<string, RangeStats> bySurface,
        RangeStats aggregate)
    {
        Row[] baseRows = set.Rows.Where((_, index) => index % 2 == 0).ToArray();
        Row[] deltaRows = set.Rows.Where((_, index) => index % 2 != 0).ToArray();
        HnswIndex baseIndex = CreateHnsw(
            baseRows,
            set.Dimension,
            new HnswIndexOptions(M: 8, EfConstruction: 96, EfSearch: 64, RandomSeed: 0x3890_2000UL + (ulong)set.Dimension));
        var mutable = new HnswMutableIndex(baseIndex);

        foreach (Row row in deltaRows)
        {
            Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(row.Id, row.Vector).Status);
        }

        Dictionary<ulong, string> allKinds = RowKinds(set.Rows);
        SearchResult[] beforeDeletes = Search(mutable, set.Query, topK: set.Rows.Length, efSearch: 96);
        aggregate = Include(
            "mutable-hnsw-base-plus-exact-delta-search",
            set,
            allKinds,
            beforeDeletes,
            bySurface,
            aggregate);

        ulong deletedBaseId = set.OppositeId;
        ulong deletedDeltaId = set.NearOppositeId;
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(deletedBaseId).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(deletedDeltaId).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(deletedBaseId, set.Rows[0].Vector).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(deletedDeltaId, set.Rows[0].Vector).Status);

        Row[] liveRows = set.Rows
            .Where(row => row.Id != deletedBaseId && row.Id != deletedDeltaId)
            .ToArray();
        Dictionary<ulong, string> liveKinds = RowKinds(liveRows);
        SearchResult[] afterDeletes = Search(mutable, set.Query, topK: liveRows.Length, efSearch: 96);
        Assert.DoesNotContain(afterDeletes, result => result.Id == deletedBaseId || result.Id == deletedDeltaId);
        aggregate = Include("mutable-hnsw-tombstone-search", set, liveKinds, afterDeletes, bySurface, aggregate);

        ulong[] allowlist = AllowlistWithSpecialsAndDuplicates(set);
        SearchResult[] allowed = Search(mutable, set.Query, allowlist, topK: 6, efSearch: 96);
        Assert.DoesNotContain(allowed, result => result.Id == deletedBaseId || result.Id == deletedDeltaId);
        aggregate = Include("mutable-hnsw-allowlist-search", set, liveKinds, allowed, bySurface, aggregate);

        string checkpointPath = Path.Combine(tempRoot, "mutable-checkpoint-d" + set.Dimension.ToString("D3"));
        HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpoint.Status);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);

        SearchResult[] afterCheckpoint = Search(mutable, set.Query, topK: liveRows.Length, efSearch: 96);
        aggregate = Include("mutable-hnsw-after-checkpoint-search", set, liveKinds, afterCheckpoint, bySurface, aggregate);

        HnswIndex reopened = HnswIndex.OpenReadOnly(checkpointPath);
        SearchResult[] reopenedResults = Search(reopened, set.Query, topK: liveRows.Length, efSearch: 96);
        aggregate = Include("reopened-checkpoint-hnsw-search", set, liveKinds, reopenedResults, bySurface, aggregate);

        SearchResult[] reopenedAllowed = Search(reopened, set.Query, allowlist, topK: 6, efSearch: 96);
        Assert.DoesNotContain(reopenedAllowed, result => result.Id == deletedBaseId || result.Id == deletedDeltaId);
        aggregate = Include(
            "reopened-checkpoint-hnsw-allowlist-search",
            set,
            liveKinds,
            reopenedAllowed,
            bySurface,
            aggregate);

        Assert.Equal(afterCheckpoint, reopenedResults);
        Assert.Equal(allowed, reopenedAllowed);
        return aggregate;
    }

    private static void AssertWithinToleratedRange(RangeStats stats)
    {
        Assert.True(stats.DistanceCount > 0, $"{stats.Surface} did not observe any distances.");
        Assert.True(
            stats.MinimumDistance >= -PolicyTolerance,
            $"{stats.Surface} minimum distance {stats.MinimumDistance:R} was below tolerated range. Case: {stats.MinimumCase}");
        Assert.True(
            stats.MaximumDistance <= 2f + PolicyTolerance,
            $"{stats.Surface} maximum distance {stats.MaximumDistance:R} was above tolerated range. Case: {stats.MaximumCase}");
    }

    private static RangeStats Include(
        string surface,
        AuditSet set,
        IReadOnlyDictionary<ulong, string> rowKinds,
        SearchResult[] results,
        Dictionary<string, RangeStats> bySurface,
        RangeStats aggregate)
    {
        AssertSortedFinite(surface, set.Dimension, results);

        RangeStats surfaceStats = bySurface.TryGetValue(surface, out RangeStats existing)
            ? existing
            : RangeStats.Empty(surface);
        surfaceStats = surfaceStats.Include(surface, set, rowKinds, results);
        bySurface[surface] = surfaceStats;
        return aggregate.Include(surface, set, rowKinds, results);
    }

    private static void AssertSortedFinite(string surface, int dimension, SearchResult[] results)
    {
        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        for (int i = 0; i < results.Length; i++)
        {
            Assert.True(
                float.IsFinite(results[i].Distance),
                $"{surface} dimension {dimension} returned non-finite distance for ID {results[i].Id}.");

            if (i > 0)
            {
                SearchResult previous = results[i - 1];
                SearchResult current = results[i];
                Assert.True(
                    previous.Distance < current.Distance ||
                    (previous.Distance == current.Distance && previous.Id <= current.Id),
                    $"{surface} dimension {dimension} was not sorted at ranks {i - 1} and {i}: " +
                    $"{previous.Id}/{previous.Distance:R}, {current.Id}/{current.Distance:R}.");
            }
        }
    }

    private static AuditSet[] CreateAuditSets() =>
    [
        CreateAuditSet(dimension: 2, seed: 0x3890_0002),
        CreateAuditSet(dimension: 3, seed: 0x3890_0003),
        CreateAuditSet(dimension: 31, seed: 0x3890_0031),
        CreateAuditSet(dimension: 64, seed: 0x3890_0064),
        CreateAuditSet(dimension: 385, seed: 0x3890_0385)
    ];

    private static AuditSet CreateAuditSet(int dimension, int seed)
    {
        float[] query = CreateUnitVector(dimension, seed);
        float[] near = CreateNearVector(query, epsilon: 1e-4, seed + 17);
        float[] opposite = Negate(query);
        float[] nearOpposite = Negate(near);

        ulong idBase = 389_000UL + (ulong)dimension * 100UL;
        var rows = new List<Row>
        {
            new(idBase + 10, query, "identical"),
            new(idBase + 20, opposite, "opposite"),
            new(idBase + 30, near, "near-identical"),
            new(idBase + 40, nearOpposite, "near-opposite")
        };

        for (int i = 0; i < 12; i++)
        {
            rows.Add(new Row(
                idBase + 100UL + (ulong)i,
                CreateUnitVector(dimension, seed + 1_000 + i * 97),
                "generated-normalized"));
        }

        return new AuditSet(
            dimension,
            query,
            rows.ToArray(),
            IdenticalId: idBase + 10,
            OppositeId: idBase + 20,
            NearIdenticalId: idBase + 30,
            NearOppositeId: idBase + 40);
    }

    private static float[] CreateUnitVector(int dimension, int seed)
    {
        var random = new Random(seed);
        var vector = new float[dimension];
        double squaredMagnitude = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            double sample = random.NextDouble() * 2 - 1;
            if (i % 7 == 0)
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

    private static float[] CreateNearVector(float[] query, double epsilon, int seed)
    {
        float[] tangent = CreatePerpendicularUnitVector(query, seed);
        var near = new float[query.Length];
        for (int i = 0; i < near.Length; i++)
        {
            near[i] = (float)(query[i] + epsilon * tangent[i]);
        }

        return Normalize(near);
    }

    private static float[] CreatePerpendicularUnitVector(float[] query, int seed)
    {
        var random = new Random(seed);
        var basis = new float[query.Length];
        for (int i = 0; i < basis.Length; i++)
        {
            basis[i] = (float)(random.NextDouble() * 2 - 1);
        }

        double projection = 0;
        for (int i = 0; i < query.Length; i++)
        {
            projection += (double)basis[i] * query[i];
        }

        for (int i = 0; i < basis.Length; i++)
        {
            basis[i] -= (float)(projection * query[i]);
        }

        if (MagnitudeSquared(basis) == 0)
        {
            basis[0] = 1f;
            basis[^1] -= query[^1] == 0f ? 0f : query[0] / query[^1];
        }

        return Normalize(basis);
    }

    private static float[] Normalize(float[] vector)
    {
        double squaredMagnitude = MagnitudeSquared(vector);
        double magnitude = Math.Sqrt(squaredMagnitude);
        var normalized = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = (float)(vector[i] / magnitude);
        }

        return normalized;
    }

    private static double MagnitudeSquared(float[] vector)
    {
        double squaredMagnitude = 0;
        foreach (float component in vector)
        {
            squaredMagnitude += (double)component * component;
        }

        return squaredMagnitude;
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

    private static ExactFlatIndex CreateExactFlat(Row[] rows, int dimension)
    {
        var index = new ExactFlatIndex(dimension, VectorMetric.Cosine);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static HnswIndex CreateHnsw(Row[] rows, int dimension, HnswIndexOptions options)
    {
        var index = new HnswIndex(dimension, VectorMetric.Cosine, options, () => 0);
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

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchFilterWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, ExactFlatCandidateSet candidates, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, candidates, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace());
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(efSearch), efSearch);
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

    private static ulong[] AllowlistWithSpecialsAndDuplicates(AuditSet set) =>
    [
        0,
        set.NearOppositeId,
        set.IdenticalId,
        set.OppositeId,
        set.NearIdenticalId,
        set.IdenticalId,
        set.Rows[4].Id,
        set.Rows[5].Id,
        set.Rows[6].Id,
        set.Rows[7].Id,
        ulong.MaxValue
    ];

    private static Dictionary<ulong, string> RowKinds(IEnumerable<Row> rows) =>
        rows.ToDictionary(static row => row.Id, static row => row.Kind);

    private readonly record struct AuditSet(
        int Dimension,
        float[] Query,
        Row[] Rows,
        ulong IdenticalId,
        ulong OppositeId,
        ulong NearIdenticalId,
        ulong NearOppositeId);

    private readonly record struct Row(ulong Id, float[] Vector, string Kind);

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

        public RangeStats Include(
            string surface,
            AuditSet set,
            IReadOnlyDictionary<ulong, string> rowKinds,
            SearchResult[] results)
        {
            RangeStats stats = this;
            for (int rank = 0; rank < results.Length; rank++)
            {
                SearchResult result = results[rank];
                string kind = rowKinds.TryGetValue(result.Id, out string? rowKind) ? rowKind : "unknown";
                string label =
                    $"surface={surface}, dimension={set.Dimension}, query=generated-normalized, row={kind}, " +
                    $"id={result.Id}, rank={rank}, distance={result.Distance:R}";
                float belowZero = result.Distance < 0 ? -result.Distance : 0f;
                float aboveTwo = result.Distance > 2f ? result.Distance - 2f : 0f;
                bool hasNewMinimum = result.Distance < stats.MinimumDistance;
                bool hasNewMaximum = result.Distance > stats.MaximumDistance;

                stats = new RangeStats(
                    Surface,
                    stats.DistanceCount + 1,
                    hasNewMinimum ? result.Distance : stats.MinimumDistance,
                    hasNewMaximum ? result.Distance : stats.MaximumDistance,
                    belowZero > stats.MaximumBelowZero ? belowZero : stats.MaximumBelowZero,
                    aboveTwo > stats.MaximumAboveTwo ? aboveTwo : stats.MaximumAboveTwo,
                    hasNewMinimum ? label : stats.MinimumCase,
                    hasNewMaximum ? label : stats.MaximumCase);
            }

            return stats;
        }

        public override string ToString() =>
            $"{Surface}: count={DistanceCount}, min={MinimumDistance:R} ({MinimumCase}), " +
            $"max={MaximumDistance:R} ({MaximumCase}), maxBelowZero={MaximumBelowZero:R}, " +
            $"maxAboveTwo={MaximumAboveTwo:R}";
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-VEC389-" + Guid.NewGuid().ToString("N"));
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
