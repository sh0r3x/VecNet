using System.Numerics;
using System.Reflection;

namespace VecNet.Tests;

public sealed class ExactFlatIndexCheckpointIndependentTests
{
    public static TheoryData<VectorMetric, int> CheckpointParityCases()
    {
        int vectorWidth = Vector<float>.Count;
        return new TheoryData<VectorMetric, int>
        {
            { VectorMetric.SquaredEuclidean, 1 },
            { VectorMetric.SquaredEuclidean, vectorWidth + 1 },
            { VectorMetric.SquaredEuclidean, 386 },
            { VectorMetric.InnerProduct, 32 },
            { VectorMetric.Cosine, 3 },
            { VectorMetric.Cosine, 96 }
        };
    }

    [Theory]
    [MemberData(nameof(CheckpointParityCases))]
    public void Vec065_CheckpointReopenAndPostCheckpointMutationsMatchIndependentLiveTruth(
        VectorMetric metric,
        int dimension)
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var random = new Random(unchecked((int)(0x6500_0001u + (uint)metric * 1009u + (uint)dimension)));
        var index = new ExactFlatIndex(dimension, metric);
        var model = new ReferenceModel(metric, dimension);

        ulong[] baseIds =
        [
            0,
            11,
            22,
            33,
            44,
            55,
            66,
            77,
            88,
            99,
            ulong.MaxValue - 1,
            ulong.MaxValue
        ];

        for (int i = 0; i < baseIds.Length; i++)
        {
            float[] vector = CreateVector(metric, dimension, i, random);
            index.Add(baseIds[i], vector);
            model.Add(baseIds[i], vector);
        }

        float[] duplicateBaseVector = model.RequireVector(33);
        AssertCommitted(index.TryDelete(22));
        model.Delete(22);
        float[] delta400 = CreateVector(metric, dimension, 100, random);
        AssertCommitted(index.TryAdd(400, delta400));
        model.Add(400, delta400);

        float[] deletedDeltaVector = CreateVector(metric, dimension, 101, random);
        AssertCommitted(index.TryAdd(401, deletedDeltaVector));
        model.Add(401, deletedDeltaVector);
        AssertCommitted(index.TryDelete(401));
        model.Delete(401);

        float[] delta402 = CreateVector(metric, dimension, 102, random);
        AssertCommitted(index.TryAdd(402, delta402));
        model.Add(402, delta402);
        AssertCommitted(index.TryDelete(ulong.MaxValue - 1));
        model.Delete(ulong.MaxValue - 1);

        float[] highDelta = CreateVector(metric, dimension, 103, random);
        AssertCommitted(index.TryAdd(ulong.MaxValue - 50, highDelta));
        model.Add(ulong.MaxValue - 50, highDelta);
        Assert.Equal(VectorMutationStatus.UnknownId, index.TryDelete(123_456_789).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(22, CreateVector(metric, dimension, 104, random)).Status);

        AssertCommitted(index.TryAdd(403, duplicateBaseVector));
        model.Add(403, duplicateBaseVector);
        AssertCommitted(index.TryDelete(0));
        model.Delete(0);

        float[] query = CreateQuery(metric, dimension);
        ulong[] scope =
        [
            0,
            22,
            33,
            33,
            400,
            401,
            402,
            403,
            ulong.MaxValue - 50,
            ulong.MaxValue - 1,
            ulong.MaxValue,
            987_654_321
        ];
        ExactFlatCandidateSet preCheckpointCandidates = index.CreateCandidateSet(scope);
        SearchResult[] sentinel = [new(999_999, -123f), new(888_888, -456f)];
        long beforeGeneration = index.Generation;
        Assert.Equal(17, index.VectorCount);

        VerifyAllSearchSurfaces(index, model, query, scope, topK: 50);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(result.Generation, index.Generation);
        Assert.Equal(model.LiveCount, result.PhysicalVectorCount);
        Assert.Equal(model.LiveCount, result.LiveVectorCount);
        Assert.Equal(model.LiveCount, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(model.ReservedCount, result.DeletedReservedIdCount);
        Assert.Equal(4, result.FoldedDeltaVectorCount);
        Assert.Equal(4, result.FoldedTombstoneCount);
        Assert.Equal(model.LiveCount, index.VectorCount);

        Assert.Throws<InvalidOperationException>(() => index.Search(query, preCheckpointCandidates, sentinel));
        Assert.Equal(new[] { new SearchResult(999_999, -123f), new SearchResult(888_888, -456f) }, sentinel);
        VerifyAllSearchSurfaces(index, model, query, scope, topK: 50);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(model.LiveCount, opened.VectorCount);
        VerifyAllSearchSurfaces(opened, model, query, scope, topK: 50);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryAdd(700, CreateVector(metric, dimension, 700, random)).Status);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryDelete(33).Status);
        using TempIndexDirectory readOnlyCheckpointTarget = TempIndexDirectory.CreateMissing();
        Assert.Throws<InvalidOperationException>(() => opened.Checkpoint(readOnlyCheckpointTarget.Path));

        VectorMutationResult reservedBase = index.TryAdd(22, CreateVector(metric, dimension, 800, random));
        VectorMutationResult reservedDelta = index.TryAdd(401, CreateVector(metric, dimension, 801, random));
        Assert.Equal(VectorMutationStatus.DuplicateId, reservedBase.Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, reservedDelta.Status);
        Assert.Equal(0, reservedBase.TombstoneCount);
        Assert.Equal(0, reservedDelta.TombstoneCount);

        ExactFlatCandidateSet postCheckpointCandidates = index.CreateCandidateSet(scope);
        VerifyCandidateSet(index, model, query, scope, postCheckpointCandidates, topK: 50);

        float[] newVector = CreateVector(metric, dimension, 900, random);
        AssertCommitted(index.TryAdd(9000, newVector));
        model.Add(9000, newVector);
        AssertCommitted(index.TryDelete(400));
        model.Delete(400);
        Assert.Throws<InvalidOperationException>(() => index.Search(query, postCheckpointCandidates, sentinel));

        ulong[] postMutationScope = scope.Concat([9000UL]).ToArray();
        VerifyAllSearchSurfaces(index, model, query, postMutationScope, topK: 50);
    }

    [Fact]
    public void Vec065_NoChangesFromEmptyAndCleanIndexesDoesNotWriteOrInvalidateCandidates()
    {
        using TempIndexDirectory missingEmptyTarget = TempIndexDirectory.CreateMissing();
        var empty = new ExactFlatIndex(4, VectorMetric.SquaredEuclidean);

        ExactFlatCheckpointResult emptyResult = empty.Checkpoint(missingEmptyTarget.Path);

        Assert.Equal(ExactFlatCheckpointStatus.NoChanges, emptyResult.Status);
        Assert.Equal(0, emptyResult.Generation);
        Assert.Equal(0, emptyResult.PhysicalVectorCount);
        Assert.Equal(0, emptyResult.LiveVectorCount);
        Assert.Equal(0, emptyResult.BaseVectorCount);
        Assert.False(Directory.Exists(missingEmptyTarget.Path));

        using TempIndexDirectory existingEmptyTarget = TempIndexDirectory.Create();
        var clean = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        clean.Add(10, [1f, 0f]);
        clean.Add(20, [0f, 1f]);
        ExactFlatCandidateSet candidates = clean.CreateCandidateSet([20, 10, 20, 999]);
        long beforeGeneration = clean.Generation;

        ExactFlatCheckpointResult cleanResult = clean.Checkpoint(existingEmptyTarget.Path);

        Assert.Equal(ExactFlatCheckpointStatus.NoChanges, cleanResult.Status);
        Assert.Equal(beforeGeneration, cleanResult.Generation);
        Assert.Equal(beforeGeneration, clean.Generation);
        Assert.Empty(Directory.EnumerateFileSystemEntries(existingEmptyTarget.Path));

        var actual = new SearchResult[4];
        int written = clean.Search([1f, 1f], candidates, actual);
        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], actual[..written].Select(static result => result.Id).Order().ToArray());
    }

    [Fact]
    public void Vec065_InvalidFileTargetLeavesSearchMutationAndCandidateStateUsable()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        string fileTarget = Path.Combine(temp.Path, "not-a-directory");
        File.WriteAllText(fileTarget, "caller-owned");

        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        var model = new ReferenceModel(VectorMetric.SquaredEuclidean, 3);
        Add(index, model, 10, [1f, 0f, 0f]);
        Add(index, model, 20, [0f, 1f, 0f]);
        Add(index, model, 30, [0f, 0f, 1f]);
        AssertCommitted(index.TryAdd(40, [1f, 1f, 0f]));
        model.Add(40, [1f, 1f, 0f]);
        AssertCommitted(index.TryDelete(20));
        model.Delete(20);

        float[] query = [0f, 0f, 0f];
        ulong[] scope = [10, 20, 30, 40, 40, 999];
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(scope);
        long beforeGeneration = index.Generation;
        VerifyAllSearchSurfaces(index, model, query, scope, topK: 10);

        Assert.Throws<IOException>(() => index.Checkpoint(fileTarget));

        Assert.Equal(beforeGeneration, index.Generation);
        Assert.Equal("caller-owned", File.ReadAllText(fileTarget));
        VerifyAllSearchSurfaces(index, model, query, scope, topK: 10);
        VerifyCandidateSet(index, model, query, scope, candidates, topK: 10);

        AssertCommitted(index.TryAdd(50, [2f, 0f, 0f]));
        model.Add(50, [2f, 0f, 0f]);
        AssertCommitted(index.TryDelete(30));
        model.Delete(30);
        VerifyAllSearchSurfaces(index, model, query, [10, 20, 30, 40, 50], topK: 10);
    }

    [Fact]
    public void Vec065_AllDeletedDeltaOnlyRowsPublishEmptyCheckpointAndRemainReserved()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);

        AssertCommitted(index.TryAdd(10, [1f]));
        AssertCommitted(index.TryAdd(20, [2f]));
        AssertCommitted(index.TryDelete(10));
        AssertCommitted(index.TryDelete(20));
        long beforeGeneration = index.Generation;

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(0, result.PhysicalVectorCount);
        Assert.Equal(0, result.LiveVectorCount);
        Assert.Equal(0, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(0, result.FoldedDeltaVectorCount);
        Assert.Equal(2, result.FoldedTombstoneCount);
        Assert.Equal(0, index.VectorCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(10, [10f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(20, [20f]).Status);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        var results = new SearchResult[5];
        Assert.Equal(0, opened.Search([0f], results));
        ExactFlatCandidateSet candidates = opened.CreateCandidateSet([10, 20, 30]);
        Assert.Equal(0, candidates.Count);
        Assert.Equal(0, opened.Search([0f], candidates, results));
    }

    [Fact]
    public void Vec065_CheckpointResultAndStatusSurfaceStayNarrow()
    {
        Assert.Equal(
            ["NoChanges", "Published"],
            Enum.GetNames<ExactFlatCheckpointStatus>().Order(StringComparer.Ordinal).ToArray());

        Assert.Equal(
            [
                "BaseVectorCount",
                "DeletedReservedIdCount",
                "DeltaVectorCount",
                "FoldedDeltaVectorCount",
                "FoldedTombstoneCount",
                "Generation",
                "LiveVectorCount",
                "PhysicalVectorCount",
                "Status",
                "TombstoneCount"
            ],
            typeof(ExactFlatCheckpointResult)
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Select(static property => property.Name)
                .Order(StringComparer.Ordinal)
                .ToArray());
    }

    private static void Add(ExactFlatIndex index, ReferenceModel model, ulong id, float[] vector)
    {
        index.Add(id, vector);
        model.Add(id, vector);
    }

    private static void VerifyAllSearchSurfaces(
        ExactFlatIndex index,
        ReferenceModel model,
        float[] query,
        ulong[] scope,
        int topK)
    {
        VerifyUnfiltered(index, model, query, topK);
        VerifyRawAllowlist(index, model, query, scope, topK);
        VerifyCandidateSet(index, model, query, scope, index.CreateCandidateSet(scope), topK);
    }

    private static void VerifyUnfiltered(ExactFlatIndex index, ReferenceModel model, float[] query, int topK)
    {
        var actual = new SearchResult[topK];
        int written = index.Search(query, actual);
        AssertResultsEqual(model.BruteForce(query, scope: null, topK), actual.AsSpan(0, written), model.Metric, model.Dimension);
    }

    private static void VerifyRawAllowlist(
        ExactFlatIndex index,
        ReferenceModel model,
        float[] query,
        ulong[] scope,
        int topK)
    {
        var actual = new SearchResult[topK];
        int written = index.Search(query, scope, actual, new ExactFlatSearchFilterWorkspace(index.VectorCount));
        AssertResultsEqual(model.BruteForce(query, scope, topK), actual.AsSpan(0, written), model.Metric, model.Dimension);
    }

    private static void VerifyCandidateSet(
        ExactFlatIndex index,
        ReferenceModel model,
        float[] query,
        ulong[] scope,
        ExactFlatCandidateSet candidates,
        int topK)
    {
        Assert.Equal(scope.Where(model.IsLive).Distinct().Count(), candidates.Count);

        var actual = new SearchResult[topK];
        int written = index.Search(query, candidates, actual);
        AssertResultsEqual(model.BruteForce(query, scope, topK), actual.AsSpan(0, written), model.Metric, model.Dimension);
    }

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

    private static float[] CreateQuery(VectorMetric metric, int dimension)
    {
        var query = new float[dimension];
        query[0] = 1f;
        for (int i = 1; i < dimension; i++)
        {
            query[i] = metric switch
            {
                VectorMetric.SquaredEuclidean => ((i % 5) - 2) * 0.125f,
                VectorMetric.InnerProduct => ((i & 1) == 0 ? 0.25f : -0.125f),
                VectorMetric.Cosine => ((i % 3) - 1) * 0.0625f,
                _ => throw new ArgumentOutOfRangeException(nameof(metric))
            };
        }

        return query;
    }

    private static float[] CreateVector(VectorMetric metric, int dimension, int salt, Random random)
    {
        var vector = new float[dimension];
        switch (metric)
        {
            case VectorMetric.SquaredEuclidean:
                for (int i = 0; i < dimension; i++)
                {
                    vector[i] = salt * 4f + i * 0.03125f + random.Next(0, 3) * 0.00390625f;
                }

                break;

            case VectorMetric.InnerProduct:
                vector[0] = salt + 1f;
                for (int i = 1; i < dimension; i++)
                {
                    vector[i] = ((salt + i) % 9 - 4) * 0.03125f;
                }

                break;

            case VectorMetric.Cosine:
                vector[0] = 1f;
                for (int i = 1; i < dimension; i++)
                {
                    vector[i] = ((salt * 3 + i) % 7 - 3) * 0.015625f;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(metric));
        }

        return vector;
    }

    private static void AssertResultsEqual(
        SearchResult[] expected,
        ReadOnlySpan<SearchResult> actual,
        VectorMetric metric,
        int dimension)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float tolerance = metric == VectorMetric.SquaredEuclidean
                ? D026Tolerance(dimension, expected[i].Distance)
                : 5e-5f;
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, tolerance);
        }
    }

    private static float D026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private sealed class ReferenceModel
    {
        private readonly HashSet<ulong> _deletedIds = [];

        public ReferenceModel(VectorMetric metric, int dimension)
        {
            Metric = metric;
            Dimension = dimension;
        }

        public VectorMetric Metric { get; }

        public int Dimension { get; }

        public List<Row> Rows { get; } = [];

        public int LiveCount => Rows.Count(row => !_deletedIds.Contains(row.Id));

        public int ReservedCount => _deletedIds.Count;

        public void Add(ulong id, float[] vector) => Rows.Add(new Row(id, vector.ToArray()));

        public void Delete(ulong id) => _deletedIds.Add(id);

        public bool IsLive(ulong id) => Rows.Any(row => row.Id == id) && !_deletedIds.Contains(id);

        public float[] RequireVector(ulong id) =>
            Rows.Single(row => row.Id == id).Vector.ToArray();

        public SearchResult[] BruteForce(float[] query, ulong[]? scope, int topK)
        {
            HashSet<ulong>? allowed = scope is null ? null : scope.ToHashSet();
            return Rows
                .Where(row => !_deletedIds.Contains(row.Id))
                .Where(row => allowed is null || allowed.Contains(row.Id))
                .Select(row => new SearchResult(row.Id, Distance(query, row.Vector)))
                .OrderBy(static result => result.Distance)
                .ThenBy(static result => result.Id)
                .Take(topK)
                .ToArray();
        }

        private float Distance(float[] query, float[] vector) =>
            Metric switch
            {
                VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
                VectorMetric.InnerProduct => InnerProduct(query, vector),
                VectorMetric.Cosine => Cosine(query, vector),
                _ => throw new ArgumentOutOfRangeException(nameof(Metric))
            };

        private static float SquaredEuclidean(float[] query, float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < query.Length; i++)
            {
                float difference = query[i] - vector[i];
                sum += difference * difference;
            }

            return sum;
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
            double queryMagnitudeSquared = 0;
            double vectorMagnitudeSquared = 0;
            double dot = 0;
            for (int i = 0; i < query.Length; i++)
            {
                queryMagnitudeSquared += (double)query[i] * query[i];
                vectorMagnitudeSquared += (double)vector[i] * vector[i];
                dot += (double)query[i] * vector[i];
            }

            return (float)(1 - dot / (Math.Sqrt(queryMagnitudeSquared) * Math.Sqrt(vectorMagnitudeSquared)));
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
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
    }
}
