using System.Numerics;

namespace VecNet.Tests;

public sealed class ExactFlatIndexMutationIndependentTests
{
    public static TheoryData<VectorMetric, int> RandomizedMutationCases()
    {
        int vectorWidth = Vector<float>.Count;
        int[] dimensions =
        [
            1,
            2,
            3,
            Math.Max(1, vectorWidth - 1),
            vectorWidth,
            vectorWidth + 1,
            32,
            96,
            128,
            386,
            768
        ];

        var data = new TheoryData<VectorMetric, int>();
        foreach (VectorMetric metric in Enum.GetValues<VectorMetric>())
        {
            foreach (int dimension in dimensions.Distinct())
            {
                data.Add(metric, dimension);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(RandomizedMutationCases))]
    public void Vec059_RandomizedMutationSequencesMatchBruteForceLiveRows(
        VectorMetric metric,
        int dimension)
    {
        int seed = unchecked((int)(0x5900_0000u + ((uint)metric * 1_000_003u) + (uint)dimension));
        var random = new Random(seed);
        var index = new ExactFlatIndex(dimension, metric);
        var model = new ReferenceModel(metric, dimension);

        ulong[] initialIds =
        [
            0,
            1,
            2,
            17,
            99,
            ulong.MaxValue - 1,
            ulong.MaxValue
        ];

        for (int i = 0; i < initialIds.Length; i++)
        {
            float[] vector = CreateVector(metric, dimension, i, random);
            if (i == 2)
            {
                vector = CreateEqualDistanceVector(metric, dimension);
            }

            index.Add(initialIds[i], vector);
            model.Add(initialIds[i], vector);
        }

        VerifyCurrentView(index, model, random, topK: 11);

        int operationCount = dimension >= 386 ? 32 : 56;
        for (int operation = 0; operation < operationCount; operation++)
        {
            ExactFlatCandidateSet staleCandidateSet = index.CreateCandidateSet(model.LiveIds().Concat([123456789UL]).ToArray());
            var staleDestination = new[] { new SearchResult(777, 888) };
            long beforeGeneration = index.Generation;
            int selector = random.Next(10);

            if (selector <= 4)
            {
                ulong id = SelectInsertId(operation, random, model);
                float[] vector = selector == 4 && model.Rows.Count != 0
                    ? model.Rows[random.Next(model.Rows.Count)].Vector.ToArray()
                    : CreateVector(metric, dimension, operation + 100, random);

                VectorMutationResult result = index.TryAdd(id, vector);
                if (model.IsKnownOrReserved(id))
                {
                    Assert.Equal(VectorMutationStatus.DuplicateId, result.Status);
                    Assert.Equal(beforeGeneration, result.Generation);
                    Assert.Equal(beforeGeneration, index.Generation);
                }
                else
                {
                    Assert.Equal(VectorMutationStatus.Committed, result.Status);
                    Assert.Equal(beforeGeneration + 1, result.Generation);
                    Assert.Equal(result.Generation, index.Generation);
                    model.Add(id, vector);
                    AssertStaleCandidateSetRejected(index, staleCandidateSet, staleDestination, metric, dimension);
                }
            }
            else
            {
                ulong id = SelectDeleteId(selector, random, model);
                VectorMutationResult result = index.TryDelete(id);
                if (model.IsLive(id))
                {
                    Assert.Equal(VectorMutationStatus.Committed, result.Status);
                    Assert.Equal(beforeGeneration + 1, result.Generation);
                    Assert.Equal(result.Generation, index.Generation);
                    model.Delete(id);
                    AssertStaleCandidateSetRejected(index, staleCandidateSet, staleDestination, metric, dimension);
                }
                else if (model.IsReserved(id))
                {
                    Assert.Equal(VectorMutationStatus.AlreadyDeleted, result.Status);
                    Assert.Equal(beforeGeneration, result.Generation);
                    Assert.Equal(beforeGeneration, index.Generation);
                }
                else
                {
                    Assert.Equal(VectorMutationStatus.UnknownId, result.Status);
                    Assert.Equal(beforeGeneration, result.Generation);
                    Assert.Equal(beforeGeneration, index.Generation);
                }
            }

            VerifyCurrentView(index, model, random, topK: random.Next(0, 15));
        }

        foreach (ulong id in model.LiveIds().ToArray())
        {
            ExactFlatCandidateSet staleCandidateSet = index.CreateCandidateSet(model.LiveIds().ToArray());
            var staleDestination = new[] { new SearchResult(999, 111) };
            long beforeGeneration = index.Generation;

            VectorMutationResult result = index.TryDelete(id);

            Assert.Equal(VectorMutationStatus.Committed, result.Status);
            Assert.Equal(beforeGeneration + 1, result.Generation);
            model.Delete(id);
            AssertStaleCandidateSetRejected(index, staleCandidateSet, staleDestination, metric, dimension);
        }

        VerifyCurrentView(index, model, random, topK: 8);
        Assert.Empty(model.LiveRows());
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(initialIds[0]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(initialIds[0], CreateVector(metric, dimension, 9000, random)).Status);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Vec059_SaveOpenUsesLiveViewAfterTombstonesAndDeltaWithoutCheckpoint(VectorMetric metric)
    {
        const int dimension = 5;
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var random = new Random(0x5905_A0E + (int)metric);
        var index = new ExactFlatIndex(dimension, metric);
        var model = new ReferenceModel(metric, dimension);

        (ulong Id, float[] Vector)[] rows =
        [
            (0, CreateEqualDistanceVector(metric, dimension)),
            (1, CreateEqualDistanceVector(metric, dimension)),
            (42, CreateVector(metric, dimension, 42, random)),
            (ulong.MaxValue - 7, CreateVector(metric, dimension, 77, random))
        ];

        foreach ((ulong id, float[] vector) in rows)
        {
            index.Add(id, vector);
            model.Add(id, vector);
        }

        float[] deltaVector = CreateVector(metric, dimension, 100, random);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(ulong.MaxValue, deltaVector).Status);
        model.Add(ulong.MaxValue, deltaVector);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(1).Status);
        model.Delete(1);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(42).Status);
        model.Delete(42);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(42).Status);
        Assert.Equal(VectorMutationStatus.UnknownId, index.TryDelete(123456789).Status);

        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        VerifySpecificScope(opened, model, [0, 1, 42, ulong.MaxValue, ulong.MaxValue - 7, 123456789], topK: 10);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryAdd(123, CreateVector(metric, dimension, 123, random)).Status);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryDelete(0).Status);
        Assert.Throws<InvalidOperationException>(() => opened.Add(123, CreateVector(metric, dimension, 124, random)));
    }

    [Fact]
    public void Vec059_CandidateSetsRejectWrongIndexAndStaleAfterInsertAndDeleteBeforeWriting()
    {
        var first = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        first.Add(10, [1f]);
        var second = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        second.Add(10, [1f]);

        ExactFlatCandidateSet wrongOwner = second.CreateCandidateSet([10]);
        var wrongOwnerDestination = new[] { new SearchResult(1, 1) };

        Assert.Throws<InvalidOperationException>(() => first.Search([0f], wrongOwner, wrongOwnerDestination));
        Assert.Equal(new SearchResult(1, 1), wrongOwnerDestination[0]);

        ExactFlatCandidateSet staleAfterInsert = first.CreateCandidateSet([10]);
        var insertDestination = new[] { new SearchResult(2, 2) };
        Assert.Equal(VectorMutationStatus.Committed, first.TryAdd(20, [0.5f]).Status);
        Assert.Throws<InvalidOperationException>(() => first.Search([0f], staleAfterInsert, insertDestination));
        Assert.Equal(new SearchResult(2, 2), insertDestination[0]);

        ExactFlatCandidateSet staleAfterDelete = first.CreateCandidateSet([10, 20]);
        var deleteDestination = new[] { new SearchResult(3, 3) };
        Assert.Equal(VectorMutationStatus.Committed, first.TryDelete(20).Status);
        Assert.Throws<InvalidOperationException>(() => first.Search([0f], staleAfterDelete, deleteDestination));
        Assert.Equal(new SearchResult(3, 3), deleteDestination[0]);
    }

    [Fact]
    public void Vec059_SaveOpenAllTombstonedIndexHasEmptyLiveView()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(0, [1f, 0f]);
        index.Add(ulong.MaxValue, [0f, 1f]);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(0).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(ulong.MaxValue).Status);

        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        var results = new SearchResult[4];
        Assert.Equal(0, opened.Search([0f, 0f], results));
        Assert.Equal(0, opened.Search(
            [0f, 0f],
            [0, ulong.MaxValue, 999],
            results,
            new ExactFlatSearchFilterWorkspace(opened.VectorCount)));
        ExactFlatCandidateSet candidates = opened.CreateCandidateSet([0, ulong.MaxValue, 999]);
        Assert.Equal(0, candidates.Count);
        Assert.Equal(0, opened.Search([0f, 0f], candidates, results));
    }

    private static void VerifyCurrentView(
        ExactFlatIndex index,
        ReferenceModel model,
        Random random,
        int topK)
    {
        float[] query = CreateVector(model.Metric, model.Dimension, 50_000 + topK, random);
        VerifyUnfiltered(index, model, query, topK);

        ulong[] scope = CreateAdversarialScope(model, random);
        VerifyRawAllowlist(index, model, query, scope, topK);
        VerifyCandidateSet(index, model, query, scope, topK);
    }

    private static void VerifySpecificScope(
        ExactFlatIndex index,
        ReferenceModel model,
        ulong[] scope,
        int topK)
    {
        float[] query = CreateEqualDistanceVector(model.Metric, model.Dimension);
        VerifyUnfiltered(index, model, query, topK);
        VerifyRawAllowlist(index, model, query, scope, topK);
        VerifyCandidateSet(index, model, query, scope, topK);
    }

    private static void VerifyUnfiltered(
        ExactFlatIndex index,
        ReferenceModel model,
        float[] query,
        int topK)
    {
        var actual = new SearchResult[topK];
        int written = index.Search(query, actual);
        SearchResult[] expected = model.BruteForce(query, scope: null, topK);
        AssertResultsEqual(expected, actual.AsSpan(0, written), model.Metric);
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
        SearchResult[] expected = model.BruteForce(query, scope, topK);
        AssertResultsEqual(expected, actual.AsSpan(0, written), model.Metric);
    }

    private static void VerifyCandidateSet(
        ExactFlatIndex index,
        ReferenceModel model,
        float[] query,
        ulong[] scope,
        int topK)
    {
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(scope);
        Assert.Equal(scope.Where(model.IsLive).Distinct().Count(), candidates.Count);

        var actual = new SearchResult[topK];
        int written = index.Search(query, candidates, actual);
        SearchResult[] expected = model.BruteForce(query, scope, topK);
        AssertResultsEqual(expected, actual.AsSpan(0, written), model.Metric);
    }

    private static void AssertStaleCandidateSetRejected(
        ExactFlatIndex index,
        ExactFlatCandidateSet staleCandidateSet,
        SearchResult[] destination,
        VectorMetric metric,
        int dimension)
    {
        SearchResult sentinel = destination[0];
        float[] query = CreateEqualDistanceVector(metric, dimension);

        Assert.Throws<InvalidOperationException>(() => index.Search(query, staleCandidateSet, destination));
        Assert.Equal(sentinel, destination[0]);
    }

    private static ulong SelectInsertId(int operation, Random random, ReferenceModel model)
    {
        if (operation % 11 == 0)
        {
            return 0;
        }

        if (operation % 13 == 0)
        {
            return ulong.MaxValue;
        }

        if (operation % 7 == 0 && model.Rows.Count != 0)
        {
            return model.Rows[random.Next(model.Rows.Count)].Id;
        }

        return ((ulong)operation << 32) | (uint)random.Next(1, int.MaxValue);
    }

    private static ulong SelectDeleteId(int selector, Random random, ReferenceModel model)
    {
        if (selector == 5 && model.LiveIds().Any())
        {
            ulong[] liveIds = model.LiveIds().ToArray();
            return liveIds[random.Next(liveIds.Length)];
        }

        if (selector <= 7 && model.ReservedIds().Any())
        {
            ulong[] reservedIds = model.ReservedIds().ToArray();
            return reservedIds[random.Next(reservedIds.Length)];
        }

        return random.Next(0, 2) == 0 ? 123456789UL : ulong.MaxValue - 1234UL;
    }

    private static ulong[] CreateAdversarialScope(ReferenceModel model, Random random)
    {
        var scope = new List<ulong>
        {
            0,
            0,
            1,
            ulong.MaxValue,
            ulong.MaxValue,
            ulong.MaxValue - 1,
            123456789,
            ulong.MaxValue - 1234
        };

        foreach (ulong id in model.LiveIds().OrderBy(_ => random.Next()).Take(6))
        {
            scope.Add(id);
            scope.Add(id);
        }

        foreach (ulong id in model.ReservedIds().OrderBy(_ => random.Next()).Take(4))
        {
            scope.Add(id);
        }

        return scope.OrderBy(_ => random.Next()).ToArray();
    }

    private static float[] CreateVector(VectorMetric metric, int dimension, int salt, Random random)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            int value = ((salt + i * 17 + random.Next(0, 7)) % 7) - 3;
            vector[i] = value;
        }

        if (metric == VectorMetric.Cosine && vector.All(static value => value == 0))
        {
            vector[0] = 1;
        }

        if (metric != VectorMetric.Cosine && salt % 19 == 0)
        {
            Array.Fill(vector, 0f);
        }

        return vector;
    }

    private static float[] CreateEqualDistanceVector(VectorMetric metric, int dimension)
    {
        var vector = new float[dimension];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = i % 2 == 0 ? 1f : -1f;
        }

        if (metric == VectorMetric.Cosine && dimension == 1)
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static void AssertResultsEqual(
        SearchResult[] expected,
        ReadOnlySpan<SearchResult> actual,
        VectorMetric metric)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float tolerance = metric == VectorMetric.SquaredEuclidean ? 5e-4f : 5e-5f;
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, tolerance);
        }
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

        public void Add(ulong id, float[] vector) => Rows.Add(new Row(id, vector.ToArray()));

        public void Delete(ulong id) => _deletedIds.Add(id);

        public bool IsKnownOrReserved(ulong id) => Rows.Any(row => row.Id == id) || _deletedIds.Contains(id);

        public bool IsLive(ulong id) => Rows.Any(row => row.Id == id) && !_deletedIds.Contains(id);

        public bool IsReserved(ulong id) => _deletedIds.Contains(id);

        public IEnumerable<ulong> LiveIds() => LiveRows().Select(static row => row.Id);

        public IEnumerable<ulong> ReservedIds() => _deletedIds;

        public IEnumerable<Row> LiveRows() => Rows.Where(row => !_deletedIds.Contains(row.Id));

        public SearchResult[] BruteForce(float[] query, ulong[]? scope, int topK)
        {
            HashSet<ulong>? allowed = scope is null ? null : scope.ToHashSet();
            return LiveRows()
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
            for (int i = 0; i < query.Length; i++)
            {
                queryMagnitudeSquared += (double)query[i] * query[i];
            }

            double vectorMagnitudeSquared = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                vectorMagnitudeSquared += (double)vector[i] * vector[i];
            }

            double vectorMagnitude = Math.Sqrt(vectorMagnitudeSquared);
            double queryMagnitude = Math.Sqrt(queryMagnitudeSquared);
            double dot = 0;
            for (int i = 0; i < query.Length; i++)
            {
                float stored = (float)(vector[i] / vectorMagnitude);
                dot += stored * (query[i] / queryMagnitude);
            }

            return (float)(1 - dot);
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
