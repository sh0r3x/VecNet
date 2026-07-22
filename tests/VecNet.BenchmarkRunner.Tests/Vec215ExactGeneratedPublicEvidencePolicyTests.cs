using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec215ExactGeneratedPublicEvidencePolicyTests
{
    [Fact]
    public void Evaluate_TopK100NearTieOrderOnlyClassifiesAcceptableAndKeepsOrderedAgreementDiagnostic()
    {
        const int topK = 100;
        const int queryCount = 10;
        DistanceTable distances = DistanceTable.Create(queryCount, id => id * 0.000001f);
        TruthSet truth = CreateTruth(queryCount, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount, topK, distances);
        (actual[0][98], actual[0][99]) = (actual[0][99], actual[0][98]);
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.True(validation.Acceptable, string.Join(Environment.NewLine, validation.Diagnostics));
        Assert.Equal("accepted-near-tie-order-only", validation.Status);
        Assert.Equal("deterministic-near-tie-order-only", validation.Classification);
        Assert.Equal(1.0, validation.RecallAtK);
        Assert.True(validation.OrderedAgreement < 1.0);
        Assert.Equal(strict.OrderedAgreement, validation.OrderedAgreement);
        Assert.Equal("passed", validation.DistanceToleranceStatus);
        Assert.Equal(0, validation.MissingResultCount);
        Assert.Equal(0, validation.DistanceMismatchCount);
        Assert.Equal(0, validation.WrongIdAwayFromNearTieCount);
        Assert.True(validation.OrderMismatchCount > 0);
    }

    [Fact]
    public void Evaluate_TopK100NearBoundaryRecallWithinAcceptedFloorClassifiesAcceptable()
    {
        const int topK = 100;
        const int queryCount = 10;
        DistanceTable distances = DistanceTable.Create(queryCount, id => id == 100 ? 99.0001f : id);
        TruthSet truth = CreateTruth(queryCount, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount, topK, distances);
        actual[0][99] = new SearchResult(100, distances.Get(0, 100));
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.Equal(0.999, validation.RecallAtK, precision: 12);
        Assert.True(validation.Acceptable, string.Join(Environment.NewLine, validation.Diagnostics));
        Assert.Equal("accepted-near-tie-order-only", validation.Status);
        Assert.Equal("passed", validation.DistanceToleranceStatus);
        Assert.True(validation.BoundaryNearTieMismatchCount > 0);
        Assert.Equal(0, validation.WrongIdAwayFromNearTieCount);
    }

    [Fact]
    public void Evaluate_DuplicateReturnedIdsFailPublicEvidencePolicy()
    {
        const int topK = 100;
        DistanceTable distances = DistanceTable.Create(queryCount: 1, id => id);
        TruthSet truth = CreateTruth(queryCount: 1, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount: 1, topK, distances);
        actual[0][99] = actual[0][98];
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.False(validation.Acceptable);
        Assert.Equal("failed", validation.Status);
        Assert.Equal(1, validation.DuplicateResultCount);
    }

    [Fact]
    public void Evaluate_RecallBelowAcceptedFloorFailsEvenWhenMismatchesAreNearTopKBoundary()
    {
        const int topK = 100;
        const int queryCount = 10;
        DistanceTable distances = DistanceTable.Create(queryCount, id => id == 100 ? 99.0001f : id);
        TruthSet truth = CreateTruth(queryCount, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount, topK, distances);
        actual[0][99] = new SearchResult(100, distances.Get(0, 100));
        actual[1][99] = new SearchResult(100, distances.Get(1, 100));
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.Equal(0.998, validation.RecallAtK, precision: 12);
        Assert.True(validation.RecallAtK < validation.AcceptedRecallFloor);
        Assert.False(validation.Acceptable, string.Join(Environment.NewLine, validation.Diagnostics));
        Assert.Equal("failed", validation.Status);
        Assert.True(validation.BoundaryNearTieMismatchCount > 0);
        Assert.Equal(0, validation.WrongIdAwayFromNearTieCount);
    }

    [Fact]
    public void Evaluate_MissingResultsStillFailPublicEvidencePolicy()
    {
        const int topK = 100;
        DistanceTable distances = DistanceTable.Create(queryCount: 1, id => id);
        TruthSet truth = CreateTruth(queryCount: 1, topK, distances);
        SearchResult[][] actual = [CreatePerfectActual(queryCount: 1, topK, distances)[0].Take(99).ToArray()];
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.False(validation.Acceptable);
        Assert.Equal("failed", validation.Status);
        Assert.Equal(1, validation.MissingResultCount);
    }

    [Fact]
    public void Evaluate_DistanceMismatchOutsideToleranceStillFailsPublicEvidencePolicy()
    {
        const int topK = 100;
        DistanceTable distances = DistanceTable.Create(queryCount: 1, id => id);
        TruthSet truth = CreateTruth(queryCount: 1, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount: 1, topK, distances);
        actual[0][5] = actual[0][5] with { Distance = 1000f };
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.False(validation.Acceptable);
        Assert.Equal("failed", validation.Status);
        Assert.Equal("failed", validation.DistanceToleranceStatus);
        Assert.Equal(1, validation.DistanceMismatchCount);
    }

    [Fact]
    public void Evaluate_ClearWrongOrderAwayFromNearTiesStillFailsPublicEvidencePolicy()
    {
        const int topK = 100;
        DistanceTable distances = DistanceTable.Create(queryCount: 1, id => id);
        TruthSet truth = CreateTruth(queryCount: 1, topK, distances);
        SearchResult[][] actual = CreatePerfectActual(queryCount: 1, topK, distances);
        (actual[0][0], actual[0][99]) = (actual[0][99], actual[0][0]);
        ResultComparison strict = ResultComparer.Compare(truth, actual, topK, dimension: 384, VectorMetric.SquaredEuclidean);

        ExactGeneratedPublicEvidenceValidationInfo validation = Evaluate(truth, actual, topK, distances, strict);

        Assert.Equal(1.0, validation.RecallAtK);
        Assert.True(validation.OrderedAgreement < 1.0);
        Assert.False(validation.Acceptable);
        Assert.Equal("failed", validation.Status);
        Assert.True(validation.WrongIdAwayFromNearTieCount > 0);
    }

    private static ExactGeneratedPublicEvidenceValidationInfo Evaluate(
        TruthSet truth,
        SearchResult[][] actual,
        int topK,
        DistanceTable distances,
        ResultComparison strict) =>
        ExactGeneratedPublicEvidencePolicy.Evaluate(
            truth,
            actual,
            VectorMetric.SquaredEuclidean,
            topK,
            dimension: 384,
            strict,
            distances.TryGet);

    private static TruthSet CreateTruth(int queryCount, int topK, DistanceTable distances)
    {
        var results = new TruthItem[queryCount][];
        for (int query = 0; query < queryCount; query++)
        {
            results[query] = Enumerable
                .Range(0, topK)
                .Select(id => new TruthItem((ulong)id, distances.Get(query, (ulong)id)))
                .ToArray();
        }

        return new TruthSet(results, topK);
    }

    private static SearchResult[][] CreatePerfectActual(int queryCount, int topK, DistanceTable distances)
    {
        var results = new SearchResult[queryCount][];
        for (int query = 0; query < queryCount; query++)
        {
            results[query] = Enumerable
                .Range(0, topK)
                .Select(id => new SearchResult((ulong)id, distances.Get(query, (ulong)id)))
                .ToArray();
        }

        return results;
    }

    private sealed class DistanceTable
    {
        private readonly Dictionary<(int Query, ulong Id), float> _distances;

        private DistanceTable(Dictionary<(int Query, ulong Id), float> distances)
        {
            _distances = distances;
        }

        public static DistanceTable Create(int queryCount, Func<ulong, float> distanceFactory)
        {
            var distances = new Dictionary<(int Query, ulong Id), float>();
            for (int query = 0; query < queryCount; query++)
            {
                for (ulong id = 0; id <= 100; id++)
                {
                    distances[(query, id)] = distanceFactory(id);
                }
            }

            return new DistanceTable(distances);
        }

        public float Get(int query, ulong id) => _distances[(query, id)];

        public float? TryGet(int query, ulong id) =>
            _distances.TryGetValue((query, id), out float distance) ? distance : null;
    }
}
