using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class ResultComparerTests
{
    [Fact]
    public void Compare_ComputesRecallAndOrderedAgreementSeparately()
    {
        var truth = new TruthSet(
            [
                [new TruthItem(1, 0f), new TruthItem(2, 1f)],
                [new TruthItem(3, 0f), new TruthItem(4, 1f)]
            ],
            depth: 2);
        SearchResult[][] actual =
        [
            [new SearchResult(1, 0f), new SearchResult(2, 1f)],
            [new SearchResult(4, 1f), new SearchResult(3, 0f)]
        ];

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            actual,
            topK: 2,
            dimension: 2,
            VectorMetric.SquaredEuclidean);

        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.5, comparison.OrderedAgreement);
        Assert.Equal("passed", comparison.DistanceToleranceStatus);
        Assert.Equal(0, comparison.DistanceMismatchCount);
        Assert.Equal(0, comparison.MissingResultCount);
    }

    [Fact]
    public void Compare_ReportsMissingResultsAsValidationFailure()
    {
        var truth = new TruthSet(
            [
                [new TruthItem(10, 0f), new TruthItem(11, 1f), new TruthItem(12, 2f)]
            ],
            depth: 3);
        SearchResult[][] actual =
        [
            [new SearchResult(10, 0f)]
        ];

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            actual,
            topK: 3,
            dimension: 2,
            VectorMetric.SquaredEuclidean);

        Assert.Equal(1.0 / 3.0, comparison.RecallAtK, precision: 12);
        Assert.Equal(1.0 / 3.0, comparison.OrderedAgreement, precision: 12);
        Assert.Equal("failed", comparison.DistanceToleranceStatus);
        Assert.Equal(0, comparison.DistanceMismatchCount);
        Assert.Equal(2, comparison.MissingResultCount);
    }

    [Fact]
    public void Compare_FailsDistanceToleranceEvenWhenIdsAndOrderMatch()
    {
        var truth = new TruthSet(
            [
                [new TruthItem(20, 1f)]
            ],
            depth: 1);
        SearchResult[][] actual =
        [
            [new SearchResult(20, 1.00002f)]
        ];

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            actual,
            topK: 1,
            dimension: 2,
            VectorMetric.InnerProduct);

        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(1.0, comparison.OrderedAgreement);
        Assert.Equal("failed", comparison.DistanceToleranceStatus);
        Assert.Equal(1, comparison.DistanceMismatchCount);
        Assert.Equal(0, comparison.MissingResultCount);
    }

    [Fact]
    public void Compare_UsesRequestedTopKAsRecallDenominatorWhenTruthIsDeeper()
    {
        var truth = new TruthSet(
            [
                [
                    new TruthItem(1, 0f),
                    new TruthItem(2, 1f),
                    new TruthItem(3, 2f),
                    new TruthItem(4, 3f)
                ]
            ],
            depth: 4);
        SearchResult[][] actual =
        [
            [
                new SearchResult(2, 1f),
                new SearchResult(1, 0f),
                new SearchResult(99, 9f)
            ]
        ];

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            actual,
            topK: 3,
            dimension: 2,
            VectorMetric.SquaredEuclidean);

        Assert.Equal(2.0 / 3.0, comparison.RecallAtK, precision: 12);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal("passed", comparison.DistanceToleranceStatus);
        Assert.Equal(0, comparison.DistanceMismatchCount);
        Assert.Equal(0, comparison.MissingResultCount);
    }
}
