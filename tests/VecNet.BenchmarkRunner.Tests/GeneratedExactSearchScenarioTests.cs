using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class GeneratedExactSearchScenarioTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Run_ProducesPassingPrivateReportForGeneratedExactScenario(VectorMetric metric)
    {
        string[] arguments =
        [
            "exact-generated",
            "--metric", metric.ToString(),
            "--dimension", "11",
            "--vectors", "17",
            "--queries", "5",
            "--top-k", "6",
            "--seed", "0x00001010",
            "--output", "VecNet.BenchmarkRunner.Artifacts/test-report.json"
        ];
        var options = new GeneratedExactSearchOptions(
            metric,
            Dimension: 11,
            VectorCount: 17,
            QueryCount: 5,
            TopK: 6,
            Seed: 0x1010,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-report.json",
            BaselineReportId: "baseline-smoke");

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, arguments);

        Assert.Equal("VecNet.BenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-012", report.TaskId);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("local-evidence", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.NotEmpty(report.Evidence.Limitations);
        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal("scalar-reference-generated", report.Truth.Kind);
        Assert.Equal(6, report.Truth.Depth);
        Assert.Equal("exact-generated", report.Command.Scenario);
        Assert.Equal(arguments, report.Command.Arguments);
        Assert.Equal(5, report.Scenario.MeasuredQueryCount);
        Assert.Equal(1, report.Scenario.Concurrency);
        Assert.Contains("excluded", report.Scenario.ExcludedFromSearchTiming, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(nameof(ExactFlatIndex), report.Index.Type);
        Assert.Equal(5, report.Search.MeasuredQueryCount);
        Assert.True(report.Search.ElapsedMilliseconds >= 0);
        Assert.True(report.Search.LatencyP50Milliseconds >= 0);
        Assert.True(report.Search.LatencyP95Milliseconds >= report.Search.LatencyP50Milliseconds);
        Assert.True(report.Search.LatencyP99Milliseconds >= report.Search.LatencyP95Milliseconds);
        Assert.True(report.Search.Qps > 0);
        Assert.Single(report.Search.Runs);
        Assert.Equal(1, report.Search.Aggregate.RunCount);
        Assert.Equal(5, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal(report.Search.Runs[0].ElapsedMilliseconds, report.Search.Aggregate.MeanElapsedMilliseconds);
        Assert.Equal("notMeasured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("absent", report.Measurement.ManagedAllocations.Value);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("singleRun", report.Measurement.RepeatedRuns.Status);
        Assert.Equal(1, report.Measurement.RepeatedRuns.RunCount);
        Assert.False(report.Measurement.RepeatedRuns.VarianceMeasured);
        Assert.Equal("absent", report.Measurement.Warmup.Status);
        Assert.Equal(0, report.Measurement.Warmup.WarmupCount);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(0, report.Metrics.MissingResultCount);
        Assert.Equal("baseline-smoke", report.Baseline.BaselineReportId);
        Assert.Equal("smoke", report.Baseline.Suitability);
        Assert.False(report.Baseline.BaselineCandidateEligible);
        Assert.False(report.Baseline.RegressionGateEligible);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("smoke", report.Validation.EvidenceStatus);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.TruthGenerated);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
    }

    [Fact]
    public void Run_WithMultipleRunsAndWarmup_RecordsPerRunAndAggregateTimingMetadata()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 9,
            VectorCount: 23,
            QueryCount: 4,
            TopK: 5,
            Seed: 0x5EED012A,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/multi-run-test.json",
            BaselineReportId: null,
            Runs: 3,
            WarmupQueries: 2);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--runs", "3",
                "--warmup-queries", "2"
            ]);

        Assert.Equal("VEC-012", report.TaskId);
        Assert.Equal(4, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal(4, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal<int>([1, 2, 3], report.Search.Runs.Select(run => run.RunNumber).ToArray());
        Assert.All(report.Search.Runs, run =>
        {
            Assert.Equal(4, run.MeasuredQueryCount);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.LatencyP50Milliseconds >= 0);
            Assert.True(run.LatencyP95Milliseconds >= run.LatencyP50Milliseconds);
            Assert.True(run.LatencyP99Milliseconds >= run.LatencyP95Milliseconds);
            Assert.True(run.Qps > 0);
        });

        Assert.Equal(
            report.Search.Runs.Average(run => run.ElapsedMilliseconds),
            report.Search.Aggregate.MeanElapsedMilliseconds);
        Assert.Equal(
            report.Search.Runs.Min(run => run.ElapsedMilliseconds),
            report.Search.Aggregate.MinElapsedMilliseconds);
        Assert.Equal(
            report.Search.Runs.Max(run => run.ElapsedMilliseconds),
            report.Search.Aggregate.MaxElapsedMilliseconds);
        Assert.Equal(report.Search.Aggregate.MeanElapsedMilliseconds, report.Search.ElapsedMilliseconds);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal(3, report.Measurement.RepeatedRuns.RunCount);
        Assert.True(report.Measurement.RepeatedRuns.VarianceMeasured);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(2, report.Measurement.Warmup.WarmupCount);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Baseline.RegressionGateEligible);
        Assert.Equal("passed", report.Validation.Status);
    }

    [Fact]
    public void Run_ReportIdIncludesScenarioParametersButNotOutputPath()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 3,
            VectorCount: 4,
            QueryCount: 2,
            TopK: 4,
            Seed: 0xCAFE,
            OutputPath: @"C:\private\owner\absolute-report.json",
            BaselineReportId: null);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);

        Assert.Contains("SquaredEuclidean-3d-4v-2q-4k-1r-0w-0000CAFE", report.ReportId);
        Assert.DoesNotContain("private", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("absolute-report", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("private-raw", report.PrivacyClass);
    }
}
