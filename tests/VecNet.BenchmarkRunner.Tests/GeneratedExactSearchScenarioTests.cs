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
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-report.json");

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, arguments);

        Assert.Equal("VecNet.BenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-010", report.TaskId);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
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
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(0, report.Metrics.MissingResultCount);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.TruthGenerated);
        Assert.True(report.Validation.ReportIsPrivateRaw);
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
            OutputPath: @"C:\private\owner\absolute-report.json");

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);

        Assert.Contains("SquaredEuclidean-3d-4v-2q-4k-0000CAFE", report.ReportId);
        Assert.DoesNotContain("private", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("absolute-report", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("private-raw", report.PrivacyClass);
    }
}
