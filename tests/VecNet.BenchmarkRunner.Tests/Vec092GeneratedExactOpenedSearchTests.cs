using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec092GeneratedExactOpenedSearchTests
{
    [Fact]
    public void ParseGeneratedExactOpenedSearch_UsesPrivateSmokeDefaults()
    {
        GeneratedExactOpenedSearchOptions options = CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2092u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.IndexDirectory);
        Assert.False(Path.IsPathRooted(options.IndexDirectory));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-opened-search", "--dimension")]
    [InlineData("generated-exact-opened-search", "dimension", "8")]
    [InlineData("generated-exact-opened-search", "--metric", "Unknown")]
    [InlineData("generated-exact-opened-search", "--dimension", "0")]
    [InlineData("generated-exact-opened-search", "--vectors", "0")]
    [InlineData("generated-exact-opened-search", "--queries", "0")]
    [InlineData("generated-exact-opened-search", "--top-k", "11", "--vectors", "10")]
    [InlineData("generated-exact-opened-search", "--runs", "0")]
    [InlineData("generated-exact-opened-search", "--runs", "6")]
    [InlineData("generated-exact-opened-search", "--warmup-queries", "-1")]
    [InlineData("generated-exact-opened-search", "--seed", "0xNOTHEX")]
    [InlineData("generated-exact-opened-search", "--output", "")]
    [InlineData("generated-exact-opened-search", "--index-directory", "")]
    [InlineData("generated-exact-opened-search", "--insertions", "1")]
    [InlineData("generated-exact-opened-search", "--deletes", "1")]
    [InlineData("generated-exact-opened-search", "--checkpoint-directory", "checkpoint")]
    [InlineData("generated-exact-opened-search", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-opened-search", "--preset", "smoke")]
    [InlineData("generated-exact-opened-search", "--filter", "broad")]
    public void ParseGeneratedExactOpenedSearch_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactOpenedSearch(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Run_MeasuresOpenedReadOnlySearchAllocationAndKeepsSetupExcluded(VectorMetric metric)
    {
        string outputPath = NewArtifactPath("opened-search-report.json");
        string indexDirectory = Path.Combine(Path.GetDirectoryName(outputPath)!, "saved-index");
        string[] arguments =
        [
            "generated-exact-opened-search",
            "--metric", metric.ToString(),
            "--dimension", "11",
            "--vectors", "30",
            "--queries", "5",
            "--top-k", "6",
            "--runs", "3",
            "--warmup-queries", "2",
            "--seed", "0x5EED092A",
            "--output", outputPath,
            "--index-directory", indexDirectory
        ];
        GeneratedExactOpenedSearchOptions options = CommandLine.ParseGeneratedExactOpenedSearch(arguments);

        GeneratedExactOpenedSearchBenchmarkReport report = GeneratedExactOpenedSearchScenario.Run(options, arguments);
        GeneratedExactOpenedSearchScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.True(File.Exists(Path.Combine(indexDirectory, "exact-flat.manifest.json")));
        Assert.Equal("VecNet.ExactOpenedReadOnlySearchAllocationReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-092", report.TaskId);
        Assert.Equal("generated-exact-opened-search", report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-opened-read-only-search-allocation-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonArtifactEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal(11, report.Dataset.Dimension);
        Assert.Equal(30, report.Dataset.VectorCount);
        Assert.Equal(5, report.Dataset.QueryCount);
        Assert.Equal("scalar-reference-generated", report.Truth.Kind);
        Assert.Equal(nameof(ExactFlatIndex), report.Index.Type);
        Assert.Contains("OpenReadOnly", report.Index.Configuration, StringComparison.Ordinal);
        Assert.Equal(indexDirectory, report.Lifecycle.SavedIndexDirectoryPath);
        Assert.True(report.Lifecycle.SourceIndexBuiltBeforeMeasurement);
        Assert.True(report.Lifecycle.SavedBeforeMeasurement);
        Assert.True(report.Lifecycle.OpenedReadOnlyBeforeMeasurement);
        Assert.True(report.Lifecycle.CallerOwnedResultBuffers);

        Assert.Equal(5, report.OpenedReadOnlySearch.MeasuredQueryCount);
        Assert.Equal(3, report.OpenedReadOnlySearch.Runs.Length);
        Assert.Equal(3, report.OpenedReadOnlySearch.Aggregate.RunCount);
        Assert.All(report.OpenedReadOnlySearch.Runs, run =>
        {
            Assert.Equal(5, run.MeasuredQueryCount);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.ManagedAllocatedBytes >= 0);
            Assert.True(run.ManagedAllocatedBytesPerQuery >= 0);
        });
        Assert.Equal(
            report.OpenedReadOnlySearch.Runs.Average(run => run.ManagedAllocatedBytesPerQuery),
            report.OpenedReadOnlySearch.Aggregate.MeanManagedAllocatedBytesPerQuery);

        Assert.Equal("measured", report.Measurement.OpenedReadOnlySearchLatency.Status);
        Assert.Equal("perOpenedReadOnlySearchCall", report.Measurement.OpenedReadOnlySearchLatency.SampleScope);
        Assert.Contains("OpenReadOnly", report.Measurement.OpenedReadOnlySearchLatency.TimedOperation, StringComparison.Ordinal);
        Assert.Contains("Save", report.Measurement.OpenedReadOnlySearchLatency.ExcludedOperations, StringComparison.Ordinal);
        Assert.Equal("measured", report.Measurement.OpenedReadOnlySearchManagedAllocations.Status);
        Assert.Equal("bytesPerOpenedReadOnlySearchCall", report.Measurement.OpenedReadOnlySearchManagedAllocations.Unit);
        Assert.True(double.Parse(report.Measurement.OpenedReadOnlySearchManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Contains("GC.GetAllocatedBytesForCurrentThread", report.Measurement.OpenedReadOnlySearchManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.SourceIndexConstruction.Status);
        Assert.Equal("notMeasured", report.Measurement.Save.Status);
        Assert.Equal("notMeasured", report.Measurement.OpenReadOnly.Status);
        Assert.Equal("notMeasured", report.Measurement.TruthConstruction.Status);
        Assert.Equal("notMeasured", report.Measurement.Validation.Status);
        Assert.Equal("notMeasured", report.Measurement.ResultCaptureComparison.Status);
        Assert.Equal("notMeasured", report.Measurement.ReportWriting.Status);
        Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Contains("Save", report.Measurement.SharedExcludedOperations, StringComparison.Ordinal);
        Assert.Contains("OpenReadOnly", report.Measurement.SharedExcludedOperations, StringComparison.Ordinal);
        Assert.Contains("warmup", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.OpenedReadOnlySearchComparedToTruth);
        Assert.True(report.Validation.SaveOpenSetupExcludedFromMeasurement);
        Assert.True(report.Validation.WarmupExcludedFromMeasurement);
        Assert.True(report.Validation.ResultCaptureComparisonExcludedFromMeasurement);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactOpenedReadOnlySearchAllocationReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-092", root.GetProperty("taskId").GetString());
        Assert.Equal("generated-exact-opened-search", root.GetProperty("scenarioName").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("openedReadOnlySearchManagedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerOpenedReadOnlySearchCall", root.GetProperty("measurement").GetProperty("openedReadOnlySearchManagedAllocations").GetProperty("unit").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("save").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("openReadOnly").GetProperty("status").GetString());
        Assert.True(root.GetProperty("lifecycle").GetProperty("savedBeforeMeasurement").GetBoolean());
        Assert.True(root.GetProperty("lifecycle").GetProperty("openedReadOnlyBeforeMeasurement").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndOpenedSearchModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search", "--vectors", "12", "--queries", "1", "--top-k", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search", "--insertions", "2"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--index-directory", "index"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--index-directory", "index"]));
        Assert.Equal("generated-exact-opened-search", GeneratedExactOpenedSearchOptions.ScenarioName);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec092-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
