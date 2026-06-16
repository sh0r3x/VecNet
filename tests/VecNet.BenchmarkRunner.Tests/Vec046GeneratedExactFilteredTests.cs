using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec046GeneratedExactFilteredTests
{
    [Fact]
    public void ParseGeneratedExactFiltered_UsesPrivateDefaults()
    {
        GeneratedExactFilteredOptions options = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal("broad", options.FilterKind);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2046u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("exact-generated-filtered", "--dimension")]
    [InlineData("exact-generated-filtered", "dimension", "8")]
    [InlineData("exact-generated-filtered", "--metric", "Unknown")]
    [InlineData("exact-generated-filtered", "--dimension", "0")]
    [InlineData("exact-generated-filtered", "--vectors", "0")]
    [InlineData("exact-generated-filtered", "--queries", "0")]
    [InlineData("exact-generated-filtered", "--top-k", "3", "--vectors", "2")]
    [InlineData("exact-generated-filtered", "--runs", "0")]
    [InlineData("exact-generated-filtered", "--runs", "6")]
    [InlineData("exact-generated-filtered", "--warmup-queries", "-1")]
    [InlineData("exact-generated-filtered", "--filter", "unknown")]
    [InlineData("exact-generated-filtered", "--filter", "very-selective", "--top-k", "1")]
    [InlineData("exact-generated-filtered", "--duplicate-ids", "-1")]
    [InlineData("exact-generated-filtered", "--unknown-ids", "-1")]
    [InlineData("exact-generated-filtered", "--baseline", "old.json")]
    [InlineData("exact-generated-filtered", "--preset", "smoke")]
    [InlineData("exact-generated-filtered", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("exact-generated-filtered", "--output", "")]
    public void ParseGeneratedExactFiltered_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedExactFilteredReport()
    {
        string outputPath = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec046-direct-" + Guid.NewGuid().ToString("N") + ".json");
        string[] arguments =
        [
            "exact-generated-filtered",
            "--metric", "SquaredEuclidean",
            "--dimension", "13",
            "--vectors", "41",
            "--queries", "7",
            "--top-k", "6",
            "--filter", "broad",
            "--duplicate-ids", "2",
            "--unknown-ids", "3",
            "--runs", "3",
            "--warmup-queries", "4",
            "--seed", "0x5EED046A",
            "--output", outputPath
        ];
        GeneratedExactFilteredOptions options = CommandLine.ParseGeneratedExactFiltered(arguments);

        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(options, arguments);
        GeneratedExactFilteredScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-046", report.TaskId);
        Assert.Equal("exact-generated-filtered", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-filtered-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal("scalar-reference-generated-filtered", report.Truth.Kind);
        Assert.Equal(6, report.Truth.Depth);
        Assert.Equal("ExactFiltered", report.Index.Profile);
        Assert.Equal(nameof(ExactFlatIndex), report.Index.Type);
        Assert.Contains("Search(query, allowedIds, results, workspace)", report.Index.Configuration, StringComparison.Ordinal);
        Assert.Equal("broad", report.Filter.Kind);
        Assert.Equal("approximately 50% of indexed rows visible", report.Filter.SelectivityTarget);
        Assert.Equal(21, report.Filter.VisibleCountPerQuery);
        Assert.Equal(21, report.Filter.KnownIdCountPerQuery);
        Assert.Equal(2, report.Filter.DuplicateIdCountPerQuery);
        Assert.Equal(3, report.Filter.UnknownIdCountPerQuery);
        Assert.Equal(26, report.Filter.AllowlistLengthPerQuery);
        Assert.Equal(147, report.Filter.TotalKnownIdCount);
        Assert.Equal(14, report.Filter.TotalDuplicateIdCount);
        Assert.Equal(21, report.Filter.TotalUnknownIdCount);
        Assert.Contains("visibleCount", report.Filter.GenerationFormula, StringComparison.Ordinal);
        Assert.Contains("coalesced", report.Filter.DuplicatePolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ignored", report.Filter.UnknownIdPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(7, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal(7, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("allowlist generation", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("workspace construction", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.True(double.Parse(report.Measurement.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Contains("ExactFlatSearchFilterWorkspace", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(4, report.Measurement.Warmup.WarmupCount);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.MissingResultCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.ExtraResultCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.WrongIdCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.OrderMismatchCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.DistanceMismatchCount);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FinalRunComparedToTruth);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("broad", root.GetProperty("filter").GetProperty("kind").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", root.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("latencyTicks", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("all", 32, 32)]
    [InlineData("broad", 16, 16)]
    [InlineData("selective", 4, 4)]
    [InlineData("very-selective", 3, 3)]
    [InlineData("empty", 0, 0)]
    public void Run_SupportsRequiredFilterKinds(string filterKind, int expectedVisibleCount, int expectedKnownCount)
    {
        var options = new GeneratedExactFilteredOptions(
            VectorMetric.InnerProduct,
            Dimension: 7,
            VectorCount: 32,
            QueryCount: 3,
            TopK: 4,
            Seed: 0x5EED046B,
            FilterKind: filterKind,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 1,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec046-filter-kind.json",
            Runs: 1,
            WarmupQueries: 0);

        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(options, ["exact-generated-filtered"]);

        Assert.Equal(filterKind, report.Filter.Kind);
        Assert.Equal(expectedVisibleCount, report.Filter.VisibleCountPerQuery);
        Assert.Equal(expectedKnownCount, report.Filter.KnownIdCountPerQuery);
        Assert.Equal(1, report.Filter.DuplicateIdCountPerQuery);
        Assert.Equal(1, report.Filter.UnknownIdCountPerQuery);
        Assert.Equal(expectedKnownCount + 2, report.Filter.AllowlistLengthPerQuery);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        if (filterKind == "empty")
        {
            Assert.Equal(0, report.Metrics.FilteredResultIntegrity.CheckedResultCount);
            Assert.Equal(1.0, report.Metrics.RecallAtK);
            Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        }

        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ValidateFilteredResults_FailsIncorrectCountIdOrderAndDistance()
    {
        var truth = new TruthSet(
            [
                [
                    new TruthItem(1, 1),
                    new TruthItem(2, 2)
                ],
                []
            ],
            depth: 2);
        SearchResult[][] actual =
        [
            [
                new SearchResult(2, 2),
                new SearchResult(1, 100),
                new SearchResult(99, float.NaN)
            ],
            [
                new SearchResult(7, 1)
            ]
        ];

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            actual,
            topK: 2,
            dimension: 4,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(0, comparison.Integrity.QueryCountMismatchCount);
        Assert.Equal(4, comparison.Integrity.CheckedResultCount);
        Assert.Equal(0, comparison.Integrity.MissingResultCount);
        Assert.Equal(2, comparison.Integrity.ExtraResultCount);
        Assert.Equal(2, comparison.Integrity.WrongIdCount);
        Assert.Equal(2, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.NonFiniteDistanceCount);
        Assert.Equal(2, comparison.Integrity.DistanceMismatchCount);
        Assert.InRange(comparison.RecallAtK, 0.99, 1.0);
        Assert.Equal(0, comparison.OrderedAgreement);
    }

    [Fact]
    public void CompareGeneratedExact_TreatsGeneratedExactFilteredReportAsUnsupportedSchema()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec046-comparison-" + Guid.NewGuid().ToString("N"));
        string reportPath = Path.Combine(outputDirectory, "filtered-report.json");
        var options = new GeneratedExactFilteredOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 6,
            VectorCount: 16,
            QueryCount: 2,
            TopK: 3,
            Seed: 0x5EED046C,
            FilterKind: "selective",
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputPath: reportPath);
        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(options, ["exact-generated-filtered"]);
        GeneratedExactFilteredScenario.Write(report, reportPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(reportPath, reportPath, Path.Combine(outputDirectory, "comparison.json")),
            ["compare-generated-exact"]);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        Assert.Empty(comparison.Metrics);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
    }
}
