using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec053GeneratedExactCandidateSetTests
{
    [Fact]
    public void ParseGeneratedExactCandidateSet_UsesPrivateDefaults()
    {
        GeneratedExactCandidateSetOptions options = CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal("broad", options.CandidateSetKind);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2053u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-candidate-set", "--dimension")]
    [InlineData("generated-exact-candidate-set", "dimension", "8")]
    [InlineData("generated-exact-candidate-set", "--metric", "Unknown")]
    [InlineData("generated-exact-candidate-set", "--dimension", "0")]
    [InlineData("generated-exact-candidate-set", "--vectors", "0")]
    [InlineData("generated-exact-candidate-set", "--queries", "0")]
    [InlineData("generated-exact-candidate-set", "--top-k", "3", "--vectors", "2")]
    [InlineData("generated-exact-candidate-set", "--runs", "0")]
    [InlineData("generated-exact-candidate-set", "--runs", "6")]
    [InlineData("generated-exact-candidate-set", "--warmup-queries", "-1")]
    [InlineData("generated-exact-candidate-set", "--candidate-set", "unknown")]
    [InlineData("generated-exact-candidate-set", "--candidate-set", "very-selective", "--top-k", "1")]
    [InlineData("generated-exact-candidate-set", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-candidate-set", "--unknown-ids", "-1")]
    [InlineData("generated-exact-candidate-set", "--filter", "broad")]
    [InlineData("generated-exact-candidate-set", "--preset", "smoke")]
    [InlineData("generated-exact-candidate-set", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-candidate-set", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-exact-candidate-set", "--m", "8")]
    [InlineData("generated-exact-candidate-set", "--output", "")]
    public void ParseGeneratedExactCandidateSet_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedExactCandidateSetReport()
    {
        string outputPath = NewArtifactPath("candidate-set-report.json");
        string[] arguments =
        [
            "generated-exact-candidate-set",
            "--metric", "SquaredEuclidean",
            "--dimension", "13",
            "--vectors", "41",
            "--queries", "7",
            "--top-k", "6",
            "--candidate-set", "broad",
            "--duplicate-ids", "2",
            "--unknown-ids", "3",
            "--runs", "3",
            "--warmup-queries", "4",
            "--seed", "0x5EED053A",
            "--output", outputPath
        ];
        GeneratedExactCandidateSetOptions options = CommandLine.ParseGeneratedExactCandidateSet(arguments);

        GeneratedExactCandidateSetBenchmarkReport report = GeneratedExactCandidateSetScenario.Run(options, arguments);
        GeneratedExactCandidateSetScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExactCandidateSetBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-053", report.TaskId);
        Assert.Equal("generated-exact-candidate-set", report.ScenarioName);
        Assert.Equal("generated-exact-candidate-set", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-candidate-set-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.Contains("candidate-set construction", string.Join(" ", report.Evidence.Limitations), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal("scalar-reference-generated-candidate-set-filtered", report.Truth.Kind);
        Assert.Equal("ExactCandidateSet", report.Index.Profile);
        Assert.Contains("Search(query, candidateSet, results)", report.Index.Configuration, StringComparison.Ordinal);
        Assert.Equal("broad", report.CandidateInput.Kind);
        Assert.Equal(21, report.CandidateInput.KnownIdCountPerQuery);
        Assert.Equal(2, report.CandidateInput.DuplicateIdCountPerQuery);
        Assert.Equal(3, report.CandidateInput.UnknownIdCountPerQuery);
        Assert.Equal(26, report.CandidateInput.InputIdCountPerQuery);
        Assert.Equal(147, report.CandidateInput.TotalKnownIdCount);
        Assert.Equal(14, report.CandidateInput.TotalDuplicateIdCount);
        Assert.Equal(21, report.CandidateInput.TotalUnknownIdCount);
        Assert.Contains("authorization", report.CandidateInput.ApplicationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("constructedOutsideMeasuredSearch", report.CandidateSet.ConstructionStatus);
        Assert.Equal("public ExactFlatIndex.CreateCandidateSet(allowedIds)", report.CandidateSet.ConstructionOperation);
        Assert.True(report.CandidateSet.ConstructedBeforeMeasuredSearch);
        Assert.Equal(7, report.CandidateSet.ConstructedSetCount);
        Assert.Equal(21, report.CandidateSet.CountPerQuery);
        Assert.Equal(21, report.CandidateSet.MinCount);
        Assert.Equal(21, report.CandidateSet.MaxCount);
        Assert.Equal(147, report.CandidateSet.TotalCandidateCount);
        Assert.Contains("excluded", report.CandidateSet.ConstructionTimingScope, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("excluded", report.CandidateSet.ConstructionAllocationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(7, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("candidate-set construction", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.True(double.Parse(report.Measurement.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Contains("prebuilt ExactFlatCandidateSet", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Contains("candidate-set construction", report.Measurement.ManagedAllocations.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.CandidateSetsConstructed);
        Assert.True(report.Validation.FinalRunComparedToTruth);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactCandidateSetBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("broad", root.GetProperty("candidateInput").GetProperty("kind").GetString());
        Assert.Equal("constructedOutsideMeasuredSearch", root.GetProperty("candidateSet").GetProperty("constructionStatus").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", root.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Contains("candidate-set construction", root.GetProperty("measurement").GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("latencyTicks", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
        AssertNoPropertyNamed(root, "baseline", "candidateEligibility", "comparisonResult", "regressionDecision", "preset", "cases", "manifest", "hnsw", "storedLabels", "recordStore", "authorizationDecision");
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, "all", 24, 24, 1.0)]
    [InlineData(VectorMetric.InnerProduct, "broad", 12, 12, 0.5)]
    [InlineData(VectorMetric.Cosine, "selective", 3, 3, 0.125)]
    [InlineData(VectorMetric.SquaredEuclidean, "very-selective", 4, 4, 1.0 / 6.0)]
    [InlineData(VectorMetric.InnerProduct, "empty", 0, 0, 0.0)]
    public void Run_CandidateSetKindsAcrossMetricsCarryValidationAndPrivatePosture(
        VectorMetric metric,
        string candidateSetKind,
        int expectedKnown,
        int expectedCandidateCount,
        double expectedSelectivity)
    {
        GeneratedExactCandidateSetBenchmarkReport report = GeneratedExactCandidateSetScenario.Run(
            new GeneratedExactCandidateSetOptions(
                metric,
                Dimension: 9,
                VectorCount: 24,
                QueryCount: 4,
                TopK: 5,
                Seed: 0x5EED5310,
                CandidateSetKind: candidateSetKind,
                DuplicateIdsPerQuery: 2,
                UnknownIdsPerQuery: 3,
                OutputPath: NewArtifactPath("candidate-kind.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-candidate-set", "--candidate-set", candidateSetKind]);

        Assert.Equal(candidateSetKind, report.CandidateInput.Kind);
        Assert.Equal(expectedKnown, report.CandidateInput.KnownIdCountPerQuery);
        Assert.Equal(expectedSelectivity, report.CandidateInput.ActualSelectivity, precision: 12);
        Assert.Equal(expectedCandidateCount, report.CandidateSet.CountPerQuery);
        Assert.Equal(expectedCandidateCount, report.CandidateSet.MinCount);
        Assert.Equal(expectedCandidateCount, report.CandidateSet.MaxCount);
        Assert.Equal(expectedCandidateCount, report.CandidateSet.MeanCount);
        Assert.Equal(expectedCandidateCount * 4, report.CandidateSet.TotalCandidateCount);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal(4 * Math.Min(5, expectedCandidateCount), report.Metrics.FilteredResultIntegrity.CheckedResultCount);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void Run_RepeatedRunsWarmupAndAllocationMetadataExcludeCandidateSetConstruction()
    {
        GeneratedExactCandidateSetBenchmarkReport report = GeneratedExactCandidateSetScenario.Run(
            new GeneratedExactCandidateSetOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 10,
                VectorCount: 40,
                QueryCount: 3,
                TopK: 6,
                Seed: 0x5EED5311,
                CandidateSetKind: "selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("runs-warmup.json"),
                Runs: 5,
                WarmupQueries: 17),
            ["generated-exact-candidate-set", "--runs", "5", "--warmup-queries", "17"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement aggregate = root.GetProperty("search").GetProperty("aggregate");
        JsonElement runs = root.GetProperty("search").GetProperty("runs");

        Assert.Equal(3, root.GetProperty("scenario").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(5, runs.GetArrayLength());
        Assert.Equal(5, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(3, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", measurement.GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Contains("candidate ID generation", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("candidate-set construction", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("filtered truth", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("candidate-set construction", measurement.GetProperty("managedAllocations").GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", measurement.GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.True(measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("measured", measurement.GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("executed", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(17, measurement.GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.DoesNotContain("\"measuredQueryCount\":20", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"measuredQueryCountPerRun\":20", json, StringComparison.Ordinal);
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndCandidateSetIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--candidate-set", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--candidate-set", "broad"]));
        Assert.Equal("generated-exact-candidate-set", GeneratedExactCandidateSetOptions.ScenarioName);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec053-" + Path.GetFileNameWithoutExtension(fileName) + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, string.Create(CultureInfo.InvariantCulture, $"Unexpected field '{property.Name}' was present."));
                AssertNoPropertyNamed(property.Value, disallowedNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, disallowedNames);
            }
        }
    }
}
