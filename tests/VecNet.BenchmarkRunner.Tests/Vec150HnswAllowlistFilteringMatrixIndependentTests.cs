using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec150HnswAllowlistFilteringMatrixIndependentTests
{
    [Fact]
    public void ParseHnswAllowlistFilteringMatrix_AcceptsCaseInsensitivePrivateMatrixOptions()
    {
        string outputDirectory = NewArtifactDirectory("parse");
        string manifestPath = Path.Combine(outputDirectory, "nested", "matrix.json");

        HnswAllowlistFilteringMatrixOptions options =
            CommandLine.ParseHnswAllowlistFilteringMatrix(
                [
                    "GENERATED-HNSW-ALLOWLIST-FILTERED-MATRIX",
                    "--PRESET", "STANDARD",
                    "--QUERIES", "5",
                    "--RUNS", "2",
                    "--WARMUP-QUERIES", "0",
                    "--SEED", "0xA150C0DE",
                    "--DUPLICATE-INSERTS", "2",
                    "--UNKNOWN-DELETES", "3",
                    "--REPEATED-DELETES", "4",
                    "--METRIC", "COSINE",
                    "--OUTPUT-DIR", outputDirectory,
                    "--MANIFEST", manifestPath
                ]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(5, options.QueryCount);
        Assert.Equal(2, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0xA150C0DEu, options.Seed);
        Assert.Equal(2, options.DuplicateInsertAttempts);
        Assert.Equal(3, options.UnknownDeleteAttempts);
        Assert.Equal(4, options.RepeatedDeleteAttempts);
        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal(outputDirectory, options.OutputDirectory);
        Assert.Equal(manifestPath, options.ManifestPath);
    }

    [Theory]
    [InlineData("--metric", "InnerProduct")]
    [InlineData("--dimension", "384")]
    [InlineData("--vectors", "2048")]
    [InlineData("--top-k", "100")]
    [InlineData("--insertions", "256")]
    [InlineData("--deletes", "64")]
    [InlineData("--delta-deletes", "16")]
    [InlineData("--filter", "all")]
    [InlineData("--output", "case-report.json")]
    [InlineData("--opened-index-directory", "opened")]
    [InlineData("--checkpoint-directory", "checkpoint")]
    [InlineData("--m", "16")]
    [InlineData("--ef-construction", "128")]
    [InlineData("--ef-search", "192")]
    [InlineData("--hnsw-seed", "0x484E535700014801")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--comparison-output", "comparison.json")]
    [InlineData("--regression-threshold", "0.05")]
    [InlineData("--cache-root", "VecNet.DatasetCache")]
    [InlineData("--truth-path", "truth.json")]
    [InlineData("--download", "true")]
    [InlineData("--sample-interval-ms", "10")]
    [InlineData("--memory-output", "memory.json")]
    [InlineData("--public-claim", "true")]
    public void ParseHnswAllowlistFilteringMatrix_RejectsCaseLevelExternalMemoryAndReportOptions(
        string option,
        string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswAllowlistFilteringMatrix(
                [HnswAllowlistFilteringMatrixOptions.ScenarioName, option, value]));

        if (option.Equals("--metric", StringComparison.OrdinalIgnoreCase))
        {
            Assert.Contains("supports", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        else
        {
            Assert.Contains(option.TrimStart('-'), exception.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ExpandCases_UsesRequiredSmokeAndStandardOrdering()
    {
        HnswAllowlistFilteringMatrixOptions baseOptions = MatrixOptions(
            preset: "standard",
            outputDirectory: NewArtifactDirectory("expand"),
            manifestName: "manifests/matrix.json",
            seed: 0x5EED2148u,
            queryCount: 32,
            runs: 3,
            warmupQueries: 3);

        HnswAllowlistFilteringMatrixScenario.MatrixCase[] smokeCases =
            HnswAllowlistFilteringMatrixScenario.ExpandCases(baseOptions with
            {
                PresetName = "smoke",
                QueryCount = 8,
                Runs = 1,
                WarmupQueries = 1
            });
        HnswAllowlistFilteringMatrixScenario.MatrixCase[] standardCases =
            HnswAllowlistFilteringMatrixScenario.ExpandCases(baseOptions);

        Assert.Equal(
            [
                "smoke-empty-k10",
                "smoke-fallback-boundary-k10",
                "smoke-broad-k10",
                "smoke-broad-tombstone-heavy-k10"
            ],
            smokeCases.Select(item => item.CaseId).ToArray());
        Assert.Equal(["empty", "fallback-boundary", "broad", "broad"], smokeCases.Select(item => item.FilterProfile).ToArray());
        Assert.Equal(["low-churn", "low-churn", "low-churn", "tombstone-heavy"], smokeCases.Select(item => item.UpdateProfileName).ToArray());
        Assert.All(smokeCases, item => Assert.Equal(10, item.Options.TopK));

        Assert.Equal(
            [
                "standard-empty-32d-k10",
                "standard-very-selective-32d-k10",
                "standard-fallback-boundary-32d-k10",
                "standard-broad-32d-k10",
                "standard-broad-tombstone-heavy-32d-k100",
                "standard-all-tombstone-heavy-32d-k100",
                "standard-empty-96d-k10",
                "standard-very-selective-96d-k10",
                "standard-fallback-boundary-96d-k10",
                "standard-broad-96d-k10",
                "standard-broad-tombstone-heavy-96d-k100",
                "standard-all-tombstone-heavy-96d-k100",
                "standard-empty-384d-k10",
                "standard-very-selective-384d-k10",
                "standard-fallback-boundary-384d-k10",
                "standard-broad-384d-k10",
                "standard-broad-tombstone-heavy-384d-k100",
                "standard-all-tombstone-heavy-384d-k100"
            ],
            standardCases.Select(item => item.CaseId).ToArray());
        Assert.Equal([32, 96, 384], standardCases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal(["all", "broad", "empty", "fallback-boundary", "very-selective"], standardCases.Select(item => item.FilterProfile).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], standardCases.Select(item => item.UpdateProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.Equal([10, 100], standardCases.Select(item => item.Options.TopK).Distinct().Order().ToArray());

        for (int i = 0; i < standardCases.Length; i++)
        {
            HnswAllowlistFilteringMatrixScenario.MatrixCase matrixCase = standardCases[i];
            uint expectedSeed = unchecked(0x5EED2148u + (uint)i);
            ulong expectedHnswSeed = unchecked(0x484E535700014800UL + (ulong)i + 1);

            Assert.Equal(expectedSeed, matrixCase.Options.Seed);
            Assert.Equal(expectedHnswSeed, matrixCase.Options.HnswSeed);
            Assert.Equal(VectorMetric.SquaredEuclidean, matrixCase.Options.Metric);
            Assert.Equal(2048, matrixCase.Options.BaseVectorCount);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.Equal(192, matrixCase.Options.EfSearch);
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeOpenedIndexDirectoryPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeCheckpointDirectoryPath));
        }
    }

    [Fact]
    public void Run_SmokeManifestSerializesRequiredShapeSummariesAndRelativeLinkedReports()
    {
        string outputDirectory = NewArtifactDirectory("json-smoke");
        string manifestDirectory = Path.Combine(outputDirectory, "manifest");
        string manifestPath = Path.Combine(manifestDirectory, "hnsw-filter-matrix.json");
        string[] arguments =
        [
            HnswAllowlistFilteringMatrixOptions.ScenarioName,
            "--preset", "smoke",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED2500",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        HnswAllowlistFilteringMatrixManifest manifest =
            HnswAllowlistFilteringMatrixScenario.Run(
                CommandLine.ParseHnswAllowlistFilteringMatrix(arguments),
                arguments);
        HnswAllowlistFilteringMatrixScenario.WriteManifest(manifest, manifestPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswAllowlistFilteringMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-150", root.GetProperty("taskId").GetString());
        Assert.Equal(HnswAllowlistFilteringMatrixOptions.ScenarioName, root.GetProperty("scenarioName").GetString());
        Assert.Equal("smoke", root.GetProperty("presetName").GetString());
        Assert.Equal("private-raw", root.GetProperty("eligibility").GetProperty("privacyClass").GetString());
        Assert.Equal("local-evidence", root.GetProperty("eligibility").GetProperty("claimClass").GetString());
        AssertFalseEligibility(root.GetProperty("eligibility"));
        AssertNoEligibilityFlagTrue(root);

        JsonElement aggregate = root.GetProperty("aggregate");
        Assert.Equal(4, aggregate.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, aggregate.GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, aggregate.GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal("recorded", aggregate.GetProperty("branchCoverage").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("exactFallbackParity").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("broadEmission").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("underfill").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("allowlist").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("mutationCounts").GetProperty("status").GetString());
        Assert.True(aggregate.GetProperty("mutationCounts").GetProperty("totalTombstoneCount").GetInt32() > 0);
        Assert.Equal("recorded", aggregate.GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("recorded", aggregate.GetProperty("allocations").GetProperty("status").GetString());
        Assert.True(aggregate.GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("passed", matrixCase.GetProperty("status").GetString());
            Assert.Equal("passed", matrixCase.GetProperty("validationStatus").GetString());
            AssertHasRequiredCaseShape(matrixCase);

            string relativeReportPath = matrixCase.GetProperty("relativeReportPath").GetString()!;
            Assert.False(Path.IsPathFullyQualified(relativeReportPath));
            string linkedReportPath = Path.GetFullPath(Path.Combine(manifestDirectory, relativeReportPath.Replace('/', Path.DirectorySeparatorChar)));
            Assert.True(File.Exists(linkedReportPath), linkedReportPath);

            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(linkedReportPath));
            JsonElement report = reportDocument.RootElement;
            Assert.Equal("VecNet.HnswAllowlistFilteringBenchmarkReport", report.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", report.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-149", report.GetProperty("taskId").GetString());
            Assert.Equal(HnswAllowlistFilteringOptions.ScenarioName, report.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.GetProperty("linkedReportId").GetString(), report.GetProperty("reportId").GetString());
            Assert.Equal(matrixCase.GetProperty("filterProfile").GetString(), report.GetProperty("allowlist").GetProperty("profile").GetString());
            Assert.Equal("notMeasured", report.GetProperty("memory").GetProperty("status").GetString());
            AssertLinkedReportSearchSections(report.GetProperty("searches"));
            AssertFalseEligibility(report.GetProperty("eligibility"));
            AssertFalseEligibility(report.GetProperty("validation"));
        }
    }

    [Fact]
    public void Run_InvalidPerCaseWorkloadRetainsFailedCasesWithReasons()
    {
        HnswAllowlistFilteringMatrixOptions options = MatrixOptions(
            preset: "smoke",
            outputDirectory: NewArtifactDirectory("failed"),
            manifestName: "matrix.json",
            seed: 0x5EED2501u,
            queryCount: 1,
            runs: 0,
            warmupQueries: 0);

        HnswAllowlistFilteringMatrixManifest manifest =
            HnswAllowlistFilteringMatrixScenario.Run(
                options,
                [HnswAllowlistFilteringMatrixOptions.ScenarioName]);

        Assert.Equal("failed", manifest.ValidationStatus);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal(0, manifest.Aggregate.LinkedReportCount);
        Assert.Equal("partial", manifest.Aggregate.RecursiveEligibility.Status);
        Assert.False(manifest.Aggregate.RecursiveEligibility.AllEligibilityFlagsFalse);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.Contains("runs", matrixCase.ErrorMessage, StringComparison.OrdinalIgnoreCase);
            Assert.Equal("notAvailable", matrixCase.BranchSummary.Status);
            Assert.Equal("notAvailable", matrixCase.Allowlist.Status);
            Assert.Equal("notAvailable", matrixCase.Tombstones.Status);
            Assert.Equal("notAvailable", matrixCase.ExactFilteredDeltaScan.Status);
            Assert.Equal("notAvailable", matrixCase.ReturnedResultIntegrity.Status);
            Assert.Equal("notAvailable", matrixCase.Allocations.Status);
            Assert.False(matrixCase.RecursiveEligibility.LinkedReportInspected);
        });
    }

    [Fact]
    public void Run_BlockedReportWriteRetainsBlockedCaseAndRelativeOutputPaths()
    {
        string outputDirectory = NewArtifactDirectory("blocked");
        Directory.CreateDirectory(Path.Combine(outputDirectory, "smoke-empty-k10", "allowlist-filtered-report.json"));
        HnswAllowlistFilteringMatrixOptions options = MatrixOptions(
            preset: "smoke",
            outputDirectory,
            manifestName: "matrix.json",
            seed: 0x5EED2502u,
            queryCount: 1,
            runs: 1,
            warmupQueries: 0);

        HnswAllowlistFilteringMatrixManifest manifest =
            HnswAllowlistFilteringMatrixScenario.Run(
                options,
                [HnswAllowlistFilteringMatrixOptions.ScenarioName]);

        Assert.Equal("failed", manifest.ValidationStatus);
        Assert.Equal(3, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(1, manifest.Aggregate.BlockedCaseCount);

        HnswAllowlistFilteringMatrixCaseManifest blocked = Assert.Single(manifest.Cases, item => item.Status == "blocked");
        Assert.Equal("smoke-empty-k10", blocked.CaseId);
        Assert.Equal("blocked", blocked.ValidationStatus);
        Assert.Null(blocked.LinkedReportId);
        Assert.False(string.IsNullOrWhiteSpace(blocked.ErrorMessage));
        Assert.False(Path.IsPathFullyQualified(blocked.RelativeReportPath));
        Assert.False(Path.IsPathFullyQualified(blocked.RelativeOpenedIndexDirectoryPath));
        Assert.False(Path.IsPathFullyQualified(blocked.RelativeCheckpointDirectoryPath));
        Assert.Equal("notAvailable", blocked.BranchSummary.Status);
        Assert.Equal("notAvailable", blocked.RecursiveEligibility.Status);
        Assert.False(blocked.RecursiveEligibility.AllEligibilityFlagsFalse);
    }

    private static HnswAllowlistFilteringMatrixOptions MatrixOptions(
        string preset,
        string outputDirectory,
        string manifestName,
        uint seed,
        int queryCount,
        int runs,
        int warmupQueries) =>
        new(
            preset,
            queryCount,
            runs,
            warmupQueries,
            seed,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            outputDirectory,
            Path.Combine(outputDirectory, manifestName));

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"test-agent-vec150-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertHasRequiredCaseShape(JsonElement matrixCase)
    {
        Assert.True(matrixCase.TryGetProperty("caseNumber", out _));
        Assert.True(matrixCase.TryGetProperty("caseId", out _));
        Assert.True(matrixCase.TryGetProperty("scenarioName", out _) || matrixCase.TryGetProperty("filterProfile", out _));
        Assert.True(matrixCase.TryGetProperty("presetName", out _) || matrixCase.TryGetProperty("updateProfileName", out _));
        Assert.Equal("recorded", matrixCase.GetProperty("branchSummary").GetProperty("status").GetString());
        Assert.True(matrixCase.TryGetProperty("exactFallbackParity", out _));
        Assert.True(matrixCase.TryGetProperty("broadEmission", out _));
        Assert.Equal("recorded", matrixCase.GetProperty("underfill").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("allowlist").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("tombstones").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("exactFilteredDeltaScan").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("mutations").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("counts").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("recorded", matrixCase.GetProperty("allocations").GetProperty("status").GetString());
        Assert.True(matrixCase.GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());
    }

    private static void AssertLinkedReportSearchSections(JsonElement searches)
    {
        Assert.True(searches.TryGetProperty("immutableHnsw", out _));
        Assert.True(searches.TryGetProperty("openedHnsw", out _));
        Assert.True(searches.TryGetProperty("sourceComposite", out _));
        Assert.True(searches.TryGetProperty("rebuiltComposite", out _));
        Assert.True(searches.TryGetProperty("checkpointOpenedHnsw", out _));
    }

    private static void AssertFalseEligibility(JsonElement element)
    {
        if (element.TryGetProperty("publicClaimEligible", out JsonElement publicClaim))
        {
            Assert.False(publicClaim.GetBoolean());
        }

        if (element.TryGetProperty("baselineCandidateEligible", out JsonElement baseline))
        {
            Assert.False(baseline.GetBoolean());
        }

        if (element.TryGetProperty("comparisonArtifactEligible", out JsonElement comparison))
        {
            Assert.False(comparison.GetBoolean());
        }

        if (element.TryGetProperty("regressionGateEligible", out JsonElement regression))
        {
            Assert.False(regression.GetBoolean());
        }
    }

    private static void AssertNoEligibilityFlagTrue(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    (string.Equals(property.Name, "publicClaimEligible", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(property.Name, "baselineCandidateEligible", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(property.Name, "comparisonArtifactEligible", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(property.Name, "regressionGateEligible", StringComparison.OrdinalIgnoreCase)))
                {
                    Assert.Fail($"Eligibility flag '{property.Name}' must remain false.");
                }

                AssertNoEligibilityFlagTrue(property.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoEligibilityFlagTrue(item);
            }
        }
    }
}
