using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec150HnswAllowlistFilteringMatrixTests
{
    [Fact]
    public void ParseHnswAllowlistFilteringMatrix_UsesPrivateSmokeDefaults()
    {
        HnswAllowlistFilteringMatrixOptions options =
            CommandLine.ParseHnswAllowlistFilteringMatrix([HnswAllowlistFilteringMatrixOptions.ScenarioName]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(8, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED2148u, options.Seed);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathFullyQualified(options.OutputDirectory));
        Assert.EndsWith("hnsw-allowlist-filtered-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseHnswAllowlistFilteringMatrix_UsesStandardDefaults()
    {
        HnswAllowlistFilteringMatrixOptions options =
            CommandLine.ParseHnswAllowlistFilteringMatrix(
                [HnswAllowlistFilteringMatrixOptions.ScenarioName, "--preset", "standard"]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(32, options.QueryCount);
        Assert.Equal(3, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--preset", "large")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--preset", " ")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--queries", "0")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--runs", "0")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--runs", "6")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--duplicate-inserts", "-1")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--unknown-deletes", "-1")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--repeated-deletes", "-1")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--output-dir", "")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--manifest", "")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--metric", "InnerProduct")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--dimension", "32")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--vectors", "512")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--top-k", "10")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--filter", "broad")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-hnsw-allowlist-filtered-matrix", "--sample-interval-ms", "10")]
    [InlineData("generated-hnsw-allowlist-filtered", "--output-dir", "matrix")]
    public void ParseHnswAllowlistFilteringMatrix_RejectsInvalidOrOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswAllowlistFilteringMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseHnswAllowlistFilteringMatrix_AcceptsCosine(string metric)
    {
        HnswAllowlistFilteringMatrixOptions options =
            CommandLine.ParseHnswAllowlistFilteringMatrix(
                [HnswAllowlistFilteringMatrixOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
    }

    [Fact]
    public void ExpandCases_SmokeAndStandardPresetsUseDeterministicShapesAndSeeds()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        string manifestPath = Path.Combine(outputDirectory, "manifests", "manifest.json");
        var smokeOptions = new HnswAllowlistFilteringMatrixOptions(
            "smoke",
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 1,
            Seed: 0xFFFF_FFFEu,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        var standardOptions = smokeOptions with
        {
            PresetName = "standard",
            QueryCount = 32,
            Runs = 3,
            WarmupQueries = 3,
            Metric = VectorMetric.Cosine
        };

        HnswAllowlistFilteringMatrixScenario.MatrixCase[] smokeCases =
            HnswAllowlistFilteringMatrixScenario.ExpandCases(smokeOptions);
        HnswAllowlistFilteringMatrixScenario.MatrixCase[] standardCases =
            HnswAllowlistFilteringMatrixScenario.ExpandCases(standardOptions);

        Assert.Equal(4, smokeCases.Length);
        Assert.Equal(["smoke-empty-k10", "smoke-fallback-boundary-k10", "smoke-broad-k10", "smoke-broad-tombstone-heavy-k10"], smokeCases.Select(item => item.CaseId).ToArray());
        Assert.Equal(["empty", "fallback-boundary", "broad"], smokeCases.Select(item => item.FilterProfile).Distinct().ToArray());
        Assert.Equal(96, smokeCases[^1].Options.Dimension);
        Assert.Equal(1024, smokeCases[^1].Options.BaseVectorCount);
        Assert.Equal(128, smokeCases[^1].Options.InsertedDeltaCount);
        Assert.Equal(128, smokeCases[^1].Options.DeletedBaseCount);
        Assert.Equal(48, smokeCases[^1].Options.DeletedDeltaCount);

        Assert.Equal(18, standardCases.Length);
        Assert.Equal([32, 96, 384], standardCases.Select(item => item.Options.Dimension).Distinct().Order().ToArray());
        Assert.Equal(["all", "broad", "empty", "fallback-boundary", "very-selective"], standardCases.Select(item => item.FilterProfile).Distinct().Order().ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], standardCases.Select(item => item.UpdateProfileName).Distinct().Order().ToArray());
        Assert.Equal([10, 100], standardCases.Select(item => item.Options.TopK).Distinct().Order().ToArray());

        Assert.All(standardCases, matrixCase =>
        {
            Assert.Equal(VectorMetric.Cosine, matrixCase.Options.Metric);
            Assert.Equal(32, matrixCase.Options.QueryCount);
            Assert.Equal(3, matrixCase.Options.Runs);
            Assert.Equal(3, matrixCase.Options.WarmupQueries);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.Equal(192, matrixCase.Options.EfSearch);
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeOpenedIndexDirectoryPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.OpenedIndexDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);
        });

        Assert.Equal(0xFFFF_FFFEu, standardCases[0].Options.Seed);
        Assert.Equal(15u, standardCases[^1].Options.Seed);
        Assert.Equal("0x484E535700014801", FormatHex(standardCases[0].Options.HnswSeed));
        Assert.Equal("0x484E535700014812", FormatHex(standardCases[^1].Options.HnswSeed));
    }

    [Fact]
    public void Run_SmokeManifestLinksVec149ReportsWithSummariesAndFalseEligibility()
    {
        string outputDirectory = NewArtifactDirectory("smoke-manifest");
        string manifestDirectory = Path.Combine(outputDirectory, "manifests");
        string manifestPath = Path.Combine(manifestDirectory, "manifest.json");
        string[] arguments =
        [
            HnswAllowlistFilteringMatrixOptions.ScenarioName,
            "--preset", "smoke",
            "--queries", "2",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED1500",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "1",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];
        HnswAllowlistFilteringMatrixOptions options =
            CommandLine.ParseHnswAllowlistFilteringMatrix(arguments);

        HnswAllowlistFilteringMatrixManifest manifest =
            HnswAllowlistFilteringMatrixScenario.Run(options, arguments);
        HnswAllowlistFilteringMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.HnswAllowlistFilteringMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-150", manifest.TaskId);
        Assert.Equal(HnswAllowlistFilteringMatrixOptions.ScenarioName, manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal("passed", manifest.ValidationStatus);
        Assert.Equal(4, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.SkippedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal(4, manifest.Aggregate.LinkedReportCount);
        Assert.Equal("recorded", manifest.Aggregate.BranchCoverage.Status);
        Assert.Equal(2, manifest.Aggregate.BranchCoverage.ExactFallbackCaseCount);
        Assert.Equal(2, manifest.Aggregate.BranchCoverage.BroadEmissionCaseCount);
        Assert.Contains("empty", manifest.Aggregate.BranchCoverage.CoveredFilterProfiles);
        Assert.Contains("fallback-boundary", manifest.Aggregate.BranchCoverage.CoveredFilterProfiles);
        Assert.Contains("broad", manifest.Aggregate.BranchCoverage.CoveredFilterProfiles);
        Assert.Equal("recorded", manifest.Aggregate.ExactFallbackParity.Status);
        Assert.Equal(2, manifest.Aggregate.ExactFallbackParity.PassedCaseCount);
        Assert.Equal("recorded", manifest.Aggregate.BroadEmission.Status);
        Assert.Equal(2, manifest.Aggregate.BroadEmission.RecordedCaseCount);
        Assert.Equal("recorded", manifest.Aggregate.Underfill.Status);
        Assert.True(manifest.Aggregate.Underfill.TotalRequestedResultSlots > 0);
        Assert.Equal("recorded", manifest.Aggregate.Allowlist.Status);
        Assert.True(manifest.Aggregate.Allowlist.TotalUnknownInputIds > 0);
        Assert.True(manifest.Aggregate.Allowlist.TotalTombstonedInputIds > 0);
        Assert.Equal("recorded", manifest.Aggregate.MutationCounts.Status);
        Assert.True(manifest.Aggregate.MutationCounts.TotalTombstoneCount > 0);
        Assert.Equal("recorded", manifest.Aggregate.ReturnedResultIntegrity.Status);
        Assert.Equal(4, manifest.Aggregate.ReturnedResultIntegrity.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.ReturnedResultIntegrity.TotalUnknownIdCount);
        Assert.Equal(0, manifest.Aggregate.ReturnedResultIntegrity.TotalTombstonedIdCount);
        Assert.Equal(0, manifest.Aggregate.ReturnedResultIntegrity.TotalNotAllowedIdCount);
        Assert.Equal("recorded", manifest.Aggregate.Allocations.Status);
        Assert.NotNull(manifest.Aggregate.Allocations.MaxMeanManagedAllocatedBytesPerSearchCall);
        Assert.True(manifest.Aggregate.RecursiveEligibility.AllEligibilityFlagsFalse);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonArtifactEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        foreach (HnswAllowlistFilteringMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeOpenedIndexDirectoryPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.NotNull(matrixCase.LinkedReportId);
            Assert.Equal("recorded", matrixCase.BranchSummary.Status);
            Assert.Equal("recorded", matrixCase.Allowlist.Status);
            Assert.Equal("recorded", matrixCase.Tombstones.Status);
            Assert.True(matrixCase.Tombstones.SuppressionPassedForAllSearches);
            Assert.Equal("recorded", matrixCase.ExactFilteredDeltaScan.Status);
            Assert.Equal("recorded", matrixCase.Mutations.Status);
            Assert.Equal("recorded", matrixCase.Counts.Status);
            Assert.Equal("recorded", matrixCase.ReturnedResultIntegrity.Status);
            Assert.True(matrixCase.ReturnedResultIntegrity.PassedForAllSearches);
            Assert.Equal("recorded", matrixCase.Allocations.Status);
            Assert.True(matrixCase.RecursiveEligibility.AllEligibilityFlagsFalse);

            if (matrixCase.FilterProfile is "empty" or "fallback-boundary")
            {
                Assert.Equal("recorded", matrixCase.ExactFallbackParity.Status);
                Assert.True(matrixCase.ExactFallbackParity.AllSearchesPassed);
                Assert.Equal("notApplicable", matrixCase.BroadEmission.Status);
            }
            else
            {
                Assert.Equal("notApplicable", matrixCase.ExactFallbackParity.Status);
                Assert.Equal("recorded", matrixCase.BroadEmission.Status);
            }

            string linkedReportPath = ResolveRelative(manifestDirectory, matrixCase.RelativeReportPath);
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(linkedReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            Assert.Equal("VecNet.HnswAllowlistFilteringBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-149", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal(HnswAllowlistFilteringOptions.ScenarioName, reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.PhysicalVectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.FilterProfile, reportRoot.GetProperty("allowlist").GetProperty("profile").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("comparisonArtifactEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        }

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifestDocument.RootElement;
        Assert.Equal("VecNet.HnswAllowlistFilteringMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal(HnswAllowlistFilteringMatrixOptions.ScenarioName, root.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal(4, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(4, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.True(root.GetProperty("aggregate").GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());
        Assert.Equal("private-raw", root.GetProperty("eligibility").GetProperty("privacyClass").GetString());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "cacheRoot", "fashionMnist", "hnswlib");
    }

    [Fact]
    public void Run_WhenReportPathIsBlocked_RecordsBlockedCaseAndContinues()
    {
        string outputDirectory = NewArtifactDirectory("blocked-case");
        var options = new HnswAllowlistFilteringMatrixOptions(
            "smoke",
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED1501,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        Directory.CreateDirectory(Path.Combine(outputDirectory, "smoke-broad-k10", "allowlist-filtered-report.json"));

        HnswAllowlistFilteringMatrixManifest manifest =
            HnswAllowlistFilteringMatrixScenario.Run(
                options,
                [HnswAllowlistFilteringMatrixOptions.ScenarioName]);

        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(3, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(1, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal("failed", manifest.ValidationStatus);

        HnswAllowlistFilteringMatrixCaseManifest blocked =
            Assert.Single(manifest.Cases, item => item.Status == "blocked");
        Assert.Equal("smoke-broad-k10", blocked.CaseId);
        Assert.Null(blocked.LinkedReportId);
        Assert.Equal("blocked", blocked.ValidationStatus);
        Assert.Equal("notAvailable", blocked.BranchSummary.Status);
        Assert.Equal("notAvailable", blocked.Allowlist.Status);
        Assert.Equal("notAvailable", blocked.ReturnedResultIntegrity.Status);
        Assert.Equal("notAvailable", blocked.Allocations.Status);
        Assert.False(string.IsNullOrWhiteSpace(blocked.ErrorMessage));
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec150-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string ResolveRelative(string root, string relativePath) =>
        Path.GetFullPath(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static void AssertNoBooleanPropertyTrueForNames(JsonElement element, params string[] propertyNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    propertyNames.Any(name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    Assert.Fail($"Property '{property.Name}' must not be true.");
                }

                AssertNoBooleanPropertyTrueForNames(property.Value, propertyNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoBooleanPropertyTrueForNames(item, propertyNames);
            }
        }
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
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
