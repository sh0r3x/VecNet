using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec136HnswBasePlusExactDeltaCheckpointMatrixTests
{
    [Fact]
    public void ParseHnswBasePlusExactDeltaCheckpointMatrix_UsesPrivateSmokeDefaults()
    {
        HnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(64, options.BaseVectorCount);
        Assert.Equal(4, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED2136u, options.Seed);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathFullyQualified(options.OutputDirectory));
        Assert.EndsWith("hnsw-base-plus-exact-delta-checkpoint-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseHnswBasePlusExactDeltaCheckpointMatrix_StandardDefaultsUseTwoCheckpointRuns()
    {
        HnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, "--preset", "standard"]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(256, options.BaseVectorCount);
        Assert.Equal(2, options.Runs);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "large")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", " ")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--vectors", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "standard", "--vectors", "163")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "standard", "--runs", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--queries", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--runs", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--runs", "6")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--duplicate-inserts", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--unknown-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--repeated-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--output-dir", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--manifest", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--dimension", "128")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--top-k", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--insertions", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--deletes", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--delta-deletes", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--m", "16")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--ef-search", "192")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint-matrix", "--checkpoint-directory", "checkpoint")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--output-dir", "matrix")]
    public void ParseHnswBasePlusExactDeltaCheckpointMatrix_RejectsInvalidOrOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    [InlineData("InnerProduct")]
    [InlineData("innerproduct")]
    public void ParseHnswBasePlusExactDeltaCheckpointMatrix_AcceptsGeneratedHnswMetrics(string metric)
    {
        HnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, "--metric", metric]);

        Assert.True(options.Metric is VectorMetric.Cosine or VectorMetric.InnerProduct);
    }

    [Fact]
    public void ExpandCases_SmokeAndStandardPresetsUseAcceptedShapes()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        var smokeOptions = new HnswBasePlusExactDeltaCheckpointMatrixOptions(
            "smoke",
            BaseVectorCount: 64,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 1,
            Seed: 0xFFFF_FFFEu,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        var standardOptions = smokeOptions with
        {
            PresetName = "standard",
            BaseVectorCount = 256,
            Runs = 2,
            Metric = VectorMetric.Cosine
        };

        HnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] smokeCases =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(smokeOptions);
        HnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] standardCases =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(standardOptions);

        Assert.Equal(2, smokeCases.Length);
        Assert.Equal(16, standardCases.Length);
        Assert.Equal([32, 128, 386, 768], standardCases.Select(item => item.Options.Dimension).Distinct().Order().ToArray());
        Assert.Equal([1, 10, 100], standardCases.Select(item => item.Options.TopK).Distinct().Order().ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], standardCases.Select(item => item.UpdateProfileName).Distinct().Order().ToArray());

        Assert.All(standardCases, matrixCase =>
        {
            Assert.Equal(VectorMetric.Cosine, matrixCase.Options.Metric);
            Assert.Equal(256, matrixCase.Options.BaseVectorCount);
            Assert.Equal(2, matrixCase.Options.Runs);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.Equal(192, matrixCase.Options.EfSearch);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.False(Path.IsPathRooted(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathRooted(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);
        });

        Assert.Equal(0xFFFF_FFFEu, standardCases[0].Options.Seed);
        Assert.Equal(13u, standardCases[^1].Options.Seed);
        Assert.Equal("0x484EACA8FFFF3601", FormatHex(standardCases[0].Options.HnswSeed));
        Assert.Equal("0x484EACA8FFFF3610", FormatHex(standardCases[^1].Options.HnswSeed));
    }

    [Fact]
    public void Run_SmokeManifestLinksVec134ReportsWithCheckpointSummariesAndFalseEligibility()
    {
        string outputDirectory = NewArtifactDirectory("smoke-manifest");
        string manifestDirectory = Path.Combine(outputDirectory, "manifests");
        string manifestPath = Path.Combine(manifestDirectory, "manifest.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
            "--preset", "smoke",
            "--vectors", "24",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "1",
            "--seed", "0x5EED1360",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "4",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];
        HnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(arguments);

        HnswBasePlusExactDeltaCheckpointMatrixManifest manifest =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.Run(options, arguments);
        HnswBasePlusExactDeltaCheckpointMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-136", manifest.TaskId);
        Assert.Equal(HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(2, manifest.CaseCount);
        Assert.Equal("passed", manifest.ValidationStatus);
        Assert.Equal(2, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal(2, manifest.Aggregate.LinkedReportCount);
        Assert.Equal(4, manifest.Aggregate.TotalCheckpointRunCount);
        Assert.Equal(2, manifest.Aggregate.ValidationPassedCaseCount);
        Assert.Equal(2, manifest.Aggregate.RepeatedCheckpointRunEvidenceCaseCount);
        Assert.Equal("recorded", manifest.Aggregate.Checkpoint.Status);
        Assert.Equal(2, manifest.Aggregate.Checkpoint.PublishedCaseCount);
        Assert.True(manifest.Aggregate.Checkpoint.TotalOutputBytes > 0);
        Assert.True(manifest.Aggregate.RecursiveEligibility.AllEligibilityFlagsFalse);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("smoke", manifest.Eligibility.EvidenceStatus);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonArtifactEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        foreach (HnswBasePlusExactDeltaCheckpointMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.False(Path.IsPathRooted(matrixCase.LinkedReportPath));
            Assert.False(Path.IsPathRooted(matrixCase.LinkedCheckpointDirectoryPath));
            Assert.NotNull(matrixCase.LinkedReportId);
            Assert.Equal(2, matrixCase.Runs);
            Assert.Equal("passed", matrixCase.ValidationSummary.Status);
            Assert.True(matrixCase.ValidationSummary.CheckpointRepeatedRunEvidencePresent);
            Assert.True(matrixCase.ValidationSummary.DetailedValidationUsesFinalRun);
            Assert.Equal("recorded", matrixCase.RepeatedCheckpointRuns.Status);
            Assert.Equal(2, matrixCase.RepeatedCheckpointRuns.RunCount);
            Assert.Equal(2, matrixCase.RepeatedCheckpointRuns.DetailedValidationRunNumber);
            Assert.Equal("Published", matrixCase.CheckpointSummary.Status);
            Assert.True(matrixCase.CheckpointSummary.OutputTotalBytes > 0);
            Assert.Equal("outsideCheckpointDuration", matrixCase.CheckpointSummary.OutputScanTimingScope);
            Assert.Equal("recorded", matrixCase.PreCheckpointSearch.Status);
            Assert.Equal("recorded", matrixCase.PostCheckpointSearch.Status);
            Assert.Equal("recorded", matrixCase.OpenedReadOnlySearch.Status);
            Assert.Equal("passed", matrixCase.OpenedReadOnlySearch.ReturnedResultIntegrityStatus);
            Assert.Equal("recorded", matrixCase.CountSummary.Status);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, matrixCase.CountSummary.PreCheckpointLiveVectorCount);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, matrixCase.CountSummary.PostCheckpointLiveVectorCount);
            Assert.Equal(0, matrixCase.CountSummary.PostCheckpointTombstoneCount);
            Assert.True(matrixCase.RecursiveEligibility.AllEligibilityFlagsFalse);

            string linkedReportPath = ResolveRelative(manifestDirectory, matrixCase.LinkedReportPath);
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(linkedReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-134", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal(HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.PhysicalVectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal(2, reportRoot.GetProperty("checkpointRuns").GetProperty("runCount").GetInt32());
            Assert.Equal(2, reportRoot.GetProperty("checkpointRuns").GetProperty("runs").GetArrayLength());
            Assert.Equal("Measured", reportRoot.GetProperty("checkpoint").GetProperty("phases").GetProperty("rebuildBuild").GetProperty("status").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("comparisonArtifactEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());

            string checkpointDirectory = ResolveRelative(manifestDirectory, matrixCase.LinkedCheckpointDirectoryPath);
            Assert.True(Directory.Exists(Path.Combine(checkpointDirectory, "checkpoint-run-001")));
            Assert.True(Directory.Exists(Path.Combine(checkpointDirectory, "checkpoint-run-002")));
        }

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifestDocument.RootElement;
        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal(HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, root.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal(2, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(2, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.True(root.GetProperty("aggregate").GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
    }

    [Fact]
    public void Run_WhenReportPathIsBlocked_RecordsBlockedCaseAndContinues()
    {
        string outputDirectory = NewArtifactDirectory("blocked-case");
        var options = new HnswBasePlusExactDeltaCheckpointMatrixOptions(
            "smoke",
            BaseVectorCount: 24,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED1361,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        Directory.CreateDirectory(Path.Combine(outputDirectory, "case-002-tombstone-heavy-32d-10k", "checkpoint-report.json"));

        HnswBasePlusExactDeltaCheckpointMatrixManifest manifest =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.Run(
                options,
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName]);

        Assert.Equal(2, manifest.CaseCount);
        Assert.Equal(1, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(1, manifest.Aggregate.BlockedCaseCount);

        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest blocked =
            Assert.Single(manifest.Cases, item => item.Status == "blocked");
        Assert.Equal(2, blocked.CaseNumber);
        Assert.Equal("blocked", blocked.ValidationStatus);
        Assert.Null(blocked.LinkedReportId);
        Assert.Equal("notAvailable", blocked.ValidationSummary.Status);
        Assert.Equal("notAvailable", blocked.RepeatedCheckpointRuns.Status);
        Assert.Equal("notAvailable", blocked.CheckpointSummary.Status);
        Assert.Equal("notAvailable", blocked.PreCheckpointSearch.Status);
        Assert.Equal("notAvailable", blocked.CountSummary.Status);
        Assert.False(string.IsNullOrWhiteSpace(blocked.ErrorMessage));
    }

    [Fact]
    public void Run_WhenCaseOptionsAreInvalid_RecordsFailedCases()
    {
        string outputDirectory = NewArtifactDirectory("failed-case");
        var options = new HnswBasePlusExactDeltaCheckpointMatrixOptions(
            "smoke",
            BaseVectorCount: 1,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED1362,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswBasePlusExactDeltaCheckpointMatrixManifest manifest =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.Run(
                options,
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName]);

        Assert.Equal(2, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(2, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.Equal("notAvailable", matrixCase.ValidationSummary.Status);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
        });
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec136-{prefix}-{Guid.NewGuid():N}"));
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
}
