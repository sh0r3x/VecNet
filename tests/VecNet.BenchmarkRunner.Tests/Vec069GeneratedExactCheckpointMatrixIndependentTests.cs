using System.Globalization;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec069GeneratedExactCheckpointMatrixIndependentTests
{
    [Fact]
    public void ParseGeneratedExactCheckpointMatrix_NormalizesEmptyAndCasedCommandLines()
    {
        GeneratedExactCheckpointMatrixOptions defaults = CommandLine.ParseGeneratedExactCheckpointMatrix([]);

        Assert.Equal("smoke", defaults.PresetName);
        Assert.Equal(1, defaults.Runs);
        Assert.Equal(0, defaults.WarmupQueries);
        Assert.Equal(0x5EED2069u, defaults.Seed);
        Assert.Equal(1, defaults.DuplicateInsertAttempts);
        Assert.Equal(1, defaults.UnknownDeleteAttempts);
        Assert.Equal(1, defaults.RepeatedDeleteAttempts);
        Assert.Equal(0, defaults.DuplicateIdsPerQuery);
        Assert.Equal(0, defaults.UnknownIdsPerQuery);
        AssertUnderArtifactRoot(defaults.OutputDirectory);
        Assert.Equal(Path.Combine(defaults.OutputDirectory, "exact-checkpoint-matrix-manifest.json"), defaults.ManifestPath);

        string outputDirectory = CreateArtifactDirectory("parse-casing");
        string manifestPath = Path.Combine(outputDirectory, "nested", "manifest.json");
        GeneratedExactCheckpointMatrixOptions cased = CommandLine.ParseGeneratedExactCheckpointMatrix(
            [
                "GENERATED-EXACT-CHECKPOINT-MATRIX",
                "--PRESET", "StAnDaRd",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "4",
                "--SEED", "4294967295",
                "--DUPLICATE-INSERTS", "0",
                "--UNKNOWN-DELETES", "2",
                "--REPEATED-DELETES", "3",
                "--DUPLICATE-IDS", "4",
                "--UNKNOWN-IDS", "5",
                "--OUTPUT-DIR", outputDirectory,
                "--MANIFEST", manifestPath
            ]);

        Assert.Equal("standard", cased.PresetName);
        Assert.Equal(5, cased.Runs);
        Assert.Equal(4, cased.WarmupQueries);
        Assert.Equal(uint.MaxValue, cased.Seed);
        Assert.Equal(0, cased.DuplicateInsertAttempts);
        Assert.Equal(2, cased.UnknownDeleteAttempts);
        Assert.Equal(3, cased.RepeatedDeleteAttempts);
        Assert.Equal(4, cased.DuplicateIdsPerQuery);
        Assert.Equal(5, cased.UnknownIdsPerQuery);
        Assert.Equal(outputDirectory, cased.OutputDirectory);
        Assert.Equal(manifestPath, cased.ManifestPath);
    }

    [Theory]
    [InlineData("--metric", "SquaredEuclidean")]
    [InlineData("--dimension", "32")]
    [InlineData("--vectors", "64")]
    [InlineData("--queries", "2")]
    [InlineData("--top-k", "10")]
    [InlineData("--insertions", "4")]
    [InlineData("--deletes", "2")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--output", "case.json")]
    [InlineData("--filter", "broad")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--cache-root", "VecNet.DatasetCache")]
    [InlineData("--query-count", "3")]
    [InlineData("--truth-depth", "10")]
    [InlineData("--download", "false")]
    [InlineData("--m", "8")]
    [InlineData("--ef-construction", "64")]
    [InlineData("--ef-search", "50")]
    [InlineData("--hnsw-seed", "0x1234")]
    public void ParseGeneratedExactCheckpointMatrix_RejectsPerCaseAndUnrelatedScenarioOptions(string optionName, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseGeneratedExactCheckpointMatrix(["generated-exact-checkpoint-matrix", optionName, value]));

        Assert.Contains(optionName, exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExpandCases_StandardPresetHasAcceptedCountsBoundsAndDimensions()
    {
        string outputDirectory = CreateArtifactDirectory("standard-bounds");
        var options = new GeneratedExactCheckpointMatrixOptions(
            "standard",
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED_69A1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactCheckpointMatrixScenario.GeneratedExactCheckpointMatrixCase[] cases =
            GeneratedExactCheckpointMatrixScenario.ExpandCases(options);

        Assert.Equal(12, cases.Length);
        Assert.Equal(4, cases.Count(item => item.Options.Metric == VectorMetric.SquaredEuclidean));
        Assert.Equal(4, cases.Count(item => item.Options.Metric == VectorMetric.InnerProduct));
        Assert.Equal(4, cases.Count(item => item.Options.Metric == VectorMetric.Cosine));
        Assert.Equal([32, 128, 386, 768], cases.Select(item => item.Options.Dimension).Distinct().Order().ToArray());
        Assert.Contains(cases, item => item.Options.Dimension == 386 && item.Options.Metric == VectorMetric.SquaredEuclidean);
        Assert.Contains(cases, item => item.Options.Dimension == 386 && item.Options.Metric == VectorMetric.InnerProduct);
        Assert.Contains(cases, item => item.Options.Dimension == 386 && item.Options.Metric == VectorMetric.Cosine);
        Assert.Contains(cases, item => item.Options.Dimension == 768 && item.Options.Metric == VectorMetric.SquaredEuclidean);
        Assert.Contains(cases, item => item.Options.Dimension == 768 && item.Options.Metric == VectorMetric.InnerProduct);
        Assert.Contains(cases, item => item.Options.Dimension == 768 && item.Options.Metric == VectorMetric.Cosine);

        Assert.All(cases, matrixCase =>
        {
            GeneratedExactCheckpointOptions caseOptions = matrixCase.Options;
            Assert.True(caseOptions.InsertedDeltaCount > 0);
            Assert.InRange(caseOptions.DeletedBaseCount, 1, caseOptions.BaseVectorCount);
            Assert.Equal(caseOptions.BaseVectorCount + caseOptions.InsertedDeltaCount, caseOptions.PhysicalVectorCount);
            Assert.Equal(caseOptions.BaseVectorCount + caseOptions.InsertedDeltaCount - caseOptions.DeletedBaseCount, caseOptions.LiveVectorCount);
            Assert.InRange(caseOptions.TopK, 1, caseOptions.LiveVectorCount);
            Assert.True(caseOptions.AllowlistKind != "very-selective" || caseOptions.TopK > 1);
            Assert.True(caseOptions.CandidateSetKind != "very-selective" || caseOptions.TopK > 1);
            AssertUnderArtifactRoot(caseOptions.OutputPath);
            Assert.StartsWith(outputDirectory, caseOptions.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(Path.GetFullPath(outputDirectory), matrixCase.CaseId, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void Run_SmokeManifestRoundTripsEveryLinkedCheckpointReport()
    {
        string outputDirectory = CreateArtifactDirectory("linked-reconcile");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        var options = new GeneratedExactCheckpointMatrixOptions(
            "SMOKE",
            Runs: 2,
            WarmupQueries: 2,
            Seed: 0x5EED_69A2,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 2,
            DuplicateIdsPerQuery: 3,
            UnknownIdsPerQuery: 4,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] commandArguments =
        [
            "generated-exact-checkpoint-matrix",
            "--preset", "SMOKE",
            "--runs", "2",
            "--warmup-queries", "2",
            "--seed", "0x5EED69A2",
            "--duplicate-inserts", "0",
            "--unknown-deletes", "1",
            "--repeated-deletes", "2",
            "--duplicate-ids", "3",
            "--unknown-ids", "4",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactCheckpointMatrixManifest manifest = GeneratedExactCheckpointMatrixScenario.Run(options, commandArguments);
        GeneratedExactCheckpointMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.ExactCheckpointBenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-069", manifest.TaskId);
        Assert.Equal("generated-exact-checkpoint-matrix", manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal("passed", manifest.ValidationStatus);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(4, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal("local-evidence", manifest.Eligibility.ClaimClass);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("smoke", manifest.Eligibility.EvidenceStatus);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.False(manifest.Eligibility.PreviewReadinessEligible);
        Assert.Contains(manifest.Notes, note => note.Contains("VecNet.ExactCheckpointBenchmarkReport schema 0.1", StringComparison.Ordinal));
        AssertUnderArtifactRoot(manifest.OutputDirectory);
        AssertUnderArtifactRoot(manifestPath);

        foreach (GeneratedExactCheckpointMatrixCaseManifest matrixCase in manifest.Cases)
        {
            GeneratedExactCheckpointOptions parsed = CommandLine.ParseGeneratedExactCheckpoint(matrixCase.CommandArguments);
            Assert.Equal(matrixCase.Metric, parsed.Metric.ToString());
            Assert.Equal(matrixCase.Dimension, parsed.Dimension);
            Assert.Equal(matrixCase.BaseVectorCount, parsed.BaseVectorCount);
            Assert.Equal(matrixCase.QueryCount, parsed.QueryCount);
            Assert.Equal(matrixCase.TopK, parsed.TopK);
            Assert.Equal(matrixCase.Seed, string.Create(CultureInfo.InvariantCulture, $"0x{parsed.Seed:X8}"));
            Assert.Equal(matrixCase.ReportPath, parsed.OutputPath);
            Assert.Equal(matrixCase.InsertedDeltaVectorCount, parsed.InsertedDeltaCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, parsed.DeletedBaseCount);
            Assert.Equal(matrixCase.AllowlistKind, parsed.AllowlistKind);
            Assert.Equal(matrixCase.CandidateSetKind, parsed.CandidateSetKind);
            Assert.Equal(matrixCase.DuplicateInsertAttempts, parsed.DuplicateInsertAttempts);
            Assert.Equal(matrixCase.UnknownDeleteAttempts, parsed.UnknownDeleteAttempts);
            Assert.Equal(matrixCase.RepeatedDeleteAttempts, parsed.RepeatedDeleteAttempts);
            Assert.Equal(matrixCase.DuplicateIdCountPerQuery, parsed.DuplicateIdsPerQuery);
            Assert.Equal(matrixCase.UnknownIdCountPerQuery, parsed.UnknownIdsPerQuery);
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ErrorMessage);
            AssertUnderArtifactRoot(matrixCase.ReportPath);

            GeneratedExactCheckpointBenchmarkReport report = ReadReport(matrixCase.ReportPath);
            Assert.Equal("VecNet.ExactCheckpointBenchmarkReport", report.SchemaName);
            Assert.Equal("0.1", report.SchemaVersion);
            Assert.Equal("VEC-067", report.TaskId);
            Assert.Equal("generated-exact-checkpoint", report.ScenarioName);
            Assert.Equal(matrixCase.ReportId, report.ReportId);
            Assert.Equal(matrixCase.CommandArguments, report.Command.Arguments);
            Assert.Equal("private-raw", report.PrivacyClass);
            Assert.Equal("local-evidence", report.ClaimClass);

            Assert.Equal(matrixCase.PhysicalVectorCount, report.Dataset.VectorCount);
            Assert.Equal(matrixCase.Dimension, report.Dataset.Dimension);
            Assert.Equal(matrixCase.BaseVectorCount, report.Workload.BaseVectorCount);
            Assert.Equal(matrixCase.InsertedDeltaVectorCount, report.Workload.InsertedDeltaVectorCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.Workload.DeletedBaseVectorCount);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, report.PreCheckpointCounts.LiveVectorCount);
            Assert.Equal(matrixCase.PhysicalVectorCount, report.PreCheckpointCounts.PhysicalVectorCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.PreCheckpointCounts.VisibilityTombstoneCount);
            Assert.Equal(matrixCase.ExpectedTombstoneRatio, report.PreCheckpointCounts.TombstoneRatio, precision: 12);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, report.PostCheckpointCounts.LiveVectorCount);
            Assert.Equal(0, report.PostCheckpointCounts.DeltaVectorCount);
            Assert.Equal(0, report.PostCheckpointCounts.VisibilityTombstoneCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.PostCheckpointCounts.DeletedReservedIdCount);

            Assert.Equal("public ExactFlatIndex.Checkpoint(directoryPath)", report.Operations.Checkpoint.TimedOperation);
            Assert.Equal(2, report.Operations.Checkpoint.Aggregate.RunCount);
            Assert.Equal("measured", report.Measurement.Checkpoint.RepeatedRuns.Status);
            Assert.Equal("measured", report.Measurement.Checkpoint.RunToRunNoise.Status);
            Assert.Equal("notMeasured", report.Measurement.CheckpointManagedAllocations.Status);
            Assert.Equal("notMeasured", report.Measurement.LiveViewSave.Status);
            Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
            Assert.Equal("passed", report.Validation.Status);
            Assert.True(report.Validation.CheckpointResultCountsMatched);
            Assert.True(report.Validation.PostCheckpointCountsMatched);
            Assert.True(report.Validation.ReopenedCheckpointOutputComparedToTruth);
            Assert.True(report.Validation.PreCheckpointCandidateSetsRejectedAsStale);
            Assert.False(report.Evidence.PublicClaimEligible);
            Assert.False(report.Validation.BaselineCandidateEligible);
            Assert.False(report.Eligibility.RegressionGateEligible);
            Assert.False(report.Eligibility.PreviewReadinessEligible);
            AssertUnderArtifactRoot(report.Outputs.CheckpointOutput.DirectoryPath);
        }
    }

    [Fact]
    public void ProgramRun_ParseOrOptionFailureAbortsBeforeManifestCreation()
    {
        string directory = CreateArtifactDirectory("parse-abort");
        string manifestPath = Path.Combine(directory, "manifest-should-not-exist.json");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "generated-exact-checkpoint-matrix",
                "--preset", "unsupported",
                "--output-dir", directory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(manifestPath));
        Assert.Empty(Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories));
    }

    [Fact]
    public void ProgramRun_WithBlockedOutputPathRecordsRecoverableFailuresAndReturnsNonZero()
    {
        string directory = CreateArtifactDirectory("blocked-output");
        string blockedOutputDirectory = Path.Combine(directory, "not-a-directory");
        string manifestPath = Path.Combine(directory, "failed-manifest.json");
        File.WriteAllText(blockedOutputDirectory, "this file intentionally blocks per-case report paths");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "generated-exact-checkpoint-matrix",
                "--preset", "smoke",
                "--runs", "1",
                "--warmup-queries", "0",
                "--seed", "0x5EED69A3",
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--output-dir", blockedOutputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        GeneratedExactCheckpointMatrixManifest manifest = ReadManifest(manifestPath);
        Assert.Equal("failed", manifest.ValidationStatus);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.False(manifest.Eligibility.PreviewReadinessEligible);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
            Assert.StartsWith(blockedOutputDirectory, matrixCase.ReportPath, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(matrixCase.ReportPath));
            _ = CommandLine.ParseGeneratedExactCheckpoint(matrixCase.CommandArguments);
        });
    }

    [Fact]
    public void ComparisonScenario_TreatsCheckpointMatrixManifestAsUnsupportedSchema()
    {
        string directory = CreateArtifactDirectory("comparison-isolation");
        string baselineManifestPath = Path.Combine(directory, "baseline-checkpoint-matrix.json");
        string currentManifestPath = Path.Combine(directory, "current-checkpoint-matrix.json");
        GeneratedExactCheckpointMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "baseline")), baselineManifestPath);
        GeneratedExactCheckpointMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "current")), currentManifestPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselineManifestPath, currentManifestPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact", "--baseline", baselineManifestPath, "--current", currentManifestPath]);

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Metrics);
        Assert.Empty(comparison.Cases);
        Assert.Null(comparison.MatrixSummary);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal(2, comparison.Compatibility.Reasons.Count(reason => reason.Code == "unsupportedSchema"));
        Assert.All(
            comparison.Compatibility.Reasons.Where(reason => reason.Code == "unsupportedSchema"),
            reason =>
            {
                Assert.Equal("schemaName", reason.Field);
                Assert.Equal("VecNet.ExactCheckpointBenchmarkMatrixManifest", reason.Actual);
                Assert.Contains("VecNet.BenchmarkMatrixManifest", reason.Expected, StringComparison.Ordinal);
            });
    }

    private static GeneratedExactCheckpointMatrixManifest CreateFastFailedManifest(string outputDirectory)
    {
        var options = new GeneratedExactCheckpointMatrixOptions(
            "smoke",
            Runs: 0,
            WarmupQueries: 0,
            Seed: 0x5EED_69A4,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        return GeneratedExactCheckpointMatrixScenario.Run(options, ["generated-exact-checkpoint-matrix"]);
    }

    private static GeneratedExactCheckpointMatrixManifest ReadManifest(string path)
    {
        GeneratedExactCheckpointMatrixManifest? manifest =
            ReportWriter.Deserialize<GeneratedExactCheckpointMatrixManifest>(File.ReadAllText(path));
        Assert.NotNull(manifest);
        return manifest;
    }

    private static GeneratedExactCheckpointBenchmarkReport ReadReport(string path)
    {
        GeneratedExactCheckpointBenchmarkReport? report =
            ReportWriter.Deserialize<GeneratedExactCheckpointBenchmarkReport>(File.ReadAllText(path));
        Assert.NotNull(report);
        return report;
    }

    private static void AssertUnderArtifactRoot(string path)
    {
        string fullPath = Path.GetFullPath(path);
        string artifactRoot = Path.GetFullPath("VecNet.BenchmarkRunner.Artifacts");
        Assert.StartsWith(artifactRoot + Path.DirectorySeparatorChar, fullPath, StringComparison.OrdinalIgnoreCase);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec069-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
