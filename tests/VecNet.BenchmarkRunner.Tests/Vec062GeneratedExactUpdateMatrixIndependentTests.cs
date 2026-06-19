using System.Globalization;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec062GeneratedExactUpdateMatrixIndependentTests
{
    [Fact]
    public void ParseGeneratedExactUpdateMatrix_NormalizesEmptyAndCasedCommandLines()
    {
        GeneratedExactUpdateMatrixOptions defaults = CommandLine.ParseGeneratedExactUpdateMatrix([]);

        Assert.Equal("smoke", defaults.PresetName);
        Assert.Equal(1, defaults.Runs);
        Assert.Equal(0, defaults.WarmupQueries);
        Assert.Equal(0x5EED2062u, defaults.Seed);
        Assert.Equal(Path.Combine(defaults.OutputDirectory, "exact-update-matrix-manifest.json"), defaults.ManifestPath);

        string outputDirectory = CreateArtifactDirectory("parse-casing");
        string manifestPath = Path.Combine(outputDirectory, "nested", "manifest.json");
        GeneratedExactUpdateMatrixOptions cased = CommandLine.ParseGeneratedExactUpdateMatrix(
            [
                "GENERATED-EXACT-UPDATE-MATRIX",
                "--PRESET", "StAnDaRd",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "4",
                "--SEED", "3735928559",
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
        Assert.Equal(0xDEADBEEFu, cased.Seed);
        Assert.Equal(0, cased.DuplicateInsertAttempts);
        Assert.Equal(2, cased.UnknownDeleteAttempts);
        Assert.Equal(3, cased.RepeatedDeleteAttempts);
        Assert.Equal(4, cased.DuplicateIdsPerQuery);
        Assert.Equal(5, cased.UnknownIdsPerQuery);
        Assert.Equal(outputDirectory, cased.OutputDirectory);
        Assert.Equal(manifestPath, cased.ManifestPath);
    }

    [Fact]
    public void Run_SmokeManifestCasesRoundTripAndReconcileWithLinkedVec061Reports()
    {
        string outputDirectory = CreateArtifactDirectory("linked-reconcile");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        var options = new GeneratedExactUpdateMatrixOptions(
            "SMOKE",
            Runs: 2,
            WarmupQueries: 2,
            Seed: 0x5EED_62A1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 2,
            DuplicateIdsPerQuery: 3,
            UnknownIdsPerQuery: 4,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] commandArguments =
        [
            "generated-exact-update-matrix",
            "--preset", "SMOKE",
            "--runs", "2",
            "--warmup-queries", "2",
            "--seed", "0x5EED62A1",
            "--duplicate-inserts", "0",
            "--unknown-deletes", "1",
            "--repeated-deletes", "2",
            "--duplicate-ids", "3",
            "--unknown-ids", "4",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactUpdateMatrixManifest manifest = GeneratedExactUpdateMatrixScenario.Run(options, commandArguments);
        GeneratedExactUpdateMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(4, manifest.Cases.Count(item => item.Status == "passed"));
        Assert.Equal(manifest.Cases.Length, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("local-evidence", manifest.Eligibility.ClaimClass);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Contains(manifest.Notes, note => note.Contains("checkpoint/rebuild", StringComparison.OrdinalIgnoreCase));

        foreach (GeneratedExactUpdateMatrixCaseManifest matrixCase in manifest.Cases)
        {
            GeneratedExactUpdateOptions parsed = CommandLine.ParseGeneratedExactUpdate(matrixCase.CommandArguments);
            Assert.Equal(matrixCase.Metric, parsed.Metric.ToString());
            Assert.Equal(matrixCase.Dimension, parsed.Dimension);
            Assert.Equal(matrixCase.BaseVectorCount, parsed.BaseVectorCount);
            Assert.Equal(matrixCase.QueryCount, parsed.QueryCount);
            Assert.Equal(matrixCase.TopK, parsed.TopK);
            Assert.Equal(matrixCase.Seed, string.Create(CultureInfo.InvariantCulture, $"0x{parsed.Seed:X8}"));
            Assert.Equal(matrixCase.ReportPath, parsed.OutputPath);
            Assert.Equal(matrixCase.AllowlistKind, parsed.AllowlistKind);
            Assert.Equal(matrixCase.CandidateSetKind, parsed.CandidateSetKind);
            Assert.Equal(matrixCase.DuplicateIdCountPerQuery, parsed.DuplicateIdsPerQuery);
            Assert.Equal(matrixCase.UnknownIdCountPerQuery, parsed.UnknownIdsPerQuery);

            GeneratedExactUpdateBenchmarkReport report = ReadReport(matrixCase.ReportPath);
            Assert.Equal("VecNet.ExactUpdateBenchmarkReport", report.SchemaName);
            Assert.Equal("0.1", report.SchemaVersion);
            Assert.Equal("VEC-061", report.TaskId);
            Assert.Equal("generated-exact-update", report.ScenarioName);
            Assert.Equal(matrixCase.ReportId, report.ReportId);
            Assert.Equal(matrixCase.CommandArguments, report.Command.Arguments);

            Assert.Equal(matrixCase.PhysicalVectorCount, report.Counts.PhysicalVectorCount);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, report.Counts.LiveVectorCount);
            Assert.Equal(matrixCase.BaseVectorCount, report.Counts.BaseVectorCount);
            Assert.Equal(matrixCase.InsertedDeltaVectorCount, report.Counts.DeltaVectorCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.Counts.TombstoneCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.Counts.DeletedOrReservedIdCount);
            Assert.Equal(matrixCase.ExpectedTombstoneRatio, report.Counts.TombstoneRatio, precision: 12);
            Assert.Equal("physicalVectorCount", report.Counts.TombstoneRatioDenominator);

            Assert.Equal(matrixCase.InsertedDeltaVectorCount, report.Mutations.InsertedCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount, report.Mutations.DeletedCount);
            Assert.Equal(0, report.Mutations.DuplicateInsertAttempts);
            Assert.Equal(matrixCase.UnknownDeleteAttempts, report.Mutations.UnknownDeleteAttempts);
            Assert.Equal(matrixCase.RepeatedDeleteAttempts, report.Mutations.RepeatedDeleteAttempts);
            Assert.Equal(matrixCase.InsertedDeltaVectorCount + matrixCase.DeletedBaseVectorCount, report.Mutations.CommittedMutationCount);
            Assert.Equal(report.Mutations.CommittedMutationCount, report.Mutations.GenerationDelta);
            Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
            Assert.Equal(report.Mutations.CommittedMutationCount, report.Mutations.StatusCounts.Committed);

            Assert.Equal(matrixCase.AllowlistKind, report.RawAllowlistInput.Kind);
            Assert.Equal(matrixCase.CandidateSetKind, report.CandidateSetInput.Kind);
            Assert.Equal(matrixCase.QueryCount, report.CandidateSet.ConstructedSetCount);
            Assert.Equal(report.CandidateSetInput.KnownLiveIdCountPerQuery, report.CandidateSet.CountPerQuery);
            Assert.Equal(report.CandidateSetInput.TotalKnownLiveIdCount, report.CandidateSet.TotalCandidateCount);
            Assert.True(report.CandidateSet.ConstructedAfterMutations);
            Assert.True(report.CandidateSet.ConstructedBeforeWarmupAndMeasuredSearch);

            Assert.Equal("passed", report.Validation.Status);
            Assert.True(report.Validation.MutationStatusCountsMatched);
            Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
            Assert.True(report.Validation.CandidateSetsConstructedAfterMutations);
            Assert.True(report.Validation.FinalRunUnfilteredComparedToTruth);
            Assert.True(report.Validation.FinalRunRawAllowlistComparedToTruth);
            Assert.True(report.Validation.FinalRunCandidateSetComparedToTruth);
            Assert.Equal("notMeasured", report.Measurement.MutationLatencyAndAllocation.Status);
            Assert.Equal("notMeasured", report.Measurement.LiveViewSave.Status);
            Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);
            Assert.Equal("measured", report.Measurement.UnfilteredSearch.RunToRunNoise.Status);
            Assert.Equal("measured", report.Measurement.RawAllowlistSearch.RunToRunNoise.Status);
            Assert.Equal("measured", report.Measurement.CandidateSetSearch.RunToRunNoise.Status);
            Assert.False(report.Evidence.PublicClaimEligible);
            Assert.False(report.Eligibility.BaselineCandidateEligible);
            Assert.False(report.Validation.RegressionGateEligible);
        }
    }

    [Fact]
    public void ProgramRun_WithBlockedOutputDirectoryWritesFailedManifestAndReturnsNonZero()
    {
        string directory = CreateArtifactDirectory("blocked-output");
        string blockedOutputDirectory = Path.Combine(directory, "not-a-directory");
        string manifestPath = Path.Combine(directory, "failed-manifest.json");
        File.WriteAllText(blockedOutputDirectory, "this file intentionally blocks per-case report directories");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "generated-exact-update-matrix",
                "--preset", "smoke",
                "--runs", "1",
                "--warmup-queries", "0",
                "--seed", "0x5EED62A2",
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--output-dir", blockedOutputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        GeneratedExactUpdateMatrixManifest manifest = ReadManifest(manifestPath);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
            Assert.StartsWith(blockedOutputDirectory, matrixCase.ReportPath, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(matrixCase.ReportPath));
            _ = CommandLine.ParseGeneratedExactUpdate(matrixCase.CommandArguments);
        });
    }

    [Fact]
    public void ComparisonScenario_TreatsUpdateMatrixManifestAsUnsupportedSchema()
    {
        string directory = CreateArtifactDirectory("comparison-isolation");
        string baselineManifestPath = Path.Combine(directory, "baseline-update-matrix.json");
        string currentManifestPath = Path.Combine(directory, "current-update-matrix.json");
        GeneratedExactUpdateMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "baseline")), baselineManifestPath);
        GeneratedExactUpdateMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "current")), currentManifestPath);

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
                Assert.Equal("VecNet.ExactUpdateBenchmarkMatrixManifest", reason.Actual);
                Assert.Contains("VecNet.BenchmarkMatrixManifest", reason.Expected, StringComparison.Ordinal);
            });
    }

    private static GeneratedExactUpdateMatrixManifest CreateFastFailedManifest(string outputDirectory)
    {
        var options = new GeneratedExactUpdateMatrixOptions(
            "smoke",
            Runs: 0,
            WarmupQueries: 0,
            Seed: 0x5EED_62A3,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        return GeneratedExactUpdateMatrixScenario.Run(options, ["generated-exact-update-matrix"]);
    }

    private static GeneratedExactUpdateMatrixManifest ReadManifest(string path)
    {
        GeneratedExactUpdateMatrixManifest? manifest =
            ReportWriter.Deserialize<GeneratedExactUpdateMatrixManifest>(File.ReadAllText(path));
        Assert.NotNull(manifest);
        return manifest;
    }

    private static GeneratedExactUpdateBenchmarkReport ReadReport(string path)
    {
        GeneratedExactUpdateBenchmarkReport? report =
            ReportWriter.Deserialize<GeneratedExactUpdateBenchmarkReport>(File.ReadAllText(path));
        Assert.NotNull(report);
        return report;
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec062-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
