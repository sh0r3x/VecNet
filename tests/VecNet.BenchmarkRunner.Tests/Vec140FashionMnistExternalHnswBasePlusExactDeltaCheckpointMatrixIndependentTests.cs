using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec140FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixIndependentTests
{
    [Fact]
    public void ParserAndExpansion_KeepOnlyMatrixLevelOptionsAndAcceptedFourCaseShape()
    {
        string outputDirectory = NewArtifactDirectory("shape");
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-BASE-PLUS-EXACT-DELTA-CHECKPOINT-MATRIX",
                    "--PRESET", "STANDARD",
                    "--CACHE-ROOT", "VecNet.DatasetCache",
                    "--OUTPUT-DIR", outputDirectory,
                    "--MANIFEST", Path.Combine(outputDirectory, "manifests", "manifest.json")
                ]);

        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] standard =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(options);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase smoke =
            Assert.Single(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(options with { PresetName = "smoke" }));

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(standard[0].CaseId, smoke.CaseId);
        Assert.Equal(
            [
                "case-001-10k-low-churn-wide-m16-ef192",
                "case-002-10k-tombstone-heavy-wide-m16-ef192",
                "case-003-100k-low-churn-wide-m16-ef192",
                "case-004-100k-tombstone-heavy-wide-m16-ef192"
            ],
            standard.Select(item => item.CaseId).ToArray());
        Assert.Equal([10, 10, 100, 100], standard.Select(item => item.Options.TopK).ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy", "low-churn", "tombstone-heavy"], standard.Select(item => item.UpdateProfileName).ToArray());
        Assert.Equal(["0x5EED2139", "0x5EED213A", "0x5EED213B", "0x5EED213C"], standard.Select(item => FormatHex(item.Options.Seed)).ToArray());
        Assert.Equal(
            ["0x484E535700013901", "0x484E535700013902", "0x484E535700013903", "0x484E535700013904"],
            standard.Select(item => FormatHex(item.Options.HnswSeed)).ToArray());

        Assert.All(standard, matrixCase =>
        {
            Assert.Equal(VectorMetric.SquaredEuclidean, matrixCase.Options.Metric);
            Assert.Equal(50, matrixCase.Options.QueryCount);
            Assert.Equal(3, matrixCase.Options.WarmupQueries);
            Assert.Equal(2, matrixCase.Options.Runs);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.Equal(192, matrixCase.Options.EfSearch);
            Assert.Equal(1, matrixCase.Options.DuplicateInsertAttempts);
            Assert.Equal(1, matrixCase.Options.UnknownDeleteAttempts);
            Assert.Equal(1, matrixCase.Options.RepeatedDeleteAttempts);
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.EndsWith("/checkpoint-report.json", matrixCase.RelativeReportPath, StringComparison.Ordinal);
            Assert.EndsWith("/checkpoint-output", matrixCase.RelativeCheckpointDirectoryPath, StringComparison.Ordinal);
        });

        AssertProfile(standard[0], 59_000, 500, 100, 0, 59_500, 59_400, 100);
        AssertProfile(standard[1], 56_000, 2_000, 5_000, 500, 58_000, 52_500, 5_500);
        AssertProfile(standard[2], 59_000, 500, 100, 0, 59_500, 59_400, 100);
        AssertProfile(standard[3], 56_000, 2_000, 5_000, 500, 58_000, 52_500, 5_500);

        string[] caseArguments = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.CreateCaseArguments(standard[2].Options);
        Assert.Contains(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, caseArguments);
        Assert.Contains("--top-k", caseArguments);
        Assert.Contains("100", caseArguments);
        Assert.Contains("--warmup-queries", caseArguments);
        Assert.Contains("3", caseArguments);
        Assert.Contains("--runs", caseArguments);
        Assert.Contains("2", caseArguments);
        Assert.DoesNotContain("--preset", caseArguments);
        Assert.DoesNotContain("--output-dir", caseArguments);
        Assert.DoesNotContain("--manifest", caseArguments);
    }

    [Theory]
    [InlineData("--query-count", "50")]
    [InlineData("--queries", "50")]
    [InlineData("--runs", "2")]
    [InlineData("--warmup-queries", "3")]
    [InlineData("--metric", "squared-euclidean")]
    [InlineData("--seed", "0x5EED2139")]
    [InlineData("--top-k", "10")]
    [InlineData("--base-vectors", "59000")]
    [InlineData("--insertions", "500")]
    [InlineData("--deletes", "100")]
    [InlineData("--delta-deletes", "0")]
    [InlineData("--duplicate-inserts", "1")]
    [InlineData("--unknown-deletes", "1")]
    [InlineData("--repeated-deletes", "1")]
    [InlineData("--m", "16")]
    [InlineData("--ef-construction", "128")]
    [InlineData("--ef-search", "192")]
    [InlineData("--hnsw-seed", "0x484E535700013901")]
    [InlineData("--checkpoint-directory", "checkpoint")]
    [InlineData("--output", "case.json")]
    [InlineData("--download", "false")]
    [InlineData("--download-raw-files", "false")]
    [InlineData("--truth-refresh", "true")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--actual-memory", "true")]
    [InlineData("--peak-memory", "true")]
    [InlineData("--concurrency", "4")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--public-claim", "true")]
    [InlineData("--public-claim-eligible", "true")]
    [InlineData("--comparison-artifact", "true")]
    [InlineData("--comparison-publication", "true")]
    [InlineData("--regression-gate", "true")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--faiss-index", "index.faiss")]
    public void Parser_RejectsOutOfScopeSingleCaseRefreshMemoryConcurrencyComparisonAndClaimFamilies(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(option, exception.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("not-an-option")]
    [InlineData("--preset")]
    [InlineData("--preset", "--manifest")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "not-an-option")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "--manifest")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "large")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--cache-root", " ")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--output-dir", " ")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--manifest", " ")]
    public void Parser_RejectsMalformedBoundaries(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ProgramRun_MissingCacheWritesAllCasesBlockedManifestAndNoFakeLinkedReports()
    {
        string outputDirectory = NewArtifactDirectory("missing-cache-program");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        string missingCache = Path.Combine(outputDirectory, "missing-cache");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
                "--preset", "standard",
                "--cache-root", missingCache,
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest", GetString(root, "schemaName"));
        Assert.Equal("0.1", GetString(root, "schemaVersion"));
        Assert.Equal("VEC-140", GetString(root, "taskId"));
        Assert.Equal("standard", GetString(root, "presetName"));
        Assert.Equal("failed", GetString(root, "validationStatus"));
        Assert.Equal("unavailable", GetString(root.GetProperty("cacheTruth"), "status"));
        Assert.Contains("must not download", GetString(root.GetProperty("cacheTruth"), "cachePolicy"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("must not refresh", GetString(root.GetProperty("cacheTruth"), "truthPolicy"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(4, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(4, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("linkedReportCount").GetInt32());
        Assert.True(root.GetProperty("aggregate").GetProperty("cacheTruth").GetProperty("allCasesBlockedBySharedReadiness").GetBoolean());
        Assert.Equal("notAvailable", GetString(root.GetProperty("aggregate").GetProperty("checkpointRuns"), "status"));
        Assert.Equal(8, root.GetProperty("aggregate").GetProperty("checkpointRuns").GetProperty("requestedRunCountTotal").GetInt32());
        Assert.Equal("notMeasured", GetString(root.GetProperty("aggregate").GetProperty("memory"), "status"));
        Assert.Equal("partial", GetString(root.GetProperty("aggregate").GetProperty("recursiveEligibility"), "status"));

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", GetString(matrixCase, "status"));
            Assert.Equal("blocked", GetString(matrixCase, "validationStatus"));
            Assert.Equal("cacheTruthReadiness", GetString(matrixCase, "errorCategory"));
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportPath").ValueKind);
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedCheckpointDirectoryPath").ValueKind);
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("linkedReportValidation"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("repeatedCheckpointRuns"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("phaseDiagnostics"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("outputSummary"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("preCheckpointSourceCompositeSearch"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("openedValidation"), "status"));
            Assert.Equal("notAvailable", GetString(matrixCase.GetProperty("countSummary"), "status"));
            Assert.Equal("notMeasured", GetString(matrixCase.GetProperty("memory"), "status"));
        }

        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "comparisonPublicationEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "downloadRawFiles", "truthRefresh", "checkpointDirectory", "actualMemory", "peakMemory", "hnswlibPython", "packageMetadata", "nugetPublication");
        Assert.DoesNotContain("\"taskId\": \"VEC-138\"", File.ReadAllText(manifestPath), StringComparison.OrdinalIgnoreCase);
        Assert.Empty(Directory.EnumerateFiles(outputDirectory, "checkpoint-report.json", SearchOption.AllDirectories));
        Assert.Empty(Directory.EnumerateDirectories(outputDirectory, "checkpoint-run-*", SearchOption.AllDirectories));
    }

    [Fact]
    public void LinkedReportValidation_RejectsSchemaVersionScenarioParameterCheckpointPhaseParityReservationAndEligibilityMismatches()
    {
        (FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase matrixCase, ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report) =
            CreateSyntheticMatrixCaseAndReport("linked-validation");

        ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary valid = InvokeValidateLinkedReport(matrixCase, report);
        Assert.Equal("passed", valid.Status);

        AssertValidationFails(report with { SchemaVersion = "9.9" }, item => Assert.False(item.SchemaMatched));
        AssertValidationFails(report with { TaskId = "VEC-999" }, item => Assert.False(item.ScenarioMatched));
        AssertValidationFails(report with { Workload = report.Workload with { TopK = report.Workload.TopK + 1 } }, item => Assert.False(item.CaseParametersMatched));
        AssertValidationFails(report with { CheckpointRuns = report.CheckpointRuns with { RunCount = 1 } }, item => Assert.False(item.RequiredCheckpointSectionsPresent));
        AssertValidationFails(report with { Checkpoint = report.Checkpoint with { Phases = report.Checkpoint.Phases with { Save = report.Checkpoint.Phases.Save with { Status = "NotExecuted" } } } }, item => Assert.False(item.PhaseDiagnosticsPresent));
        AssertValidationFails(report with { OpenedValidation = report.OpenedValidation with { Status = "failed" } }, item => Assert.False(item.OpenedValidationPresent));
        AssertValidationFails(report with { OpenedValidation = report.OpenedValidation with { RebuiltCompositeOpenedSearchParity = report.OpenedValidation.RebuiltCompositeOpenedSearchParity with { AllResultsMatched = false } } }, item => Assert.False(item.RebuiltOpenedParityPassed));
        AssertValidationFails(report with { Validation = report.Validation with { DeletedReservedIdsRejectedAfterCheckpoint = false } }, item => Assert.False(item.DeletedReservationValidated));
        AssertValidationFails(report with { Evidence = report.Evidence with { PublicClaimEligible = true } }, item => Assert.False(item.EligibilityFalse));
        AssertValidationFails(report with { Validation = report.Validation with { BaselineCandidateEligible = true } }, item => Assert.False(item.EligibilityFalse));
        AssertValidationFails(report with { Eligibility = report.Eligibility with { RegressionGateEligible = true } }, item => Assert.False(item.EligibilityFalse));

        void AssertValidationFails(
            ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport mutated,
            Action<ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary> assertSpecificFlag)
        {
            ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary summary =
                InvokeValidateLinkedReport(matrixCase, mutated);

            Assert.Equal("failed", summary.Status);
            Assert.True(summary.LinkedReportInspected);
            assertSpecificFlag(summary);
        }
    }

    [Fact]
    public void AggregateSummaries_CopyRecordedValuesAndKeepFailedBlockedAndNonFalseEligibilityVisible()
    {
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest passed = MatrixCase(
            caseNumber: 1,
            status: "passed",
            validationStatus: "passed",
            linkedReportId: "linked-pass",
            completedRuns: 2,
            publishedRuns: 2,
            failedRuns: 0,
            detailedValidationFinal: true,
            measuredPhases: 2,
            outputBytes: 1_000,
            recall: 0.99,
            ordered: 0.98,
            integrity: "passed",
            distance: "passed",
            underfilledQueries: 0,
            underfilledSlots: 0,
            openedStatus: "passed",
            idMismatches: 0,
            vectorMismatches: 0,
            parityMatched: true,
            parityIdMismatches: 0,
            parityOrderMismatches: 0,
            parityDistanceMismatches: 0,
            deletedReservationPassed: true,
            deletedReservedIds: 100,
            noChangesStatus: "passed",
            noChangesGenerationUnchanged: true,
            noChangesOutputEmpty: true,
            nonFalseEligibility: 0);
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest failed = MatrixCase(
            caseNumber: 2,
            status: "failed",
            validationStatus: "failed",
            linkedReportId: "linked-failed",
            completedRuns: 2,
            publishedRuns: 1,
            failedRuns: 1,
            detailedValidationFinal: false,
            measuredPhases: 1,
            outputBytes: 2_000,
            recall: 0.72,
            ordered: 0.51,
            integrity: "failed",
            distance: "failed",
            underfilledQueries: 3,
            underfilledSlots: 9,
            openedStatus: "failed",
            idMismatches: 2,
            vectorMismatches: 1,
            parityMatched: false,
            parityIdMismatches: 4,
            parityOrderMismatches: 5,
            parityDistanceMismatches: 6,
            deletedReservationPassed: false,
            deletedReservedIds: 5_400,
            noChangesStatus: "failed",
            noChangesGenerationUnchanged: false,
            noChangesOutputEmpty: false,
            nonFalseEligibility: 3);
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest blocked = MatrixCase(
            caseNumber: 3,
            status: "blocked",
            validationStatus: "blocked",
            linkedReportId: null,
            completedRuns: null,
            publishedRuns: null,
            failedRuns: null,
            detailedValidationFinal: null,
            measuredPhases: null,
            outputBytes: null,
            recall: null,
            ordered: null,
            integrity: "notAvailable",
            distance: "notAvailable",
            underfilledQueries: null,
            underfilledSlots: null,
            openedStatus: "notAvailable",
            idMismatches: null,
            vectorMismatches: null,
            parityMatched: null,
            parityIdMismatches: null,
            parityOrderMismatches: null,
            parityDistanceMismatches: null,
            deletedReservationPassed: null,
            deletedReservedIds: null,
            noChangesStatus: "notAvailable",
            noChangesGenerationUnchanged: null,
            noChangesOutputEmpty: null,
            nonFalseEligibility: 0);

        ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate aggregate =
            InvokeCreateAggregate([passed, failed, blocked], passed: 1, failed: 1, blocked: 1);

        Assert.Equal(1, aggregate.PassedCaseCount);
        Assert.Equal(1, aggregate.FailedCaseCount);
        Assert.Equal(1, aggregate.BlockedCaseCount);
        Assert.Equal(0, aggregate.SkippedCaseCount);
        Assert.Equal(2, aggregate.LinkedReportCount);
        Assert.Equal("available", aggregate.CacheTruth.Status);
        Assert.False(aggregate.CacheTruth.AllCasesBlockedBySharedReadiness);
        Assert.Equal("recorded", aggregate.CheckpointRuns.Status);
        Assert.Equal(2, aggregate.CheckpointRuns.RecordedCaseCount);
        Assert.Equal(6, aggregate.CheckpointRuns.RequestedRunCountTotal);
        Assert.Equal(4, aggregate.CheckpointRuns.CompletedRunCount);
        Assert.Equal(3, aggregate.CheckpointRuns.PublishedRunCount);
        Assert.Equal(1, aggregate.CheckpointRuns.FailedRunCount);
        Assert.Equal(1, aggregate.CheckpointRuns.FinalRunDetailedValidationCaseCount);
        Assert.Equal("recorded", aggregate.PhaseDiagnostics.Status);
        Assert.Equal(3, aggregate.PhaseDiagnostics.RebuildBuild.MeasuredCount);
        Assert.Equal("recorded", aggregate.OutputBytes.Status);
        Assert.Equal(3_000, aggregate.OutputBytes.TotalBytes);
        Assert.Equal("outsideCheckpointDuration", aggregate.OutputBytes.ScanTimingScope);
        Assert.Equal("recorded", aggregate.PreCheckpointSourceCompositeSearch.Status);
        Assert.Equal(0.72, aggregate.PreCheckpointSourceCompositeSearch.MinRecallAtK);
        Assert.Equal(0.99, aggregate.PreCheckpointSourceCompositeSearch.MaxRecallAtK);
        Assert.Equal(0.51, aggregate.PreCheckpointSourceCompositeSearch.MinOrderedAgreement);
        Assert.Equal(0.98, aggregate.PreCheckpointSourceCompositeSearch.MaxOrderedAgreement);
        Assert.Equal(1, aggregate.PreCheckpointSourceCompositeSearch.ReturnedResultIntegrityNotPassedCaseCount);
        Assert.Equal(1, aggregate.PreCheckpointSourceCompositeSearch.DistanceToleranceNotPassedCaseCount);
        Assert.Equal(3, aggregate.PreCheckpointSourceCompositeSearch.TotalUnderfilledQueryCount);
        Assert.Equal(9, aggregate.PreCheckpointSourceCompositeSearch.TotalUnderfilledSlotCount);
        Assert.Equal(1, aggregate.OpenedValidation.PassedCaseCount);
        Assert.Equal(2, aggregate.OpenedValidation.IdMismatchCount);
        Assert.Equal(1, aggregate.RebuiltOpenedParity.PassedCaseCount);
        Assert.Equal(4, aggregate.RebuiltOpenedParity.IdMismatchCount);
        Assert.Equal(5, aggregate.RebuiltOpenedParity.OrderMismatchCount);
        Assert.Equal(6, aggregate.RebuiltOpenedParity.DistanceMismatchCount);
        Assert.Equal(1, aggregate.DeletedReservation.PassedCaseCount);
        Assert.Equal(5_500, aggregate.DeletedReservation.ExpectedDeletedReservedIdCountTotal);
        Assert.Equal(5_500, aggregate.DeletedReservation.ActualDeletedReservedIdCountTotal);
        Assert.Equal(1, aggregate.NoChanges.PassedCaseCount);
        Assert.Equal(1, aggregate.NoChanges.GenerationChangedCaseCount);
        Assert.Equal(1, aggregate.NoChanges.OutputDirectoryNotEmptyCaseCount);
        Assert.Equal("notMeasured", aggregate.Memory.Status);
        Assert.Equal("partial", aggregate.RecursiveEligibility.Status);
        Assert.False(aggregate.RecursiveEligibility.LinkedReportInspected);
        Assert.Equal(3, aggregate.RecursiveEligibility.NonFalseEligibilityFlagCount);
        Assert.False(aggregate.RecursiveEligibility.AllEligibilityFlagsFalse);
    }

    [Fact]
    public void DefaultAndExpandedArtifactPathsStayUnderIgnoredRunnerArtifactRoot()
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions defaults =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix([]);

        AssertUnderArtifactRoot(defaults.OutputDirectory);
        AssertUnderArtifactRoot(defaults.ManifestPath);
        AssertIgnoredByGit(defaults.ManifestPath);

        string outputDirectory = NewArtifactDirectory("ignore-policy");
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options =
            new("standard", "VecNet.DatasetCache", outputDirectory, Path.Combine(outputDirectory, "manifest.json"));
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase matrixCase =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(options)[0];

        AssertUnderArtifactRoot(matrixCase.Options.OutputPath);
        AssertUnderArtifactRoot(matrixCase.Options.CheckpointDirectory);
        AssertIgnoredByGit(matrixCase.Options.OutputPath);
        AssertIgnoredByGit(Path.Combine(matrixCase.Options.CheckpointDirectory, "checkpoint-run-001", "manifest.json"));
    }

    private static void AssertProfile(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase matrixCase,
        int baseRows,
        int deltaRows,
        int baseDeletes,
        int deltaDeletes,
        int physicalCandidates,
        int liveRows,
        int deletedReservedIds)
    {
        Assert.Equal(baseRows, matrixCase.Options.BaseVectorCount);
        Assert.Equal(deltaRows, matrixCase.Options.InsertedDeltaCount);
        Assert.Equal(baseDeletes, matrixCase.Options.DeletedBaseCount);
        Assert.Equal(deltaDeletes, matrixCase.Options.DeletedDeltaCount);
        Assert.Equal(physicalCandidates, matrixCase.Options.PhysicalCandidateVectorCount);
        Assert.Equal(liveRows, matrixCase.Options.LiveVectorCount);
        Assert.Equal(deletedReservedIds, matrixCase.UpdateProfile.ExpectedDeletedReservedIdCount);
    }

    private static (FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase MatrixCase, ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport Report)
        CreateSyntheticMatrixCaseAndReport(string prefix)
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(prefix, baseCount: 44, queryCount: 6, truthDepth: 6);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputDirectory = Path.Combine(cacheRoot, "matrix");
        string caseDirectory = Path.Combine(outputDirectory, "case-001-10k-low-churn-wide-m16-ef192");
        var generatedOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            Path.Combine(caseDirectory, "checkpoint-report.json"),
            Path.Combine(caseDirectory, "checkpoint-output"),
            QueryCount: 4,
            TopK: 4,
            BaseVectorCount: 30,
            InsertedDeltaCount: 6,
            DeletedBaseCount: 4,
            DeletedDeltaCount: 2,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 2,
            WarmupQueries: 1,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED2140,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x484E535700013940);
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport generatedReport =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                generatedOptions,
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.CreateCaseArguments(generatedOptions));

        var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            Path.Combine(caseDirectory, "checkpoint-report.json"),
            Path.Combine(caseDirectory, "checkpoint-output"),
            QueryCount: 50,
            TopK: 10,
            BaseVectorCount: 59_000,
            InsertedDeltaCount: 500,
            DeletedBaseCount: 100,
            DeletedDeltaCount: 0,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 2,
            WarmupQueries: 3,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED2139,
            M: 16,
            EfConstruction: 128,
            EfSearch: 192,
            HnswSeed: 0x484E535700013901);
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report = generatedReport with
        {
            Workload = generatedReport.Workload with
            {
                MeasuredQueryCount = 50,
                TopK = 10,
                ImmutableBaseRowCount = 59_000,
                DeltaRowCount = 500,
                DeletedBaseVectorCount = 100,
                DeletedDeltaVectorCount = 0,
                CheckpointRunCount = 2,
                WarmupQueryCount = 3
            },
            Hnsw = generatedReport.Hnsw with
            {
                M = 16,
                MMax = 16,
                MMax0 = 32,
                EfConstruction = 128,
                EfSearch = 192,
                RandomSeed = "0x484E535700013901"
            }
        };
        var matrixCase = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase(
            "case-001-10k-low-churn-wide-m16-ef192",
            "low-churn",
            "wide-m16-ef192",
            "case-001-10k-low-churn-wide-m16-ef192/checkpoint-report.json",
            "case-001-10k-low-churn-wide-m16-ef192/checkpoint-output",
            new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExternalCheckpointUpdateProfile(
                "low-churn",
                BaseRowCount: 30,
                DeltaRowCount: 6,
                DeletedBaseCount: 4,
                DeletedDeltaCount: 2,
                ExpectedPhysicalCandidateCount: 36,
                ExpectedLiveCount: 30,
                ExpectedDeletedReservedIdCount: 6,
                "independent synthetic profile"),
            caseOptions);

        return (matrixCase, report);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary InvokeValidateLinkedReport(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase matrixCase,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario).GetMethod(
            "ValidateLinkedReport",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        return Assert.IsType<ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary>(
            method.Invoke(null, [matrixCase, report]));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate InvokeCreateAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario).GetMethod(
            "CreateAggregate",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        var cacheTruth = new ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo(
            "available",
            "VecNet.DatasetCache",
            "fashion-mnist-784-euclidean",
            784,
            VectorMetric.SquaredEuclidean.ToString(),
            "cache available",
            "truth guard available",
            "manifest.json",
            "manifest-sha",
            "truth.json",
            "truth-sha",
            60_000,
            10_000,
            50,
            100,
            50,
            100,
            59_500,
            ErrorMessage: null);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate>(
            method.Invoke(null, [cases, passed, failed, blocked, cacheTruth, null]));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest MatrixCase(
        int caseNumber,
        string status,
        string validationStatus,
        string? linkedReportId,
        int? completedRuns,
        int? publishedRuns,
        int? failedRuns,
        bool? detailedValidationFinal,
        int? measuredPhases,
        long? outputBytes,
        double? recall,
        double? ordered,
        string integrity,
        string distance,
        int? underfilledQueries,
        int? underfilledSlots,
        string openedStatus,
        int? idMismatches,
        int? vectorMismatches,
        bool? parityMatched,
        int? parityIdMismatches,
        int? parityOrderMismatches,
        int? parityDistanceMismatches,
        bool? deletedReservationPassed,
        int? deletedReservedIds,
        string noChangesStatus,
        bool? noChangesGenerationUnchanged,
        bool? noChangesOutputEmpty,
        int nonFalseEligibility)
    {
        string caseId = string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}");
        bool recorded = linkedReportId is not null;
        int expectedDeletedReservedIds = caseNumber == 1 ? 100 : 5_400;

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
            caseNumber,
            caseId,
            caseNumber == 2 ? "tombstone-heavy" : "low-churn",
            "wide-m16-ef192",
            "fashion-mnist-784-euclidean",
            VectorMetric.SquaredEuclidean.ToString(),
            Dimension: 784,
            QueryCount: 50,
            TopK: caseNumber == 3 ? 100 : 10,
            CheckpointRunCount: 2,
            WarmupQueries: 3,
            WorkloadSeed: "0x5EED2139",
            HnswSeed: "0x484E535700013901",
            M: 16,
            EfConstruction: 128,
            EfSearch: 192,
            ImmutableBaseStartRow: 0,
            ImmutableBaseEndRowInclusive: 58_999,
            ImmutableBaseRowCount: 59_000,
            DeltaStartRow: 59_000,
            DeltaEndRowInclusive: 59_499,
            DeltaRowCount: 500,
            UnusedStartRow: 59_500,
            UnusedEndRowInclusive: 59_999,
            UnusedCandidateRowCount: 500,
            DeletedBaseVectorCount: 100,
            DeletedDeltaVectorCount: 0,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            ExpectedPhysicalCandidateVectorCount: 59_500,
            ExpectedLiveVectorCount: 59_400,
            expectedDeletedReservedIds,
            LinkedReportPath: recorded ? $"{caseId}/checkpoint-report.json" : null,
            LinkedCheckpointDirectoryPath: recorded ? $"{caseId}/checkpoint-output" : null,
            CommandArguments: [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName],
            linkedReportId,
            status,
            validationStatus,
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary(
                recorded ? validationStatus : "notAvailable",
                recorded,
                recorded,
                recorded,
                recorded,
                recorded,
                recorded,
                recorded,
                parityMatched,
                deletedReservationPassed,
                nonFalseEligibility == 0),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
                recorded ? "recorded" : "notAvailable",
                RequestedRunCount: 2,
                completedRuns,
                publishedRuns,
                NoChangesRunCount: 0,
                failedRuns,
                DetailedValidationRunNumber: recorded ? 2 : null,
                DetailedValidationUsesFinalRun: detailedValidationFinal,
                MeanElapsedMilliseconds: recorded ? caseNumber : null,
                MinElapsedMilliseconds: recorded ? caseNumber : null,
                MaxElapsedMilliseconds: recorded ? caseNumber + 0.5 : null,
                MeanManagedAllocatedBytes: recorded ? caseNumber * 10 : null,
                MinManagedAllocatedBytes: recorded ? caseNumber * 10 : null,
                MaxManagedAllocatedBytes: recorded ? caseNumber * 20 : null),
            PhaseSummary(measuredPhases),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary(
                recorded ? "recorded" : "notAvailable",
                FileCount: outputBytes is null ? null : 5,
                TotalBytes: outputBytes,
                ManifestBytes: outputBytes is null ? null : 100,
                IdsBytes: outputBytes is null ? null : 200,
                VectorsBytes: outputBytes is null ? null : 300,
                LevelsBytes: outputBytes is null ? null : 400,
                GraphBytes: outputBytes is null ? null : outputBytes - 1_000,
                OutputVectorCount: outputBytes is null ? null : 59_400,
                BytesPerLiveVector: outputBytes is null ? null : outputBytes.Value / 59_400.0,
                ValidationOpenStatus: outputBytes is null ? null : openedStatus,
                ScanTimingScope: outputBytes is null ? null : "outsideCheckpointDuration"),
            SearchSummary(recall, ordered, integrity, distance, underfilledQueries, underfilledSlots),
            SearchSummary(recall, ordered, integrity, distance, underfilledQueries, underfilledSlots),
            SearchSummary(recall, ordered, integrity, distance, underfilledQueries, underfilledSlots),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary(
                openedStatus,
                ExpectedVectorCount: recorded ? 59_400 : null,
                OpenedVectorCount: recorded ? 59_400 : null,
                idMismatches,
                vectorMismatches),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary(
                parityMatched is null ? "notAvailable" : parityMatched.Value ? "passed" : "failed",
                QueryCount: recorded ? 50 : null,
                WrittenCountMismatchCount: recorded ? 0 : null,
                parityIdMismatches,
                parityOrderMismatches,
                parityDistanceMismatches,
                parityMatched),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary(
                deletedReservationPassed is null ? "notAvailable" : deletedReservationPassed.Value ? "passed" : "failed",
                deletedReservationPassed,
                expectedDeletedReservedIds,
                deletedReservedIds),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary(
                noChangesStatus,
                noChangesGenerationUnchanged,
                noChangesOutputEmpty,
                PhaseSummary(recorded ? 0 : null)),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary(
                recorded ? "recorded" : "notAvailable",
                ExpectedBasePhysicalVectorCount: 59_000,
                ExpectedDeltaPhysicalVectorCount: 500,
                ExpectedPhysicalCandidateVectorCount: 59_500,
                ExpectedLiveVectorCount: 59_400,
                expectedDeletedReservedIds,
                PreCheckpointLiveVectorCount: recorded ? 59_400 : null,
                PreCheckpointTombstoneCount: recorded ? 100 : null,
                PreCheckpointDeletedReservedIdCount: recorded ? expectedDeletedReservedIds : null,
                PostCheckpointBasePhysicalVectorCount: recorded ? 59_400 : null,
                PostCheckpointLiveVectorCount: recorded ? 59_400 : null,
                PostCheckpointTombstoneCount: recorded ? 0 : null,
                PostCheckpointDeletedReservedIdCount: deletedReservedIds,
                PreCheckpointTombstoneRatio: recorded ? 100.0 / 59_500 : null,
                PreCheckpointDeltaInsertRatio: recorded ? 500.0 / 59_000 : null),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixMemorySummary(
                "notMeasured",
                "bytes",
                "Actual/process/resident/peak memory is not measured by VEC-140."),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
                recorded ? "recorded" : "notAvailable",
                LinkedReportInspected: recorded,
                nonFalseEligibility,
                PublicClaimEligible: nonFalseEligibility > 0,
                BaselineCandidateEligible: nonFalseEligibility > 1,
                ComparisonArtifactEligible: nonFalseEligibility > 2,
                ComparisonPublicationEligible: false,
                RegressionGateEligible: false,
                AllEligibilityFlagsFalse: recorded && nonFalseEligibility == 0),
            status == "blocked" ? "caseRuntimeBlock" : null,
            status == "blocked" ? "blocked by independent test" : null);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary PhaseSummary(int? measuredCount) =>
        new(
            measuredCount is null ? "notAvailable" : "recorded",
            Phase(measuredCount),
            Phase(measuredCount),
            Phase(measuredCount),
            Phase(measuredCount),
            Phase(measuredCount));

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary Phase(int? measuredCount) =>
        new(
            measuredCount is null ? "notAvailable" : "recorded",
            measuredCount ?? 0,
            measuredCount == 0 ? 2 : 0,
            measuredCount is null ? 1 : 0,
            measuredCount is null ? null : measuredCount * 1.25,
            measuredCount is null ? null : measuredCount * 10);

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary SearchSummary(
        double? recall,
        double? ordered,
        string integrity,
        string distance,
        int? underfilledQueries,
        int? underfilledSlots) =>
        new(
            recall is null ? "notAvailable" : "recorded",
            recall,
            ordered,
            recall is null ? null : distance,
            DistanceMismatchCount: recall is null ? null : distance == "passed" ? 0 : 1,
            MissingResultCount: recall is null ? null : 0,
            ExtraResultCount: recall is null ? null : 0,
            recall is null ? null : integrity,
            CheckedResultCount: recall is null ? null : 50,
            UnknownIdCount: recall is null ? null : integrity == "passed" ? 0 : 1,
            TombstonedIdCount: recall is null ? null : 0,
            IntegrityDistanceMismatchCount: recall is null ? null : integrity == "passed" ? 0 : 1,
            underfilledQueries,
            underfilledSlots,
            MeanQps: recall is null ? null : 1000,
            MeanLatencyP95Milliseconds: recall is null ? null : 0.5,
            MeanManagedAllocatedBytesPerQuery: recall is null ? null : 0);

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = NewArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 3, columns: 5);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false),
            ["external-fashion-mnist", "--download", "false"],
            spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 41)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 2)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 73)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 5)).ToArray());

        static FashionMnistRawFileSpec Spec(string path, string fileName, string role, int expectedCount) =>
            new(fileName, role, expectedCount, FileChecksum.ComputeMd5(path), downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            datasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: baseCount,
            QueryCount: queryCount,
            ImageRows: rows,
            ImageColumns: columns,
            Dimension: checked(rows * columns),
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", baseCount),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", baseCount),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", queryCount),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", queryCount));
    }

    private static byte[] CreatePixels(int count, int dimension, int offset)
    {
        var payload = new byte[checked(count * dimension)];
        for (int row = 0; row < count; row++)
        {
            for (int column = 0; column < dimension; column++)
            {
                payload[(row * dimension) + column] = (byte)((row * 19 + column * 29 + offset + (row % 5) * 7) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count, int offset)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)((i + offset) % 10);
        }

        return labels;
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2051);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2049);
        WriteInt32BigEndian(decoded, count);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream Gzip(byte[] decoded)
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            gzip.Write(decoded);
        }

        compressed.Position = 0;
        return compressed;
    }

    private static void WriteInt32BigEndian(Stream stream, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, value);
        stream.Write(buffer);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec140-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static string GetString(JsonElement element, string propertyName) =>
        element.GetProperty(propertyName).GetString() ?? string.Empty;

    private static void AssertUnderArtifactRoot(string path)
    {
        string relative = Path.GetRelativePath(Directory.GetCurrentDirectory(), Path.GetFullPath(path));
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", relative, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertIgnoredByGit(string path)
    {
        using var process = new Process();
        process.StartInfo = new ProcessStartInfo("git")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        process.StartInfo.ArgumentList.Add("check-ignore");
        process.StartInfo.ArgumentList.Add("-q");
        process.StartInfo.ArgumentList.Add(path);
        process.Start();
        process.WaitForExit(5000);

        Assert.True(process.ExitCode == 0, $"Expected '{path}' to be ignored by git.");
    }

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

    private static void AssertNoPropertyNamed(JsonElement element, params string[] propertyNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(propertyNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                AssertNoPropertyNamed(property.Value, propertyNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, propertyNames);
            }
        }
    }
}
