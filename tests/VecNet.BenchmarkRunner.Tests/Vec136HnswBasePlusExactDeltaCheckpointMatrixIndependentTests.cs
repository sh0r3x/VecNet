using System.Globalization;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec136HnswBasePlusExactDeltaCheckpointMatrixIndependentTests
{
    [Fact]
    public void ParserAndExpansion_KeepAcceptedStandardOrderAndBoundedAxes()
    {
        HnswBasePlusExactDeltaCheckpointMatrixOptions defaults =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix([]);

        Assert.Equal("smoke", defaults.PresetName);
        Assert.Equal(64, defaults.BaseVectorCount);
        Assert.Equal(1, defaults.Runs);
        Assert.Equal(1, defaults.WarmupQueries);
        Assert.Equal(0x5EED2136u, defaults.Seed);
        AssertUnderArtifactRoot(defaults.OutputDirectory);
        Assert.Equal(
            Path.Combine(defaults.OutputDirectory, "hnsw-base-plus-exact-delta-checkpoint-matrix-manifest.json"),
            defaults.ManifestPath);
        Assert.Equal(12, HnswBasePlusExactDeltaCheckpointMatrixScenario.GetMinimumBaseVectorCount("smoke"));
        Assert.Equal(164, HnswBasePlusExactDeltaCheckpointMatrixScenario.GetMinimumBaseVectorCount("standard"));

        string directory = NewArtifactDirectory("standard-order");
        HnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [
                    "GENERATED-HNSW-BASE-PLUS-EXACT-DELTA-CHECKPOINT-MATRIX",
                    "--PRESET", "STANDARD",
                    "--VECTORS", "256",
                    "--QUERIES", "7",
                    "--RUNS", "2",
                    "--WARMUP-QUERIES", "3",
                    "--SEED", "0xFFFFfff0",
                    "--DUPLICATE-INSERTS", "0",
                    "--UNKNOWN-DELETES", "2",
                    "--REPEATED-DELETES", "3",
                    "--OUTPUT-DIR", directory,
                    "--MANIFEST", Path.Combine(directory, "nested", "manifest.json")
                ]);

        HnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] cases =
            HnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(options);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(16, cases.Length);
        Assert.Equal(
            [
                "case-001-low-churn-32d-1k",
                "case-002-low-churn-32d-10k",
                "case-003-tombstone-heavy-32d-10k",
                "case-004-tombstone-heavy-32d-100k",
                "case-005-low-churn-128d-1k",
                "case-006-low-churn-128d-10k",
                "case-007-tombstone-heavy-128d-10k",
                "case-008-tombstone-heavy-128d-100k",
                "case-009-low-churn-386d-1k",
                "case-010-low-churn-386d-10k",
                "case-011-tombstone-heavy-386d-10k",
                "case-012-tombstone-heavy-386d-100k",
                "case-013-low-churn-768d-1k",
                "case-014-low-churn-768d-10k",
                "case-015-tombstone-heavy-768d-10k",
                "case-016-tombstone-heavy-768d-100k"
            ],
            cases.Select(matrixCase => matrixCase.CaseId).ToArray());
        Assert.Equal(
            Enumerable.Range(0, 16).Select(offset => FormatHex(unchecked(0xFFFFfff0u + (uint)offset))).ToArray(),
            cases.Select(matrixCase => FormatHex(matrixCase.Options.Seed)).ToArray());
        Assert.All(cases, matrixCase =>
        {
            HnswBasePlusExactDeltaCheckpointOptions caseOptions = matrixCase.Options;
            Assert.Equal(VectorMetric.SquaredEuclidean, caseOptions.Metric);
            Assert.Equal(256, caseOptions.BaseVectorCount);
            Assert.Equal(7, caseOptions.QueryCount);
            Assert.Equal(2, caseOptions.Runs);
            Assert.Equal(3, caseOptions.WarmupQueries);
            Assert.Equal(16, caseOptions.M);
            Assert.Equal(128, caseOptions.EfConstruction);
            Assert.Equal(192, caseOptions.EfSearch);
            Assert.Equal("fixed-hnsw", matrixCase.HnswProfileName);
            Assert.Contains(caseOptions.Dimension, new[] { 32, 128, 386, 768 });
            Assert.Contains(caseOptions.TopK, new[] { 1, 10, 100 });
            Assert.Contains(matrixCase.UpdateProfileName, new[] { "low-churn", "tombstone-heavy" });
            Assert.False(Path.IsPathRooted(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathRooted(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.Contains("/checkpoint-output", matrixCase.RelativeCheckpointDirectoryPath, StringComparison.Ordinal);
        });
        Assert.All(
            cases.GroupBy(matrixCase => matrixCase.Options.Dimension),
            group => Assert.Equal([1, 10, 10, 100], group.Select(matrixCase => matrixCase.Options.TopK).ToArray()));
    }

    [Theory]
    [InlineData("--base-vectors", "256")]
    [InlineData("--insertions", "32")]
    [InlineData("--deletes", "16")]
    [InlineData("--delta-deletes", "0")]
    [InlineData("--output", "case.json")]
    [InlineData("--m", "16")]
    [InlineData("--ef-construction", "128")]
    [InlineData("--ef-search", "192")]
    [InlineData("--hnsw-seed", "0x484E535700013601")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--cache-root", "VecNet.DatasetCache")]
    [InlineData("--download", "false")]
    [InlineData("--truth-refresh", "true")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--actual-memory", "true")]
    [InlineData("--peak-memory", "true")]
    [InlineData("--concurrency", "4")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--public-claim", "true")]
    [InlineData("--comparison-artifact", "true")]
    [InlineData("--regression-gate", "true")]
    public void Parser_RejectsSingleCaseExternalMemoryConcurrencyComparisonAndClaimFamilies(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(option, exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ProgramRun_SmokeManifestJsonLinksRelativeVec134ReportsAndPreservesCheckpointRunDirectories()
    {
        string outputDirectory = NewArtifactDirectory("program-smoke");
        string manifestDirectory = Path.Combine(outputDirectory, "manifests");
        string manifestPath = Path.Combine(manifestDirectory, "manifest.json");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
                "--preset", "smoke",
                "--vectors", "24",
                "--queries", "2",
                "--runs", "2",
                "--warmup-queries", "1",
                "--seed", "0x5EED1366",
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(0, exitCode);
        Assert.True(File.Exists(manifestPath));

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifestDocument.RootElement;
        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-136", root.GetProperty("taskId").GetString());
        Assert.Equal("smoke", root.GetProperty("presetName").GetString());
        Assert.Equal("passed", root.GetProperty("validationStatus").GetString());
        AssertStatusCountsMatchAggregate(root);
        Assert.Equal(2, root.GetProperty("aggregate").GetProperty("linkedReportCount").GetInt32());
        Assert.Equal(4, root.GetProperty("aggregate").GetProperty("totalCheckpointRunCount").GetInt32());
        Assert.Equal(2, root.GetProperty("aggregate").GetProperty("repeatedCheckpointRunEvidenceCaseCount").GetInt32());
        Assert.Equal("local-evidence", root.GetProperty("eligibility").GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("eligibility").GetProperty("privacyClass").GetString());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("passed", matrixCase.GetProperty("status").GetString());
            Assert.Equal("passed", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal("recorded", matrixCase.GetProperty("repeatedCheckpointRuns").GetProperty("status").GetString());
            Assert.Equal(2, matrixCase.GetProperty("repeatedCheckpointRuns").GetProperty("runCount").GetInt32());
            Assert.Equal(2, matrixCase.GetProperty("validationSummary").GetProperty("detailedValidationRunNumber").GetInt32());
            Assert.True(matrixCase.GetProperty("validationSummary").GetProperty("detailedValidationUsesFinalRun").GetBoolean());
            Assert.True(matrixCase.GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());

            string relativeReportPath = matrixCase.GetProperty("linkedReportPath").GetString()!;
            string relativeCheckpointPath = matrixCase.GetProperty("linkedCheckpointDirectoryPath").GetString()!;
            Assert.False(Path.IsPathFullyQualified(relativeReportPath));
            Assert.False(Path.IsPathFullyQualified(relativeCheckpointPath));
            Assert.StartsWith("../case-", relativeReportPath, StringComparison.Ordinal);
            Assert.StartsWith("../case-", relativeCheckpointPath, StringComparison.Ordinal);

            string linkedReportPath = ResolveRelative(manifestDirectory, relativeReportPath);
            string checkpointDirectory = ResolveRelative(manifestDirectory, relativeCheckpointPath);
            Assert.True(File.Exists(linkedReportPath));
            Assert.True(Directory.Exists(checkpointDirectory));
            Assert.True(Directory.Exists(Path.Combine(checkpointDirectory, "checkpoint-run-001")));
            Assert.True(Directory.Exists(Path.Combine(checkpointDirectory, "checkpoint-run-002")));

            using JsonDocument linkedReportDocument = JsonDocument.Parse(File.ReadAllText(linkedReportPath));
            JsonElement linkedReport = linkedReportDocument.RootElement;
            Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport", linkedReport.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", linkedReport.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-134", linkedReport.GetProperty("taskId").GetString());
            Assert.Equal(2, linkedReport.GetProperty("checkpointRuns").GetProperty("runCount").GetInt32());
            Assert.Equal(2, linkedReport.GetProperty("checkpointRuns").GetProperty("runs").GetArrayLength());
            Assert.All(
                linkedReport.GetProperty("checkpointRuns").GetProperty("runs").EnumerateArray(),
                checkpointRun =>
                {
                    string runDirectory = checkpointRun.GetProperty("checkpointDirectory").GetString()!;
                    Assert.StartsWith(checkpointDirectory, Path.GetFullPath(runDirectory), StringComparison.OrdinalIgnoreCase);
                    Assert.StartsWith("checkpoint-run-", Path.GetFileName(runDirectory), StringComparison.Ordinal);
                    Assert.Equal("Published", checkpointRun.GetProperty("status").GetString());
                    Assert.Equal("Measured", checkpointRun.GetProperty("phases").GetProperty("rebuildBuild").GetProperty("status").GetString());
                });
        }
    }

    [Fact]
    public void ProgramRun_BlockedCaseOutputReturnsNonZeroAndWritesHonestManifest()
    {
        string directory = NewArtifactDirectory("blocked-program");
        string blockedOutputDirectory = Path.Combine(directory, "blocked-output-root");
        string manifestPath = Path.Combine(directory, "manifest.json");
        File.WriteAllText(blockedOutputDirectory, "file blocks case subdirectories");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
                "--preset", "smoke",
                "--vectors", "24",
                "--queries", "1",
                "--runs", "1",
                "--warmup-queries", "0",
                "--seed", "0x5EED1367",
                "--duplicate-inserts", "0",
                "--unknown-deletes", "0",
                "--repeated-deletes", "0",
                "--output-dir", blockedOutputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("failed", root.GetProperty("validationStatus").GetString());
        AssertStatusCountsMatchAggregate(root);
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(2, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("linkedReportCount").GetInt32());
        Assert.Equal("partial", root.GetProperty("aggregate").GetProperty("recursiveEligibility").GetProperty("status").GetString());
        Assert.False(root.GetProperty("aggregate").GetProperty("recursiveEligibility").GetProperty("linkedReportInspected").GetBoolean());
        Assert.False(root.GetProperty("aggregate").GetProperty("recursiveEligibility").GetProperty("allEligibilityFlagsFalse").GetBoolean());

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal("blocked", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.Equal("notAvailable", matrixCase.GetProperty("validationSummary").GetProperty("status").GetString());
            Assert.Equal("notAvailable", matrixCase.GetProperty("repeatedCheckpointRuns").GetProperty("status").GetString());
            Assert.Equal("notAvailable", matrixCase.GetProperty("checkpointSummary").GetProperty("status").GetString());
            Assert.Equal("notAvailable", matrixCase.GetProperty("preCheckpointSearch").GetProperty("status").GetString());
            Assert.Equal("notAvailable", matrixCase.GetProperty("countSummary").GetProperty("status").GetString());
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.GetProperty("errorMessage").GetString()));
            Assert.False(File.Exists(ResolveRelative(directory, matrixCase.GetProperty("linkedReportPath").GetString()!)));
        }
    }

    [Fact]
    public void AggregateSummaries_CountPassedFailedBlockedAndNonFalseLinkedEligibilityWithoutDowngrading()
    {
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest passed = MatrixCase(
            caseNumber: 1,
            status: "passed",
            validationStatus: "passed",
            linkedReportId: "passed-report",
            validationSummaryStatus: "passed",
            repeatedRunCount: 2,
            repeatedEvidencePresent: true,
            checkpointStatus: "Published",
            recallAtK: 0.99,
            orderedAgreement: 0.95,
            underfilledSlots: 0,
            outputBytes: 1000,
            nonFalseEligibility: false);
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest failed = MatrixCase(
            caseNumber: 2,
            status: "failed",
            validationStatus: "failed",
            linkedReportId: "failed-report",
            validationSummaryStatus: "failed",
            repeatedRunCount: 3,
            repeatedEvidencePresent: true,
            checkpointStatus: "Published",
            recallAtK: 0.75,
            orderedAgreement: 0.50,
            underfilledSlots: 4,
            outputBytes: 2000,
            nonFalseEligibility: true);
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest blocked = MatrixCase(
            caseNumber: 3,
            status: "blocked",
            validationStatus: "blocked",
            linkedReportId: null,
            validationSummaryStatus: "notAvailable",
            repeatedRunCount: null,
            repeatedEvidencePresent: null,
            checkpointStatus: "notAvailable",
            recallAtK: null,
            orderedAgreement: null,
            underfilledSlots: null,
            outputBytes: null,
            nonFalseEligibility: false);

        HnswBasePlusExactDeltaCheckpointMatrixAggregate aggregate =
            InvokeCreateAggregate([passed, failed, blocked], passed: 1, failed: 1, blocked: 1);

        Assert.Equal(1, aggregate.PassedCaseCount);
        Assert.Equal(1, aggregate.FailedCaseCount);
        Assert.Equal(1, aggregate.BlockedCaseCount);
        Assert.Equal(0, aggregate.SkippedCaseCount);
        Assert.Equal(2, aggregate.LinkedReportCount);
        Assert.Equal(5, aggregate.TotalCheckpointRunCount);
        Assert.Equal(1, aggregate.ValidationPassedCaseCount);
        Assert.Equal(2, aggregate.RepeatedCheckpointRunEvidenceCaseCount);
        Assert.Equal("recorded", aggregate.PreCheckpointSearch.Status);
        Assert.Equal(2, aggregate.PreCheckpointSearch.RecordedCaseCount);
        Assert.Equal(0.75, aggregate.PreCheckpointSearch.MinRecallAtK);
        Assert.Equal(0.99, aggregate.PreCheckpointSearch.MaxRecallAtK);
        Assert.Equal(0.50, aggregate.PreCheckpointSearch.MinOrderedAgreement);
        Assert.Equal(0.95, aggregate.PreCheckpointSearch.MaxOrderedAgreement);
        Assert.Equal(4, aggregate.PreCheckpointSearch.TotalUnderfilledSlotCount);
        Assert.Equal("recorded", aggregate.Checkpoint.Status);
        Assert.Equal(2, aggregate.Checkpoint.RecordedCaseCount);
        Assert.Equal(2, aggregate.Checkpoint.PublishedCaseCount);
        Assert.Equal(3000, aggregate.Checkpoint.TotalOutputBytes);
        Assert.Equal("partial", aggregate.RecursiveEligibility.Status);
        Assert.False(aggregate.RecursiveEligibility.LinkedReportInspected);
        Assert.True(aggregate.RecursiveEligibility.PublicClaimEligible);
        Assert.False(aggregate.RecursiveEligibility.AllEligibilityFlagsFalse);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixCaseManifest MatrixCase(
        int caseNumber,
        string status,
        string validationStatus,
        string? linkedReportId,
        string validationSummaryStatus,
        int? repeatedRunCount,
        bool? repeatedEvidencePresent,
        string checkpointStatus,
        double? recallAtK,
        double? orderedAgreement,
        int? underfilledSlots,
        long? outputBytes,
        bool nonFalseEligibility)
    {
        string caseId = string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}");
        return new HnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
            caseNumber,
            caseId,
            "fixed-hnsw",
            caseNumber == 2 ? "tombstone-heavy" : "low-churn",
            VectorMetric.SquaredEuclidean.ToString(),
            Dimension: 32,
            BaseVectorCount: 256,
            PhysicalVectorCount: 288,
            ExpectedLiveVectorCount: 272,
            QueryCount: 7,
            TopK: caseNumber == 1 ? 10 : 100,
            Runs: repeatedRunCount ?? 0,
            WarmupQueries: 3,
            DataSeed: "0x5EED1366",
            HnswSeed: "0x484E535700013601",
            M: 16,
            EfConstruction: 128,
            EfSearch: 192,
            InsertedDeltaVectorCount: 32,
            DeletedBaseVectorCount: 16,
            DeletedDeltaVectorCount: 0,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            LinkedReportPath: $"{caseId}/checkpoint-report.json",
            LinkedCheckpointDirectoryPath: $"{caseId}/checkpoint-output",
            CommandArguments: [HnswBasePlusExactDeltaCheckpointOptions.ScenarioName],
            linkedReportId,
            status,
            validationStatus,
            new HnswBasePlusExactDeltaCheckpointMatrixValidationSummary(
                validationSummaryStatus,
                CheckpointResultStatusPublished: repeatedRunCount is null ? null : checkpointStatus == "Published",
                CheckpointResultCountsMatched: repeatedRunCount is null ? null : validationStatus == "passed",
                CheckpointGenerationAdvancedExactlyOnce: repeatedRunCount is null ? null : true,
                PhaseDiagnosticsMeasuredForPublishedCheckpoint: repeatedRunCount is null ? null : true,
                CheckpointRepeatedRunEvidencePresent: repeatedEvidencePresent,
                DetailedValidationRunNumber: repeatedRunCount,
                DetailedValidationUsesFinalRun: repeatedRunCount is null ? null : true,
                OpenedReadOnlyHnswIdVectorValidationPassed: repeatedRunCount is null ? null : validationStatus == "passed",
                RebuiltCompositeOpenedHnswSearchParityPassed: repeatedRunCount is null ? null : validationStatus == "passed",
                ReturnedResultIntegrityPassedForAllSearches: repeatedRunCount is null ? null : validationStatus == "passed",
                NoChangesCheckpointProbePassed: repeatedRunCount is null ? null : true,
                DeletedReservedIdsRejectedAfterCheckpoint: repeatedRunCount is null ? null : true,
                OutputBytesScannedOutsideCheckpointDuration: repeatedRunCount is null ? null : true),
            new HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
                repeatedRunCount is null ? "notAvailable" : "recorded",
                repeatedRunCount,
                repeatedRunCount,
                MeanElapsedMilliseconds: repeatedRunCount is null ? null : caseNumber,
                MinElapsedMilliseconds: repeatedRunCount is null ? null : caseNumber,
                MaxElapsedMilliseconds: repeatedRunCount is null ? null : caseNumber + 1,
                MeanManagedAllocatedBytes: repeatedRunCount is null ? null : caseNumber * 10,
                MinManagedAllocatedBytes: repeatedRunCount is null ? null : caseNumber * 10,
                MaxManagedAllocatedBytes: repeatedRunCount is null ? null : caseNumber * 20),
            new HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary(
                checkpointStatus,
                FinalRunElapsedMilliseconds: outputBytes is null ? null : caseNumber,
                FinalRunManagedAllocatedBytes: outputBytes is null ? null : caseNumber * 10,
                GenerationBeforeCheckpoint: outputBytes is null ? null : 1,
                GenerationAfterCheckpoint: outputBytes is null ? null : 2,
                GenerationAdvancedExactlyOnce: outputBytes is null ? null : true,
                OutputFileCount: outputBytes is null ? null : 5,
                OutputTotalBytes: outputBytes,
                OutputScanTimingScope: outputBytes is null ? null : "outsideCheckpointDuration"),
            SearchSummary(recallAtK, orderedAgreement, underfilledSlots),
            SearchSummary(recallAtK, orderedAgreement, underfilledSlots),
            SearchSummary(recallAtK, orderedAgreement, underfilledSlots),
            new HnswBasePlusExactDeltaCheckpointMatrixCountSummary(
                repeatedRunCount is null ? "notAvailable" : "recorded",
                BasePhysicalVectorCount: 256,
                PhysicalVectorCount: 288,
                ExpectedLiveVectorCount: 272,
                PreCheckpointLiveVectorCount: repeatedRunCount is null ? null : 272,
                PreCheckpointTombstoneCount: repeatedRunCount is null ? null : 16,
                PostCheckpointBasePhysicalVectorCount: repeatedRunCount is null ? null : 272,
                PostCheckpointLiveVectorCount: repeatedRunCount is null ? null : 272,
                PostCheckpointTombstoneCount: repeatedRunCount is null ? null : 0,
                DeletedReservedIdCount: repeatedRunCount is null ? null : 16,
                PreCheckpointTombstoneRatio: repeatedRunCount is null ? null : 16.0 / 288,
                PreCheckpointDeltaInsertRatio: repeatedRunCount is null ? null : 32.0 / 256),
            new HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
                repeatedRunCount is null ? "notAvailable" : "recorded",
                LinkedReportInspected: repeatedRunCount is not null,
                PublicClaimEligible: nonFalseEligibility,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                AllEligibilityFlagsFalse: !nonFalseEligibility && repeatedRunCount is not null),
            status == "blocked" ? "blocked by test" : null);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixSearchSummary SearchSummary(
        double? recallAtK,
        double? orderedAgreement,
        int? underfilledSlots) =>
        new(
            recallAtK is null ? "notAvailable" : "recorded",
            recallAtK,
            orderedAgreement,
            recallAtK is null ? null : "passed",
            underfilledSlots is null ? null : underfilledSlots > 0 ? 1 : 0,
            underfilledSlots,
            MeanQps: recallAtK is null ? null : 1000,
            MeanLatencyP95Milliseconds: recallAtK is null ? null : 0.25,
            MeanManagedAllocatedBytesPerQuery: recallAtK is null ? null : 0);

    private static HnswBasePlusExactDeltaCheckpointMatrixAggregate InvokeCreateAggregate(
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        MethodInfo? method = typeof(HnswBasePlusExactDeltaCheckpointMatrixScenario).GetMethod(
            "CreateAggregate",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<HnswBasePlusExactDeltaCheckpointMatrixAggregate>(
            method.Invoke(null, [cases, passed, failed, blocked]));
    }

    private static void AssertStatusCountsMatchAggregate(JsonElement root)
    {
        JsonElement cases = root.GetProperty("cases");
        int passed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "passed");
        int failed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "failed");
        int skipped = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "skipped");
        int blocked = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "blocked");
        JsonElement aggregate = root.GetProperty("aggregate");

        Assert.Equal(root.GetProperty("caseCount").GetInt32(), passed + failed + skipped + blocked);
        Assert.Equal(passed, aggregate.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(failed, aggregate.GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(skipped, aggregate.GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(blocked, aggregate.GetProperty("blockedCaseCount").GetInt32());
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

    private static string ResolveRelative(string root, string relativePath) =>
        Path.GetFullPath(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));

    private static void AssertUnderArtifactRoot(string path)
    {
        string fullPath = Path.GetFullPath(path);
        string artifactRoot = Path.GetFullPath("VecNet.BenchmarkRunner.Artifacts");
        Assert.StartsWith(artifactRoot + Path.DirectorySeparatorChar, fullPath, StringComparison.OrdinalIgnoreCase);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec136-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");
}
