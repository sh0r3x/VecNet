using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec079GeneratedExactPracticalUpdateIndependentTests
{
    [Fact]
    public void Parser_AcceptsBoundaryAliasesAndRejectsOtherModeOptions()
    {
        string directory = NewArtifactDirectory("parser");
        string outputPath = Path.Combine(directory, "report.json");
        string checkpointDirectory = Path.Combine(directory, "checkpoint");

        GeneratedExactPracticalUpdateOptions parsed = CommandLine.ParseGeneratedExactPracticalUpdate(
            [
                "GENERATED-EXACT-PRACTICAL-UPDATE",
                "--METRIC", "innerproduct",
                "--DIMENSION", "386",
                "--VECTORS", "9",
                "--QUERIES", "1",
                "--TOP-K", "2",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "7",
                "--SEED", "4294967295",
                "--INSERTIONS", "1",
                "--DELETES", "8",
                "--DUPLICATE-INSERTS", "0",
                "--UNKNOWN-DELETES", "0",
                "--REPEATED-DELETES", "0",
                "--ALLOWLIST", "verySelective",
                "--CANDIDATE-SET", "verySelective",
                "--DUPLICATE-IDS", "0",
                "--UNKNOWN-IDS", "0",
                "--OUTPUT", outputPath,
                "--CHECKPOINT-DIRECTORY", checkpointDirectory
            ]);

        Assert.Equal(VectorMetric.InnerProduct, parsed.Metric);
        Assert.Equal(386, parsed.Dimension);
        Assert.Equal(9, parsed.BaseVectorCount);
        Assert.Equal(10, parsed.PhysicalVectorCount);
        Assert.Equal(2, parsed.LiveVectorCount);
        Assert.Equal(2, parsed.TopK);
        Assert.Equal(5, parsed.Runs);
        Assert.Equal(7, parsed.WarmupQueries);
        Assert.Equal(uint.MaxValue, parsed.Seed);
        Assert.Equal(1, parsed.InsertedDeltaCount);
        Assert.Equal(8, parsed.DeletedBaseCount);
        Assert.Equal("very-selective", parsed.AllowlistKind);
        Assert.Equal("very-selective", parsed.CandidateSetKind);
        Assert.Equal(outputPath, parsed.OutputPath);
        Assert.Equal(checkpointDirectory, parsed.CheckpointDirectory);

        string[][] rejected =
        [
            ["generated-exact-practical-update", "--seed", "-1"],
            ["generated-exact-practical-update", "--seed", "0x100000000"],
            ["generated-exact-practical-update", "--dimension", "1.5"],
            ["generated-exact-practical-update", "--allowlist", " "],
            ["generated-exact-practical-update", "--candidate-set", " "],
            ["generated-exact-practical-update", "--baseline", "baseline.json"],
            ["generated-exact-practical-update", "--current", "current.json"],
            ["generated-exact-practical-update", "--baseline-report-id", "baseline-id"],
            ["generated-exact-practical-update", "--query-count", "3"],
            ["generated-exact-practical-update", "--truth-depth", "10"],
            ["generated-exact-practical-update", "--download", "false"],
            ["generated-exact-practical-update", "--output-dir", "matrix"],
            ["generated-exact-practical-update", "--manifest", "manifest.json"],
            ["generated-exact-practical-update", "--snapshot-directory", "snapshot"],
            ["generated-exact-practical-update", "--m", "8"],
            ["generated-exact-practical-update", "--ef-construction", "64"],
            ["generated-exact-practical-update", "--ef-search", "50"],
            ["generated-exact-practical-update", "--hnsw-seed", "0x1234"]
        ];

        foreach (string[] args in rejected)
        {
            ArgumentException exception = Assert.Throws<ArgumentException>(
                () => CommandLine.ParseGeneratedExactPracticalUpdate(args));
            Assert.NotEmpty(exception.Message);
        }

        Assert.Throws<ArgumentException>(() => CommandLine.Parse(["exact-generated", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--vectors", "32"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpointMatrix(["generated-exact-checkpoint-matrix", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--deletes", "1"]));
    }

    [Fact]
    public void MinimalWorkload_AllowsZeroFailureAttemptsAndEmptyVisibilityInputs()
    {
        string directory = NewArtifactDirectory("minimal");
        GeneratedExactPracticalUpdateBenchmarkReport report = GeneratedExactPracticalUpdateScenario.Run(
            new GeneratedExactPracticalUpdateOptions(
                VectorMetric.InnerProduct,
                Dimension: 3,
                BaseVectorCount: 2,
                QueryCount: 2,
                TopK: 2,
                Seed: 0x5EED_7901,
                InsertedDeltaCount: 1,
                DeletedBaseCount: 1,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "empty",
                CandidateSetKind: "empty",
                DuplicateIdsPerQuery: 3,
                UnknownIdsPerQuery: 2,
                OutputPath: Path.Combine(directory, "minimal.json"),
                CheckpointDirectory: Path.Combine(directory, "checkpoint"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-practical-update", "--allowlist", "empty", "--candidate-set", "empty"]);

        Assert.Equal("VecNet.ExactPracticalUpdateModeReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-079", report.TaskId);
        Assert.Equal("generated-exact-practical-update", report.ScenarioName);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("passed", report.Validation.Status);

        Assert.Equal(2, report.Counts.InitialBaseCount);
        Assert.Equal(3, report.Counts.PhysicalVectorCountAfterMutation);
        Assert.Equal(2, report.Counts.FinalLiveCountBeforeCheckpoint);
        Assert.Equal(1, report.Counts.LiveBaseCountBeforeCheckpoint);
        Assert.Equal(1, report.Counts.LiveDeltaCountBeforeCheckpoint);
        Assert.Equal(1, report.Counts.TombstoneCountBeforeCheckpoint);
        Assert.Equal(1.0 / 3.0, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCountAfterMutation", report.Counts.TombstoneRatioDenominator);
        Assert.Equal(1.0 / 2.0, report.Counts.DeltaInsertRatio, precision: 12);
        Assert.Equal("initialBaseCount", report.Counts.DeltaInsertRatioDenominator);
        Assert.Equal(2, report.Counts.PhysicalVectorCountAfterCheckpoint);
        Assert.Equal(2, report.Counts.FinalLiveCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.DeltaCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.TombstoneCountAfterCheckpoint);

        Assert.Equal(1, report.Mutations.InsertAttemptCount);
        Assert.Equal(1, report.Mutations.InsertSuccessCount);
        Assert.Equal(1, report.Mutations.DeleteAttemptCount);
        Assert.Equal(1, report.Mutations.DeleteSuccessCount);
        Assert.Equal(0, report.Mutations.DuplicateInsertFailures);
        Assert.Equal(0, report.Mutations.UnknownDeleteFailures);
        Assert.Equal(0, report.Mutations.RepeatedDeleteFailures);
        Assert.Equal(2, report.Mutations.CommittedMutationCount);
        Assert.Equal(2, report.Mutations.StatusCounts.Committed);

        Assert.Equal(2, report.Generations.BeforeMutation);
        Assert.Equal(4, report.Generations.AfterMutation);
        Assert.Equal(4, report.Generations.BeforeCheckpoint);
        Assert.Equal(5, report.Generations.AfterCheckpoint);
        Assert.True(report.Generations.MutationDeltaMatchesCommittedMutations);
        Assert.True(report.Generations.CheckpointAdvancedExactlyOnce);

        AssertEmptyInput(report.RawAllowlistInput);
        AssertEmptyInput(report.CandidateSetInput);
        Assert.Equal(2, report.CandidateSet.FreshConstructedSetCount);
        Assert.Equal(0, report.CandidateSet.FreshCountPerQuery);
        Assert.Equal(0, report.CandidateSet.FreshTotalCandidateCount);
        Assert.True(report.CandidateSet.StaleCandidateSetConstructedBeforeMutation);
        Assert.True(report.CandidateSet.StaleCandidateSetRejectedAfterMutation);
        Assert.True(report.CandidateSet.FreshCandidateSetConstructedAfterMutation);

        AssertPassed(report.Metrics.PostMutationExactSearch, expectedCheckedResults: 4);
        AssertPassed(report.Metrics.RawAllowlistSearch, expectedCheckedResults: 0);
        AssertPassed(report.Metrics.FreshCandidateSetSearch, expectedCheckedResults: 0);
        AssertPassed(report.Metrics.ReopenedOutputSearch, expectedCheckedResults: 4);

        Assert.Equal("Published", report.Outputs.CheckpointStatus);
        Assert.Equal(3, report.Outputs.CheckpointFileCount);
        Assert.Equal(32 + (2 * sizeof(ulong)), report.Outputs.CheckpointIdsBytes);
        Assert.Equal(48 + (2 * 3 * sizeof(float)), report.Outputs.CheckpointVectorsBytes);
        Assert.Equal(2, report.Outputs.CheckpointOutputVectorCount);
        Assert.Contains("outside", report.Outputs.OutputByteScanTimingScope, StringComparison.OrdinalIgnoreCase);

        Assert.Single(report.Operations.Mutations.Runs);
        Assert.Single(report.Operations.PostMutationExactSearch.Runs);
        Assert.Equal("singleRun", report.Measurement.Mutations.RepeatedRuns.Status);
        Assert.Equal("notMeasured", report.Measurement.Mutations.RunToRunNoise.Status);
        Assert.Equal("absent", report.Measurement.Warmup.Status);
        Assert.Equal("notMeasured", report.Operations.RawAllowlistValidationSearch.Status);
        Assert.Equal("notMeasured", report.Operations.FreshCandidateSetValidationSearch.Status);
        Assert.Equal("notMeasured", report.Operations.StaleCandidateSetRejectionValidation.Status);
        AssertResourceAndEligibilityPosture(report);
    }

    [Fact]
    public void AllLiveScopes_ReportDeltaTombstoneCheckpointAndSearchParityForAdversarialDimension()
    {
        const int baseCount = 9;
        const int insertions = 2;
        const int deletes = 8;
        const int queryCount = 2;
        const int liveCount = baseCount + insertions - deletes;
        const int physicalCount = baseCount + insertions;
        string directory = NewArtifactDirectory("high-tombstone");

        GeneratedExactPracticalUpdateBenchmarkReport report = GeneratedExactPracticalUpdateScenario.Run(
            new GeneratedExactPracticalUpdateOptions(
                VectorMetric.Cosine,
                Dimension: 386,
                BaseVectorCount: baseCount,
                QueryCount: queryCount,
                TopK: liveCount,
                Seed: 0x5EED_7902,
                InsertedDeltaCount: insertions,
                DeletedBaseCount: deletes,
                DuplicateInsertAttempts: 2,
                UnknownDeleteAttempts: 1,
                RepeatedDeleteAttempts: 3,
                AllowlistKind: "all",
                CandidateSetKind: "all",
                DuplicateIdsPerQuery: 2,
                UnknownIdsPerQuery: 1,
                OutputPath: Path.Combine(directory, "high-tombstone.json"),
                CheckpointDirectory: Path.Combine(directory, "checkpoint"),
                Runs: 2,
                WarmupQueries: 5),
            ["generated-exact-practical-update", "--metric", "Cosine", "--dimension", "386"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("Cosine", report.Dataset.Metric);
        Assert.Equal(386, report.Dataset.Dimension);
        Assert.Equal(physicalCount, report.Dataset.VectorCount);
        Assert.Equal(physicalCount, report.Index.VectorCount);
        Assert.Equal(liveCount, report.Workload.TopK);

        Assert.Equal(physicalCount, report.Counts.PhysicalVectorCountAfterMutation);
        Assert.Equal(liveCount, report.Counts.FinalLiveCountBeforeCheckpoint);
        Assert.Equal(baseCount - deletes, report.Counts.LiveBaseCountBeforeCheckpoint);
        Assert.Equal(insertions, report.Counts.LiveDeltaCountBeforeCheckpoint);
        Assert.Equal(deletes, report.Counts.TombstoneCountBeforeCheckpoint);
        Assert.Equal(deletes, report.Counts.DeletedReservedIdCountBeforeCheckpoint);
        Assert.Equal((double)deletes / physicalCount, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal((double)insertions / baseCount, report.Counts.DeltaInsertRatio, precision: 12);
        Assert.Equal(liveCount, report.Counts.PhysicalVectorCountAfterCheckpoint);
        Assert.Equal(liveCount, report.Counts.FinalLiveCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.DeltaCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.TombstoneCountAfterCheckpoint);

        Assert.Equal(insertions + 2, report.Mutations.InsertAttemptCount);
        Assert.Equal(insertions, report.Mutations.InsertSuccessCount);
        Assert.Equal(deletes + 1 + 3, report.Mutations.DeleteAttemptCount);
        Assert.Equal(deletes, report.Mutations.DeleteSuccessCount);
        Assert.Equal(2, report.Mutations.DuplicateInsertFailures);
        Assert.Equal(1, report.Mutations.UnknownDeleteFailures);
        Assert.Equal(3, report.Mutations.RepeatedDeleteFailures);
        Assert.Equal(insertions + deletes, report.Mutations.CommittedMutationCount);
        Assert.Equal(insertions + deletes, report.Mutations.StatusCounts.Committed);

        Assert.Equal(baseCount, report.Generations.BeforeMutation);
        Assert.Equal(baseCount + insertions + deletes, report.Generations.AfterMutation);
        Assert.Equal(report.Generations.AfterMutation, report.Generations.BeforeCheckpoint);
        Assert.Equal(report.Generations.BeforeCheckpoint + 1, report.Generations.AfterCheckpoint);

        Assert.Equal("all", report.RawAllowlistInput.Kind);
        Assert.Equal(liveCount, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Equal(queryCount * liveCount, report.RawAllowlistInput.TotalKnownLiveIdCount);
        Assert.Contains("include live delta rows", report.RawAllowlistInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("excluding tombstoned base rows", report.RawAllowlistInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("all", report.CandidateSetInput.Kind);
        Assert.Equal(liveCount, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.Equal(liveCount, report.CandidateSet.FreshCountPerQuery);
        Assert.Equal(queryCount * liveCount, report.CandidateSet.FreshTotalCandidateCount);
        Assert.Contains("generation-bound", report.CandidateSet.Binding, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("stale search must fail", report.CandidateSet.StalePolicy, StringComparison.OrdinalIgnoreCase);

        AssertPassed(report.Metrics.PostMutationExactSearch, expectedCheckedResults: queryCount * liveCount);
        AssertPassed(report.Metrics.RawAllowlistSearch, expectedCheckedResults: queryCount * liveCount);
        AssertPassed(report.Metrics.FreshCandidateSetSearch, expectedCheckedResults: queryCount * liveCount);
        AssertPassed(report.Metrics.ReopenedOutputSearch, expectedCheckedResults: queryCount * liveCount);

        Assert.Equal(2, report.Operations.Mutations.Aggregate.RunCount);
        Assert.Equal(2, report.Operations.PostMutationExactSearch.Aggregate.RunCount);
        Assert.Equal(2, report.Operations.Checkpoint.Aggregate.RunCount);
        Assert.Equal(2, report.Operations.Open.Aggregate.RunCount);
        Assert.Equal("measured", report.Measurement.PostMutationExactSearch.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(5, report.Measurement.Warmup.WarmupCount);
        Assert.Equal("public ExactFlatIndex.Search(query, results)", report.Operations.PostMutationExactSearch.TimedOperation);
        Assert.All(report.Operations.PostMutationExactSearch.Runs, run =>
        {
            Assert.Equal(queryCount, run.OperationCount);
            Assert.Contains("summed per-query Stopwatch", run.TimingScope, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("result allocation/capture/copying", run.TimingScope, StringComparison.OrdinalIgnoreCase);
        });
        Assert.Contains("one elapsed Stopwatch sample per public ExactFlatIndex.Search", report.Measurement.PostMutationExactSearch.Latency.PercentileEstimator, StringComparison.Ordinal);
        Assert.Contains("query lookup", report.Measurement.PostMutationExactSearch.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("result allocation/capture/copying", report.Measurement.PostMutationExactSearch.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("bookkeeping", report.Measurement.Mutations.Latency.TimedOperation, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(3, report.Outputs.CheckpointFileCount);
        Assert.Equal(32 + (liveCount * sizeof(ulong)), report.Outputs.CheckpointIdsBytes);
        Assert.Equal(48 + (liveCount * 386 * sizeof(float)), report.Outputs.CheckpointVectorsBytes);
        Assert.Equal(liveCount, report.Outputs.CheckpointOutputVectorCount);
        Assert.Equal(
            report.Outputs.CheckpointManifestBytes + report.Outputs.CheckpointIdsBytes + report.Outputs.CheckpointVectorsBytes,
            report.Outputs.CheckpointOutputBytes);
        Assert.True(File.Exists(Path.Combine(report.Outputs.CheckpointDirectoryPath, "exact-flat.manifest.json")));
        Assert.True(File.Exists(Path.Combine(report.Outputs.CheckpointDirectoryPath, "exact-flat.ids.u64")));
        Assert.True(File.Exists(Path.Combine(report.Outputs.CheckpointDirectoryPath, "exact-flat.vectors.f32")));
        AssertResourceAndEligibilityPosture(report);
    }

    [Fact]
    public void ProgramRun_WritesJsonWithPrivatePostureAndNoComparisonOrPreviewSurface()
    {
        string directory = NewArtifactDirectory("program-json");
        string outputPath = Path.Combine(directory, "practical-update.json");
        string checkpointDirectory = Path.Combine(directory, "checkpoint");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "generated-exact-practical-update",
                "--metric", "SquaredEuclidean",
                "--dimension", "5",
                "--vectors", "12",
                "--queries", "2",
                "--top-k", "4",
                "--insertions", "3",
                "--deletes", "2",
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--allowlist", "broad",
                "--candidate-set", "verySelective",
                "--duplicate-ids", "1",
                "--unknown-ids", "2",
                "--runs", "1",
                "--warmup-queries", "0",
                "--seed", "0x5EED7903",
                "--output", outputPath,
                "--checkpoint-directory", checkpointDirectory
            ]);

        Assert.Equal(0, exitCode);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactPracticalUpdateModeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-079", root.GetProperty("taskId").GetString());
        Assert.Equal("generated-exact-practical-update", root.GetProperty("scenarioName").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        AssertFalseEligibility(root.GetProperty("evidence"));
        AssertFalseEligibility(root.GetProperty("eligibility"));
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualResidentMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualProcessMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualGcMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualPrivateMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualPeakMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("peakTemporaryDisk").GetProperty("status").GetString());
        Assert.Equal("measuredFinalOutputBytesOnly", root.GetProperty("resources").GetProperty("finalCheckpointOutputBytesStatus").GetString());
        Assert.Equal("notMeasured", root.GetProperty("operations").GetProperty("rawAllowlistValidationSearch").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("operations").GetProperty("freshCandidateSetValidationSearch").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("operations").GetProperty("staleCandidateSetRejectionValidation").GetProperty("status").GetString());
        AssertNoForbiddenScopeFields(root);
    }

    [Fact]
    public void GeneratedExactComparisonRejectsPracticalUpdateReportSchema()
    {
        string directory = NewArtifactDirectory("comparison");
        string reportPath = Path.Combine(directory, "practical-update.json");
        GeneratedExactPracticalUpdateBenchmarkReport report = GeneratedExactPracticalUpdateScenario.Run(
            new GeneratedExactPracticalUpdateOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 4,
                BaseVectorCount: 5,
                QueryCount: 1,
                TopK: 3,
                Seed: 0x5EED_7904,
                InsertedDeltaCount: 2,
                DeletedBaseCount: 2,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "all",
                CandidateSetKind: "all",
                DuplicateIdsPerQuery: 0,
                UnknownIdsPerQuery: 0,
                OutputPath: reportPath,
                CheckpointDirectory: Path.Combine(directory, "checkpoint"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-practical-update"]);
        GeneratedExactPracticalUpdateScenario.Write(report, reportPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(reportPath, reportPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact", "--baseline", reportPath, "--current", reportPath]);

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
                Assert.Equal("VecNet.ExactPracticalUpdateModeReport", reason.Actual);
                Assert.Contains("VecNet.BenchmarkReport", reason.Expected, StringComparison.Ordinal);
            });
    }

    private static void AssertEmptyInput(GeneratedExactUpdateFilterInputInfo input)
    {
        Assert.Equal("empty", input.Kind);
        Assert.Equal(0, input.KnownLiveIdCountPerQuery);
        Assert.Equal(3, input.DuplicateIdCountPerQuery);
        Assert.Equal(2, input.UnknownIdCountPerQuery);
        Assert.Equal(5, input.InputIdCountPerQuery);
        Assert.Equal(0, input.TotalKnownLiveIdCount);
        Assert.Equal(6, input.TotalDuplicateIdCount);
        Assert.Equal(4, input.TotalUnknownIdCount);
        Assert.Equal(0, input.ActualLiveSelectivity);
        Assert.Contains("ignored", input.UnknownIdPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("post-mutation live view", input.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertPassed(
        GeneratedExactCheckpointOperationMetricsInfo metrics,
        int expectedCheckedResults)
    {
        Assert.Equal(1.0, metrics.RecallAtK);
        Assert.Equal(1.0, metrics.OrderedAgreement);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
        Assert.Equal("passed", metrics.ResultIntegrity.Status);
        Assert.Equal(expectedCheckedResults, metrics.ResultIntegrity.CheckedResultCount);
    }

    private static void AssertResourceAndEligibilityPosture(GeneratedExactPracticalUpdateBenchmarkReport report)
    {
        Assert.Equal("notMeasured", report.Measurement.MutationManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.SearchManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.OpenManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Resources.ActualResidentMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualProcessMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualGcMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualPrivateMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualPeakMemory.Status);
        Assert.Equal("notMeasured", report.Resources.PeakTemporaryDisk.Status);
        Assert.Equal("measuredFinalOutputBytesOnly", report.Resources.FinalCheckpointOutputBytesStatus);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonArtifactEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    private static void AssertFalseEligibility(JsonElement section)
    {
        Assert.False(section.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(section.GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(section.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(section.GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(section.GetProperty("regressionGateEligible").GetBoolean());
    }

    private static void AssertNoForbiddenScopeFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baselineReportId",
            "baselineReportPath",
            "candidateEligibility",
            "comparisonResult",
            "comparisonWarning",
            "regressionPassed",
            "regressionDecision",
            "regressionThreshold",
            "threshold",
            "publicClaimPassed",
            "publicClaimStatus",
            "previewReadinessPassed",
            "previewReadinessStatus",
            "cacheRoot",
            "download",
            "truthDepth",
            "residentMemoryBytes",
            "processMemoryBytes",
            "workingSetBytes",
            "privateBytes",
            "peakMemoryBytes",
            "peakTemporaryDiskBytes",
            "snapshotDirectory",
            "hnswSeed",
            "efSearch",
            "efConstruction",
            "storedLabel",
            "labelFilter");
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

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec079-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
