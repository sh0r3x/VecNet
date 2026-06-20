using System.Globalization;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec067GeneratedExactCheckpointIndependentTests
{
    [Fact]
    public void MinimalCheckpointWorkload_AllowsZeroFailureAttemptsAndEmptyFilters()
    {
        GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(
            new GeneratedExactCheckpointOptions(
                VectorMetric.InnerProduct,
                Dimension: 3,
                BaseVectorCount: 2,
                QueryCount: 2,
                TopK: 2,
                Seed: 0x5EED_6701,
                InsertedDeltaCount: 1,
                DeletedBaseCount: 1,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "empty",
                CandidateSetKind: "empty",
                DuplicateIdsPerQuery: 3,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("minimal-empty.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-checkpoint", "--allowlist", "empty", "--candidate-set", "empty"]);

        Assert.Equal("VecNet.ExactCheckpointBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-067", report.TaskId);
        Assert.Equal("generated-exact-checkpoint", report.ScenarioName);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("passed", report.Validation.Status);

        Assert.Equal(3, report.PreCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(2, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(1, report.PreCheckpointCounts.BaseVectorCount);
        Assert.Equal(1, report.PreCheckpointCounts.DeltaVectorCount);
        Assert.Equal(1, report.PreCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(1, report.PreCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(1.0 / 3.0, report.PreCheckpointCounts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCount", report.PreCheckpointCounts.TombstoneRatioDenominator);
        Assert.Equal(4, report.PreCheckpointCounts.Generation);
        Assert.Contains("base vectors plus inserted delta vectors", report.PreCheckpointCounts.VectorCountSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(5, report.CheckpointResult.Generation);
        Assert.Equal(2, report.CheckpointResult.PhysicalVectorCount);
        Assert.Equal(2, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(2, report.CheckpointResult.BaseVectorCount);
        Assert.Equal(0, report.CheckpointResult.DeltaVectorCount);
        Assert.Equal(0, report.CheckpointResult.TombstoneCount);
        Assert.Equal(1, report.CheckpointResult.DeletedReservedIdCount);
        Assert.Equal(1, report.CheckpointResult.FoldedDeltaVectorCount);
        Assert.Equal(1, report.CheckpointResult.FoldedTombstoneCount);

        Assert.Equal(2, report.PostCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(2, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal(2, report.PostCheckpointCounts.BaseVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(1, report.PostCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(0, report.PostCheckpointCounts.TombstoneRatio);
        Assert.Equal(5, report.PostCheckpointCounts.Generation);
        Assert.Contains("not persisted", report.PostCheckpointCounts.DeletedReservedIdSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(1, report.Mutations.InsertedCount);
        Assert.Equal(1, report.Mutations.DeletedCount);
        Assert.Equal(0, report.Mutations.DuplicateInsertAttempts);
        Assert.Equal(0, report.Mutations.UnknownDeleteAttempts);
        Assert.Equal(0, report.Mutations.RepeatedDeleteAttempts);
        Assert.Equal(2, report.Mutations.CommittedMutationCount);
        Assert.Equal(2, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(4, report.Mutations.GenerationAfterMutations);
        Assert.Equal(2, report.Mutations.GenerationDelta);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(2, report.Mutations.StatusCounts.Committed);
        Assert.Equal(0, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(0, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(0, report.Mutations.StatusCounts.AlreadyDeleted);

        AssertEmptyFilterInput(report.RawAllowlistInput);
        AssertEmptyFilterInput(report.CandidateSetInput);
        Assert.Equal("preCheckpointSetsStaleAfterPublishedCheckpointAndPostCheckpointSetsConstructedOutsideTiming", report.CandidateSet.ConstructionStatus);
        Assert.True(report.CandidateSet.PreCheckpointCandidateSetsConstructed);
        Assert.True(report.CandidateSet.PreCheckpointCandidateSetsStaleAfterPublishedCheckpoint);
        Assert.True(report.CandidateSet.PostCheckpointCandidateSetsConstructed);
        Assert.Equal(2, report.CandidateSet.ConstructedSetCount);
        Assert.Equal(0, report.CandidateSet.CountPerQuery);
        Assert.Equal(0, report.CandidateSet.TotalCandidateCount);

        AssertPassed(report.Metrics.PreCheckpointInMemorySearch, expectedCheckedResults: 4);
        AssertPassed(report.Metrics.PostCheckpointInMemorySearch, expectedCheckedResults: 4);
        AssertPassed(report.Metrics.ReopenedCheckpointOutputSearch, expectedCheckedResults: 4);
        AssertPassed(report.Metrics.PostCheckpointRawAllowlistSearch, expectedCheckedResults: 0);
        AssertPassed(report.Metrics.PostCheckpointCandidateSetSearch, expectedCheckedResults: 0);

        Assert.Equal("checkpoint", report.Operations.Checkpoint.Name);
        Assert.Equal("public ExactFlatIndex.Checkpoint(directoryPath)", report.Operations.Checkpoint.TimedOperation);
        Assert.Single(report.Operations.Checkpoint.Runs);
        Assert.Equal("Published", report.Operations.Checkpoint.Runs[0].Status);
        Assert.Equal(4, report.Operations.Checkpoint.Runs[0].GenerationBeforeCheckpoint);
        Assert.Equal(5, report.Operations.Checkpoint.Runs[0].GenerationAfterCheckpoint);
        Assert.Equal(1, report.Operations.Checkpoint.Aggregate.RunCount);
        Assert.Equal("singleRun", report.Measurement.Checkpoint.RepeatedRuns.Status);
        Assert.Equal("notMeasured", report.Measurement.Checkpoint.RunToRunNoise.Status);
        Assert.Equal("absent", report.Measurement.Warmup.Status);
        Assert.Equal("notMeasured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Operations.LiveViewSave.Status);
        Assert.Equal("notMeasured", report.Operations.PostCheckpointUnfilteredSearch.Status);
        Assert.Equal("notMeasured", report.Operations.PostCheckpointRawAllowlistSearch.Status);
        Assert.Equal("notMeasured", report.Operations.PostCheckpointCandidateSetSearch.Status);
        Assert.Equal("notIncluded", report.Operations.NoChanges.Status);
        Assert.Equal("notIncluded", report.Operations.FailureCases.Status);

        Assert.Equal("written", report.Outputs.CheckpointOutput.Status);
        Assert.True(Directory.Exists(report.Outputs.CheckpointOutput.DirectoryPath));
        Assert.Equal(3, report.Outputs.CheckpointOutput.FileCount);
        Assert.Equal(32 + (2 * sizeof(ulong)), report.Outputs.CheckpointOutput.IdsBytes);
        Assert.Equal(48 + (2 * 3 * sizeof(float)), report.Outputs.CheckpointOutput.VectorsBytes);
        Assert.Equal(report.Outputs.CheckpointOutput.ManifestBytes + report.Outputs.CheckpointOutput.IdsBytes + report.Outputs.CheckpointOutput.VectorsBytes, report.Outputs.CheckpointOutput.TotalBytes);
        Assert.Equal(2, report.Outputs.CheckpointOutput.OutputVectorCount);
        Assert.Equal("passed", report.Outputs.CheckpointOutput.ValidationOpenStatus);
        Assert.Contains("outside checkpoint duration", report.Outputs.CheckpointOutput.ScanTimingScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Outputs.SaveOutput.Status);

        Assert.True(report.Validation.PreCheckpointInMemoryComparedToTruth);
        Assert.True(report.Validation.CheckpointResultStatusPublished);
        Assert.True(report.Validation.CheckpointResultCountsMatched);
        Assert.True(report.Validation.PostCheckpointCountsMatched);
        Assert.True(report.Validation.GenerationAdvancedExactlyOnce);
        Assert.True(report.Validation.PostCheckpointInMemoryComparedToTruth);
        Assert.True(report.Validation.ReopenedCheckpointOutputComparedToTruth);
        Assert.True(report.Validation.RawAllowlistComparedToTruth);
        Assert.True(report.Validation.CandidateSetComparedToTruth);
        Assert.True(report.Validation.PreCheckpointCandidateSetsRejectedAsStale);
        Assert.True(report.Validation.DeletedReservedIdsRejectedAfterCheckpoint);
        Assert.True(report.Validation.OutputBytesScannedOutsideCheckpointDuration);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);

        Assert.Equal(3L * sizeof(ulong), report.MemoryEstimates.PreCheckpointPhysicalIdPayloadLowerBoundBytes);
        Assert.Equal(3L * 3L * sizeof(float), report.MemoryEstimates.PreCheckpointPhysicalVectorPayloadLowerBoundBytes);
        Assert.Equal(2L * 3L * sizeof(float), report.MemoryEstimates.PreCheckpointLiveVectorPayloadLowerBoundBytes);
        Assert.Equal(2L * sizeof(ulong), report.MemoryEstimates.PostCheckpointCompactIdPayloadLowerBoundBytes);
        Assert.Equal(2L * 3L * sizeof(float), report.MemoryEstimates.PostCheckpointCompactVectorPayloadLowerBoundBytes);
        Assert.Equal(2L * sizeof(ulong) + 2L * 3L * sizeof(float), report.MemoryEstimates.CheckpointSnapshotPayloadLowerBoundBytes);
        Assert.Equal(0, report.MemoryEstimates.CandidateSetOrdinalPayloadLowerBoundBytes);
        Assert.Equal("notAvailable", report.MemoryEstimates.TombstoneDeletedReservationRetainedMemory.Status);
        Assert.Equal("notAvailable", report.MemoryEstimates.RetainedHashSetCapacity.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.GcHeap.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.WorkingSet.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PrivateBytes.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PeakMemory.Status);
    }

    [Fact]
    public void HighTombstonePressureCheckpoint_ReconcilesRunsCountsAndOutputForAdversarialDimension()
    {
        const int baseCount = 9;
        const int insertions = 2;
        const int deletes = 8;
        const int liveCount = baseCount + insertions - deletes;
        const int physicalCount = baseCount + insertions;

        GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(
            new GeneratedExactCheckpointOptions(
                VectorMetric.Cosine,
                Dimension: 386,
                BaseVectorCount: baseCount,
                QueryCount: 2,
                TopK: liveCount,
                Seed: 0x5EED_6702,
                InsertedDeltaCount: insertions,
                DeletedBaseCount: deletes,
                DuplicateInsertAttempts: 2,
                UnknownDeleteAttempts: 1,
                RepeatedDeleteAttempts: 3,
                AllowlistKind: "all",
                CandidateSetKind: "very-selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("high-pressure.json"),
                Runs: 2,
                WarmupQueries: 5),
            ["generated-exact-checkpoint", "--metric", "Cosine", "--dimension", "386"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("Cosine", report.Dataset.Metric);
        Assert.Equal(386, report.Dataset.Dimension);
        Assert.Equal(physicalCount, report.Dataset.VectorCount);
        Assert.Equal(physicalCount, report.Index.VectorCount);
        Assert.Equal(liveCount, report.Workload.TopK);
        Assert.Equal(physicalCount, report.PreCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(liveCount, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(baseCount - deletes, report.PreCheckpointCounts.BaseVectorCount);
        Assert.Equal(insertions, report.PreCheckpointCounts.DeltaVectorCount);
        Assert.Equal(deletes, report.PreCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(deletes, report.PreCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal((double)deletes / physicalCount, report.PreCheckpointCounts.TombstoneRatio, precision: 12);

        Assert.Equal(liveCount, report.CheckpointResult.PhysicalVectorCount);
        Assert.Equal(liveCount, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(liveCount, report.CheckpointResult.BaseVectorCount);
        Assert.Equal(0, report.CheckpointResult.DeltaVectorCount);
        Assert.Equal(0, report.CheckpointResult.TombstoneCount);
        Assert.Equal(deletes, report.CheckpointResult.DeletedReservedIdCount);
        Assert.Equal(insertions, report.CheckpointResult.FoldedDeltaVectorCount);
        Assert.Equal(deletes, report.CheckpointResult.FoldedTombstoneCount);
        Assert.Equal(report.PreCheckpointCounts.Generation + 1, report.CheckpointResult.Generation);

        Assert.Equal(liveCount, report.PostCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(liveCount, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal(liveCount, report.PostCheckpointCounts.BaseVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(deletes, report.PostCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(report.CheckpointResult.Generation, report.PostCheckpointCounts.Generation);

        Assert.Equal(2, report.Operations.Checkpoint.Runs.Length);
        Assert.Equal(2, report.Operations.Checkpoint.Aggregate.RunCount);
        Assert.Equal("measured", report.Measurement.Checkpoint.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.Checkpoint.RunToRunNoise.Status);
        Assert.Equal("notApplicable", report.Measurement.Warmup.Status);
        Assert.Equal(5, report.Measurement.Warmup.WarmupCount);
        foreach (GeneratedExactCheckpointOperationRunInfo run in report.Operations.Checkpoint.Runs)
        {
            Assert.Equal("Published", run.Status);
            Assert.Equal(report.PreCheckpointCounts.Generation, run.GenerationBeforeCheckpoint);
            Assert.Equal(report.CheckpointResult.Generation, run.GenerationAfterCheckpoint);
            Assert.Contains("fresh ignored artifact directory", run.OutputDirectoryPolicy, StringComparison.OrdinalIgnoreCase);
            Assert.True(run.ElapsedMilliseconds >= 0);
        }

        Assert.Equal("all", report.RawAllowlistInput.Kind);
        Assert.Equal(liveCount, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Equal("very-selective", report.CandidateSetInput.Kind);
        Assert.Equal(liveCount - 1, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.Equal(report.CandidateSetInput.KnownLiveIdCountPerQuery, report.CandidateSet.CountPerQuery);
        Assert.Equal(report.CandidateSetInput.TotalKnownLiveIdCount, report.CandidateSet.TotalCandidateCount);
        Assert.Contains("generation-bound", report.CandidateSet.Binding, StringComparison.OrdinalIgnoreCase);

        AssertPassed(report.Metrics.PreCheckpointInMemorySearch, expectedCheckedResults: report.Workload.QueryCount * liveCount);
        AssertPassed(report.Metrics.PostCheckpointInMemorySearch, expectedCheckedResults: report.Workload.QueryCount * liveCount);
        AssertPassed(report.Metrics.ReopenedCheckpointOutputSearch, expectedCheckedResults: report.Workload.QueryCount * liveCount);
        AssertPassed(report.Metrics.PostCheckpointRawAllowlistSearch, expectedCheckedResults: report.Workload.QueryCount * liveCount);
        AssertPassed(report.Metrics.PostCheckpointCandidateSetSearch, expectedCheckedResults: report.Workload.QueryCount * (liveCount - 1));

        Assert.Equal(3, report.Outputs.CheckpointOutput.FileCount);
        Assert.Equal(32 + (liveCount * sizeof(ulong)), report.Outputs.CheckpointOutput.IdsBytes);
        Assert.Equal(48 + (liveCount * 386 * sizeof(float)), report.Outputs.CheckpointOutput.VectorsBytes);
        Assert.Equal(liveCount, report.Outputs.CheckpointOutput.OutputVectorCount);
        Assert.Equal("passed", report.Outputs.CheckpointOutput.ValidationOpenStatus);
        Assert.True(report.Outputs.CheckpointOutput.BytesPerLiveVector > 0);

        Assert.Equal(physicalCount * sizeof(ulong), report.MemoryEstimates.PreCheckpointPhysicalIdPayloadLowerBoundBytes);
        Assert.Equal((long)physicalCount * 386L * sizeof(float), report.MemoryEstimates.PreCheckpointPhysicalVectorPayloadLowerBoundBytes);
        Assert.Equal((long)liveCount * 386L * sizeof(float), report.MemoryEstimates.PreCheckpointLiveVectorPayloadLowerBoundBytes);
        Assert.Equal((long)liveCount * sizeof(ulong) + (long)liveCount * 386L * sizeof(float), report.MemoryEstimates.CheckpointSnapshotPayloadLowerBoundBytes);
        Assert.Equal((long)report.CandidateSet.TotalCandidateCount * sizeof(int), report.MemoryEstimates.CandidateSetOrdinalPayloadLowerBoundBytes);
    }

    [Fact]
    public void ParserAcceptsCheckpointAliasesAtBoundsAndOtherModesRejectCheckpointOptions()
    {
        GeneratedExactCheckpointOptions parsed = CommandLine.ParseGeneratedExactCheckpoint(
            [
                "GENERATED-EXACT-CHECKPOINT",
                "--METRIC", "innerproduct",
                "--DIMENSION", "386",
                "--VECTORS", "9",
                "--QUERIES", "1",
                "--TOP-K", "2",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "7",
                "--SEED", "0xFFFFFFFF",
                "--INSERTIONS", "1",
                "--DELETES", "8",
                "--DUPLICATE-INSERTS", "0",
                "--UNKNOWN-DELETES", "0",
                "--REPEATED-DELETES", "0",
                "--ALLOWLIST", "verySelective",
                "--CANDIDATE-SET", "verySelective",
                "--DUPLICATE-IDS", "0",
                "--UNKNOWN-IDS", "0",
                "--OUTPUT", "VecNet.BenchmarkRunner.Artifacts/vec067-independent-parse.json"
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

        string[][] checkpointRejects =
        [
            ["generated-exact-checkpoint", "--seed", "-1"],
            ["generated-exact-checkpoint", "--seed", "0x100000000"],
            ["generated-exact-checkpoint", "--dimension", "1.5"],
            ["generated-exact-checkpoint", "--allowlist", " "],
            ["generated-exact-checkpoint", "--candidate-set", " "],
            ["generated-exact-checkpoint", "--baseline", "baseline.json"],
            ["generated-exact-checkpoint", "--current", "current.json"],
            ["generated-exact-checkpoint", "--query-count", "3"],
            ["generated-exact-checkpoint", "--truth-depth", "10"],
            ["generated-exact-checkpoint", "--download", "false"],
            ["generated-exact-checkpoint", "--output-dir", "matrix"],
            ["generated-exact-checkpoint", "--manifest", "manifest.json"],
            ["generated-exact-checkpoint", "--ef-construction", "64"],
            ["generated-exact-checkpoint", "--ef-search", "50"],
            ["generated-exact-checkpoint", "--hnsw-seed", "0x1234"]
        ];

        foreach (string[] args in checkpointRejects)
        {
            ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(args));
            Assert.NotEmpty(exception.Message);
        }

        Assert.Throws<ArgumentException>(() => CommandLine.Parse(["exact-generated", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--checkpoint-mode", "new-or-empty-directory"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSetMatrix(["generated-exact-candidate-set-matrix", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--vectors", "32"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--allowlist", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--duplicate-inserts", "1"]));
    }

    [Fact]
    public void GeneratedExactComparisonRejectsCheckpointReportSchema()
    {
        string directory = NewArtifactDirectory("comparison-schema");
        string reportPath = Path.Combine(directory, "checkpoint-report.json");
        GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(
            new GeneratedExactCheckpointOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 5,
                BaseVectorCount: 5,
                QueryCount: 1,
                TopK: 3,
                Seed: 0x5EED_6703,
                InsertedDeltaCount: 1,
                DeletedBaseCount: 3,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "all",
                CandidateSetKind: "all",
                DuplicateIdsPerQuery: 0,
                UnknownIdsPerQuery: 0,
                OutputPath: reportPath,
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-checkpoint"]);
        GeneratedExactCheckpointScenario.Write(report, reportPath);

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
                Assert.Equal("VecNet.ExactCheckpointBenchmarkReport", reason.Actual);
                Assert.Contains("VecNet.BenchmarkReport", reason.Expected, StringComparison.Ordinal);
            });
    }

    private static void AssertEmptyFilterInput(GeneratedExactUpdateFilterInputInfo input)
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
        Assert.Contains("tombstoned IDs are excluded", input.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("committed delta IDs are eligible", input.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertPassed(GeneratedExactCheckpointOperationMetricsInfo metrics, int expectedCheckedResults)
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

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec067-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName));
        return Path.Combine(directory, fileName);
    }
}
