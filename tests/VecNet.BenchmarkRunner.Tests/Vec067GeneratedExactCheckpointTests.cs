using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec067GeneratedExactCheckpointTests
{
    [Fact]
    public void ParseGeneratedExactCheckpoint_UsesPrivateSmokeDefaults()
    {
        GeneratedExactCheckpointOptions options = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.BaseVectorCount);
        Assert.Equal(11_000, options.PhysicalVectorCount);
        Assert.Equal(10_000, options.LiveVectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal("broad", options.AllowlistKind);
        Assert.Equal("selective", options.CandidateSetKind);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2067u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-checkpoint", "--dimension")]
    [InlineData("generated-exact-checkpoint", "dimension", "8")]
    [InlineData("generated-exact-checkpoint", "--metric", "Unknown")]
    [InlineData("generated-exact-checkpoint", "--dimension", "0")]
    [InlineData("generated-exact-checkpoint", "--vectors", "0")]
    [InlineData("generated-exact-checkpoint", "--queries", "0")]
    [InlineData("generated-exact-checkpoint", "--top-k", "11", "--vectors", "10", "--insertions", "1", "--deletes", "1")]
    [InlineData("generated-exact-checkpoint", "--runs", "0")]
    [InlineData("generated-exact-checkpoint", "--runs", "6")]
    [InlineData("generated-exact-checkpoint", "--warmup-queries", "-1")]
    [InlineData("generated-exact-checkpoint", "--insertions", "0")]
    [InlineData("generated-exact-checkpoint", "--deletes", "0")]
    [InlineData("generated-exact-checkpoint", "--deletes", "11", "--vectors", "10")]
    [InlineData("generated-exact-checkpoint", "--duplicate-inserts", "-1")]
    [InlineData("generated-exact-checkpoint", "--unknown-deletes", "-1")]
    [InlineData("generated-exact-checkpoint", "--repeated-deletes", "-1")]
    [InlineData("generated-exact-checkpoint", "--allowlist", "unknown")]
    [InlineData("generated-exact-checkpoint", "--candidate-set", "unknown")]
    [InlineData("generated-exact-checkpoint", "--allowlist", "very-selective", "--top-k", "1")]
    [InlineData("generated-exact-checkpoint", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-checkpoint", "--unknown-ids", "-1")]
    [InlineData("generated-exact-checkpoint", "--filter", "broad")]
    [InlineData("generated-exact-checkpoint", "--preset", "smoke")]
    [InlineData("generated-exact-checkpoint", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-checkpoint", "--output-dir", "matrix")]
    [InlineData("generated-exact-checkpoint", "--manifest", "manifest.json")]
    [InlineData("generated-exact-checkpoint", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-exact-checkpoint", "--m", "8")]
    [InlineData("generated-exact-checkpoint", "--output", "")]
    public void ParseGeneratedExactCheckpoint_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedExactCheckpointReportWithCountsOutputBytesAndValidation()
    {
        string outputPath = NewArtifactPath("exact-checkpoint-report.json");
        string[] arguments =
        [
            "generated-exact-checkpoint",
            "--metric", "SquaredEuclidean",
            "--dimension", "11",
            "--vectors", "40",
            "--queries", "5",
            "--top-k", "6",
            "--insertions", "7",
            "--deletes", "5",
            "--duplicate-inserts", "3",
            "--unknown-deletes", "4",
            "--repeated-deletes", "2",
            "--allowlist", "broad",
            "--candidate-set", "selective",
            "--duplicate-ids", "2",
            "--unknown-ids", "3",
            "--runs", "3",
            "--warmup-queries", "2",
            "--seed", "0x5EED067A",
            "--output", outputPath
        ];
        GeneratedExactCheckpointOptions options = CommandLine.ParseGeneratedExactCheckpoint(arguments);

        GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(options, arguments);
        GeneratedExactCheckpointScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExactCheckpointBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-067", report.TaskId);
        Assert.Equal("generated-exact-checkpoint", report.ScenarioName);
        Assert.Equal("generated-exact-checkpoint", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-checkpoint-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);

        Assert.Equal(47, report.PreCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(42, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(35, report.PreCheckpointCounts.BaseVectorCount);
        Assert.Equal(7, report.PreCheckpointCounts.DeltaVectorCount);
        Assert.Equal(5, report.PreCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(5, report.PreCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(5.0 / 47.0, report.PreCheckpointCounts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCount", report.PreCheckpointCounts.TombstoneRatioDenominator);
        Assert.Equal(52, report.PreCheckpointCounts.Generation);
        Assert.Contains("not persisted", report.PreCheckpointCounts.DeletedReservedIdSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(53, report.CheckpointResult.Generation);
        Assert.Equal(42, report.CheckpointResult.PhysicalVectorCount);
        Assert.Equal(42, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(42, report.CheckpointResult.BaseVectorCount);
        Assert.Equal(0, report.CheckpointResult.DeltaVectorCount);
        Assert.Equal(0, report.CheckpointResult.TombstoneCount);
        Assert.Equal(5, report.CheckpointResult.DeletedReservedIdCount);
        Assert.Equal(7, report.CheckpointResult.FoldedDeltaVectorCount);
        Assert.Equal(5, report.CheckpointResult.FoldedTombstoneCount);

        Assert.Equal(42, report.PostCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(42, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal(42, report.PostCheckpointCounts.BaseVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(5, report.PostCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(0, report.PostCheckpointCounts.TombstoneRatio);
        Assert.Equal(53, report.PostCheckpointCounts.Generation);

        Assert.Equal(7, report.Mutations.InsertedCount);
        Assert.Equal(5, report.Mutations.DeletedCount);
        Assert.Equal(12, report.Mutations.CommittedMutationCount);
        Assert.Equal(40, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(52, report.Mutations.GenerationAfterMutations);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(12, report.Mutations.StatusCounts.Committed);
        Assert.Equal(3, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(4, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);

        Assert.Equal("broad", report.RawAllowlistInput.Kind);
        Assert.Equal(21, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Equal("selective", report.CandidateSetInput.Kind);
        Assert.Equal(5, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.True(report.CandidateSet.PreCheckpointCandidateSetsConstructed);
        Assert.True(report.CandidateSet.PreCheckpointCandidateSetsStaleAfterPublishedCheckpoint);
        Assert.True(report.CandidateSet.PostCheckpointCandidateSetsConstructed);
        Assert.Equal(5, report.CandidateSet.ConstructedSetCount);
        Assert.Equal(5, report.CandidateSet.CountPerQuery);
        Assert.Equal(25, report.CandidateSet.TotalCandidateCount);

        Assert.Equal("checkpoint", report.Operations.Checkpoint.Name);
        Assert.Equal("public ExactFlatIndex.Checkpoint(directoryPath)", report.Operations.Checkpoint.TimedOperation);
        Assert.Equal(3, report.Operations.Checkpoint.Runs.Length);
        Assert.Equal(3, report.Operations.Checkpoint.Aggregate.RunCount);
        Assert.True(report.Operations.Checkpoint.Aggregate.MeanElapsedMilliseconds >= 0);
        Assert.All(report.Operations.Checkpoint.Runs, run =>
        {
            Assert.Equal("Published", run.Status);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.Equal(52, run.GenerationBeforeCheckpoint);
            Assert.Equal(53, run.GenerationAfterCheckpoint);
        });
        Assert.Equal("measured", report.Measurement.Checkpoint.Latency.Status);
        Assert.Equal("perCheckpointCall", report.Measurement.Checkpoint.Latency.SampleScope);
        Assert.Equal("measured", report.Measurement.Checkpoint.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.Checkpoint.RunToRunNoise.Status);
        Assert.Equal("notMeasured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.LiveViewSave.Status);
        Assert.Equal("notMeasured", report.Measurement.PostCheckpointSearchTiming.Status);
        Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);
        Assert.Equal("notApplicable", report.Measurement.Warmup.Status);
        Assert.Contains("output-byte scans", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("written", report.Outputs.CheckpointOutput.Status);
        Assert.True(Directory.Exists(report.Outputs.CheckpointOutput.DirectoryPath));
        Assert.Equal(3, report.Outputs.CheckpointOutput.FileCount);
        Assert.True(report.Outputs.CheckpointOutput.TotalBytes > 0);
        Assert.True(report.Outputs.CheckpointOutput.ManifestBytes > 0);
        Assert.Equal(32 + (42 * 8), report.Outputs.CheckpointOutput.IdsBytes);
        Assert.Equal(48 + (42 * 11 * 4), report.Outputs.CheckpointOutput.VectorsBytes);
        Assert.Equal(42, report.Outputs.CheckpointOutput.OutputVectorCount);
        Assert.True(report.Outputs.CheckpointOutput.BytesPerLiveVector > 0);
        Assert.Equal("passed", report.Outputs.CheckpointOutput.ValidationOpenStatus);
        Assert.Contains("outside checkpoint duration", report.Outputs.CheckpointOutput.ScanTimingScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Outputs.SaveOutput.Status);

        AssertPassed(report.Metrics.PreCheckpointInMemorySearch);
        AssertPassed(report.Metrics.PostCheckpointInMemorySearch);
        AssertPassed(report.Metrics.ReopenedCheckpointOutputSearch);
        AssertPassed(report.Metrics.PostCheckpointRawAllowlistSearch);
        AssertPassed(report.Metrics.PostCheckpointCandidateSetSearch);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.LiveTruthGenerated);
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

        Assert.Equal("estimatedPayloadLowerBounds", report.MemoryEstimates.Status);
        Assert.Equal(47L * sizeof(ulong), report.MemoryEstimates.PreCheckpointPhysicalIdPayloadLowerBoundBytes);
        Assert.Equal(47L * 11L * sizeof(float), report.MemoryEstimates.PreCheckpointPhysicalVectorPayloadLowerBoundBytes);
        Assert.Equal(42L * 11L * sizeof(float), report.MemoryEstimates.PreCheckpointLiveVectorPayloadLowerBoundBytes);
        Assert.Equal(42L * sizeof(ulong), report.MemoryEstimates.PostCheckpointCompactIdPayloadLowerBoundBytes);
        Assert.Equal(42L * 11L * sizeof(float), report.MemoryEstimates.PostCheckpointCompactVectorPayloadLowerBoundBytes);
        Assert.Equal(42L * sizeof(ulong) + 42L * 11L * sizeof(float), report.MemoryEstimates.CheckpointSnapshotPayloadLowerBoundBytes);
        Assert.Equal(25L * sizeof(int), report.MemoryEstimates.CandidateSetOrdinalPayloadLowerBoundBytes);
        Assert.Equal("notAvailable", report.MemoryEstimates.TombstoneDeletedReservationRetainedMemory.Status);
        Assert.Equal("notAvailable", report.MemoryEstimates.RetainedHashSetCapacity.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.GcHeap.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.WorkingSet.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PrivateBytes.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PeakMemory.Status);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactCheckpointBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("generated-exact-checkpoint", root.GetProperty("scenarioName").GetString());
        Assert.Equal("Published", root.GetProperty("checkpointResult").GetProperty("status").GetString());
        Assert.Equal(47, root.GetProperty("preCheckpointCounts").GetProperty("physicalVectorCount").GetInt32());
        Assert.Equal(42, root.GetProperty("postCheckpointCounts").GetProperty("physicalVectorCount").GetInt32());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("checkpoint").GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("public ExactFlatIndex.Checkpoint(directoryPath)", root.GetProperty("measurement").GetProperty("checkpoint").GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.True(root.GetProperty("outputs").GetProperty("checkpointOutput").GetProperty("totalBytes").GetInt64() > 0);
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Run_ValidatesCheckpointParityAcrossExactMetrics(VectorMetric metric)
    {
        GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(
            new GeneratedExactCheckpointOptions(
                metric,
                Dimension: 7,
                BaseVectorCount: 24,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED_6720,
                InsertedDeltaCount: 5,
                DeletedBaseCount: 3,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "all",
                CandidateSetKind: "very-selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputPath: NewArtifactPath("metric.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-checkpoint"]);

        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(29, report.PreCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(26, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(26, report.PostCheckpointCounts.PhysicalVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.VisibilityTombstoneCount);
        Assert.Equal(5, report.CheckpointResult.FoldedDeltaVectorCount);
        Assert.Equal(3, report.CheckpointResult.FoldedTombstoneCount);
        AssertPassed(report.Metrics.PreCheckpointInMemorySearch);
        AssertPassed(report.Metrics.PostCheckpointInMemorySearch);
        AssertPassed(report.Metrics.ReopenedCheckpointOutputSearch);
        AssertPassed(report.Metrics.PostCheckpointRawAllowlistSearch);
        AssertPassed(report.Metrics.PostCheckpointCandidateSetSearch);
        Assert.True(report.Validation.GenerationAdvancedExactlyOnce);
        Assert.Equal("singleRun", report.Measurement.Checkpoint.RepeatedRuns.Status);
        Assert.Equal("notMeasured", report.Measurement.Checkpoint.RunToRunNoise.Status);
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndCheckpointModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--output-dir", "matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--checkpoint-mode", "new-or-empty-directory"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--insertions", "2"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--deletes", "2"]));
        Assert.Equal("generated-exact-checkpoint", GeneratedExactCheckpointOptions.ScenarioName);
    }

    private static void AssertPassed(GeneratedExactCheckpointOperationMetricsInfo metrics)
    {
        Assert.Equal(1.0, metrics.RecallAtK);
        Assert.Equal(1.0, metrics.OrderedAgreement);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
        Assert.Equal("passed", metrics.ResultIntegrity.Status);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec067-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
