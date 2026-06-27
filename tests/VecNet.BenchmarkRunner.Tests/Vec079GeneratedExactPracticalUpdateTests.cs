using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec079GeneratedExactPracticalUpdateTests
{
    [Fact]
    public void ParseGeneratedExactPracticalUpdate_UsesPrivateSmokeDefaults()
    {
        GeneratedExactPracticalUpdateOptions options =
            CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update"]);

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
        Assert.Equal(0x5EED2079u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.CheckpointDirectory);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.False(Path.IsPathRooted(options.CheckpointDirectory));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-practical-update", "--dimension")]
    [InlineData("generated-exact-practical-update", "dimension", "8")]
    [InlineData("generated-exact-practical-update", "--metric", "Unknown")]
    [InlineData("generated-exact-practical-update", "--dimension", "0")]
    [InlineData("generated-exact-practical-update", "--vectors", "0")]
    [InlineData("generated-exact-practical-update", "--queries", "0")]
    [InlineData("generated-exact-practical-update", "--top-k", "11", "--vectors", "10", "--insertions", "1", "--deletes", "1")]
    [InlineData("generated-exact-practical-update", "--runs", "0")]
    [InlineData("generated-exact-practical-update", "--runs", "6")]
    [InlineData("generated-exact-practical-update", "--warmup-queries", "-1")]
    [InlineData("generated-exact-practical-update", "--insertions", "0")]
    [InlineData("generated-exact-practical-update", "--deletes", "0")]
    [InlineData("generated-exact-practical-update", "--deletes", "11", "--vectors", "10")]
    [InlineData("generated-exact-practical-update", "--duplicate-inserts", "-1")]
    [InlineData("generated-exact-practical-update", "--unknown-deletes", "-1")]
    [InlineData("generated-exact-practical-update", "--repeated-deletes", "-1")]
    [InlineData("generated-exact-practical-update", "--allowlist", "unknown")]
    [InlineData("generated-exact-practical-update", "--candidate-set", "unknown")]
    [InlineData("generated-exact-practical-update", "--allowlist", "very-selective", "--top-k", "1")]
    [InlineData("generated-exact-practical-update", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-practical-update", "--unknown-ids", "-1")]
    [InlineData("generated-exact-practical-update", "--filter", "broad")]
    [InlineData("generated-exact-practical-update", "--preset", "smoke")]
    [InlineData("generated-exact-practical-update", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-practical-update", "--output-dir", "matrix")]
    [InlineData("generated-exact-practical-update", "--manifest", "manifest.json")]
    [InlineData("generated-exact-practical-update", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-exact-practical-update", "--m", "8")]
    [InlineData("generated-exact-practical-update", "--output", "")]
    [InlineData("generated-exact-practical-update", "--checkpoint-directory", "")]
    public void ParseGeneratedExactPracticalUpdate_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseGeneratedExactPracticalUpdate(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivatePracticalUpdateReportWithValidationRatiosCheckpointAndFalseEligibility()
    {
        string outputPath = NewArtifactPath("practical-update-report.json");
        string checkpointDirectory = Path.Combine(Path.GetDirectoryName(outputPath)!, "checkpoint");
        string[] arguments =
        [
            "generated-exact-practical-update",
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
            "--seed", "0x5EED079A",
            "--output", outputPath,
            "--checkpoint-directory", checkpointDirectory
        ];
        GeneratedExactPracticalUpdateOptions options =
            CommandLine.ParseGeneratedExactPracticalUpdate(arguments);

        GeneratedExactPracticalUpdateBenchmarkReport report =
            GeneratedExactPracticalUpdateScenario.Run(options, arguments);
        GeneratedExactPracticalUpdateScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExactPracticalUpdateModeReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-079", report.TaskId);
        Assert.Equal("generated-exact-practical-update", report.ScenarioName);
        Assert.Equal("generated-exact-practical-update", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-practical-update-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonArtifactEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        Assert.Equal(40, report.Counts.InitialBaseCount);
        Assert.Equal(47, report.Counts.PhysicalVectorCountAfterMutation);
        Assert.Equal(42, report.Counts.FinalLiveCountBeforeCheckpoint);
        Assert.Equal(35, report.Counts.LiveBaseCountBeforeCheckpoint);
        Assert.Equal(7, report.Counts.LiveDeltaCountBeforeCheckpoint);
        Assert.Equal(5, report.Counts.TombstoneCountBeforeCheckpoint);
        Assert.Equal(5.0 / 47.0, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCountAfterMutation", report.Counts.TombstoneRatioDenominator);
        Assert.Equal(7.0 / 40.0, report.Counts.DeltaInsertRatio, precision: 12);
        Assert.Equal("initialBaseCount", report.Counts.DeltaInsertRatioDenominator);
        Assert.Equal(42, report.Counts.PhysicalVectorCountAfterCheckpoint);
        Assert.Equal(42, report.Counts.FinalLiveCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.DeltaCountAfterCheckpoint);
        Assert.Equal(0, report.Counts.TombstoneCountAfterCheckpoint);

        Assert.Equal(10, report.Mutations.InsertAttemptCount);
        Assert.Equal(7, report.Mutations.InsertSuccessCount);
        Assert.Equal(11, report.Mutations.DeleteAttemptCount);
        Assert.Equal(5, report.Mutations.DeleteSuccessCount);
        Assert.Equal(3, report.Mutations.DuplicateInsertFailures);
        Assert.Equal(4, report.Mutations.UnknownDeleteFailures);
        Assert.Equal(2, report.Mutations.RepeatedDeleteFailures);
        Assert.Equal(12, report.Mutations.CommittedMutationCount);
        Assert.Equal(12, report.Mutations.StatusCounts.Committed);
        Assert.Equal(3, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(4, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);

        Assert.Equal(40, report.Generations.BeforeMutation);
        Assert.Equal(52, report.Generations.AfterMutation);
        Assert.Equal(52, report.Generations.BeforeCheckpoint);
        Assert.Equal(53, report.Generations.AfterCheckpoint);
        Assert.Equal(12, report.Generations.MutationDelta);
        Assert.Equal(1, report.Generations.CheckpointDelta);
        Assert.True(report.Generations.MutationDeltaMatchesCommittedMutations);
        Assert.True(report.Generations.CheckpointAdvancedExactlyOnce);

        Assert.Equal("broad", report.RawAllowlistInput.Kind);
        Assert.Equal(21, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Contains("post-mutation live view", report.RawAllowlistInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("selective", report.CandidateSetInput.Kind);
        Assert.Equal(5, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.True(report.CandidateSet.StaleCandidateSetConstructedBeforeMutation);
        Assert.True(report.CandidateSet.StaleCandidateSetRejectedAfterMutation);
        Assert.True(report.CandidateSet.FreshCandidateSetConstructedAfterMutation);
        Assert.Equal(5, report.CandidateSet.FreshConstructedSetCount);
        Assert.Equal(5, report.CandidateSet.FreshCountPerQuery);
        Assert.Equal(25, report.CandidateSet.FreshTotalCandidateCount);

        AssertMeasured(report.Operations.Mutations, "mutations", "TryAdd/TryDelete", 3);
        AssertMeasured(report.Operations.PostMutationExactSearch, "postMutationExactSearch", "Search(query, results)", 3);
        AssertMeasured(report.Operations.Checkpoint, "checkpoint", "Checkpoint(directoryPath)", 3);
        AssertMeasured(report.Operations.Open, "open", "OpenReadOnly(directoryPath)", 3);
        Assert.All(report.Operations.Mutations.Runs, run => Assert.Contains("bookkeeping", run.TimingScope, StringComparison.OrdinalIgnoreCase));
        Assert.All(report.Operations.PostMutationExactSearch.Runs, run =>
        {
            Assert.Contains("summed per-query Stopwatch samples", run.TimingScope, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("query lookup", run.TimingScope, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("result allocation/capture/copying", run.TimingScope, StringComparison.OrdinalIgnoreCase);
        });
        Assert.Equal("measured", report.Measurement.Mutations.Latency.Status);
        Assert.Equal("measured", report.Measurement.PostMutationExactSearch.Latency.Status);
        Assert.Equal("measured", report.Measurement.Checkpoint.Latency.Status);
        Assert.Equal("measured", report.Measurement.Open.Latency.Status);
        Assert.Contains("bookkeeping", report.Measurement.Mutations.Latency.TimedOperation, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("one elapsed Stopwatch sample per public ExactFlatIndex.Search", report.Measurement.PostMutationExactSearch.Latency.PercentileEstimator, StringComparison.Ordinal);
        Assert.Contains("query lookup", report.Measurement.PostMutationExactSearch.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("result allocation/capture/copying", report.Measurement.PostMutationExactSearch.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Measurement.MutationManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.SearchManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.OpenManagedAllocations.Status);
        Assert.Contains("truth construction", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("query lookup", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("result allocation/capture/copying", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("Published", report.Outputs.CheckpointStatus);
        Assert.True(Directory.Exists(report.Outputs.CheckpointDirectoryPath));
        Assert.Equal(3, report.Outputs.CheckpointFileCount);
        Assert.True(report.Outputs.CheckpointOutputBytes > 0);
        Assert.True(report.Outputs.CheckpointManifestBytes > 0);
        Assert.Equal(32 + (42 * 8), report.Outputs.CheckpointIdsBytes);
        Assert.Equal(48 + (42 * 11 * 4), report.Outputs.CheckpointVectorsBytes);
        Assert.Equal(42, report.Outputs.CheckpointOutputVectorCount);
        Assert.Contains("outside", report.Outputs.OutputByteScanTimingScope, StringComparison.OrdinalIgnoreCase);

        AssertPassed(report.Metrics.PostMutationExactSearch);
        AssertPassed(report.Metrics.RawAllowlistSearch);
        AssertPassed(report.Metrics.FreshCandidateSetSearch);
        AssertPassed(report.Metrics.ReopenedOutputSearch);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.MutationCountsMatched);
        Assert.True(report.Validation.GenerationBeforeAfterMutationReported);
        Assert.True(report.Validation.PostMutationExactSearchComparedToTruth);
        Assert.True(report.Validation.RawAllowlistVisibleAfterMutation);
        Assert.True(report.Validation.FreshCandidateSetVisibleAfterMutation);
        Assert.True(report.Validation.StaleCandidateSetRejectedAfterMutation);
        Assert.True(report.Validation.CheckpointPublished);
        Assert.True(report.Validation.ReopenedOutputParity);
        Assert.True(report.Validation.CheckpointOutputBytesScannedOutsideTiming);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);

        Assert.Equal("notMeasured", report.Resources.ActualResidentMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualProcessMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualGcMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualPrivateMemory.Status);
        Assert.Equal("notMeasured", report.Resources.ActualPeakMemory.Status);
        Assert.Equal("notMeasured", report.Resources.PeakTemporaryDisk.Status);
        Assert.Equal("measuredFinalOutputBytesOnly", report.Resources.FinalCheckpointOutputBytesStatus);
        Assert.Contains("Existing public ExactFlatIndex APIs only", report.Index.Configuration, StringComparison.OrdinalIgnoreCase);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactPracticalUpdateModeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-079", root.GetProperty("taskId").GetString());
        Assert.Equal("generated-exact-practical-update", root.GetProperty("scenarioName").GetString());
        Assert.Equal(42, root.GetProperty("counts").GetProperty("finalLiveCountBeforeCheckpoint").GetInt32());
        Assert.Equal(5, root.GetProperty("counts").GetProperty("tombstoneCountBeforeCheckpoint").GetInt32());
        Assert.Equal(7, root.GetProperty("counts").GetProperty("liveDeltaCountBeforeCheckpoint").GetInt32());
        Assert.Equal(40, root.GetProperty("generations").GetProperty("beforeMutation").GetInt64());
        Assert.Equal(52, root.GetProperty("generations").GetProperty("afterMutation").GetInt64());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("rawAllowlistVisibleAfterMutation").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("freshCandidateSetVisibleAfterMutation").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("staleCandidateSetRejectedAfterMutation").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("checkpointPublished").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reopenedOutputParity").GetBoolean());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("checkpoint").GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("open").GetProperty("latency").GetProperty("status").GetString());
        Assert.Contains(
            "one elapsed Stopwatch sample per public ExactFlatIndex.Search",
            root.GetProperty("measurement").GetProperty("postMutationExactSearch").GetProperty("latency").GetProperty("percentileEstimator").GetString(),
            StringComparison.Ordinal);
        Assert.True(root.GetProperty("outputs").GetProperty("checkpointOutputBytes").GetInt64() > 0);
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualResidentMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualProcessMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualGcMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualPrivateMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("actualPeakMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("resources").GetProperty("peakTemporaryDisk").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Run_ValidatesGeneratedWorkloadAcrossExactMetrics(VectorMetric metric)
    {
        string outputPath = NewArtifactPath("metric.json");
        GeneratedExactPracticalUpdateBenchmarkReport report = GeneratedExactPracticalUpdateScenario.Run(
            new GeneratedExactPracticalUpdateOptions(
                metric,
                Dimension: 7,
                BaseVectorCount: 24,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED_7920,
                InsertedDeltaCount: 5,
                DeletedBaseCount: 3,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "all",
                CandidateSetKind: "very-selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputPath: outputPath,
                CheckpointDirectory: Path.Combine(Path.GetDirectoryName(outputPath)!, "checkpoint"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-practical-update"]);

        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(29, report.Counts.PhysicalVectorCountAfterMutation);
        Assert.Equal(26, report.Counts.FinalLiveCountBeforeCheckpoint);
        Assert.Equal(5, report.Counts.LiveDeltaCountBeforeCheckpoint);
        Assert.Equal(3, report.Counts.TombstoneCountBeforeCheckpoint);
        Assert.True(report.Validation.PostMutationExactSearchComparedToTruth);
        Assert.True(report.Validation.RawAllowlistVisibleAfterMutation);
        Assert.True(report.Validation.FreshCandidateSetVisibleAfterMutation);
        Assert.True(report.Validation.StaleCandidateSetRejectedAfterMutation);
        Assert.True(report.Validation.CheckpointPublished);
        Assert.True(report.Validation.ReopenedOutputParity);
        AssertPassed(report.Metrics.PostMutationExactSearch);
        AssertPassed(report.Metrics.RawAllowlistSearch);
        AssertPassed(report.Metrics.FreshCandidateSetSearch);
        AssertPassed(report.Metrics.ReopenedOutputSearch);
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndPracticalUpdateModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--preset", "smoke"]);
        _ = CommandLine.ParseGeneratedExactCheckpointMatrix(["generated-exact-checkpoint-matrix", "--preset", "smoke"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--output-dir", "matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--baseline-report-id", "baseline"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCheckpointMatrix(["generated-exact-checkpoint-matrix", "--checkpoint-directory", "checkpoint"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--insertions", "2"]));
        Assert.Equal("generated-exact-practical-update", GeneratedExactPracticalUpdateOptions.ScenarioName);
    }

    private static void AssertMeasured(
        GeneratedExactPracticalUpdateTimedOperationInfo operation,
        string expectedName,
        string expectedTimedOperationPart,
        int expectedRunCount)
    {
        Assert.Equal(expectedName, operation.Name);
        Assert.Contains(expectedTimedOperationPart, operation.TimedOperation, StringComparison.Ordinal);
        Assert.Equal(expectedRunCount, operation.Runs.Length);
        Assert.Equal(expectedRunCount, operation.Aggregate.RunCount);
        Assert.True(operation.Aggregate.MeanElapsedMilliseconds >= 0);
        Assert.All(operation.Runs, run => Assert.True(run.ElapsedMilliseconds >= 0));
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
                $"vec079-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
