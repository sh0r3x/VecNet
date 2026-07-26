using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec134HnswBasePlusExactDeltaCheckpointTests
{
    [Fact]
    public void ParseHnswBasePlusExactDeltaCheckpoint_UsesPrivateSmokeDefaults()
    {
        HnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(["generated-hnsw-base-plus-exact-delta-checkpoint"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(1_024, options.BaseVectorCount);
        Assert.Equal(1_152, options.PhysicalVectorCount);
        Assert.Equal(1_008, options.LiveVectorCount);
        Assert.Equal(16, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(128, options.InsertedDeltaCount);
        Assert.Equal(128, options.DeletedBaseCount);
        Assert.Equal(16, options.DeletedDeltaCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(128, options.EfSearch);
        Assert.Equal(0x484E535700013200UL, options.HnswSeed);
        Assert.Equal(0x5EED2132u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.CheckpointDirectory);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.False(Path.IsPathRooted(options.CheckpointDirectory));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--metric", "InnerProduct")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--dimension", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--vectors", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--queries", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--top-k", "10", "--vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--runs", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--runs", "6")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--insertions", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "6", "--vectors", "5")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--delta-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--delta-deletes", "2", "--insertions", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--duplicate-inserts", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--unknown-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--repeated-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--top-k", "10", "--ef-search", "9")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--m", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--m", "65")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--m", "8", "--ef-construction", "7")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--ef-construction", "4097")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--ef-search", "4097")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--output", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--checkpoint-directory", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--preset", "smoke")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--output-dir", "matrix")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--manifest", "manifest.json")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--snapshot-directory", "snapshot")]
    [InlineData("generated-hnsw-base-plus-exact-delta-checkpoint", "--sample-interval-ms", "10")]
    public void ParseHnswBasePlusExactDeltaCheckpoint_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseHnswBasePlusExactDeltaCheckpoint_AcceptsCosine(string metric)
    {
        HnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(
                [HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedCheckpointReportWithDiagnosticsAndSeparatedSearches()
    {
        string outputPath = NewArtifactPath("checkpoint.json");
        string checkpointDirectory = NewArtifactDirectory("checkpoint-output");
        string[] arguments =
        [
            "generated-hnsw-base-plus-exact-delta-checkpoint",
            "--metric", "SquaredEuclidean",
            "--dimension", "11",
            "--vectors", "48",
            "--queries", "4",
            "--top-k", "5",
            "--insertions", "8",
            "--deletes", "8",
            "--delta-deletes", "2",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "4",
            "--runs", "2",
            "--warmup-queries", "1",
            "--seed", "0x5EED134A",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "16",
            "--hnsw-seed", "0x000000000000134A",
            "--output", outputPath,
            "--checkpoint-directory", checkpointDirectory
        ];
        HnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(arguments);

        HnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            HnswBasePlusExactDeltaCheckpointScenario.Run(options, arguments);
        HnswBasePlusExactDeltaCheckpointScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-134", report.TaskId);
        Assert.Equal("generated-hnsw-base-plus-exact-delta-checkpoint", report.ScenarioName);
        Assert.Equal("generated-hnsw-base-plus-exact-delta-checkpoint", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Dataset.Metric);
        Assert.Equal(56, report.Dataset.VectorCount);
        Assert.Equal(11, report.Dataset.Dimension);
        Assert.Equal("scalar-reference-generated-live-hnsw-base-plus-exact-delta-checkpoint", report.Truth.Kind);
        Assert.Equal(5, report.Truth.Depth);
        Assert.Equal("HnswBasePlusExactDeltaIndex", report.Index.Type);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(16, report.Hnsw.EfSearch);
        Assert.Equal("0x000000000000134A", report.Hnsw.RandomSeed);

        Assert.Equal(48, report.Workload.BaseVectorCount);
        Assert.Equal(8, report.Workload.InsertedDeltaVectorCount);
        Assert.Equal(8, report.Workload.DeletedBaseVectorCount);
        Assert.Equal(2, report.Workload.DeletedDeltaVectorCount);
        Assert.Equal(2, report.Workload.RunCount);
        Assert.Equal(1, report.Workload.WarmupQueryCount);
        Assert.Contains("deleted IDs remain reserved", report.Workload.IdPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(48, report.PreCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(40, report.PreCheckpointCounts.BaseLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(6, report.PreCheckpointCounts.DeltaLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.BaseTombstoneCount);
        Assert.Equal(2, report.PreCheckpointCounts.DeltaTombstoneCount);
        Assert.Equal(10, report.PreCheckpointCounts.TombstoneCount);
        Assert.Equal(46, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(10, report.PreCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(18, report.PreCheckpointCounts.Generation);

        Assert.Equal(8, report.Mutations.InsertedCount);
        Assert.Equal(8, report.Mutations.DeletedBaseCount);
        Assert.Equal(2, report.Mutations.DeletedDeltaCount);
        Assert.Equal(18, report.Mutations.CommittedMutationCount);
        Assert.Equal(0, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(18, report.Mutations.GenerationAfterMutations);
        Assert.Equal(18, report.Mutations.GenerationDelta);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(18, report.Mutations.StatusCounts.Committed);
        Assert.Equal(2, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(3, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(4, report.Mutations.StatusCounts.AlreadyDeleted);

        Assert.Equal(2, report.CheckpointRuns.RunCount);
        Assert.Equal(2, report.CheckpointRuns.DetailedValidationRunNumber);
        Assert.Contains("final checkpoint run", report.CheckpointRuns.DetailedValidationPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(2, report.CheckpointRuns.Runs.Length);
        Assert.Equal(2, report.CheckpointRuns.Aggregate.RunCount);
        Assert.True(report.CheckpointRuns.Aggregate.MeanElapsedMilliseconds >= 0);
        Assert.True(report.CheckpointRuns.Aggregate.MinElapsedMilliseconds >= 0);
        Assert.True(report.CheckpointRuns.Aggregate.MaxElapsedMilliseconds >= report.CheckpointRuns.Aggregate.MinElapsedMilliseconds);
        Assert.True(report.CheckpointRuns.Aggregate.MeanManagedAllocatedBytes >= 0);
        Assert.True(report.CheckpointRuns.Aggregate.MinManagedAllocatedBytes >= 0);
        Assert.True(report.CheckpointRuns.Aggregate.MaxManagedAllocatedBytes >= report.CheckpointRuns.Aggregate.MinManagedAllocatedBytes);
        Assert.Contains("independently rebuilt equivalent checkpoint attempts", report.CheckpointRuns.Aggregate.AggregateSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(2, report.CheckpointRuns.Runs.Select(run => run.CheckpointDirectory).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.All(report.CheckpointRuns.Runs, run =>
        {
            Assert.Equal("Published", run.Status);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.ManagedAllocatedBytes >= 0);
            Assert.Equal(18, run.GenerationBeforeCheckpoint);
            Assert.Equal(19, run.GenerationAfterCheckpoint);
            Assert.True(run.GenerationAdvancedExactlyOnce);
            Assert.StartsWith(Path.GetFullPath(checkpointDirectory), run.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);
            AssertMeasured(run.Phases.LiveSnapshot);
            AssertMeasured(run.Phases.RebuildBuild);
            AssertMeasured(run.Phases.Save);
            AssertMeasured(run.Phases.OpenValidation);
            AssertMeasured(run.Phases.Publication);
        });

        Assert.Equal("Published", report.Checkpoint.Status);
        Assert.Equal("internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)", report.Checkpoint.TimedOperation);
        Assert.True(report.Checkpoint.ElapsedMilliseconds >= 0);
        Assert.True(report.Checkpoint.ManagedAllocatedBytes >= 0);
        Assert.Equal(18, report.Checkpoint.GenerationBeforeCheckpoint);
        Assert.Equal(19, report.Checkpoint.GenerationAfterCheckpoint);
        Assert.True(report.Checkpoint.GenerationAdvancedExactlyOnce);
        AssertMeasured(report.Checkpoint.Phases.LiveSnapshot);
        AssertMeasured(report.Checkpoint.Phases.RebuildBuild);
        AssertMeasured(report.Checkpoint.Phases.Save);
        AssertMeasured(report.Checkpoint.Phases.OpenValidation);
        AssertMeasured(report.Checkpoint.Phases.Publication);

        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(19, report.CheckpointResult.Generation);
        Assert.Equal(46, report.CheckpointResult.RebuiltBaseVectorCount);
        Assert.Equal(46, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(46, report.CheckpointResult.BasePhysicalVectorCount);
        Assert.Equal(46, report.CheckpointResult.BaseLiveVectorCount);
        Assert.Equal(0, report.CheckpointResult.DeltaPhysicalVectorCount);
        Assert.Equal(0, report.CheckpointResult.TombstoneCount);
        Assert.Equal(10, report.CheckpointResult.DeletedReservedIdCount);
        Assert.Equal(6, report.CheckpointResult.FoldedDeltaVectorCount);
        Assert.Equal(8, report.CheckpointResult.FoldedBaseTombstoneCount);
        Assert.Equal(2, report.CheckpointResult.FoldedDeltaTombstoneCount);

        Assert.Equal(46, report.PostCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(46, report.PostCheckpointCounts.BaseLiveVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.TombstoneCount);
        Assert.Equal(46, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal(10, report.PostCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(19, report.PostCheckpointCounts.Generation);

        Assert.Equal("passed", report.NoChangesProbe.Status);
        Assert.Equal(19, report.NoChangesProbe.GenerationBeforeProbe);
        Assert.Equal(19, report.NoChangesProbe.GenerationAfterProbe);
        Assert.True(report.NoChangesProbe.GenerationUnchanged);
        Assert.True(report.NoChangesProbe.OutputDirectoryRemainedEmpty);
        AssertNotExecuted(report.NoChangesProbe.Phases.LiveSnapshot);
        AssertNotExecuted(report.NoChangesProbe.Phases.RebuildBuild);
        AssertNotExecuted(report.NoChangesProbe.Phases.Save);
        AssertNotExecuted(report.NoChangesProbe.Phases.OpenValidation);
        AssertNotExecuted(report.NoChangesProbe.Phases.Publication);

        Assert.Equal("recorded", report.Output.Status);
        Assert.Equal(Path.Combine(Path.GetFullPath(checkpointDirectory), "checkpoint-run-002"), report.Output.DirectoryPath, ignoreCase: true);
        Assert.True(report.Output.FileCount >= 5);
        Assert.True(report.Output.TotalBytes > 0);
        Assert.True(report.Output.ManifestBytes > 0);
        Assert.True(report.Output.IdsBytes > 0);
        Assert.True(report.Output.VectorsBytes > 0);
        Assert.True(report.Output.LevelsBytes > 0);
        Assert.True(report.Output.GraphBytes > 0);
        Assert.Equal(46, report.Output.OutputVectorCount);
        Assert.Equal("passed", report.Output.ValidationOpenStatus);
        Assert.Equal("outsideCheckpointDuration", report.Output.ScanTimingScope);

        Assert.Equal("passed", report.OpenedValidation.Status);
        Assert.Equal(46, report.OpenedValidation.ExpectedVectorCount);
        Assert.Equal(46, report.OpenedValidation.OpenedVectorCount);
        Assert.Equal(0, report.OpenedValidation.IdMismatchCount);
        Assert.Equal(0, report.OpenedValidation.VectorMismatchCount);
        Assert.True(report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched);
        Assert.Equal(0, report.OpenedValidation.RebuiltCompositeOpenedSearchParity.IdMismatchCount);
        Assert.Equal(0, report.OpenedValidation.RebuiltCompositeOpenedSearchParity.DistanceMismatchCount);

        AssertSearchSection(report.Searches.PreCheckpointComposite, "preCheckpointComposite", options);
        AssertSearchSection(report.Searches.PostCheckpointRebuiltComposite, "postCheckpointRebuiltComposite", options);
        AssertSearchSection(report.Searches.OpenedReadOnlyHnsw, "openedReadOnlyHnsw", options);

        Assert.Equal("measured", report.Measurement.CheckpointLatency.Status);
        Assert.Contains("VEC-133", report.Measurement.CheckpointLatency.PercentileEstimator, StringComparison.Ordinal);
        Assert.Equal("measured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("bytesPerCheckpointCall", report.Measurement.CheckpointManagedAllocations.Unit);
        Assert.Equal(report.CheckpointRuns.Aggregate.MeanManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture), report.Measurement.CheckpointManagedAllocations.Value);
        Assert.Equal("measured", report.Measurement.OutputBytes.Status);
        Assert.Equal(report.Output.TotalBytes.ToString(CultureInfo.InvariantCulture), report.Measurement.OutputBytes.Value);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        AssertMeasured(report.Measurement.PhaseDiagnostics.LiveSnapshot);
        AssertMeasured(report.Measurement.PhaseDiagnostics.RebuildBuild);
        AssertMeasured(report.Measurement.PhaseDiagnostics.Save);
        AssertMeasured(report.Measurement.PhaseDiagnostics.OpenValidation);
        AssertMeasured(report.Measurement.PhaseDiagnostics.Publication);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.LiveTruthGenerated);
        Assert.True(report.Validation.PreCheckpointCompositeComparedToTruth);
        Assert.True(report.Validation.CheckpointResultStatusPublished);
        Assert.True(report.Validation.CheckpointResultCountsMatched);
        Assert.True(report.Validation.CheckpointGenerationAdvancedExactlyOnce);
        Assert.True(report.Validation.PhaseDiagnosticsMeasuredForPublishedCheckpoint);
        Assert.True(report.Validation.CheckpointRepeatedRunEvidencePresent);
        Assert.Equal(2, report.Validation.DetailedValidationRunNumber);
        Assert.True(report.Validation.DetailedValidationUsesFinalRun);
        Assert.True(report.Validation.PostCheckpointCountsMatched);
        Assert.True(report.Validation.OpenedReadOnlyHnswIdVectorValidationPassed);
        Assert.True(report.Validation.RebuiltCompositeOpenedHnswSearchParityPassed);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);
        Assert.True(report.Validation.NoChangesCheckpointProbePassed);
        Assert.True(report.Validation.DeletedReservedIdsRejectedAfterCheckpoint);
        Assert.True(report.Validation.OutputBytesScannedOutsideCheckpointDuration);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-134", root.GetProperty("taskId").GetString());
        Assert.Equal("generated-hnsw-base-plus-exact-delta-checkpoint", root.GetProperty("scenarioName").GetString());
        Assert.Equal(48, root.GetProperty("workload").GetProperty("baseVectorCount").GetInt32());
        Assert.Equal(8, root.GetProperty("workload").GetProperty("insertedDeltaVectorCount").GetInt32());
        Assert.Equal(2, root.GetProperty("checkpointRuns").GetProperty("runCount").GetInt32());
        Assert.Equal(2, root.GetProperty("checkpointRuns").GetProperty("detailedValidationRunNumber").GetInt32());
        Assert.Equal(2, root.GetProperty("checkpointRuns").GetProperty("runs").GetArrayLength());
        Assert.Equal("Published", root.GetProperty("checkpointRuns").GetProperty("runs")[0].GetProperty("status").GetString());
        Assert.Equal("Published", root.GetProperty("checkpointRuns").GetProperty("runs")[1].GetProperty("status").GetString());
        Assert.NotEqual(
            root.GetProperty("checkpointRuns").GetProperty("runs")[0].GetProperty("checkpointDirectory").GetString(),
            root.GetProperty("checkpointRuns").GetProperty("runs")[1].GetProperty("checkpointDirectory").GetString());
        Assert.Equal(46, root.GetProperty("checkpointResult").GetProperty("liveVectorCount").GetInt32());
        Assert.Equal("Measured", root.GetProperty("checkpoint").GetProperty("phases").GetProperty("liveSnapshot").GetProperty("status").GetString());
        Assert.Equal("Measured", root.GetProperty("measurement").GetProperty("phaseDiagnostics").GetProperty("rebuildBuild").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("searches").GetProperty("postCheckpointRebuiltComposite").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("openedValidation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("openedValidation").GetProperty("rebuiltCompositeOpenedSearchParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("checkpointRepeatedRunEvidencePresent").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("detailedValidationUsesFinalRun").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "cacheRoot", "preset", "manifest", "outputDir", "memoryEstimates", "candidateEligibility", "regressionDecision", "publicClaimStatus");
    }

    private static void AssertSearchSection(
        HnswBasePlusExactDeltaCheckpointSearchSectionInfo section,
        string name,
        HnswBasePlusExactDeltaCheckpointOptions options)
    {
        Assert.Equal(name, section.Name);
        Assert.Equal(options.QueryCount, section.Search.MeasuredQueryCount);
        Assert.Equal(options.Runs, section.Search.Runs.Length);
        Assert.Equal(options.Runs, section.Search.Aggregate.RunCount);
        Assert.Equal("measured", section.Measurement.Latency.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerSearchCall", section.Measurement.ManagedAllocations.Unit);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.Equal("measured", section.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", section.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", section.Measurement.Warmup.Status);
        Assert.InRange(section.Metrics.RecallAtK, 0, 1);
        Assert.InRange(section.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", section.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, section.Metrics.DistanceMismatchCount);
        Assert.Equal("passed", section.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal(options.QueryCount, section.Underfill.QueryCount);
        Assert.Equal(options.TopK, section.Underfill.RequestedResultCountPerQuery);
        Assert.Equal(options.QueryCount * options.TopK, section.Underfill.TotalRequestedResultSlots);
        Assert.Equal(section.Underfill.TotalRequestedResultSlots - section.Underfill.TotalReturnedResults, section.Underfill.UnderfilledSlotCount);
    }

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointPhaseInfo phase)
    {
        Assert.Equal("Measured", phase.Status);
        Assert.True(phase.ElapsedTicks >= 0);
        Assert.True(phase.ElapsedMilliseconds >= 0);
        Assert.True(phase.ManagedAllocatedBytes >= 0);
        Assert.Contains("VEC-133", phase.Source, StringComparison.Ordinal);
    }

    private static void AssertNotExecuted(HnswBasePlusExactDeltaCheckpointPhaseInfo phase)
    {
        Assert.Equal("NotExecuted", phase.Status);
        Assert.Equal(0, phase.ElapsedTicks);
        Assert.Equal(0, phase.ElapsedMilliseconds);
        Assert.Equal(0, phase.ManagedAllocatedBytes);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName));
        return Path.Combine(directory, fileName);
    }

    private static string NewArtifactDirectory(string name)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec134-{name}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertNoBooleanPropertyTrueForNames(JsonElement element, params string[] names)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    names.Any(name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    throw new Xunit.Sdk.XunitException($"Property {property.Name} was unexpectedly true.");
                }

                AssertNoBooleanPropertyTrueForNames(property.Value, names);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoBooleanPropertyTrueForNames(item, names);
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
