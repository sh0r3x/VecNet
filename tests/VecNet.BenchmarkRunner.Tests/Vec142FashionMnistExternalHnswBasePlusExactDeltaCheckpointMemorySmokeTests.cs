using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec142FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeTests
{
    [Fact]
    public void ParseExternalFashionMnistCheckpointMemorySmoke_UsesAcceptedPrivateDefaults()
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.EndsWith("fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-memory-smoke.json", options.OutputPath);
        Assert.EndsWith(Path.Combine("vec-142-memory-smoke", "checkpoint-output"), options.CheckpointDirectory);
        Assert.Equal(10, options.SampleIntervalMilliseconds);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(100, options.TopK);
        Assert.Equal(58_000, options.BaseVectorCount);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(59_000, options.PhysicalCandidateVectorCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(100, options.DeletedDeltaCount);
        Assert.Equal(57_900, options.LiveVectorCount);
        Assert.Equal(1_100, options.DeletedReservedIdCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(1, FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.CheckpointRunCount);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(0x5EED2141u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(128, options.EfConstruction);
        Assert.Equal(192, options.EfSearch);
        Assert.Equal(0x484E535700014100UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--query-count", "50")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--top-k", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--base-vectors", "58000")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--insertions", "1000")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--deletes", "1000")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--delta-deletes", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--duplicate-inserts", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--unknown-deletes", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--repeated-deletes", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--runs", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--warmup-queries", "3")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--metric", "InnerProduct")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--seed", "0x5EED2141")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--m", "16")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--ef-construction", "128")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--ef-search", "192")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--hnsw-seed", "0x484E535700014100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--baseline-report-id", "baseline")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--comparison", "true")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--sample-interval-ms", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--sample-interval-ms", "1001")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--output", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke", "--checkpoint-directory", "")]
    public void ParseExternalFashionMnistCheckpointMemorySmoke_RejectsOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistCheckpointMemorySmoke_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsSeparatedMemoryAndCheckpointValidationReport()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("report", baseCount: 48, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = NewArtifactPath("external-checkpoint-memory-smoke.json");
        string checkpointDirectory = CreateArtifactDirectory("checkpoint-output");
        string[] arguments =
        [
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName,
            "--cache-root", cacheRoot,
            "--output", outputPath,
            "--checkpoint-directory", checkpointDirectory,
            "--sample-interval-ms", "1"
        ];
        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions(
            cacheRoot,
            outputPath,
            checkpointDirectory,
            SampleIntervalMilliseconds: 1,
            QueryCount: 5,
            TopK: 6,
            BaseVectorCount: 36,
            InsertedDeltaCount: 8,
            DeletedBaseCount: 5,
            DeletedDeltaCount: 3,
            DuplicateInsertAttempts: 2,
            UnknownDeleteAttempts: 3,
            RepeatedDeleteAttempts: 2,
            WarmupQueries: 2,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1420,
            M: 4,
            EfConstruction: 16,
            EfSearch: 10,
            HnswSeed: 0x0000000000001420UL);

        ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Run(options, arguments);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-142", report.TaskId);
        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonArtifactEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Contains("readiness guard", report.ExistingTruthGuard.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.False(report.UpdatedTruth.Persisted);
        Assert.Equal(5, report.UpdatedTruth.QueryCount);
        Assert.Equal(6, report.UpdatedTruth.TruthDepth);
        Assert.Equal(36, report.UpdatedTruth.LiveVectorCount);
        Assert.Equal(1, report.Workload.CheckpointRunCount);
        Assert.Equal(1, report.Workload.SampleIntervalMilliseconds);
        Assert.Equal(36, report.Workload.ExpectedLiveVectorCount);
        Assert.Equal(8, report.Workload.ExpectedDeletedReservedIdCount);

        AssertPhase(report.MeasuredPhases.CacheTruthLoad, "cacheTruthLoad");
        AssertPhase(report.MeasuredPhases.ImmutableHnswBaseBuild, "immutableHnswBaseBuild");
        AssertPhase(report.MeasuredPhases.CompositeCreationAndExactDeltaTombstoneMutation, "compositeCreationAndExactDeltaTombstoneMutation");
        AssertPhase(report.MeasuredPhases.ExactUpdatedTruthGeneration, "exactUpdatedTruthGeneration");
        AssertPhase(report.MeasuredPhases.PreCheckpointSourceCompositeSearch, "preCheckpointSourceCompositeSearch");
        AssertPhase(report.MeasuredPhases.CheckpointPublication, "checkpointPublication");
        AssertPhase(report.MeasuredPhases.OpenedReadOnlyHnswOpen, "openedReadOnlyHnswOpen");
        AssertPhase(report.MeasuredPhases.PostCheckpointRebuiltCompositeSearch, "postCheckpointRebuiltCompositeSearch");
        AssertPhase(report.MeasuredPhases.OpenedReadOnlyHnswSearch, "openedReadOnlyHnswSearch");
        AssertPhase(report.MeasuredPhases.FinalValidation, "finalValidation");
        AssertMeasured(report.MeasuredPhases.CheckpointPhaseDiagnostics.LiveSnapshot);
        AssertMeasured(report.MeasuredPhases.CheckpointPhaseDiagnostics.RebuildBuild);
        AssertMeasured(report.MeasuredPhases.CheckpointPhaseDiagnostics.Save);
        AssertMeasured(report.MeasuredPhases.CheckpointPhaseDiagnostics.OpenValidation);
        AssertMeasured(report.MeasuredPhases.CheckpointPhaseDiagnostics.Publication);

        Assert.Equal(1, report.CheckpointRuns.RunCount);
        Assert.Equal(1, report.CheckpointRuns.DetailedValidationRunNumber);
        Assert.Single(report.CheckpointRuns.Runs);
        Assert.Equal("Published", report.Checkpoint.Status);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(36, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(36, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal("passed", report.NoChangesProbe.Status);
        Assert.Equal("recorded", report.CheckpointOutput.Status);
        Assert.True(report.CheckpointOutput.TotalBytes > 0);
        Assert.Equal("outsideCheckpointDuration", report.CheckpointOutput.ScanTimingScope);
        Assert.Equal("passed", report.OpenedValidation.Status);
        Assert.True(report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched);

        AssertSearchSection(report.Searches.PreCheckpointSourceComposite, options);
        AssertSearchSection(report.Searches.PostCheckpointRebuiltComposite, options);
        AssertSearchSection(report.Searches.OpenedReadOnlyHnsw, options);

        Assert.Equal("measured", report.ActualMemory.Status);
        AssertMemorySample(report.ActualMemory.BaselineProcess, "baselineProcess");
        AssertMemorySample(report.ActualMemory.PostCacheTruthLoad, "cacheTruthLoadEnd");
        AssertMemorySample(report.ActualMemory.PostCheckpointPublication, "checkpointPublicationEnd");
        Assert.True(report.ActualMemory.BaselineProcess.ProcessWorkingSetBytes.ContextOnly);
        Assert.True(report.ActualMemory.BaselineProcess.ProcessPeakWorkingSetBytes.ContextOnly);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateIdMapRetainedMemory.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateTombstoneSetMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.IndexOnlyPrivateBytes.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.TrueProcessPeakMemory.Status);

        Assert.Equal("sampled", report.PeakMemory.Status);
        AssertPeak(report.PeakMemory.CacheTruthLoad, "cacheTruthLoad");
        AssertPeak(report.PeakMemory.ImmutableHnswBaseBuild, "immutableHnswBaseBuild");
        AssertPeak(report.PeakMemory.CheckpointPublication, "checkpointPublication");
        AssertPeak(report.PeakMemory.OpenedReadOnlyHnswOpen, "openedReadOnlyHnswOpen");
        Assert.Equal("notMeasured", report.PeakMemory.PeakTemporaryDiskBytes.Status);
        Assert.Contains("miss", report.PeakMemory.CheckpointPublication.MissedShortPeakCaveat, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.Status);
        Assert.Contains("payload-only", report.LayoutLowerBounds.ClaimBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(36, report.LayoutLowerBounds.SourceBasePhysicalVectorCount);
        Assert.Equal(8, report.LayoutLowerBounds.SourceDeltaPhysicalVectorCount);
        Assert.Equal(36, report.LayoutLowerBounds.SourceLiveVectorCount);
        Assert.Equal(36, report.LayoutLowerBounds.RebuiltOpenedVectorCount);
        Assert.Equal(36L * 15L * sizeof(float), report.LayoutLowerBounds.SourceBaseVectorPayloadLowerBoundBytes);
        Assert.Equal(8L * 15L * sizeof(float), report.LayoutLowerBounds.SourceDeltaVectorPayloadLowerBoundBytes);
        Assert.Equal(5L * sizeof(ulong), report.LayoutLowerBounds.BaseTombstoneIdPayloadLowerBoundBytes);
        Assert.Equal(3L * sizeof(ulong), report.LayoutLowerBounds.DeltaTombstoneIdPayloadLowerBoundBytes);
        Assert.Equal(8L * sizeof(ulong), report.LayoutLowerBounds.DeletedReservedIdPayloadLowerBoundBytes);
        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.CompositeSearchWorkspacePayloadLowerBoundBytes.Status);
        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.OpenedSearchWorkspacePayloadLowerBoundBytes.Status);
        Assert.NotEmpty(report.LayoutLowerBounds.Formula);
        Assert.Contains("Excludes", report.LayoutLowerBounds.Exclusions, StringComparison.Ordinal);

        Assert.Equal("fileFacts", report.StorageOutput.Status);
        Assert.Equal(report.CheckpointOutput.TotalBytes, report.StorageOutput.TotalBytes);
        Assert.Equal(report.CheckpointOutput.ManifestBytes, report.StorageOutput.ManifestBytes);
        Assert.Contains("not memory", report.StorageOutput.MemoryBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.StorageOutput.PeakTemporaryDiskBytes.Status);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.CacheAndTruthReadinessPassed);
        Assert.True(report.Validation.UpdatedTruthGeneratedFromLiveView);
        Assert.True(report.Validation.PreCheckpointSourceCompositeComparedToTruth);
        Assert.True(report.Validation.CheckpointRunCountIsOne);
        Assert.True(report.Validation.PostCheckpointRebuiltCompositeComparedToTruth);
        Assert.True(report.Validation.OpenedReadOnlyHnswOpened);
        Assert.True(report.Validation.OpenedReadOnlyHnswComparedToTruth);
        Assert.True(report.Validation.RebuiltCompositeOpenedHnswSearchParityPassed);
        Assert.True(report.Validation.DeletedReservedIdsRejectedAfterCheckpoint);
        Assert.True(report.Validation.ActualPeakLowerBoundAndStorageSectionsSeparated);
        Assert.True(report.Validation.OutputBytesAreSeparateFileFacts);
        Assert.True(report.Validation.UnsupportedFieldsExplicitlyMarked);
        Assert.True(report.Validation.WorkingSetContextOnly);
        Assert.True(report.Validation.SampledPeakLabelsPresent);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.True(root.TryGetProperty("actualMemory", out JsonElement actualMemory));
        Assert.True(root.TryGetProperty("peakMemory", out JsonElement peakMemory));
        Assert.True(root.TryGetProperty("layoutLowerBounds", out JsonElement layoutLowerBounds));
        Assert.True(root.TryGetProperty("storageOutput", out JsonElement storageOutput));
        Assert.Equal("measured", actualMemory.GetProperty("status").GetString());
        Assert.Equal("sampled", peakMemory.GetProperty("checkpointPublication").GetProperty("status").GetString());
        Assert.Contains("miss", peakMemory.GetProperty("checkpointPublication").GetProperty("missedShortPeakCaveat").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("estimatedLowerBound", layoutLowerBounds.GetProperty("status").GetString());
        Assert.Equal("fileFacts", storageOutput.GetProperty("status").GetString());
        Assert.Contains("not memory", storageOutput.GetProperty("memoryBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.True(root.GetProperty("validation").GetProperty("outputBytesAreSeparateFileFacts").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "previewReadinessEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "candidateEligibility", "regressionDecision", "publicClaimStatus", "comparisonPublicationEligible");
    }

    [Fact]
    public void Run_MissingCacheFailsClosedWithoutWritingReportOrCheckpointDirectory()
    {
        string cacheRoot = CreateArtifactDirectory("missing-cache");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string checkpointDirectory = Path.Combine(cacheRoot, "checkpoint-output");
        var options = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.Default with
        {
            CacheRoot = cacheRoot,
            OutputPath = outputPath,
            CheckpointDirectory = checkpointDirectory,
            QueryCount = 1,
            TopK = 1,
            BaseVectorCount = 4,
            InsertedDeltaCount = 1,
            DeletedBaseCount = 1,
            DeletedDeltaCount = 0,
            RepeatedDeleteAttempts = 0,
            SampleIntervalMilliseconds = 1,
            M = 2,
            EfConstruction = 8,
            EfSearch = 2
        };

        Assert.Throws<FileNotFoundException>(() =>
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName]));
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    private static void AssertPhase(ExternalHnswCheckpointMemorySmokePhaseInfo phase, string name)
    {
        Assert.Equal(name, phase.Name);
        Assert.Equal("measured", phase.Status);
        Assert.True(phase.ElapsedMilliseconds >= 0);
        Assert.True(phase.ManagedAllocatedBytes >= 0);
        Assert.Contains("sampled", phase.MemorySampling, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(phase.Boundary);
        Assert.NotEmpty(phase.ExcludedOperations);
    }

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointPhaseInfo phase)
    {
        Assert.Equal("Measured", phase.Status);
        Assert.True(phase.ElapsedTicks >= 0);
        Assert.True(phase.ElapsedMilliseconds >= 0);
        Assert.True(phase.ManagedAllocatedBytes >= 0);
        Assert.Contains("VEC-133", phase.Source, StringComparison.Ordinal);
    }

    private static void AssertSearchSection(
        HnswBasePlusExactDeltaCheckpointSearchSectionInfo section,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options)
    {
        Assert.Equal(options.QueryCount, section.Search.MeasuredQueryCount);
        Assert.Single(section.Search.Runs);
        Assert.Equal("measured", section.Measurement.Latency.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.InRange(section.Metrics.RecallAtK, 0, 1);
        Assert.InRange(section.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", section.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", section.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal(options.QueryCount, section.Underfill.QueryCount);
        Assert.Equal(options.TopK, section.Underfill.RequestedResultCountPerQuery);
    }

    private static void AssertMemorySample(HnswMemorySampleInfo sample, string name)
    {
        Assert.Equal(name, sample.Name);
        Assert.Equal("measured", sample.ManagedHeapSizeBytes.Status);
        Assert.True(sample.ManagedHeapSizeBytes.ValueBytes >= 0);
        Assert.Equal("measured", sample.GcCommittedBytes.Status);
        Assert.True(sample.GcCommittedBytes.ValueBytes >= 0);
        Assert.Equal("measured", sample.GcFragmentedBytes.Status);
        Assert.True(sample.GcFragmentedBytes.ValueBytes >= 0);
        Assert.Equal("measured", sample.ProcessPrivateBytes.Status);
        Assert.True(sample.ProcessPrivateBytes.ValueBytes > 0);
        Assert.Equal("measured", sample.ProcessWorkingSetBytes.Status);
        Assert.True(sample.ProcessWorkingSetBytes.ValueBytes > 0);
        Assert.Equal("measured", sample.ProcessPeakWorkingSetBytes.Status);
        Assert.True(sample.ProcessPeakWorkingSetBytes.ValueBytes > 0);
    }

    private static void AssertPeak(HnswMemoryPeakOperationInfo operation, string name)
    {
        Assert.Equal(name, operation.Name);
        Assert.Equal("sampled", operation.Status);
        Assert.Equal(1, operation.SampleIntervalMilliseconds);
        Assert.True(operation.SampleCount >= 2);
        AssertMemorySample(operation.StartSample, name + "Start");
        AssertMemorySample(operation.EndSample, name + "End");
        Assert.Equal("sampled", operation.PeakObservedManagedHeapSizeBytes.Status);
        Assert.Equal("sampled", operation.PeakObservedGcCommittedBytes.Status);
        Assert.Equal("sampled", operation.PeakObservedPrivateBytes.Status);
        Assert.Equal("sampled", operation.PeakObservedWorkingSetBytes.Status);
        Assert.False(operation.PeakObservedPrivateBytes.ContextOnly);
        Assert.True(operation.PeakObservedWorkingSetBytes.ContextOnly);
        Assert.Contains("observed sampled peak", operation.PeakObservedPrivateBytes.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("miss", operation.MissedShortPeakCaveat, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("whole-process", operation.WholeProcessCaveat, StringComparison.OrdinalIgnoreCase);
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
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
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 7)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 23 + column * 19 + offset + (row % 7) * 5) % 251);
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

    private static string NewArtifactPath(string fileName)
    {
        string directory = CreateArtifactDirectory(Path.GetFileNameWithoutExtension(fileName));
        return Path.Combine(directory, fileName);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec142-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

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
