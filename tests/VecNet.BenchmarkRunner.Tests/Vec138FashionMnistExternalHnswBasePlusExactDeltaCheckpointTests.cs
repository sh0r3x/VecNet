using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec138FashionMnistExternalHnswBasePlusExactDeltaCheckpointTests
{
    [Fact]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint_UsesAcceptedPrivateDefaults()
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.EndsWith("fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint.json", options.OutputPath);
        Assert.EndsWith(Path.Combine("vec-138-smoke", "checkpoint-output"), options.CheckpointDirectory);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(100, options.TopK);
        Assert.Equal(58_000, options.BaseVectorCount);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(59_000, options.PhysicalCandidateVectorCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(100, options.DeletedDeltaCount);
        Assert.Equal(57_900, options.LiveVectorCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(2, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(0x5EED2137u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(128, options.EfConstruction);
        Assert.Equal(192, options.EfSearch);
        Assert.Equal(0x484E535700013700UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--snapshot-directory", "snapshot")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--hnswlib-python", "python")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--metric", "InnerProduct")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--base-vectors", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--insertions", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "11", "--base-vectors", "10")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--delta-deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--delta-deletes", "2", "--insertions", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--top-k", "10", "--base-vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--m", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--m", "65")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--m", "8", "--ef-construction", "7")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--output", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--checkpoint-directory", "")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsPrivateExternalCheckpointReport()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("report", baseCount: 48, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = NewArtifactPath("external-checkpoint-report.json");
        string checkpointDirectory = CreateArtifactDirectory("checkpoint-output");
        string[] arguments =
        [
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "--cache-root", cacheRoot,
            "--output", outputPath,
            "--checkpoint-directory", checkpointDirectory,
            "--query-count", "5",
            "--top-k", "6",
            "--base-vectors", "36",
            "--insertions", "8",
            "--deletes", "5",
            "--delta-deletes", "3",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "2",
            "--runs", "2",
            "--warmup-queries", "2",
            "--metric", "squared-euclidean",
            "--seed", "0x5EED1380",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "10",
            "--hnsw-seed", "0x0000000000001380"
        ];
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(arguments);

        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(options, arguments);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-138", report.TaskId);
        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Truth.Sha256, report.ExistingTruthGuard.Sha256);
        Assert.Contains("readiness guard", report.ExistingTruthGuard.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("scalar-reference-external-live-hnsw-base-plus-exact-delta-checkpoint", report.UpdatedTruth.Kind);
        Assert.False(report.UpdatedTruth.Persisted);
        Assert.Equal(5, report.UpdatedTruth.QueryCount);
        Assert.Equal(6, report.UpdatedTruth.TruthDepth);
        Assert.Equal(36, report.UpdatedTruth.LiveVectorCount);

        Assert.Equal(48, report.Workload.AdmittedBaseMatrixRowCount);
        Assert.Equal(6, report.Workload.QueryMatrixCount);
        Assert.Equal(36, report.Workload.ImmutableBaseRowCount);
        Assert.Equal(8, report.Workload.DeltaRowCount);
        Assert.Equal(4, report.Workload.UnusedCandidateRowCount);
        Assert.Equal(2, report.Workload.CheckpointRunCount);
        Assert.Contains("deleted IDs remain reserved", report.Workload.ExternalIdPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(36, report.PreCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(31, report.PreCheckpointCounts.BaseLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(5, report.PreCheckpointCounts.DeltaLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.TombstoneCount);
        Assert.Equal(36, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.DeletedReservedIdCount);
        Assert.Equal(16, report.PreCheckpointCounts.Generation);

        Assert.Equal(8, report.Mutations.InsertedCount);
        Assert.Equal(5, report.Mutations.DeletedBaseCount);
        Assert.Equal(3, report.Mutations.DeletedDeltaCount);
        Assert.Equal(16, report.Mutations.CommittedMutationCount);
        Assert.Equal(16, report.Mutations.StatusCounts.Committed);
        Assert.Equal(2, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(3, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);

        Assert.Equal(2, report.CheckpointRuns.RunCount);
        Assert.Equal(2, report.CheckpointRuns.DetailedValidationRunNumber);
        Assert.Equal(2, report.CheckpointRuns.Runs.Length);
        Assert.Equal(2, report.CheckpointRuns.Runs.Select(run => run.CheckpointDirectory).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.All(report.CheckpointRuns.Runs, run =>
        {
            Assert.Equal("Published", run.Status);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.ManagedAllocatedBytes >= 0);
            Assert.Equal(16, run.GenerationBeforeCheckpoint);
            Assert.Equal(17, run.GenerationAfterCheckpoint);
            Assert.True(run.GenerationAdvancedExactlyOnce);
            AssertMeasured(run.Phases.LiveSnapshot);
            AssertMeasured(run.Phases.RebuildBuild);
            AssertMeasured(run.Phases.Save);
            AssertMeasured(run.Phases.OpenValidation);
            AssertMeasured(run.Phases.Publication);
        });

        Assert.Equal("Published", report.Checkpoint.Status);
        Assert.Equal(16, report.Checkpoint.GenerationBeforeCheckpoint);
        Assert.Equal(17, report.Checkpoint.GenerationAfterCheckpoint);
        Assert.True(report.Checkpoint.GenerationAdvancedExactlyOnce);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(36, report.CheckpointResult.RebuiltBaseVectorCount);
        Assert.Equal(36, report.CheckpointResult.LiveVectorCount);
        Assert.Equal(36, report.PostCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.TombstoneCount);
        Assert.Equal(8, report.PostCheckpointCounts.DeletedReservedIdCount);

        Assert.Equal("passed", report.NoChangesProbe.Status);
        AssertNotExecuted(report.NoChangesProbe.Phases.LiveSnapshot);
        Assert.Equal("recorded", report.Output.Status);
        Assert.Equal("passed", report.Output.ValidationOpenStatus);
        Assert.True(report.Output.TotalBytes > 0);
        Assert.Equal(36, report.Output.OutputVectorCount);
        Assert.Equal("outsideCheckpointDuration", report.Output.ScanTimingScope);

        Assert.Equal("passed", report.OpenedValidation.Status);
        Assert.Equal(36, report.OpenedValidation.ExpectedVectorCount);
        Assert.Equal(36, report.OpenedValidation.OpenedVectorCount);
        Assert.Equal(0, report.OpenedValidation.IdMismatchCount);
        Assert.Equal(0, report.OpenedValidation.VectorMismatchCount);
        Assert.True(report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched);

        AssertSearchSection(report.Searches.PreCheckpointSourceComposite, "preCheckpointSourceComposite", options);
        AssertSearchSection(report.Searches.PostCheckpointRebuiltComposite, "postCheckpointRebuiltComposite", options);
        AssertSearchSection(report.Searches.OpenedReadOnlyHnsw, "openedReadOnlyHnsw", options);

        Assert.Equal("measured", report.Measurement.CheckpointLatency.Status);
        Assert.Contains("VEC-133", report.Measurement.CheckpointLatency.PercentileEstimator, StringComparison.Ordinal);
        Assert.Equal("measured", report.Measurement.CheckpointManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.CacheAndTruthReadinessPassed);
        Assert.True(report.Validation.ExistingTruthGuardLoaded);
        Assert.True(report.Validation.UpdatedTruthGeneratedFromLiveView);
        Assert.True(report.Validation.PreCheckpointSourceCompositeComparedToTruth);
        Assert.True(report.Validation.CheckpointRepeatedRunEvidencePresent);
        Assert.True(report.Validation.DetailedValidationUsesFinalRun);
        Assert.True(report.Validation.PostCheckpointRebuiltCompositeComparedToTruth);
        Assert.True(report.Validation.OpenedReadOnlyHnswIdVectorValidationPassed);
        Assert.True(report.Validation.RebuiltCompositeOpenedHnswSearchParityPassed);
        Assert.True(report.Validation.DeletedReservedIdsRejectedAfterCheckpoint);
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
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-138", root.GetProperty("taskId").GetString());
        Assert.Equal(2, root.GetProperty("checkpointRuns").GetProperty("runCount").GetInt32());
        Assert.Equal("Measured", root.GetProperty("checkpoint").GetProperty("phases").GetProperty("liveSnapshot").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("searches").GetProperty("preCheckpointSourceComposite").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("searches").GetProperty("postCheckpointRebuiltComposite").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("searches").GetProperty("openedReadOnlyHnsw").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.True(root.GetProperty("openedValidation").GetProperty("rebuiltCompositeOpenedSearchParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("deletedReservedIdsRejectedAfterCheckpoint").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "preset", "manifest", "outputDir", "snapshotDirectory", "hnswlibPython", "candidateEligibility", "regressionDecision", "publicClaimStatus");
    }

    [Fact]
    public void Run_MissingCacheFailsClosedWithoutWritingReport()
    {
        string cacheRoot = CreateArtifactDirectory("missing-cache");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string checkpointDirectory = Path.Combine(cacheRoot, "checkpoint-output");

        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            outputPath,
            checkpointDirectory,
            QueryCount: 1,
            TopK: 1,
            BaseVectorCount: 4,
            InsertedDeltaCount: 1,
            DeletedBaseCount: 1,
            DeletedDeltaCount: 0,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1381,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x1381);

        Assert.Throws<FileNotFoundException>(() =>
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName]));
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    [Fact]
    public void UpdatedTruthLiveIds_ArePostUpdateBasePlusDeltaMinusTombstones()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("live-truth", baseCount: 24, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            NewArtifactPath("live-truth-report.json"),
            CreateArtifactDirectory("live-truth-checkpoint"),
            QueryCount: 3,
            TopK: 4,
            BaseVectorCount: 12,
            InsertedDeltaCount: 5,
            DeletedBaseCount: 3,
            DeletedDeltaCount: 2,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1382,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x1382);

        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName]);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadDataset(admission, report);
        ulong[] liveIds = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildLiveIds(options);
        TruthSet truth = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.GenerateLiveTruth(dataset, options, liveIds);

        Assert.Equal<ulong>([3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16], liveIds);
        Assert.Equal(liveIds.Length, report.UpdatedTruth.LiveVectorCount);
        Assert.Equal(liveIds.Length, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(liveIds.Length, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal(4, truth.Depth);
        Assert.All(truth.Results.SelectMany(row => row), item =>
        {
            Assert.Contains(item.Id, liveIds);
            Assert.DoesNotContain(item.Id, new ulong[] { 0, 1, 2, 12, 13 });
        });
    }

    private static void AssertSearchSection(
        HnswBasePlusExactDeltaCheckpointSearchSectionInfo section,
        string name,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
    {
        Assert.Equal(name, section.Name);
        Assert.Equal(options.QueryCount, section.Search.MeasuredQueryCount);
        Assert.Equal(options.Runs, section.Search.Runs.Length);
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

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 3, columns: 5);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false),
            ["external-fashion-mnist", "--download", "false"],
            spec);
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadDataset(
        FashionMnistAdmissionResult admission,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report)
    {
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        ExternalConvertedMatrixEntry baseEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "base");
        ExternalConvertedMatrixEntry queryEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "query");
        float[] baseVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le"),
            (ulong)baseEntry.RowCount,
            (uint)baseEntry.Dimension);
        float[] queryVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "query.f32le"),
            (ulong)queryEntry.RowCount,
            (uint)queryEntry.Dimension);

        return new FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset(
            new FashionMnistExternalHnswBenchmarkScenario.DatasetPaths(cacheRoot, admission.Manifest.DatasetId, admission.ManifestPath),
            admission.Manifest,
            report.Dataset.AdmissionManifest.Sha256,
            ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(admission.TruthPath))!,
            admission.Manifest.Truth.Sha256,
            baseVectors,
            queryVectors,
            baseEntry.RowCount,
            queryEntry.RowCount,
            baseEntry.Dimension);
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
            string.Create(CultureInfo.InvariantCulture, $"vec138-{prefix}-{Guid.NewGuid():N}"));
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
