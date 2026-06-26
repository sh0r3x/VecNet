using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec074DurableHnswGeneratedTests
{
    [Fact]
    public void ParseDurableHnswGenerated_UsesPrivateSmokeDefaults()
    {
        DurableHnswGeneratedOptions options = CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(1024, options.VectorCount);
        Assert.Equal(25, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2073u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(200, options.EfConstruction);
        Assert.Equal(50, options.EfSearch);
        Assert.Equal(0x564543_034UL, options.HnswSeed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.SnapshotDirectory);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.False(Path.IsPathRooted(options.SnapshotDirectory));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnsw-generated-durable", "--metric", "Cosine")]
    [InlineData("hnsw-generated-durable", "--metric", "InnerProduct")]
    [InlineData("hnsw-generated-durable", "--dimension", "0")]
    [InlineData("hnsw-generated-durable", "--vectors", "0")]
    [InlineData("hnsw-generated-durable", "--queries", "0")]
    [InlineData("hnsw-generated-durable", "--top-k", "3", "--vectors", "2")]
    [InlineData("hnsw-generated-durable", "--runs", "0")]
    [InlineData("hnsw-generated-durable", "--runs", "6")]
    [InlineData("hnsw-generated-durable", "--warmup-queries", "-1")]
    [InlineData("hnsw-generated-durable", "--m", "1")]
    [InlineData("hnsw-generated-durable", "--m", "65")]
    [InlineData("hnsw-generated-durable", "--m", "8", "--ef-construction", "7")]
    [InlineData("hnsw-generated-durable", "--ef-construction", "4097")]
    [InlineData("hnsw-generated-durable", "--top-k", "10", "--ef-search", "9")]
    [InlineData("hnsw-generated-durable", "--ef-search", "4097")]
    [InlineData("hnsw-generated-durable", "--hnsw-seed", "0xNOTHEX")]
    [InlineData("hnsw-generated-durable", "--output", "")]
    [InlineData("hnsw-generated-durable", "--snapshot-directory", "")]
    [InlineData("hnsw-generated-durable", "--preset", "smoke")]
    [InlineData("hnsw-generated-durable", "--output-dir", "matrix")]
    [InlineData("hnsw-generated-durable", "--manifest", "manifest.json")]
    [InlineData("hnsw-generated-durable", "--baseline-report-id", "baseline")]
    [InlineData("hnsw-generated-durable", "--cache-root", "VecNet.DatasetCache")]
    public void ParseDurableHnswGenerated_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateDurableHnswReportWithSeparatedOperationsValidationAndStorageBytes()
    {
        string directory = NewArtifactDirectory("direct");
        string outputPath = Path.Combine(directory, "durable-hnsw.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        string[] arguments =
        [
            "hnsw-generated-durable",
            "--metric", "SquaredEuclidean",
            "--dimension", "12",
            "--vectors", "48",
            "--queries", "4",
            "--top-k", "5",
            "--runs", "3",
            "--warmup-queries", "2",
            "--seed", "0x5EED074A",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x000000000000074A",
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory
        ];
        DurableHnswGeneratedOptions options = CommandLine.ParseDurableHnswGenerated(arguments);

        DurableHnswBenchmarkReport report = DurableHnswGeneratedScenario.Run(options, arguments);
        DurableHnswGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-074", report.TaskId);
        Assert.Equal("hnsw-generated-durable", report.ScenarioName);
        Assert.Equal("hnsw-generated-durable", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-durable-hnsw-smoke", report.Evidence.Scope);
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

        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Workload.Metric);
        Assert.Equal(12, report.Workload.Dimension);
        Assert.Equal(48, report.Workload.VectorCount);
        Assert.Equal(4, report.Workload.QueryCount);
        Assert.Equal(5, report.Workload.TopK);
        Assert.Equal("0x000000000000074A", report.Workload.HnswSeed);
        Assert.Equal(4, report.Workload.M);
        Assert.Equal(16, report.Workload.EfConstruction);
        Assert.Equal(8, report.Workload.EfSearch);
        Assert.Equal(3, report.Workload.RunCount);
        Assert.Equal(2, report.Workload.WarmupQueryCount);
        Assert.Equal("hnsw", report.Workload.DurableFileFamilyName);
        Assert.Contains("opened read-only", report.Workload.SaveOpenLifecycle, StringComparison.OrdinalIgnoreCase);

        AssertOperation(report.Operations.Build, "build", "internal HnswIndex construction and Add calls", 3);
        AssertOperation(report.Operations.Save, "save", "internal HnswIndex.Save(directoryPath)", 3);
        AssertOperation(report.Operations.Open, "open", "internal HnswIndex.OpenReadOnly(directoryPath)", 3);
        Assert.Equal("openedSearch", report.Operations.OpenedSearch.Name);
        Assert.Equal("internal opened HnswIndex.Search(query, results, workspace)", report.Operations.OpenedSearch.TimedOperation);
        Assert.Equal(3, report.Operations.OpenedSearch.Runs.Length);
        Assert.Equal(3, report.Operations.OpenedSearch.Aggregate.RunCount);
        Assert.Equal(4, report.Operations.OpenedSearch.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("notMeasured", report.Operations.SourceSearch.Status);
        Assert.Contains("validation only", report.Operations.SourceSearch.Reason, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("measured", report.Measurement.Build.Latency.Status);
        Assert.Equal("measured", report.Measurement.Build.ManagedAllocations.Status);
        Assert.True(double.Parse(report.Measurement.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Equal("notMeasured", report.Measurement.Save.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.Open.ManagedAllocations.Status);
        Assert.Equal("measured", report.Measurement.OpenedSearch.Latency.Status);
        Assert.Equal("perMeasuredOpenedQuery", report.Measurement.OpenedSearch.Latency.SampleScope);
        Assert.Equal("internal opened HnswIndex.Search(query, results, workspace)", report.Measurement.OpenedSearch.Latency.TimedOperation);
        Assert.Contains("build, save, open", report.Measurement.OpenedSearch.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.OpenedSearch.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.OpenedSearch.ManagedAllocations.Unit);
        Assert.Equal("measured", report.Measurement.OpenedSearch.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.OpenedSearch.RunToRunNoise.Status);
        Assert.Equal("notMeasured", report.Measurement.SourceSearch.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Contains("output-byte scans", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("written", report.Outputs.SnapshotOutput.Status);
        Assert.True(Directory.Exists(report.Outputs.SnapshotOutput.DirectoryPath));
        Assert.Equal(5, report.Outputs.SnapshotOutput.FileCount);
        Assert.True(report.Outputs.SnapshotOutput.TotalBytes > 0);
        Assert.True(report.Outputs.SnapshotOutput.ManifestBytes > 0);
        Assert.Equal(32 + (48 * 8), report.Outputs.SnapshotOutput.IdsBytes);
        Assert.Equal(48 + (48 * 12 * 4), report.Outputs.SnapshotOutput.VectorsBytes);
        Assert.Equal(32 + (48 * 4), report.Outputs.SnapshotOutput.LevelsBytes);
        Assert.True(report.Outputs.SnapshotOutput.GraphBytes > 0);
        Assert.Equal(48, report.Outputs.SnapshotOutput.VectorCount);
        Assert.True(report.Outputs.SnapshotOutput.BytesPerVector > 0);
        Assert.Equal("passed", report.Outputs.SnapshotOutput.ValidationOpenStatus);
        Assert.Equal("outsideSaveAndOpenDuration", report.Outputs.SnapshotOutput.ScanTimingScope);
        Assert.True(File.Exists(Path.Combine(report.Outputs.SnapshotOutput.DirectoryPath, "hnsw.manifest.json")));
        Assert.True(File.Exists(Path.Combine(report.Outputs.SnapshotOutput.DirectoryPath, "hnsw.graph.bin")));

        AssertMetrics(report.Metrics.SourceHnsw);
        AssertMetrics(report.Metrics.OpenedHnsw);
        Assert.True(report.Metrics.SourceAndOpenedRecallEqual);
        Assert.True(report.Metrics.SourceAndOpenedOrderedAgreementEqual);
        Assert.True(report.Metrics.SourceAndOpenedDistanceIntegrityEqual);
        Assert.Contains("parity", report.Metrics.RecallEquivalenceReason, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.ExactTruthGenerated);
        Assert.True(report.Validation.SourceHnswBuilt);
        Assert.True(report.Validation.SourceHnswSaved);
        Assert.True(report.Validation.OpenedHnswOpened);
        Assert.True(report.Validation.OpenedIndexReadOnly);
        Assert.True(report.Validation.SourceHnswComparedToTruth);
        Assert.True(report.Validation.OpenedHnswComparedToTruth);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForSource);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForOpened);
        Assert.True(report.Validation.SavedOpenedParity.AllResultsMatched);
        Assert.Equal(0, report.Validation.SavedOpenedParity.WrittenCountMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.IdMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.OrderMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.DistanceMismatchCount);
        Assert.Equal("passed", report.Validation.OpenedReadOnlyMutation.Status);
        Assert.Equal("InvalidOperationException", report.Validation.OpenedReadOnlyMutation.ExceptionType);
        Assert.True(report.Validation.OpenedReadOnlyMutation.RejectedBeforeVectorValidation);
        Assert.True(report.Validation.OutputBytesScannedOutsideSaveOpenDuration);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);

        Assert.Equal("estimatedPayloadLowerBoundsAndFileFacts", report.MemoryEstimates.Status);
        Assert.Equal(48L * 12L * sizeof(float), report.MemoryEstimates.VectorPayloadBytes);
        Assert.Equal(48L * sizeof(ulong), report.MemoryEstimates.IdPayloadBytes);
        Assert.Equal(48L * sizeof(int), report.MemoryEstimates.LevelPayloadBytes);
        Assert.True(report.MemoryEstimates.GraphNeighborPayloadBytes > 0);
        Assert.True(report.MemoryEstimates.GraphCountPayloadBytes > 0);
        Assert.True(report.MemoryEstimates.SearchWorkspaceBytes > 0);
        Assert.Equal(report.Outputs.SnapshotOutput.TotalBytes, report.MemoryEstimates.DurableOutputBytes);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.GcHeap.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.WorkingSet.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PrivateBytes.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PeakMemory.Status);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-074", root.GetProperty("taskId").GetString());
        Assert.Equal("hnsw-generated-durable", root.GetProperty("scenarioName").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("openedSearch").GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("sourceSearch").GetProperty("status").GetString());
        Assert.True(root.GetProperty("outputs").GetProperty("snapshotOutput").GetProperty("totalBytes").GetInt64() > 0);
        Assert.True(root.GetProperty("validation").GetProperty("savedOpenedParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("openedReadOnlyMutation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("\"publicClaimEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndDurableModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--snapshot-directory", "snap"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--output-dir", "matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--insertions", "2"]));
        Assert.Equal("hnsw-generated-durable", DurableHnswGeneratedOptions.ScenarioName);
        Assert.Equal("hnsw-generated", HnswGeneratedOptions.ScenarioName);
    }

    private static void AssertOperation(DurableHnswOperationInfo operation, string name, string timedOperationContains, int runCount)
    {
        Assert.Equal(name, operation.Name);
        Assert.Contains(timedOperationContains, operation.TimedOperation, StringComparison.Ordinal);
        Assert.Equal(runCount, operation.Runs.Length);
        Assert.Equal(runCount, operation.Aggregate.RunCount);
        Assert.True(operation.Aggregate.MeanElapsedMilliseconds >= 0);
        Assert.All(operation.Runs, run =>
        {
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.NotEmpty(run.Status);
        });
    }

    private static void AssertMetrics(DurableHnswOperationMetricsInfo metrics)
    {
        Assert.InRange(metrics.RecallAtK, 0, 1);
        Assert.InRange(metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
        Assert.Equal("passed", metrics.ReturnedResultIntegrity.Status);
        Assert.True(metrics.ReturnedResultIntegrity.CheckedResultCount > 0);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.ResultCountViolationCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Contains("set recall@k", metrics.RecallDefinition, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Every returned", metrics.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec074-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
