using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec113HnswMemorySmokeTests
{
    [Fact]
    public void ParseHnswMemorySmoke_UsesPrivateSmokeDefaults()
    {
        HnswMemorySmokeOptions options = CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(4096, options.VectorCount);
        Assert.Equal(32, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(4, options.WarmupQueries);
        Assert.Equal(0x5EED2112u, options.Seed);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(128, options.EfSearch);
        Assert.Equal(0x484E535700011212UL, options.HnswSeed);
        Assert.Equal(10, options.SampleIntervalMilliseconds);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.SnapshotDirectory);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.False(Path.IsPathRooted(options.SnapshotDirectory));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-memory-smoke", "--metric", "Cosine")]
    [InlineData("generated-hnsw-memory-smoke", "--dimension", "0")]
    [InlineData("generated-hnsw-memory-smoke", "--vectors", "0")]
    [InlineData("generated-hnsw-memory-smoke", "--queries", "0")]
    [InlineData("generated-hnsw-memory-smoke", "--top-k", "3", "--vectors", "2")]
    [InlineData("generated-hnsw-memory-smoke", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-memory-smoke", "--m", "1")]
    [InlineData("generated-hnsw-memory-smoke", "--m", "65")]
    [InlineData("generated-hnsw-memory-smoke", "--m", "8", "--ef-construction", "7")]
    [InlineData("generated-hnsw-memory-smoke", "--ef-construction", "4097")]
    [InlineData("generated-hnsw-memory-smoke", "--top-k", "10", "--ef-search", "9")]
    [InlineData("generated-hnsw-memory-smoke", "--ef-search", "4097")]
    [InlineData("generated-hnsw-memory-smoke", "--hnsw-seed", "0xNOTHEX")]
    [InlineData("generated-hnsw-memory-smoke", "--sample-interval-ms", "0")]
    [InlineData("generated-hnsw-memory-smoke", "--sample-interval-ms", "1001")]
    [InlineData("generated-hnsw-memory-smoke", "--output", "")]
    [InlineData("generated-hnsw-memory-smoke", "--snapshot-directory", "")]
    [InlineData("generated-hnsw-memory-smoke", "--runs", "2")]
    [InlineData("generated-hnsw-memory-smoke", "--save-directory", "save")]
    [InlineData("generated-hnsw-memory-smoke", "--checkpoint-directory", "checkpoint")]
    [InlineData("generated-hnsw-memory-smoke", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-hnsw-memory-smoke", "--preset", "smoke")]
    [InlineData("generated-hnsw-memory-smoke", "--baseline-report-id", "baseline")]
    public void ParseHnswMemorySmoke_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesSeparatedActualPeakLowerBoundAndStorageReportWithFalseEligibility()
    {
        string directory = NewArtifactDirectory("report");
        string outputPath = Path.Combine(directory, "generated-hnsw-memory-smoke.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        string[] arguments =
        [
            "generated-hnsw-memory-smoke",
            "--metric", "SquaredEuclidean",
            "--dimension", "9",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--warmup-queries", "2",
            "--seed", "0x5EED1130",
            "--m", "4",
            "--ef-construction", "12",
            "--ef-search", "6",
            "--hnsw-seed", "0x0000000000011300",
            "--sample-interval-ms", "1",
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory
        ];
        HnswMemorySmokeOptions options = CommandLine.ParseHnswMemorySmoke(arguments);

        HnswMemorySmokeReport report = HnswMemorySmokeScenario.Run(options, arguments);
        HnswMemorySmokeScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.True(File.Exists(Path.Combine(snapshotDirectory, "hnsw.manifest.json")));
        Assert.True(File.Exists(Path.Combine(snapshotDirectory, "hnsw.graph.bin")));
        Assert.Equal("VecNet.HnswMemorySmokeReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-113", report.TaskId);
        Assert.Equal("generated-hnsw-memory-smoke", report.ScenarioName);
        Assert.Equal("generated-hnsw-memory-smoke", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-hnsw-memory-smoke", report.Evidence.Scope);
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
        Assert.Equal(9, report.Workload.Dimension);
        Assert.Equal(32, report.Workload.VectorCount);
        Assert.Equal(3, report.Workload.QueryCount);
        Assert.Equal(4, report.Workload.TopK);
        Assert.Equal(2, report.Workload.WarmupQueries);
        Assert.Equal("0x0000000000011300", report.Workload.HnswSeed);
        Assert.Equal(1, report.Workload.SampleIntervalMilliseconds);

        Assert.Equal("measured", report.ActualMemory.Status);
        AssertMemorySample(report.ActualMemory.BaselineProcess, "baselineProcess");
        AssertMemorySample(report.ActualMemory.PostDatasetGeneration, "postDatasetGeneration");
        AssertMemorySample(report.ActualMemory.PostSourceBuildRetained, "postSourceBuildRetained");
        AssertMemorySample(report.ActualMemory.PostSourceWarmSearchRetained, "postSourceWarmSearchRetained");
        AssertMemorySample(report.ActualMemory.PostSaveRetained, "postSaveRetained");
        AssertMemorySample(report.ActualMemory.PostOpenReadOnlyRetained, "postOpenReadOnlyRetained");
        AssertMemorySample(report.ActualMemory.PostOpenedWarmSearchRetained, "postOpenedWarmSearchRetained");
        AssertMemorySample(report.ActualMemory.PostValidationRetained, "postValidationRetained");
        Assert.True(report.ActualMemory.BaselineProcess.ProcessWorkingSetBytes.ContextOnly);
        Assert.True(report.ActualMemory.BaselineProcess.ProcessPeakWorkingSetBytes.ContextOnly);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateIdMapRetainedMemory.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateGraphLayerObjectMemory.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectHeadersArrayHeadersAlignmentAndSlack.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.NeighborCandidateRetainedLayout.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.IndexOnlyPrivateBytes.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.OpenedOnlyRetainedMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.SaveManagedAllocations.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.OpenManagedAllocations.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.TrueProcessPeakMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.PeakTemporaryDisk.Status);

        Assert.Equal("sampled", report.PeakMemory.Status);
        AssertPeak(report.PeakMemory.Build, "build");
        AssertPeak(report.PeakMemory.Save, "save");
        AssertPeak(report.PeakMemory.Open, "open");
        Assert.Equal("notMeasured", report.PeakMemory.SourceSearchWarmupPeakMemory.Status);
        Assert.Equal("notMeasured", report.PeakMemory.OpenedSearchWarmupPeakMemory.Status);
        Assert.Equal("notMeasured", report.PeakMemory.PeakTemporaryDiskBytes.Status);

        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.Status);
        Assert.Contains("not actual retained memory", report.LayoutLowerBounds.ClaimBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(32L * 9L * sizeof(float), report.LayoutLowerBounds.VectorPayloadLowerBoundBytes);
        Assert.Equal(32L * sizeof(ulong), report.LayoutLowerBounds.IdPayloadLowerBoundBytes);
        Assert.Equal(32L * sizeof(int), report.LayoutLowerBounds.LevelPayloadLowerBoundBytes);
        Assert.True(report.LayoutLowerBounds.GraphCountPayloadLowerBoundBytes > 0);
        Assert.True(report.LayoutLowerBounds.GraphNeighborPayloadLowerBoundBytes > 0);
        Assert.Equal(report.LayoutLowerBounds.GraphCountPayloadLowerBoundBytes + report.LayoutLowerBounds.GraphNeighborPayloadLowerBoundBytes, report.LayoutLowerBounds.GraphPayloadLowerBoundBytes);
        Assert.Equal(32L * (sizeof(ulong) + sizeof(int)), report.LayoutLowerBounds.IdMapEntryPayloadLowerBoundBytes);
        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.SearchWorkspacePayloadLowerBoundBytes.Status);
        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.BuildScratchPayloadLowerBoundBytes.Status);
        Assert.True(report.LayoutLowerBounds.SourceRetainedPayloadLowerBoundBytes > report.LayoutLowerBounds.OpenedRetainedPayloadLowerBoundBytes);
        Assert.NotEmpty(report.LayoutLowerBounds.Layers);

        Assert.Equal("fileFacts", report.StorageSize.Status);
        Assert.Equal("private ignored benchmark-runner artifact path", report.StorageSize.SnapshotDirectoryPathPolicy);
        Assert.Equal(snapshotDirectory, report.StorageSize.SnapshotDirectory);
        Assert.Equal(5, report.StorageSize.FileCount);
        Assert.Equal(report.StorageSize.ManifestBytes + report.StorageSize.IdsBytes + report.StorageSize.VectorsBytes + report.StorageSize.LevelsBytes + report.StorageSize.GraphBytes, report.StorageSize.TotalBytes);
        Assert.Equal(32 + (32 * 8), report.StorageSize.IdsBytes);
        Assert.Equal(48 + (32 * 9 * 4), report.StorageSize.VectorsBytes);
        Assert.Equal(32 + (32 * 4), report.StorageSize.LevelsBytes);
        Assert.True(report.StorageSize.BytesPerVector > 0);
        Assert.Contains("outside", report.StorageSize.ScanTimingScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.StorageSize.PeakObservedOutputDirectoryBytes.Status);
        Assert.Equal("notMeasured", report.StorageSize.PeakTemporaryDiskBytes.Status);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.SourceHnswBuilt);
        Assert.True(report.Validation.SourceWarmSearchExecuted);
        Assert.True(report.Validation.SourceHnswSaved);
        Assert.True(report.Validation.OpenedHnswOpened);
        Assert.True(report.Validation.OpenedIndexReadOnly);
        Assert.True(report.Validation.OpenedWarmSearchExecuted);
        Assert.True(report.Validation.SourceOpenedParityChecked);
        Assert.True(report.Validation.SourceOpenedParity.AllResultsMatched);
        Assert.Equal("passed", report.Validation.SourceReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Validation.OpenedReturnedResultIntegrity.Status);
        Assert.True(report.Validation.ActualPeakLowerBoundAndStorageSectionsSeparated);
        Assert.True(report.Validation.UnsupportedFieldsExplicitlyMarked);
        Assert.True(report.Validation.WorkingSetContextOnly);
        Assert.True(report.Validation.SampledPeakLabelsPresent);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);

        Assert.Contains(report.Notes, note => note.Contains("observed sampled peaks", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(report.Notes, note => note.Contains("Working set", StringComparison.OrdinalIgnoreCase) && note.Contains("context", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(report.Notes, note => note.Contains("storageSize", StringComparison.Ordinal));
        Assert.Contains(report.Notes, note => note.Contains("notMeasured", StringComparison.Ordinal) || note.Contains("notAvailable", StringComparison.Ordinal));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswMemorySmokeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("generated-hnsw-memory-smoke", root.GetProperty("scenarioName").GetString());
        Assert.True(root.TryGetProperty("actualMemory", out JsonElement actualMemory));
        Assert.True(root.TryGetProperty("peakMemory", out JsonElement peakMemory));
        Assert.True(root.TryGetProperty("layoutLowerBounds", out JsonElement layoutLowerBounds));
        Assert.True(root.TryGetProperty("storageSize", out JsonElement storageSize));
        Assert.Equal("measured", actualMemory.GetProperty("baselineProcess").GetProperty("managedHeapSizeBytes").GetProperty("status").GetString());
        Assert.True(actualMemory.GetProperty("postSourceWarmSearchRetained").GetProperty("processWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        Assert.Equal("sampled", peakMemory.GetProperty("build").GetProperty("status").GetString());
        Assert.Equal("sampled", peakMemory.GetProperty("build").GetProperty("peakObservedPrivateBytes").GetProperty("status").GetString());
        Assert.True(peakMemory.GetProperty("save").GetProperty("peakObservedWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        Assert.Contains("miss", peakMemory.GetProperty("open").GetProperty("missedShortPeakCaveat").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("estimatedLowerBound", layoutLowerBounds.GetProperty("status").GetString());
        Assert.Equal("fileFacts", storageSize.GetProperty("status").GetString());
        Assert.Equal("notMeasured", storageSize.GetProperty("peakTemporaryDiskBytes").GetProperty("status").GetString());
        Assert.Equal("notAvailable", actualMemory.GetProperty("unsupported").GetProperty("objectAccurateIdMapRetainedMemory").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("\"publicClaimEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExistingHnswParsersRemainCompatibleAndMemorySmokeModeIsIsolated()
    {
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--query-count", "1", "--top-k", "1", "--ef-search", "1"]);
        _ = CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--sample-interval-ms", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--sample-interval-ms", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke", "--runs", "2"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke", "--cache-root", "VecNet.DatasetCache"]));
        Assert.Equal("generated-hnsw-memory-smoke", HnswMemorySmokeOptions.ScenarioName);
        Assert.Equal("hnsw-generated", HnswGeneratedOptions.ScenarioName);
        Assert.Equal("hnsw-generated-durable", DurableHnswGeneratedOptions.ScenarioName);
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
        Assert.NotEmpty(operation.TimedScope);
        Assert.NotEmpty(operation.ExcludedOperations);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec113-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
