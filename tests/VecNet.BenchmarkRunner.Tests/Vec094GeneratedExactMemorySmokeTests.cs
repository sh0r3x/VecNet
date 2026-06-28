using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec094GeneratedExactMemorySmokeTests
{
    [Fact]
    public void ParseGeneratedExactMemorySmoke_UsesPrivateSmokeDefaults()
    {
        GeneratedExactMemorySmokeOptions options =
            CommandLine.ParseGeneratedExactMemorySmoke(["generated-exact-memory-smoke"]);

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
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED2094u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.SaveDirectory);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.CheckpointDirectory);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.False(Path.IsPathRooted(options.SaveDirectory));
        Assert.False(Path.IsPathRooted(options.CheckpointDirectory));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-memory-smoke", "--dimension")]
    [InlineData("generated-exact-memory-smoke", "dimension", "8")]
    [InlineData("generated-exact-memory-smoke", "--metric", "Unknown")]
    [InlineData("generated-exact-memory-smoke", "--dimension", "0")]
    [InlineData("generated-exact-memory-smoke", "--vectors", "0")]
    [InlineData("generated-exact-memory-smoke", "--queries", "0")]
    [InlineData("generated-exact-memory-smoke", "--top-k", "11", "--vectors", "10", "--insertions", "1", "--deletes", "1")]
    [InlineData("generated-exact-memory-smoke", "--warmup-queries", "-1")]
    [InlineData("generated-exact-memory-smoke", "--insertions", "0")]
    [InlineData("generated-exact-memory-smoke", "--deletes", "0")]
    [InlineData("generated-exact-memory-smoke", "--deletes", "11", "--vectors", "10")]
    [InlineData("generated-exact-memory-smoke", "--duplicate-inserts", "-1")]
    [InlineData("generated-exact-memory-smoke", "--unknown-deletes", "-1")]
    [InlineData("generated-exact-memory-smoke", "--repeated-deletes", "-1")]
    [InlineData("generated-exact-memory-smoke", "--allowlist", "unknown")]
    [InlineData("generated-exact-memory-smoke", "--candidate-set", "unknown")]
    [InlineData("generated-exact-memory-smoke", "--allowlist", "very-selective", "--top-k", "1")]
    [InlineData("generated-exact-memory-smoke", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-memory-smoke", "--unknown-ids", "-1")]
    [InlineData("generated-exact-memory-smoke", "--output", "")]
    [InlineData("generated-exact-memory-smoke", "--save-directory", "")]
    [InlineData("generated-exact-memory-smoke", "--checkpoint-directory", "")]
    [InlineData("generated-exact-memory-smoke", "--runs", "2")]
    [InlineData("generated-exact-memory-smoke", "--index-directory", "index")]
    [InlineData("generated-exact-memory-smoke", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-memory-smoke", "--preset", "smoke")]
    [InlineData("generated-exact-memory-smoke", "--filter", "broad")]
    public void ParseGeneratedExactMemorySmoke_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseGeneratedExactMemorySmoke(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesSeparatedActualMemoryAndLowerBoundReportWithFalseEligibility()
    {
        string outputPath = NewArtifactPath("memory-smoke-report.json");
        string saveDirectory = Path.Combine(Path.GetDirectoryName(outputPath)!, "save-output");
        string checkpointDirectory = Path.Combine(Path.GetDirectoryName(outputPath)!, "checkpoint-output");
        string[] arguments =
        [
            "generated-exact-memory-smoke",
            "--metric", "SquaredEuclidean",
            "--dimension", "7",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--insertions", "5",
            "--deletes", "4",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "2",
            "--allowlist", "broad",
            "--candidate-set", "selective",
            "--duplicate-ids", "1",
            "--unknown-ids", "1",
            "--warmup-queries", "2",
            "--seed", "0x5EED0940",
            "--output", outputPath,
            "--save-directory", saveDirectory,
            "--checkpoint-directory", checkpointDirectory
        ];
        GeneratedExactMemorySmokeOptions options =
            CommandLine.ParseGeneratedExactMemorySmoke(arguments);

        GeneratedExactMemorySmokeReport report =
            GeneratedExactMemorySmokeScenario.Run(options, arguments);
        GeneratedExactMemorySmokeScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.True(File.Exists(Path.Combine(saveDirectory, "exact-flat.manifest.json")));
        Assert.True(File.Exists(Path.Combine(checkpointDirectory, "exact-flat.manifest.json")));
        Assert.Equal("VecNet.ExactMemorySmokeReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-094", report.TaskId);
        Assert.Equal("generated-exact-memory-smoke", report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-memory-smoke", report.Evidence.Scope);
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

        Assert.Equal(29, report.Workload.PhysicalVectorCountAfterMutation);
        Assert.Equal(25, report.Workload.LiveVectorCountAfterMutation);
        Assert.Equal(20, report.Workload.LiveBaseVectorCountAfterMutation);
        Assert.Equal(5, report.Workload.LiveDeltaVectorCountAfterMutation);
        Assert.Equal(4, report.Workload.TombstoneCountAfterMutation);
        Assert.Equal(4, report.Workload.DeletedReservedIdCountAfterMutation);
        Assert.Equal(4.0 / 29.0, report.Workload.TombstoneRatio, precision: 12);
        Assert.Equal(5.0 / 24.0, report.Workload.DeltaRatio, precision: 12);
        Assert.Equal(12, report.Workload.RawAllowlistKnownIdCountPerQuery);
        Assert.Equal(3, report.Workload.CandidateSetKnownIdCountPerQuery);
        Assert.Equal(3, report.Workload.CandidateSetCount);
        Assert.Equal(9, report.Workload.CandidateSetOrdinalCount);

        Assert.Equal("measured", report.ActualMemory.Status);
        Assert.Contains("whole-process", report.ActualMemory.Limitations[0], StringComparison.OrdinalIgnoreCase);
        AssertMemorySample(report.ActualMemory.BaselineProcess, "baselineProcess");
        AssertMemorySample(report.ActualMemory.PostDatasetGeneration, "postDatasetGeneration");
        AssertMemorySample(report.ActualMemory.PostIndexBuildRetained, "postIndexBuildRetained");
        AssertMemorySample(report.ActualMemory.PostWarmSearchRetained, "postWarmSearchRetained");
        AssertMemorySample(report.ActualMemory.RawAllowlistWorkspaceRetained, "rawAllowlistWorkspaceRetained");
        AssertMemorySample(report.ActualMemory.CandidateSetRetained, "candidateSetRetained");
        AssertMemorySample(report.ActualMemory.PostMutationRetained, "postMutationRetained");
        AssertMemorySample(report.ActualMemory.PostSaveRetained, "postSaveRetained");
        AssertMemorySample(report.ActualMemory.PostOpenReadOnlyRetained, "postOpenReadOnlyRetained");
        AssertMemorySample(report.ActualMemory.OpenedReadOnlyWarmSearchRetained, "openedReadOnlyWarmSearchRetained");
        AssertMemorySample(report.ActualMemory.PostCheckpointRetained, "postCheckpointRetained");
        Assert.True(report.ActualMemory.BaselineProcess.ProcessWorkingSetBytes.ContextOnly);
        Assert.True(report.ActualMemory.BaselineProcess.ProcessPeakWorkingSetBytes.ContextOnly);
        Assert.Equal("notMeasured", report.ActualMemory.PostCheckpointRetained.PeakObservedPrivateBytes.Status);
        Assert.Equal("notMeasured", report.ActualMemory.PostCheckpointRetained.PeakObservedWorkingSetBytes.Status);

        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateIdMapRetainedMemory.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateTombstoneHashSetRetainedMemory.Status);
        Assert.Equal("notAvailable", report.ActualMemory.Unsupported.ObjectAccurateDeletedReservationHashSetRetainedMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.IndexOnlyPrivateBytes.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.OpenedOnlyRetainedMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.PeakTemporaryProcessMemory.Status);
        Assert.Equal("notMeasured", report.ActualMemory.Unsupported.PeakTemporaryDisk.Status);

        Assert.Equal("estimatedLowerBound", report.LayoutLowerBounds.Status);
        Assert.Contains("not actual retained memory", report.LayoutLowerBounds.ClaimBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(29 * 8, report.LayoutLowerBounds.PhysicalIdPayloadLowerBoundBytes);
        Assert.Equal(29 * 7 * 4, report.LayoutLowerBounds.PhysicalVectorPayloadLowerBoundBytes);
        Assert.Equal(25 * 7 * 4, report.LayoutLowerBounds.LiveVectorPayloadLowerBoundBytes);
        Assert.Equal(29 * (8 + 4), report.LayoutLowerBounds.IdMapEntryPayloadLowerBoundBytes);
        Assert.Equal(24 * 4, report.LayoutLowerBounds.RawAllowlistWorkspacePayloadLowerBoundBytes);
        Assert.Equal(9 * 4, report.LayoutLowerBounds.CandidateSetOrdinalPayloadLowerBoundBytes);
        Assert.Equal((25 * 8) + (25 * 7 * 4), report.LayoutLowerBounds.CheckpointSnapshotPayloadLowerBoundBytes);
        Assert.Equal(25 * 8, report.LayoutLowerBounds.DurableIdPayloadBytes);
        Assert.Equal(25 * 7 * 4, report.LayoutLowerBounds.DurableVectorPayloadBytes);
        Assert.Equal("notAvailable", report.LayoutLowerBounds.TombstoneIdPayloadLowerBoundBytes.Status);
        Assert.Equal("notAvailable", report.LayoutLowerBounds.DeletedReservedIdPayloadLowerBoundBytes.Status);

        Assert.Equal("written", report.Outputs.SaveOutput.Status);
        Assert.Equal("written", report.Outputs.CheckpointOutput.Status);
        Assert.True(report.Outputs.SaveOutput.FinalOutputBytes > 0);
        Assert.True(report.Outputs.CheckpointOutput.FinalOutputBytes > 0);
        Assert.Equal(25, report.Outputs.SaveOutput.OutputVectorCount);
        Assert.Equal(25, report.Outputs.CheckpointOutput.OutputVectorCount);
        Assert.Equal("notMeasured", report.Outputs.PeakObservedSaveOutputDirectoryBytes.Status);
        Assert.Equal("notMeasured", report.Outputs.PeakObservedCheckpointOutputDirectoryBytes.Status);
        Assert.Equal("notMeasured", report.Outputs.PeakTemporaryDiskBytes.Status);
        Assert.Equal("notMeasured", report.Outputs.PeakObservedPrivateBytesDuringSave.Status);
        Assert.Equal("notMeasured", report.Outputs.PeakObservedPrivateBytesDuringCheckpoint.Status);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.ActualAndEstimateSectionsSeparated);
        Assert.True(report.Validation.UnsupportedFieldsExplicitlyMarked);
        Assert.True(report.Validation.WorkingSetContextOnly);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);

        Assert.Contains(report.Notes, note => note.Contains("Working set", StringComparison.OrdinalIgnoreCase) && note.Contains("context", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(report.Notes, note => note.Contains("Lower-bound", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(report.Notes, note => note.Contains("Peak temporary", StringComparison.OrdinalIgnoreCase) && note.Contains("not measured", StringComparison.OrdinalIgnoreCase));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactMemorySmokeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("generated-exact-memory-smoke", root.GetProperty("scenarioName").GetString());
        Assert.True(root.TryGetProperty("actualMemory", out JsonElement actualMemory));
        Assert.True(root.TryGetProperty("layoutLowerBounds", out JsonElement layoutLowerBounds));
        Assert.Equal("measured", actualMemory.GetProperty("baselineProcess").GetProperty("managedHeapSizeBytes").GetProperty("status").GetString());
        Assert.Equal("measured", actualMemory.GetProperty("postIndexBuildRetained").GetProperty("processPrivateBytes").GetProperty("status").GetString());
        Assert.True(actualMemory.GetProperty("postWarmSearchRetained").GetProperty("processWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        Assert.Equal("notMeasured", actualMemory.GetProperty("postCheckpointRetained").GetProperty("peakObservedPrivateBytes").GetProperty("status").GetString());
        Assert.Equal("notAvailable", actualMemory.GetProperty("unsupported").GetProperty("objectAccurateIdMapRetainedMemory").GetProperty("status").GetString());
        Assert.Equal("estimatedLowerBound", layoutLowerBounds.GetProperty("status").GetString());
        Assert.Equal(25 * 7 * 4, layoutLowerBounds.GetProperty("durableVectorPayloadBytes").GetInt64());
        Assert.Equal("notAvailable", layoutLowerBounds.GetProperty("tombstoneIdPayloadLowerBoundBytes").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("outputs").GetProperty("peakTemporaryDiskBytes").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndMemorySmokeModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactMemorySmoke(["generated-exact-memory-smoke", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactMemorySmoke(["generated-exact-memory-smoke", "--runs", "2"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactMemorySmoke(["generated-exact-memory-smoke", "--index-directory", "index"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactOpenedSearch(["generated-exact-opened-search", "--save-directory", "save"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactPracticalUpdate(["generated-exact-practical-update", "--save-directory", "save"]));
        Assert.Equal("generated-exact-memory-smoke", GeneratedExactMemorySmokeOptions.ScenarioName);
    }

    private static void AssertMemorySample(GeneratedExactMemorySampleInfo sample, string name)
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

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec094-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
