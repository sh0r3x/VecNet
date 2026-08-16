using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec348HnswInnerProductEvidenceRunnerPrerequisiteTests
{
    [Fact]
    public void HnswMemorySmokeInnerProductRunsAndSerializesSelectedMetric()
    {
        string directory = NewArtifactDirectory("memory-ip");
        string outputPath = Path.Combine(directory, "memory-ip.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        string[] arguments =
        [
            HnswMemorySmokeOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--warmup-queries", "1",
            "--seed", "0x5EED3480",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003480",
            "--sample-interval-ms", "1",
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory
        ];

        HnswMemorySmokeReport report = HnswMemorySmokeScenario.Run(CommandLine.ParseHnswMemorySmoke(arguments), arguments);
        HnswMemorySmokeScenario.Write(report, outputPath);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Workload.Metric);
        Assert.Contains(VectorMetric.InnerProduct.ToString(), report.Hnsw.MetricScope, StringComparison.Ordinal);
        Assert.Equal("passed", report.Validation.SourceReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Validation.OpenedReturnedResultIntegrity.Status);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("InnerProduct", root.GetProperty("dataset").GetProperty("metric").GetString());
        Assert.Equal("InnerProduct", root.GetProperty("index").GetProperty("metric").GetString());
        Assert.Equal("InnerProduct", root.GetProperty("workload").GetProperty("metric").GetString());
        Assert.Contains("InnerProduct", root.GetProperty("hnsw").GetProperty("metricScope").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void HnswMemorySmokeSquaredEuclideanStillRunsAndReportsSquaredEuclidean()
    {
        string directory = NewArtifactDirectory("memory-l2");
        string[] arguments =
        [
            HnswMemorySmokeOptions.ScenarioName,
            "--metric", "SquaredEuclidean",
            "--dimension", "7",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--warmup-queries", "1",
            "--seed", "0x5EED3481",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003481",
            "--sample-interval-ms", "1",
            "--output", Path.Combine(directory, "memory-l2.json"),
            "--snapshot-directory", Path.Combine(directory, "snapshot")
        ];

        HnswMemorySmokeReport report = HnswMemorySmokeScenario.Run(CommandLine.ParseHnswMemorySmoke(arguments), arguments);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Index.Metric);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Workload.Metric);
        Assert.Contains(VectorMetric.SquaredEuclidean.ToString(), report.Hnsw.MetricScope, StringComparison.Ordinal);
    }

    [Fact]
    public void MutableGeneratedReportDistinguishesFirstPassEfSearchFromWorkspaceRetryCeiling()
    {
        string outputPath = Path.Combine(NewArtifactDirectory("mutable-diagnostics"), "mutable.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaGeneratedOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "32",
            "--queries", "4",
            "--top-k", "4",
            "--insertions", "4",
            "--deletes", "16",
            "--delta-deletes", "0",
            "--duplicate-inserts", "0",
            "--unknown-deletes", "0",
            "--repeated-deletes", "0",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED3482",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "4",
            "--workspace-ef-search", "16",
            "--hnsw-seed", "0x0000000000003482",
            "--output", outputPath
        ];

        HnswBasePlusExactDeltaGeneratedOptions options = CommandLine.ParseHnswBasePlusExactDeltaGenerated(arguments);
        HnswBasePlusExactDeltaBenchmarkReport report = HnswBasePlusExactDeltaGeneratedScenario.Run(options, arguments);
        HnswBasePlusExactDeltaGeneratedScenario.Write(report, outputPath);

        Assert.Equal(4, options.EfSearch);
        Assert.Equal(16, options.WorkspaceEfSearch);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(4, report.Hnsw.EfSearch);
        Assert.Equal("measured", report.RetryDiagnostics.Status);
        Assert.Equal(4, report.RetryDiagnostics.FirstPassEfSearch);
        Assert.Equal(16, report.RetryDiagnostics.WorkspaceEfSearchCeiling);
        Assert.Equal(16, report.RetryDiagnostics.EffectiveRetryEfSearchCeiling);
        Assert.Equal(12, report.RetryDiagnostics.RetryWidthDelta);
        Assert.True(report.RetryDiagnostics.WorkspaceCanWidenBeyondFirstPass);
        Assert.True(report.RetryDiagnostics.BaseTombstonesPresent);
        Assert.Equal(report.Underfill.UnderfilledQueryCount > 0, report.RetryDiagnostics.UnderfillRemainedAfterWidening);
        Assert.Equal(report.Underfill.UnderfilledSlotCount, report.RetryDiagnostics.UnderfilledSlotCountAfterWidening);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement retry = document.RootElement.GetProperty("retryDiagnostics");
        Assert.Equal(4, retry.GetProperty("firstPassEfSearch").GetInt32());
        Assert.Equal(16, retry.GetProperty("workspaceEfSearchCeiling").GetInt32());
        Assert.True(retry.GetProperty("workspaceCanWidenBeyondFirstPass").GetBoolean());
        Assert.True(retry.GetProperty("baseTombstonesPresent").GetBoolean());
        Assert.True(retry.TryGetProperty("retryWideningObserved", out _));
        Assert.True(retry.TryGetProperty("underfillRemainedAfterWidening", out _));
    }

    [Fact]
    public void MutableCheckpointReportExposesRetryDiagnosticsPerSearchSection()
    {
        string directory = NewArtifactDirectory("checkpoint-diagnostics");
        string outputPath = Path.Combine(directory, "checkpoint.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--insertions", "4",
            "--deletes", "16",
            "--delta-deletes", "0",
            "--duplicate-inserts", "0",
            "--unknown-deletes", "0",
            "--repeated-deletes", "0",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED3483",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "4",
            "--workspace-ef-search", "16",
            "--hnsw-seed", "0x0000000000003483",
            "--output", outputPath,
            "--checkpoint-directory", Path.Combine(directory, "checkpoint-output")
        ];

        HnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            HnswBasePlusExactDeltaCheckpointScenario.Run(CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(arguments), arguments);
        HnswBasePlusExactDeltaCheckpointScenario.Write(report, outputPath);

        HnswBasePlusExactDeltaRetryDiagnosticsInfo pre = Assert.IsType<HnswBasePlusExactDeltaRetryDiagnosticsInfo>(
            report.Searches.PreCheckpointComposite.RetryDiagnostics);
        Assert.Equal("measured", pre.Status);
        Assert.Equal(4, pre.FirstPassEfSearch);
        Assert.Equal(16, pre.WorkspaceEfSearchCeiling);
        Assert.Equal(16, pre.EffectiveRetryEfSearchCeiling);
        Assert.True(pre.WorkspaceCanWidenBeyondFirstPass);
        Assert.True(pre.BaseTombstonesPresent);

        HnswBasePlusExactDeltaRetryDiagnosticsInfo opened = Assert.IsType<HnswBasePlusExactDeltaRetryDiagnosticsInfo>(
            report.Searches.OpenedReadOnlyHnsw.RetryDiagnostics);
        Assert.Equal("notApplicable", opened.Status);
        Assert.False(opened.RetryWideningObserved);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement preJson = document.RootElement.GetProperty("searches").GetProperty("preCheckpointComposite").GetProperty("retryDiagnostics");
        Assert.Equal(4, preJson.GetProperty("firstPassEfSearch").GetInt32());
        Assert.Equal(16, preJson.GetProperty("workspaceEfSearchCeiling").GetInt32());
        Assert.True(preJson.TryGetProperty("underfillRemainedAfterWidening", out _));
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec348-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
