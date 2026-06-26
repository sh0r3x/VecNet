using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec074DurableHnswGeneratedIndependentTests
{
    [Fact]
    public void Parser_AcceptsDocumentedDurableHnswBoundaryValues()
    {
        DurableHnswGeneratedOptions options = CommandLine.ParseDurableHnswGenerated(
            [
                "hnsw-generated-durable",
                "--dimension", "1",
                "--vectors", "1",
                "--queries", "1",
                "--top-k", "1",
                "--runs", "5",
                "--warmup-queries", "0",
                "--m", "64",
                "--ef-construction", "4096",
                "--ef-search", "4096",
                "--seed", "0xFFFFFFFF",
                "--hnsw-seed", "0xFFFFFFFFFFFFFFFF",
                "--output", "VecNet.BenchmarkRunner.Artifacts/vec074-boundary/report.json",
                "--snapshot-directory", "VecNet.BenchmarkRunner.Artifacts/vec074-boundary/snapshot"
            ]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(1, options.Dimension);
        Assert.Equal(1, options.VectorCount);
        Assert.Equal(1, options.QueryCount);
        Assert.Equal(1, options.TopK);
        Assert.Equal(5, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(64, options.M);
        Assert.Equal(4096, options.EfConstruction);
        Assert.Equal(4096, options.EfSearch);
        Assert.Equal(uint.MaxValue, options.Seed);
        Assert.Equal(ulong.MaxValue, options.HnswSeed);
    }

    [Theory]
    [InlineData("--filter", "all")]
    [InlineData("--candidate-set", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--insertions", "1")]
    [InlineData("--deletes", "1")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--query-count", "2")]
    [InlineData("--truth-depth", "10")]
    [InlineData("--download", "false")]
    [InlineData("--cache-root", "VecNet.DatasetCache")]
    public void Parser_RejectsOptionsOwnedByOtherRunnerScenarios(string optionName, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() =>
            CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", optionName, value]));

        Assert.Contains(optionName, exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("hnsw-generated-durable", "--metric")]
    [InlineData("hnsw-generated-durable", "--snapshot-directory")]
    [InlineData("hnsw-generated-durable", "--output")]
    [InlineData("hnsw-generated-durable", "--ef-construction", "2", "--m", "3")]
    [InlineData("hnsw-generated-durable", "--top-k", "2", "--ef-search", "1")]
    public void Parser_RejectsMissingValuesAndInvalidCrossOptionBounds(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ProgramRun_InvalidDurableHnswOptionsAbortBeforeReportOrSnapshotCreation()
    {
        string directory = NewArtifactDirectory("abort");
        string outputPath = Path.Combine(directory, "should-not-exist.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot-should-not-exist");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable",
                "--dimension", "8",
                "--vectors", "12",
                "--queries", "2",
                "--top-k", "4",
                "--ef-search", "3",
                "--output", outputPath,
                "--snapshot-directory", snapshotDirectory
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.Empty(Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories));
    }

    [Fact]
    public void ProgramRun_WritesExplicitIgnoredReportPathAndFreshPerRunSnapshots()
    {
        string directory = NewArtifactDirectory("program");
        string outputPath = Path.Combine(directory, "durable-report.json");
        string snapshotRoot = Path.Combine(directory, "snapshot-root");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable",
                "--metric", "SquaredEuclidean",
                "--dimension", "8",
                "--vectors", "24",
                "--queries", "2",
                "--top-k", "3",
                "--runs", "2",
                "--warmup-queries", "1",
                "--seed", "0x5EED074B",
                "--m", "4",
                "--ef-construction", "12",
                "--ef-search", "4",
                "--hnsw-seed", "0x000000000000074B",
                "--output", outputPath,
                "--snapshot-directory", snapshotRoot
            ]);

        Assert.Equal(0, exitCode);
        Assert.True(File.Exists(outputPath));
        AssertUnderArtifactRoot(outputPath);
        AssertUnderArtifactRoot(snapshotRoot);

        DurableHnswBenchmarkReport report = ReadReport(outputPath);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(outputPath, report.Command.Arguments.Single(value => string.Equals(value, outputPath, StringComparison.Ordinal)));
        Assert.Equal(snapshotRoot, report.Command.Arguments.Single(value => string.Equals(value, snapshotRoot, StringComparison.Ordinal)));
        Assert.Equal(2, report.Operations.Save.Runs.Length);
        Assert.Equal(2, report.Operations.Open.Runs.Length);

        string[] saveDirectories = report.Operations.Save.Runs.Select(run => run.SnapshotDirectory).ToArray();
        string[] openDirectories = report.Operations.Open.Runs.Select(run => run.SnapshotDirectory).ToArray();
        Assert.Equal(saveDirectories, openDirectories);
        Assert.Equal(saveDirectories.Length, saveDirectories.Distinct(StringComparer.OrdinalIgnoreCase).Count());

        Assert.All(saveDirectories, path =>
        {
            Assert.StartsWith(Path.GetFullPath(snapshotRoot), Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase);
            Assert.True(Directory.Exists(path));
            Assert.True(File.Exists(Path.Combine(path, "hnsw.manifest.json")));
            Assert.True(File.Exists(Path.Combine(path, "hnsw.graph.bin")));
        });

        Assert.StartsWith(Path.GetFullPath(snapshotRoot), Path.GetFullPath(report.Outputs.SnapshotOutput.DirectoryPath), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(saveDirectories[^1], report.Outputs.SnapshotOutput.DirectoryPath);
    }

    [Fact]
    public void ReportSerialization_KeepsDurableHnswPrivateIneligibleAndSchemaSeparated()
    {
        DurableHnswBenchmarkReport report = CreateSmallReport("schema");
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-074", root.GetProperty("taskId").GetString());
        Assert.Equal("hnsw-generated-durable", root.GetProperty("scenarioName").GetString());
        Assert.True(root.TryGetProperty("outputs", out JsonElement outputs));
        Assert.True(root.TryGetProperty("operations", out JsonElement operations));
        Assert.True(root.GetProperty("metrics").TryGetProperty("sourceHnsw", out _));
        Assert.True(root.GetProperty("metrics").TryGetProperty("openedHnsw", out _));
        Assert.False(root.TryGetProperty("baseline", out _));
        Assert.False(root.TryGetProperty("comparison", out _));
        Assert.False(root.TryGetProperty("matrix", out _));
        Assert.Equal("outsideSaveAndOpenDuration", outputs.GetProperty("snapshotOutput").GetProperty("scanTimingScope").GetString());
        Assert.Equal("notMeasured", operations.GetProperty("sourceSearch").GetProperty("status").GetString());

        foreach (string sectionName in new[] { "evidence", "validation", "eligibility" })
        {
            JsonElement section = root.GetProperty(sectionName);
            Assert.False(section.GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(section.GetProperty("previewReadinessEligible").GetBoolean());
            Assert.False(section.GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(section.GetProperty("comparisonArtifactEligible").GetBoolean());
            Assert.False(section.GetProperty("regressionGateEligible").GetBoolean());
        }

        Assert.DoesNotContain("\"publicClaimEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"previewReadinessEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonArtifactEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Report_IncludesIndependentOperationValidationRecallParityIntegrityAndFileFactPosture()
    {
        DurableHnswBenchmarkReport report = CreateSmallReport("invariants", runs: 2, warmupQueries: 1);

        Assert.Equal("build", report.Operations.Build.Name);
        Assert.All(report.Operations.Build.Runs, run => Assert.Equal("notApplicable", run.SnapshotDirectory));
        Assert.Equal("save", report.Operations.Save.Name);
        Assert.Equal("open", report.Operations.Open.Name);
        Assert.Equal("openedSearch", report.Operations.OpenedSearch.Name);
        Assert.Equal("notMeasured", report.Operations.SourceSearch.Status);
        Assert.Equal("notMeasured", report.Measurement.SourceSearch.Status);
        Assert.Contains("source-HNSW search was captured", report.Measurement.SourceSearch.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("exact truth construction", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("output-byte scans", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("source result capture", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(2, report.Operations.OpenedSearch.Aggregate.RunCount);
        Assert.Equal(3, report.Operations.OpenedSearch.Aggregate.MeasuredQueryCountPerRun);
        Assert.All(report.Operations.OpenedSearch.Runs, run =>
        {
            Assert.Equal(3, run.MeasuredQueryCount);
            Assert.True(run.ManagedAllocatedBytes >= 0);
            Assert.True(run.ManagedAllocatedBytesPerQuery >= 0);
        });

        Assert.Equal(report.Metrics.SourceHnsw.RecallAtK, report.Metrics.OpenedHnsw.RecallAtK);
        Assert.Equal(report.Metrics.SourceHnsw.OrderedAgreement, report.Metrics.OpenedHnsw.OrderedAgreement);
        Assert.True(report.Metrics.SourceAndOpenedRecallEqual);
        Assert.True(report.Metrics.SourceAndOpenedOrderedAgreementEqual);
        Assert.True(report.Metrics.SourceAndOpenedDistanceIntegrityEqual);
        Assert.Equal("passed", report.Metrics.SourceHnsw.ReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Metrics.OpenedHnsw.ReturnedResultIntegrity.Status);
        Assert.Equal(report.Metrics.SourceHnsw.ReturnedResultIntegrity.CheckedResultCount, report.Metrics.OpenedHnsw.ReturnedResultIntegrity.CheckedResultCount);
        Assert.True(report.Metrics.SourceHnsw.ReturnedResultIntegrity.CheckedResultCount > 0);
        Assert.Contains("within the accepted D-026 tolerance", report.Metrics.SourceHnsw.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.SourceHnswComparedToTruth);
        Assert.True(report.Validation.OpenedHnswComparedToTruth);
        Assert.Equal(3, report.Validation.SavedOpenedParity.QueryCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.WrittenCountMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.IdMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.OrderMismatchCount);
        Assert.Equal(0, report.Validation.SavedOpenedParity.DistanceMismatchCount);
        Assert.True(report.Validation.SavedOpenedParity.AllResultsMatched);
        Assert.Equal("passed", report.Validation.OpenedReadOnlyMutation.Status);
        Assert.True(report.Validation.OpenedReadOnlyMutation.RejectedBeforeVectorValidation);
        Assert.Contains("emptyVector", report.Validation.OpenedReadOnlyMutation.Operation, StringComparison.Ordinal);

        DurableHnswSnapshotOutputInfo output = report.Outputs.SnapshotOutput;
        Assert.Equal(5, output.FileCount);
        Assert.Equal(output.ManifestBytes + output.IdsBytes + output.VectorsBytes + output.LevelsBytes + output.GraphBytes, output.TotalBytes);
        Assert.Equal(32 + (32 * 8), output.IdsBytes);
        Assert.Equal(48 + (32 * 10 * 4), output.VectorsBytes);
        Assert.Equal(32 + (32 * 4), output.LevelsBytes);
        Assert.Equal("outsideSaveAndOpenDuration", output.ScanTimingScope);
        Assert.True(report.Validation.OutputBytesScannedOutsideSaveOpenDuration);
        Assert.Equal("notMeasured", report.Outputs.TemporaryDisk.Status);
        Assert.Equal("notMeasured", report.Outputs.PeakDisk.Status);
        Assert.Equal(output.TotalBytes, report.MemoryEstimates.DurableOutputBytes);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PeakMemory.Status);
    }

    private static DurableHnswBenchmarkReport CreateSmallReport(string prefix, int runs = 1, int warmupQueries = 0)
    {
        string directory = NewArtifactDirectory(prefix);
        string outputPath = Path.Combine(directory, "report.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        string[] arguments =
        [
            "hnsw-generated-durable",
            "--dimension", "10",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--runs", runs.ToString(),
            "--warmup-queries", warmupQueries.ToString(),
            "--seed", "0x5EED074C",
            "--m", "4",
            "--ef-construction", "12",
            "--ef-search", "4",
            "--hnsw-seed", "0x000000000000074C",
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory
        ];

        DurableHnswGeneratedOptions options = CommandLine.ParseDurableHnswGenerated(arguments);
        return DurableHnswGeneratedScenario.Run(options, arguments);
    }

    private static DurableHnswBenchmarkReport ReadReport(string path) =>
        ReportWriter.Deserialize<DurableHnswBenchmarkReport>(File.ReadAllText(path))!;

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec074-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertUnderArtifactRoot(string path)
    {
        string artifactRoot = Path.GetFullPath("VecNet.BenchmarkRunner.Artifacts");
        string fullPath = Path.GetFullPath(path);
        Assert.StartsWith(artifactRoot, fullPath, StringComparison.OrdinalIgnoreCase);
    }
}
