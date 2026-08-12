using System.Globalization;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec346HnswInnerProductBenchmarkRunnerTests
{
    [Fact]
    public void CommandLineOptionsAndMatricesRepresentPrivateHnswInnerProduct()
    {
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswGenerated(
                ["hnsw-generated", "--metric", "InnerProduct", "--dimension", "6", "--vectors", "16", "--queries", "2", "--top-k", "3", "--ef-search", "4"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseDurableHnswGenerated(
                ["hnsw-generated-durable", "--metric", "InnerProduct", "--dimension", "6", "--vectors", "16", "--queries", "2", "--top-k", "3", "--ef-search", "4"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswAllowlistFiltering(
                [HnswAllowlistFilteringOptions.ScenarioName, "--metric", "InnerProduct"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswAllowlistFilteringMatrix(
                [HnswAllowlistFilteringMatrixOptions.ScenarioName, "--metric", "InnerProduct"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswBasePlusExactDeltaGenerated(
                ["generated-hnsw-base-plus-exact-delta", "--metric", "InnerProduct", "--dimension", "6", "--vectors", "16", "--queries", "2", "--top-k", "3", "--insertions", "4", "--deletes", "2", "--delta-deletes", "1", "--ef-search", "4"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(
                [HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", "InnerProduct"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(
                [HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, "--metric", "InnerProduct"]).Metric);

        string outputDirectory = NewArtifactDirectory("matrices");
        Assert.Contains(
            HnswGeneratedMatrixScenario.ExpandCases(
                    new HnswGeneratedMatrixOptions(
                        "smoke",
                        VectorCount: 64,
                        QueryCount: 1,
                        Runs: 1,
                        WarmupQueries: 0,
                        Seed: 0x5EED3460u,
                        OutputDirectory: outputDirectory,
                        ManifestPath: Path.Combine(outputDirectory, "hnsw.json")))
                .Select(item => item.Options.Metric)
                .Distinct(),
            metric => metric == VectorMetric.InnerProduct);
        Assert.Contains(
            DurableHnswGeneratedMatrixScenario.ExpandCases(
                    new DurableHnswGeneratedMatrixOptions(
                        "smoke",
                        Seed: 0x5EED3460u,
                        OutputDirectory: outputDirectory,
                        ManifestPath: Path.Combine(outputDirectory, "durable.json")))
                .Select(item => item.Options.Metric)
                .Distinct(),
            metric => metric == VectorMetric.InnerProduct);
        Assert.Contains(
            HnswBasePlusExactDeltaMatrixScenario.ExpandCases(
                    new HnswBasePlusExactDeltaMatrixOptions(
                        "smoke",
                        BaseVectorCount: 64,
                        QueryCount: 1,
                        Runs: 1,
                        WarmupQueries: 0,
                        Seed: 0x5EED3460u,
                        DuplicateInsertAttempts: 0,
                        UnknownDeleteAttempts: 0,
                        RepeatedDeleteAttempts: 0,
                        OutputDirectory: outputDirectory,
                        ManifestPath: Path.Combine(outputDirectory, "base-plus-delta.json")))
                .Select(item => item.Options.Metric)
                .Distinct(),
            metric => metric == VectorMetric.InnerProduct);
    }

    [Fact]
    public void GeneratedImmutableHnswInnerProductSmokeComparesExactTruth()
    {
        string outputPath = NewArtifactPath("generated", "hnsw-generated-inner-product.json");
        string[] arguments =
        [
            "hnsw-generated",
            "--metric", "InnerProduct",
            "--dimension", "6",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3461",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003461",
            "--output", outputPath
        ];

        HnswBenchmarkReport report = HnswGeneratedScenario.Run(CommandLine.ParseHnswGenerated(arguments), arguments);
        HnswGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal("scalar-reference-generated", report.Truth.Kind);
        AssertPassedHnswMetrics(report.Metrics);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FinalRunComparedToTruth);
        Assert.False(report.Evidence.PublicClaimEligible);
    }

    [Fact]
    public void DurableOpenedInnerProductSmokeValidatesSaveOpenParity()
    {
        string directory = NewArtifactDirectory("durable");
        string outputPath = Path.Combine(directory, "durable-hnsw-inner-product.json");
        string[] arguments =
        [
            "hnsw-generated-durable",
            "--metric", "InnerProduct",
            "--dimension", "6",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3462",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003462",
            "--output", outputPath,
            "--snapshot-directory", Path.Combine(directory, "snapshot")
        ];

        DurableHnswBenchmarkReport report =
            DurableHnswGeneratedScenario.Run(CommandLine.ParseDurableHnswGenerated(arguments), arguments);
        DurableHnswGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Workload.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.SavedOpenedParity.AllResultsMatched);
        Assert.Equal("passed", report.Outputs.SnapshotOutput.ValidationOpenStatus);
        AssertPassedDurableMetrics(report.Metrics.SourceHnsw);
        AssertPassedDurableMetrics(report.Metrics.OpenedHnsw);
        Assert.False(report.Eligibility.PublicClaimEligible);
    }

    [Fact]
    public void AllowlistFilteringInnerProductSmokeValidatesFilteringAndUnderfillWiring()
    {
        string directory = NewArtifactDirectory("allowlist");
        string[] arguments =
        [
            HnswAllowlistFilteringOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "40",
            "--queries", "4",
            "--top-k", "5",
            "--insertions", "8",
            "--deletes", "4",
            "--delta-deletes", "2",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "1",
            "--filter", "broad",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3463",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003463",
            "--output", Path.Combine(directory, "allowlist-inner-product.json"),
            "--opened-index-directory", Path.Combine(directory, "opened"),
            "--checkpoint-directory", Path.Combine(directory, "checkpoint")
        ];

        HnswAllowlistFilteringOptions options = CommandLine.ParseHnswAllowlistFiltering(arguments);
        HnswAllowlistFilteringBenchmarkReport report = HnswAllowlistFilteringScenario.Run(options, arguments);
        HnswAllowlistFilteringScenario.Write(report, options.OutputPath);

        Assert.True(File.Exists(options.OutputPath));
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal("broad", report.Allowlist.Profile);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.BroadEmissionIntegrityPassedForAllSearches);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);
        Assert.Equal("measured", report.Searches.SourceComposite.ExactFilteredDeltaScan.Status);
        Assert.Equal("measuredZeroAfterCheckpoint", report.Searches.RebuiltComposite.ExactFilteredDeltaScan.Status);
        Assert.True(report.Parity.ImmutableOpenedHnsw.AllResultsMatched);
        Assert.True(report.Parity.RebuiltCompositeCheckpointOpenedHnsw.AllResultsMatched);
        AssertBroadAllowlistSection(report.Searches.ImmutableHnsw);
        AssertBroadAllowlistSection(report.Searches.SourceComposite);
        Assert.False(report.Eligibility.PublicClaimEligible);
    }

    [Fact]
    public void MutableBasePlusExactDeltaInnerProductSmokeValidatesMetricTombstoneAndDeltaWiring()
    {
        string outputPath = NewArtifactPath("mutable", "base-plus-delta-inner-product.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaGeneratedOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--insertions", "5",
            "--deletes", "3",
            "--delta-deletes", "1",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "2",
            "--repeated-deletes", "2",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3464",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003464",
            "--output", outputPath
        ];

        HnswBasePlusExactDeltaBenchmarkReport report =
            HnswBasePlusExactDeltaGeneratedScenario.Run(
                CommandLine.ParseHnswBasePlusExactDeltaGenerated(arguments),
                arguments);
        HnswBasePlusExactDeltaGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LiveTruthGenerated);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(2, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(2, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);
        Assert.Equal(report.Underfill.TotalRequestedResultSlots - report.Underfill.TotalReturnedResults, report.Underfill.UnderfilledSlotCount);
        Assert.False(report.Eligibility.PublicClaimEligible);
    }

    [Fact]
    public void CheckpointInnerProductSmokeValidatesCheckpointTombstoneAndOpenedParityWiring()
    {
        string directory = NewArtifactDirectory("checkpoint");
        string outputPath = Path.Combine(directory, "checkpoint-inner-product.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--dimension", "7",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--insertions", "6",
            "--deletes", "4",
            "--delta-deletes", "1",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "1",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3465",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003465",
            "--output", outputPath,
            "--checkpoint-directory", Path.Combine(directory, "checkpoint-output")
        ];

        HnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            HnswBasePlusExactDeltaCheckpointScenario.Run(
                CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(arguments),
                arguments);
        HnswBasePlusExactDeltaCheckpointScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Index.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(0, report.PostCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(0, report.PostCheckpointCounts.TombstoneCount);
        Assert.Equal("passed", report.OpenedValidation.Status);
        Assert.Equal(0, report.OpenedValidation.VectorMismatchCount);
        Assert.True(report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched);
        AssertCheckpointSection(report.Searches.PreCheckpointComposite);
        AssertCheckpointSection(report.Searches.PostCheckpointRebuiltComposite);
        AssertCheckpointSection(report.Searches.OpenedReadOnlyHnsw);
        Assert.False(report.Eligibility.PublicClaimEligible);
    }

    private static void AssertPassedHnswMetrics(HnswMetricsInfo metrics)
    {
        Assert.InRange(metrics.RecallAtK, 0, 1);
        Assert.InRange(metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal("passed", metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DistanceMismatchCount);
    }

    private static void AssertPassedDurableMetrics(DurableHnswOperationMetricsInfo metrics)
    {
        Assert.InRange(metrics.RecallAtK, 0, 1);
        Assert.InRange(metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal("passed", metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DistanceMismatchCount);
    }

    private static void AssertBroadAllowlistSection(HnswAllowlistSearchSectionInfo section)
    {
        Assert.Equal("notApplicable", section.ExactFallbackValidation.Status);
        Assert.Equal("passed", section.BroadEmissionValidation.Status);
        Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
        Assert.Equal(section.Underfill.TotalRequestedResultSlots - section.Underfill.TotalReturnedResults, section.Underfill.UnderfilledSlotCount);
    }

    private static void AssertCheckpointSection(HnswBasePlusExactDeltaCheckpointSearchSectionInfo section)
    {
        Assert.Equal("passed", section.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, section.Metrics.DistanceMismatchCount);
        Assert.Equal("passed", section.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(section.Underfill.TotalRequestedResultSlots - section.Underfill.TotalReturnedResults, section.Underfill.UnderfilledSlotCount);
    }

    private static string NewArtifactPath(string prefix, string fileName)
    {
        string directory = NewArtifactDirectory(prefix);
        return Path.Combine(directory, fileName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec346-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
