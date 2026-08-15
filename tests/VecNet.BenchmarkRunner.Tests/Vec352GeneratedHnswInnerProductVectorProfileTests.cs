using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec352GeneratedHnswInnerProductVectorProfileTests
{
    [Fact]
    public void GeneratedHnswProfileParsersDefaultToUniformAndRejectInvalidProfiles()
    {
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseHnswGenerated([HnswGeneratedOptions.ScenarioName]).VectorProfile);
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseDurableHnswGenerated([DurableHnswGeneratedOptions.ScenarioName]).VectorProfile);
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseHnswAllowlistFiltering([HnswAllowlistFilteringOptions.ScenarioName]).VectorProfile);
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseHnswBasePlusExactDeltaGenerated([HnswBasePlusExactDeltaGeneratedOptions.ScenarioName]).VectorProfile);
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseHnswBasePlusExactDeltaCheckpoint([HnswBasePlusExactDeltaCheckpointOptions.ScenarioName]).VectorProfile);
        Assert.Equal(GeneratedVectorProfile.Uniform, CommandLine.ParseHnswMemorySmoke([HnswMemorySmokeOptions.ScenarioName]).VectorProfile);

        Assert.Equal(
            GeneratedVectorProfile.NormSkewed,
            CommandLine.ParseHnswGenerated([HnswGeneratedOptions.ScenarioName, "--vector-profile", "norm-skewed"]).VectorProfile);
        Assert.Equal(
            GeneratedVectorProfile.ZeroVector,
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(
                [HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", "InnerProduct", "--vector-profile", "zero-vector"]).VectorProfile);

        Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswGenerated([HnswGeneratedOptions.ScenarioName, "--vector-profile", "unknown"]));
        Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswGenerated(
                [HnswGeneratedOptions.ScenarioName, "--metric", "Cosine", "--vector-profile", "zero-vector"]));
    }

    [Fact]
    public void GeneratedDatasetProfilesAreDeterministicAndCarryMetadata()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.InnerProduct,
            Dimension: 4,
            VectorCount: 10,
            QueryCount: 6,
            TopK: 3,
            Seed: 0x5EED3520,
            OutputPath: "unused.json",
            BaselineReportId: null);

        GeneratedDataset first = GeneratedDatasetFactory.Create(options, GeneratedVectorProfile.NormSkewed);
        GeneratedDataset second = GeneratedDatasetFactory.Create(options, GeneratedVectorProfile.NormSkewed);
        GeneratedDataset zero = GeneratedDatasetFactory.Create(options, GeneratedVectorProfile.ZeroVector);

        Assert.Equal(first.Vectors, second.Vectors);
        Assert.Equal(first.Queries, second.Queries);
        Assert.Equal(GeneratedVectorProfile.NormSkewed, first.VectorProfile);
        Assert.Equal("generated-norm-skewed", first.DatasetKind);
        Assert.Contains("scaled", first.ProfileDistribution, StringComparison.OrdinalIgnoreCase);
        Assert.All(first.Vectors, AssertFinite);
        Assert.All(first.Queries, AssertFinite);
        Assert.True(RowMagnitudeSquared(first.Vectors, options.Dimension, row: 6) > RowMagnitudeSquared(first.Vectors, options.Dimension, row: 0));

        Assert.Equal(GeneratedVectorProfile.ZeroVector, zero.VectorProfile);
        Assert.Equal("generated-zero-vector", zero.DatasetKind);
        Assert.Contains("all-zero", zero.ProfileDistribution, StringComparison.OrdinalIgnoreCase);
        Assert.True(IsZeroRow(zero.Vectors, options.Dimension, row: 0));
        Assert.True(IsZeroRow(zero.Vectors, options.Dimension, row: 5));
        Assert.True(IsZeroRow(zero.Queries, options.Dimension, row: 0));
        Assert.True(IsZeroRow(zero.Queries, options.Dimension, row: 3));
        Assert.Contains(zero.Vectors, value => value != 0f);
        Assert.Contains(zero.Queries, value => value != 0f);
        Assert.All(zero.Vectors, AssertFinite);
        Assert.All(zero.Queries, AssertFinite);
    }

    [Theory]
    [InlineData("norm-skewed", "generated-norm-skewed")]
    [InlineData("zero-vector", "generated-zero-vector")]
    public void ImmutableHnswInnerProductSmokeReportsSelectedVectorProfile(string profile, string expectedKind)
    {
        string outputPath = NewArtifactPath("immutable", $"hnsw-generated-{profile}.json");
        string[] arguments =
        [
            HnswGeneratedOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--vector-profile", profile,
            "--dimension", "6",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED3521",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000003521",
            "--output", outputPath
        ];

        HnswGeneratedOptions options = CommandLine.ParseHnswGenerated(arguments);
        HnswBenchmarkReport report = HnswGeneratedScenario.Run(options, arguments);
        HnswGeneratedScenario.Write(report, outputPath);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(expectedKind, report.Dataset.Kind);
        Assert.Contains(profile, report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement dataset = document.RootElement.GetProperty("dataset");
        Assert.Equal(expectedKind, dataset.GetProperty("kind").GetString());
        Assert.Contains(profile == "norm-skewed" ? "scaled" : "all-zero", dataset.GetProperty("distribution").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("norm-skewed", "generated-norm-skewed")]
    [InlineData("zero-vector", "generated-zero-vector")]
    public void CheckpointInnerProductSmokeReportsSelectedVectorProfile(string profile, string expectedKind)
    {
        string directory = NewArtifactDirectory("checkpoint");
        string outputPath = Path.Combine(directory, $"checkpoint-{profile}.json");
        string[] arguments =
        [
            HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "--metric", "InnerProduct",
            "--vector-profile", profile,
            "--dimension", "6",
            "--vectors", "24",
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
            "--seed", "0x5EED3522",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--workspace-ef-search", "8",
            "--hnsw-seed", "0x0000000000003522",
            "--output", outputPath,
            "--checkpoint-directory", Path.Combine(directory, "checkpoint-output")
        ];

        HnswBasePlusExactDeltaCheckpointOptions options = CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(arguments);
        HnswBasePlusExactDeltaCheckpointBenchmarkReport report = HnswBasePlusExactDeltaCheckpointScenario.Run(options, arguments);
        HnswBasePlusExactDeltaCheckpointScenario.Write(report, outputPath);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(VectorMetric.InnerProduct.ToString(), report.Dataset.Metric);
        Assert.Equal(expectedKind, report.Dataset.Kind);
        Assert.Contains(profile, report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal("passed", report.OpenedValidation.Status);
        Assert.True(report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched);
        Assert.Equal("passed", report.Searches.PreCheckpointComposite.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Searches.PostCheckpointRebuiltComposite.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Searches.OpenedReadOnlyHnsw.Metrics.ReturnedResultIntegrity.Status);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement dataset = document.RootElement.GetProperty("dataset");
        Assert.Equal(expectedKind, dataset.GetProperty("kind").GetString());
        Assert.Contains(profile == "norm-skewed" ? "scaled" : "all-zero", dataset.GetProperty("distribution").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static double RowMagnitudeSquared(float[] values, int dimension, int row)
    {
        double sum = 0;
        int offset = checked(row * dimension);
        for (int i = 0; i < dimension; i++)
        {
            double value = values[offset + i];
            sum += value * value;
        }

        return sum;
    }

    private static bool IsZeroRow(float[] values, int dimension, int row)
    {
        int offset = checked(row * dimension);
        for (int i = 0; i < dimension; i++)
        {
            if (values[offset + i] != 0f)
            {
                return false;
            }
        }

        return true;
    }

    private static void AssertFinite(float value) => Assert.True(float.IsFinite(value));

    private static string NewArtifactPath(string prefix, string fileName)
    {
        string directory = NewArtifactDirectory(prefix);
        return Path.Combine(directory, fileName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec352-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
