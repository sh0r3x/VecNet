using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class CommandLineTests
{
    [Fact]
    public void Parse_UsesPrivateArtifactRootByDefault()
    {
        GeneratedExactSearchOptions options = CommandLine.Parse([]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(0x5EED2009u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("exact-generated", "--dimension")]
    [InlineData("exact-generated", "dimension", "8")]
    [InlineData("exact-generated", "--metric", "Unknown")]
    [InlineData("exact-generated", "--dimension", "0")]
    [InlineData("exact-generated", "--vectors", "-1")]
    [InlineData("exact-generated", "--queries", "1.5")]
    [InlineData("exact-generated", "--top-k", "3", "--vectors", "2")]
    [InlineData("exact-generated", "--seed", "0xNOTHEX")]
    public void Parse_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.Parse(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("SquaredEuclidean")]
    [InlineData("InnerProduct")]
    [InlineData("Cosine")]
    public void Parse_AcceptsAllMetricsKeptEnabledByRunner(string metric)
    {
        GeneratedExactSearchOptions options = CommandLine.Parse(
            [
                "exact-generated",
                "--metric", metric,
                "--dimension", "7",
                "--vectors", "9",
                "--queries", "3",
                "--top-k", "4",
                "--seed", "0x0000002A",
                "--output", "VecNet.BenchmarkRunner.Artifacts/test.json"
            ]);

        Assert.Equal(Enum.Parse<VectorMetric>(metric), options.Metric);
        Assert.Equal(7, options.Dimension);
        Assert.Equal(9, options.VectorCount);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(4, options.TopK);
        Assert.Equal(42u, options.Seed);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/test.json", options.OutputPath);
    }
}
