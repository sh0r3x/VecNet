using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class GeneratedDatasetTests
{
    [Fact]
    public void Create_IsDeterministicForSameOptions()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 5,
            VectorCount: 7,
            QueryCount: 3,
            TopK: 2,
            Seed: 0x5EED2009,
            OutputPath: "unused.json");

        GeneratedDataset first = GeneratedDatasetFactory.Create(options);
        GeneratedDataset second = GeneratedDatasetFactory.Create(options);

        Assert.Equal(first.Vectors, second.Vectors);
        Assert.Equal(first.Queries, second.Queries);
        Assert.All(first.Vectors, AssertFinite);
        Assert.All(first.Queries, AssertFinite);
    }

    private static void AssertFinite(float value) => Assert.True(float.IsFinite(value));
}
