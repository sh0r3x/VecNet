using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class ScalarGroundTruthTests
{
    [Fact]
    public void Generate_UsesScalarCanonicalDistanceAndExternalIdTieOrder()
    {
        var dataset = new GeneratedDataset(
            dimension: 2,
            vectorCount: 3,
            queryCount: 1,
            seed: 123,
            vectors:
            [
                0f, 0f,
                1f, 0f,
                0f, 2f
            ],
            queries:
            [
                0f, 1f
            ]);

        TruthSet truth = ScalarGroundTruth.Generate(dataset, VectorMetric.SquaredEuclidean, depth: 3);

        Assert.Equal(3, truth.Depth);
        Assert.Equal(
            [
                new TruthItem(0, 1f),
                new TruthItem(2, 1f),
                new TruthItem(1, 2f)
            ],
            truth.Results[0]);
    }
}
