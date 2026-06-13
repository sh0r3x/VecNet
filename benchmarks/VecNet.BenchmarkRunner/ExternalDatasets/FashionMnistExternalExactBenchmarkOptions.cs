namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalExactBenchmarkOptions(
    string CacheRoot,
    string OutputPath,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    VectorMetric Metric)
{
    public const string ScenarioName = "external-fashion-mnist-exact";

    public static readonly FashionMnistExternalExactBenchmarkOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-exact.json"),
        QueryCount: 3,
        TopK: 10,
        Runs: 3,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean);
}
