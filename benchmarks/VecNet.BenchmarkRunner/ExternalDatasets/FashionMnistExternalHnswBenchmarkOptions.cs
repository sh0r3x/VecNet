namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBenchmarkOptions(
    string CacheRoot,
    string OutputPath,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    VectorMetric Metric,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw";

    public static readonly FashionMnistExternalHnswBenchmarkOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw.json"),
        QueryCount: 3,
        TopK: 10,
        Runs: 3,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean,
        M: 8,
        EfConstruction: 64,
        EfSearch: 100,
        HnswSeed: 0x484E535700000039UL);
}
