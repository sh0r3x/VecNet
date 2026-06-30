namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalDurableHnswBenchmarkOptions(
    string CacheRoot,
    string OutputPath,
    string SnapshotDirectory,
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
    public const string ScenarioName = "external-fashion-mnist-hnsw-durable";

    public static readonly FashionMnistExternalDurableHnswBenchmarkOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-durable.json"),
        SnapshotDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-durable-snapshot"),
        QueryCount: 3,
        TopK: 10,
        Runs: 1,
        WarmupQueries: 0,
        Metric: VectorMetric.SquaredEuclidean,
        M: 8,
        EfConstruction: 64,
        EfSearch: 100,
        HnswSeed: 0x484E535700010901UL);
}
