namespace VecNet.BenchmarkRunner;

public sealed record HnswMemorySmokeOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string SnapshotDirectory,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed,
    int SampleIntervalMilliseconds)
{
    public const string ScenarioName = "generated-hnsw-memory-smoke";

    public static HnswMemorySmokeOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        VectorCount: 4096,
        QueryCount: 32,
        TopK: 10,
        Seed: 0x5EED2112,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-memory-smoke.json"),
        SnapshotDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-memory-smoke-snapshot"),
        WarmupQueries: 4,
        M: 8,
        EfConstruction: 64,
        EfSearch: 128,
        HnswSeed: 0x484E535700011212UL,
        SampleIntervalMilliseconds: 10);
}
