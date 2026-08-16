namespace VecNet.BenchmarkRunner;

public sealed record DurableHnswGeneratedOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string SnapshotDirectory,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed,
    GeneratedVectorProfile VectorProfile = GeneratedVectorProfile.Uniform)
{
    public const string ScenarioName = "hnsw-generated-durable";

    public static DurableHnswGeneratedOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        VectorCount: 1024,
        QueryCount: 25,
        TopK: 10,
        Seed: 0x5EED2073,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnsw-generated-durable.json"),
        SnapshotDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnsw-generated-durable-snapshot"),
        Runs: 1,
        WarmupQueries: 0,
        M: HnswIndexOptions.Default.M,
        EfConstruction: HnswIndexOptions.Default.EfConstruction,
        EfSearch: HnswIndexOptions.Default.EfSearch,
        HnswSeed: HnswIndexOptions.Default.RandomSeed);
}
