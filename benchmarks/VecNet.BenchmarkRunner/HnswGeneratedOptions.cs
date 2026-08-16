namespace VecNet.BenchmarkRunner;

public sealed record HnswGeneratedOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed,
    GeneratedVectorProfile VectorProfile = GeneratedVectorProfile.Uniform)
{
    public const string ScenarioName = "hnsw-generated";

    public static HnswGeneratedOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        VectorCount: 10_000,
        QueryCount: 100,
        TopK: 10,
        Seed: 0x5EED2036,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnsw-generated.json"),
        Runs: 1,
        WarmupQueries: 0,
        M: HnswIndexOptions.Default.M,
        EfConstruction: HnswIndexOptions.Default.EfConstruction,
        EfSearch: HnswIndexOptions.Default.EfSearch,
        HnswSeed: HnswIndexOptions.Default.RandomSeed);
}
