namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaGeneratedOptions(
    VectorMetric Metric,
    int Dimension,
    int BaseVectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    int InsertedDeltaCount,
    int DeletedBaseCount,
    int DeletedDeltaCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string OutputPath,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "generated-hnsw-base-plus-exact-delta";

    public int PhysicalVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static HnswBasePlusExactDeltaGeneratedOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        BaseVectorCount: 10_000,
        QueryCount: 100,
        TopK: 10,
        Seed: 0x5EED2124,
        InsertedDeltaCount: 1_000,
        DeletedBaseCount: 1_000,
        DeletedDeltaCount: 0,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-base-plus-exact-delta.json"),
        Runs: 1,
        WarmupQueries: 0,
        M: HnswIndexOptions.Default.M,
        EfConstruction: HnswIndexOptions.Default.EfConstruction,
        EfSearch: HnswIndexOptions.Default.EfSearch,
        HnswSeed: HnswIndexOptions.Default.RandomSeed);
}
