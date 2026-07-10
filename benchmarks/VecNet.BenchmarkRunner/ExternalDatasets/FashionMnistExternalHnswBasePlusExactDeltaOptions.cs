namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBasePlusExactDeltaOptions(
    string CacheRoot,
    string OutputPath,
    int QueryCount,
    int TopK,
    int BaseVectorCount,
    int InsertedDeltaCount,
    int DeletedBaseCount,
    int DeletedDeltaCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int Runs,
    int WarmupQueries,
    VectorMetric Metric,
    uint Seed,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw-base-plus-exact-delta";

    public int PhysicalCandidateVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static readonly FashionMnistExternalHnswBasePlusExactDeltaOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-base-plus-exact-delta.json"),
        QueryCount: 50,
        TopK: 100,
        BaseVectorCount: 58_000,
        InsertedDeltaCount: 1_000,
        DeletedBaseCount: 1_000,
        DeletedDeltaCount: 100,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        Runs: 1,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean,
        Seed: 0x5EED2127,
        M: 16,
        EfConstruction: 128,
        EfSearch: 192,
        HnswSeed: 0x484E535700012700UL);
}
