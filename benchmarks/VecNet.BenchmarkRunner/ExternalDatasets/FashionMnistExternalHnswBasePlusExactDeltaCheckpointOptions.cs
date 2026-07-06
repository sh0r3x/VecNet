namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
    string CacheRoot,
    string OutputPath,
    string CheckpointDirectory,
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
    public const string ScenarioName = "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint";

    public int PhysicalCandidateVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static readonly FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-138-smoke",
            "fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint.json"),
        CheckpointDirectory: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-138-smoke",
            "checkpoint-output"),
        QueryCount: 50,
        TopK: 100,
        BaseVectorCount: 58_000,
        InsertedDeltaCount: 1_000,
        DeletedBaseCount: 1_000,
        DeletedDeltaCount: 100,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        Runs: 2,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean,
        Seed: 0x5EED2137,
        M: 16,
        EfConstruction: 128,
        EfSearch: 192,
        HnswSeed: 0x484E535700013700UL);
}
