namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaCheckpointOptions(
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
    string CheckpointDirectory,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "generated-hnsw-base-plus-exact-delta-checkpoint";

    public int PhysicalVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static HnswBasePlusExactDeltaCheckpointOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        BaseVectorCount: 1_024,
        QueryCount: 16,
        TopK: 10,
        Seed: 0x5EED2132,
        InsertedDeltaCount: 128,
        DeletedBaseCount: 128,
        DeletedDeltaCount: 16,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-base-plus-exact-delta-checkpoint.json"),
        CheckpointDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-base-plus-exact-delta-checkpoint-output"),
        Runs: 1,
        WarmupQueries: 1,
        M: 8,
        EfConstruction: 64,
        EfSearch: 128,
        HnswSeed: 0x484E535700013200);
}
