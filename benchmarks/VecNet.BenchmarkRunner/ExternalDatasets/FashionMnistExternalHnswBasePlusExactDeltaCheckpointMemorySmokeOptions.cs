namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions(
    string CacheRoot,
    string OutputPath,
    string CheckpointDirectory,
    int SampleIntervalMilliseconds,
    int QueryCount,
    int TopK,
    int BaseVectorCount,
    int InsertedDeltaCount,
    int DeletedBaseCount,
    int DeletedDeltaCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int WarmupQueries,
    VectorMetric Metric,
    uint Seed,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke";

    public const int CheckpointRunCount = 1;

    public int PhysicalCandidateVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public int DeletedReservedIdCount => checked(DeletedBaseCount + DeletedDeltaCount);

    public FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions ToCheckpointOptions() =>
        new(
            CacheRoot,
            OutputPath,
            CheckpointDirectory,
            QueryCount,
            TopK,
            BaseVectorCount,
            InsertedDeltaCount,
            DeletedBaseCount,
            DeletedDeltaCount,
            DuplicateInsertAttempts,
            UnknownDeleteAttempts,
            RepeatedDeleteAttempts,
            CheckpointRunCount,
            WarmupQueries,
            Metric,
            Seed,
            M,
            EfConstruction,
            EfSearch,
            HnswSeed);

    public static readonly FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-142-memory-smoke",
            "fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-memory-smoke.json"),
        CheckpointDirectory: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-142-memory-smoke",
            "checkpoint-output"),
        SampleIntervalMilliseconds: 10,
        QueryCount: 50,
        TopK: 100,
        BaseVectorCount: 58_000,
        InsertedDeltaCount: 1_000,
        DeletedBaseCount: 1_000,
        DeletedDeltaCount: 100,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean,
        Seed: 0x5EED2141,
        M: 16,
        EfConstruction: 128,
        EfSearch: 192,
        HnswSeed: 0x484E535700014100UL);
}
