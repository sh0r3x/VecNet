namespace VecNet.BenchmarkRunner;

public sealed record HnswAllowlistFilteringOptions(
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
    string FilterProfile,
    string OutputPath,
    string OpenedIndexDirectory,
    string CheckpointDirectory,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "generated-hnsw-allowlist-filtered";
    public const string DefaultFilterProfile = "fallback-boundary";

    public int PhysicalVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static HnswAllowlistFilteringOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 32,
        BaseVectorCount: 512,
        QueryCount: 8,
        TopK: 10,
        Seed: 0x5EED2148,
        InsertedDeltaCount: 64,
        DeletedBaseCount: 32,
        DeletedDeltaCount: 8,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        FilterProfile: DefaultFilterProfile,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-allowlist-filtered.json"),
        OpenedIndexDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-allowlist-filtered-opened"),
        CheckpointDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "generated-hnsw-allowlist-filtered-checkpoint"),
        Runs: 1,
        WarmupQueries: 1,
        M: 8,
        EfConstruction: 64,
        EfSearch: 64,
        HnswSeed: 0x484E535700014800);

    public static string NormalizeFilterProfile(string value)
    {
        string normalized = value.Trim().ToLowerInvariant();
        return normalized switch
        {
            "empty" or "very-selective" or "fallback-boundary" or "broad" or "all" => normalized,
            _ => throw new ArgumentException($"Unsupported generated HNSW allowlist filter profile '{value}'.")
        };
    }
}
