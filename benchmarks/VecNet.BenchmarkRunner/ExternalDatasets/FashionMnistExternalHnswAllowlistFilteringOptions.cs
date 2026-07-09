namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswAllowlistFilteringOptions(
    string CacheRoot,
    string OutputPath,
    string OpenedIndexDirectory,
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
    string FilterProfile,
    int Runs,
    int WarmupQueries,
    VectorMetric Metric,
    uint Seed,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw-allowlist-filtered";
    public const string DefaultFilterProfile = "fallback-boundary";

    public int PhysicalCandidateVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount - DeletedDeltaCount);

    public static readonly FashionMnistExternalHnswAllowlistFilteringOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-151-smoke",
            "fashion-mnist-external-hnsw-allowlist-filtered.json"),
        OpenedIndexDirectory: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-151-smoke",
            "opened-output"),
        CheckpointDirectory: Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec-151-smoke",
            "checkpoint-output"),
        QueryCount: 50,
        TopK: 10,
        BaseVectorCount: 58_000,
        InsertedDeltaCount: 1_000,
        DeletedBaseCount: 1_000,
        DeletedDeltaCount: 100,
        DuplicateInsertAttempts: 1,
        UnknownDeleteAttempts: 1,
        RepeatedDeleteAttempts: 1,
        FilterProfile: DefaultFilterProfile,
        Runs: 1,
        WarmupQueries: 3,
        Metric: VectorMetric.SquaredEuclidean,
        Seed: 0x5EED2151,
        M: 16,
        EfConstruction: 128,
        EfSearch: 192,
        HnswSeed: 0x484E535700015100UL);

    public static string NormalizeFilterProfile(string value)
    {
        string normalized = value.Trim().ToLowerInvariant();
        return normalized switch
        {
            "fallback-boundary" or "broad" => normalized,
            _ => throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW allowlist filter profile '{value}'.")
        };
    }
}
