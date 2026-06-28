namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactMemorySmokeOptions(
    VectorMetric Metric,
    int Dimension,
    int BaseVectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    int InsertedDeltaCount,
    int DeletedBaseCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string AllowlistKind,
    string CandidateSetKind,
    int DuplicateIdsPerQuery,
    int UnknownIdsPerQuery,
    string OutputPath,
    string SaveDirectory,
    string CheckpointDirectory,
    int WarmupQueries = 0)
{
    public const string ScenarioName = "generated-exact-memory-smoke";
    public const string DefaultAllowlistKind = "broad";
    public const string DefaultCandidateSetKind = "selective";

    public int PhysicalVectorCount => checked(BaseVectorCount + InsertedDeltaCount);

    public int LiveVectorCount => checked(BaseVectorCount + InsertedDeltaCount - DeletedBaseCount);

    public static string NormalizeSelectivityKind(string value)
    {
        if (string.Equals(value, "all", StringComparison.OrdinalIgnoreCase))
        {
            return "all";
        }

        if (string.Equals(value, "broad", StringComparison.OrdinalIgnoreCase))
        {
            return "broad";
        }

        if (string.Equals(value, "selective", StringComparison.OrdinalIgnoreCase))
        {
            return "selective";
        }

        if (string.Equals(value, "very-selective", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "verySelective", StringComparison.OrdinalIgnoreCase))
        {
            return "very-selective";
        }

        if (string.Equals(value, "empty", StringComparison.OrdinalIgnoreCase))
        {
            return "empty";
        }

        throw new ArgumentException($"Unsupported generated exact memory smoke selectivity kind '{value}'.");
    }
}
