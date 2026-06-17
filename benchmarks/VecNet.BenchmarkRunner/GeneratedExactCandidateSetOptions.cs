namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCandidateSetOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string CandidateSetKind,
    int DuplicateIdsPerQuery,
    int UnknownIdsPerQuery,
    string OutputPath,
    int Runs = 1,
    int WarmupQueries = 0)
{
    public const string ScenarioName = "generated-exact-candidate-set";
    public const string DefaultCandidateSetKind = "broad";

    public static string NormalizeCandidateSetKind(string value)
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

        throw new ArgumentException($"Unsupported generated exact candidate-set kind '{value}'.");
    }
}
