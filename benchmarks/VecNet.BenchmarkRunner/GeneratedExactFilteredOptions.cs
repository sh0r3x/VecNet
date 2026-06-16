namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactFilteredOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string FilterKind,
    int DuplicateIdsPerQuery,
    int UnknownIdsPerQuery,
    string OutputPath,
    int Runs = 1,
    int WarmupQueries = 0)
{
    public const string ScenarioName = "exact-generated-filtered";
    public const string DefaultFilterKind = "broad";

    public static string NormalizeFilterKind(string value)
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

        throw new ArgumentException($"Unsupported generated exact-filter kind '{value}'.");
    }
}
