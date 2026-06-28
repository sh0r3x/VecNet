namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactOpenedSearchOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string IndexDirectory,
    int Runs = 1,
    int WarmupQueries = 0)
{
    public const string ScenarioName = "generated-exact-opened-search";
}
