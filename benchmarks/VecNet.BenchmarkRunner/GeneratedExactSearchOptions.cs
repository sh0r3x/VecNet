namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactSearchOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string? BaselineReportId,
    int Runs = 1,
    int WarmupQueries = 0)
{
    public const string ScenarioName = "exact-generated";
}
