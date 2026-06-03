namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactSearchOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string? BaselineReportId)
{
    public const string ScenarioName = "exact-generated";
}
