namespace VecNet.BenchmarkRunner;

public sealed record BenchmarkComparisonOptions(
    string BaselinePath,
    string CurrentPath,
    string OutputPath)
{
    public const string ScenarioName = "compare-generated-exact";
}
