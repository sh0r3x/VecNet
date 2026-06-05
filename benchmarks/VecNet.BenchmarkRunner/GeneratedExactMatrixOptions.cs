namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactMatrixOptions(
    string PresetName,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    uint Seed,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "exact-generated-matrix";
    public const string DefaultPresetName = "smoke";
    public const int MaxTopK = 10;
}
