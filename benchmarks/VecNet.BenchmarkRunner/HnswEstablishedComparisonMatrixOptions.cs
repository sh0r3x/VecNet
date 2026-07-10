namespace VecNet.BenchmarkRunner;

public sealed record HnswEstablishedComparisonMatrixOptions(
    string PresetName,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    uint Seed,
    string OutputDirectory,
    string ManifestPath,
    string HnswlibPythonPath)
{
    public const string ScenarioName = "hnswlib-generated-comparison-matrix";
    public const string SmokePresetName = "smoke";
    public const string StandardPresetName = "standard";
    public const string DefaultPresetName = SmokePresetName;

    public static string NormalizePresetName(string presetName)
    {
        if (string.Equals(presetName, SmokePresetName, StringComparison.OrdinalIgnoreCase))
        {
            return SmokePresetName;
        }

        if (string.Equals(presetName, StandardPresetName, StringComparison.OrdinalIgnoreCase))
        {
            return StandardPresetName;
        }

        throw new ArgumentException($"Unsupported hnswlib comparison matrix preset '{presetName}'.");
    }
}
