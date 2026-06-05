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

        throw new ArgumentException($"Unsupported matrix preset '{presetName}'.");
    }
}
