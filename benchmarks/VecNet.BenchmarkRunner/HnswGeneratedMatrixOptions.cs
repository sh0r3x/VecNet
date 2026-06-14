namespace VecNet.BenchmarkRunner;

public sealed record HnswGeneratedMatrixOptions(
    string PresetName,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    uint Seed,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "hnsw-generated-matrix";
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

        throw new ArgumentException($"Unsupported HNSW matrix preset '{presetName}'.");
    }
}
