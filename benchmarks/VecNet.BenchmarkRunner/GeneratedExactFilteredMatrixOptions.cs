namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactFilteredMatrixOptions(
    string PresetName,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    uint Seed,
    int DuplicateIdsPerQuery,
    int UnknownIdsPerQuery,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "exact-generated-filtered-matrix";
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

        throw new ArgumentException($"Unsupported generated exact-filter matrix preset '{presetName}'.");
    }
}
