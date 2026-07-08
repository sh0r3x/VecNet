namespace VecNet.BenchmarkRunner;

public sealed record HnswAllowlistFilteringMatrixOptions(
    string PresetName,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    uint Seed,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "generated-hnsw-allowlist-filtered-matrix";
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

        throw new ArgumentException($"Unsupported generated HNSW allowlist filtering matrix preset '{presetName}'.");
    }
}
