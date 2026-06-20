namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCheckpointMatrixOptions(
    string PresetName,
    int Runs,
    int WarmupQueries,
    uint Seed,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int DuplicateIdsPerQuery,
    int UnknownIdsPerQuery,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "generated-exact-checkpoint-matrix";
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

        throw new ArgumentException($"Unsupported generated exact checkpoint matrix preset '{presetName}'.");
    }
}
