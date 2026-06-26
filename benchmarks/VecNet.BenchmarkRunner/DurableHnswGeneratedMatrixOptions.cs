namespace VecNet.BenchmarkRunner;

public sealed record DurableHnswGeneratedMatrixOptions(
    string PresetName,
    uint Seed,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "hnsw-generated-durable-matrix";
    public const string SmokePresetName = "smoke";
    public const string DefaultPresetName = SmokePresetName;

    public static string NormalizePresetName(string presetName)
    {
        if (string.Equals(presetName, SmokePresetName, StringComparison.OrdinalIgnoreCase))
        {
            return SmokePresetName;
        }

        throw new ArgumentException($"Unsupported durable HNSW matrix preset '{presetName}'.");
    }
}
