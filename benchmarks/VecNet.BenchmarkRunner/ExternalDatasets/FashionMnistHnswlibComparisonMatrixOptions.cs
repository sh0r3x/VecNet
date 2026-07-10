namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistHnswlibComparisonMatrixOptions(
    string PresetName,
    string CacheRoot,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    ulong Seed,
    string OutputDirectory,
    string ManifestPath,
    string HnswlibPythonPath)
{
    public const string ScenarioName = "external-fashion-mnist-hnswlib-comparison-matrix";
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

        throw new ArgumentException($"Unsupported Fashion-MNIST hnswlib comparison matrix preset '{presetName}'.");
    }
}
