namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions(
    string PresetName,
    string CacheRoot,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix";
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

        throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta checkpoint matrix preset '{presetName}'.");
    }
}
