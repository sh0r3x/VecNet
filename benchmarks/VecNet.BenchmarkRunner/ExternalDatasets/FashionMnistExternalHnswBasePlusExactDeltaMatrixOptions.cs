namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions(
    string PresetName,
    string CacheRoot,
    int QueryCount,
    int Runs,
    int WarmupQueries,
    VectorMetric Metric,
    uint Seed,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string OutputDirectory,
    string ManifestPath)
{
    public const string ScenarioName = "external-fashion-mnist-hnsw-base-plus-exact-delta-matrix";
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

        throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta matrix preset '{presetName}'.");
    }
}
