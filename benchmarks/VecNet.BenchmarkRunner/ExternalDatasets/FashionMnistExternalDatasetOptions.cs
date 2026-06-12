namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistExternalDatasetOptions(
    string CacheRoot,
    int QueryCount,
    int TruthDepth,
    bool DownloadRawFiles)
{
    public const string ScenarioName = "external-fashion-mnist";

    public static readonly FashionMnistExternalDatasetOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        QueryCount: 100,
        TruthDepth: 10,
        DownloadRawFiles: false);
}
