namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistHnswlibComparisonOptions(
    string CacheRoot,
    string OutputPath,
    string WorkDirectory,
    string VecNetSnapshotDirectory,
    string HnswlibIndexPath,
    string HnswlibPythonPath,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong Seed)
{
    public const string ScenarioName = "external-fashion-mnist-hnswlib-comparison";

    public static readonly FashionMnistHnswlibComparisonOptions Default = new(
        CacheRoot: "VecNet.DatasetCache",
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison.json"),
        WorkDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-work"),
        VecNetSnapshotDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-vecnet-snapshot"),
        HnswlibIndexPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-hnswlib.bin"),
        HnswlibPythonPath: HnswEstablishedComparisonOptions.Default.HnswlibPythonPath,
        QueryCount: 50,
        TopK: 10,
        Runs: 1,
        WarmupQueries: 3,
        M: 8,
        EfConstruction: 64,
        EfSearch: 100,
        Seed: 0x484E535700012000UL);
}
