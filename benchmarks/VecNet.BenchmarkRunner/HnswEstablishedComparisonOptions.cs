namespace VecNet.BenchmarkRunner;

public sealed record HnswEstablishedComparisonOptions(
    VectorMetric Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    uint Seed,
    string OutputPath,
    string WorkDirectory,
    string VecNetSnapshotDirectory,
    string HnswlibIndexPath,
    string HnswlibPythonPath,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int EfSearch,
    ulong HnswSeed)
{
    public const string ScenarioName = "hnswlib-generated-comparison";
    public const string HnswlibPackageName = "hnswlib";
    public const string HnswlibPackageSource = "PyPI";
    public const string HnswlibVersion = "0.8.0";
    public const string HnswlibSourceDistributionSha256 = "cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c";
    public const string HnswlibLicense = "Apache-2.0";

    public static int[] RepresentativeDimensions { get; } = [128, 384, 768];

    public static int[] OptionalAdversarialDimensions { get; } = [386];

    public static HnswEstablishedComparisonOptions Default { get; } = new(
        VectorMetric.SquaredEuclidean,
        Dimension: 128,
        VectorCount: 4096,
        QueryCount: 100,
        TopK: 10,
        Seed: 0x5EED2118,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnswlib-generated-comparison.json"),
        WorkDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnswlib-generated-comparison-work"),
        VecNetSnapshotDirectory: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnswlib-generated-comparison-vecnet-snapshot"),
        HnswlibIndexPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnswlib-generated-comparison-hnswlib.bin"),
        HnswlibPythonPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec-118-tools", "hnswlib-venv", "Scripts", "python.exe"),
        Runs: 1,
        WarmupQueries: 3,
        M: 8,
        EfConstruction: 64,
        EfSearch: 128,
        HnswSeed: 0x484E535700011818UL);
}
