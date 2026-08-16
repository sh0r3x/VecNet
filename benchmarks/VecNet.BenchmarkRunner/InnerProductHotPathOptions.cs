namespace VecNet.BenchmarkRunner;

public sealed record InnerProductHotPathOptions(
    int[] Dimensions,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupIterations,
    uint Seed,
    string OperationShape,
    int EfConstruction,
    int EfSearch,
    string OutputPath)
{
    public const string ScenarioName = "inner-product-hot-path";
    public const string AllOperationShapes = "all";
    public const string ExactFlatSearchShape = "exact-flat-search";
    public const string HnswBuildDistanceCallsShape = "hnsw-build-distance-calls";
    public const string HnswSearchDistanceCallsShape = "hnsw-search-distance-calls";

    public static readonly int[] DefaultDimensions = [31, 33, 127, 128, 129, 384, 386, 768, 769, 1536];

    public static InnerProductHotPathOptions Default { get; } = new(
        DefaultDimensions,
        VectorCount: 512,
        QueryCount: 16,
        Runs: 1,
        WarmupIterations: 1,
        Seed: 0x5EED2360,
        OperationShape: AllOperationShapes,
        EfConstruction: 64,
        EfSearch: 64,
        OutputPath: Path.Combine("VecNet.BenchmarkRunner.Artifacts", "inner-product-hot-path.json"));

    public static string NormalizeOperationShape(string value)
    {
        string normalized = value.Trim().ToLowerInvariant();
        return normalized switch
        {
            AllOperationShapes => AllOperationShapes,
            ExactFlatSearchShape => ExactFlatSearchShape,
            HnswBuildDistanceCallsShape => HnswBuildDistanceCallsShape,
            HnswSearchDistanceCallsShape => HnswSearchDistanceCallsShape,
            _ => throw new ArgumentException($"Unsupported operation shape '{value}'.")
        };
    }

    public static string[] ExpandOperationShapes(string operationShape) =>
        NormalizeOperationShape(operationShape) == AllOperationShapes
            ? [ExactFlatSearchShape, HnswBuildDistanceCallsShape, HnswSearchDistanceCallsShape]
            : [NormalizeOperationShape(operationShape)];
}
