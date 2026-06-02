using BenchmarkDotNet.Attributes;

namespace VecNet.Benchmarks;

[MemoryDiagnoser]
public class ExactFlatIndexBenchmarks
{
    private const int ResultCount = 10;

    private ExactFlatIndex _scalarIndex = null!;
    private ExactFlatIndex _vectorIndex = null!;
    private float[] _query = null!;
    private SearchResult[] _scalarResults = null!;
    private SearchResult[] _vectorResults = null!;

    [Params(32, 128, 384, 386)]
    public int Dimension { get; set; }

    [Params(1024)]
    public int VectorCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(0x5EED);
        var vector = new float[Dimension];

        _scalarIndex = new ExactFlatIndex(Dimension, VectorMetric.SquaredEuclidean);
        _vectorIndex = new ExactFlatIndex(
            Dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.VectorFloatSquaredL2);

        for (int row = 0; row < VectorCount; row++)
        {
            FillVector(random, vector);
            _scalarIndex.Add((ulong)row, vector);
            _vectorIndex.Add((ulong)row, vector);
        }

        _query = new float[Dimension];
        FillVector(random, _query);
        _scalarResults = new SearchResult[ResultCount];
        _vectorResults = new SearchResult[ResultCount];
    }

    [Benchmark(Baseline = true)]
    public int ScalarSearchTop10()
    {
        return _scalarIndex.Search(_query, _scalarResults);
    }

    [Benchmark]
    public int VectorFloatSquaredL2SearchTop10()
    {
        return _vectorIndex.Search(_query, _vectorResults);
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }
}
