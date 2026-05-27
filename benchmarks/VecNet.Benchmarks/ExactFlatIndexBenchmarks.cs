using BenchmarkDotNet.Attributes;

namespace VecNet.Benchmarks;

[MemoryDiagnoser]
public class ExactFlatIndexBenchmarks
{
    private const int ResultCount = 10;

    private ExactFlatIndex _index = null!;
    private float[] _query = null!;
    private SearchResult[] _results = null!;

    [Params(32, 128, 384)]
    public int Dimension { get; set; }

    [Params(1024)]
    public int VectorCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(0x5EED);
        var vector = new float[Dimension];

        _index = new ExactFlatIndex(Dimension, VectorMetric.SquaredEuclidean);
        for (int row = 0; row < VectorCount; row++)
        {
            FillVector(random, vector);
            _index.Add((ulong)row, vector);
        }

        _query = new float[Dimension];
        FillVector(random, _query);
        _results = new SearchResult[ResultCount];
    }

    [Benchmark]
    public int SearchTop10()
    {
        return _index.Search(_query, _results);
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }
}
