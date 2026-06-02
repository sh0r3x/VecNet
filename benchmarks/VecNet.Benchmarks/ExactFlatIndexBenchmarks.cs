using BenchmarkDotNet.Attributes;

namespace VecNet.Benchmarks;

[MemoryDiagnoser]
public class ExactFlatIndexBenchmarks
{
    private ExactFlatIndex _scalarIndex = null!;
    private ExactFlatIndex _publicDefaultIndex = null!;
    private float[] _query = null!;
    private SearchResult[] _scalarResults = null!;
    private SearchResult[] _publicDefaultResults = null!;

    [Params(32, 96, 128, 384, 386, 768)]
    public int Dimension { get; set; }

    [Params(1024, 10000)]
    public int VectorCount { get; set; }

    [Params(1, 10, 100)]
    public int ResultCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(0x5EED);
        var vector = new float[Dimension];

        _scalarIndex = new ExactFlatIndex(
            Dimension,
            VectorMetric.SquaredEuclidean,
            ExactFlatIndexDistanceMode.ScalarDouble);
        _publicDefaultIndex = new ExactFlatIndex(Dimension, VectorMetric.SquaredEuclidean);

        for (int row = 0; row < VectorCount; row++)
        {
            FillVector(random, vector);
            _scalarIndex.Add((ulong)row, vector);
            _publicDefaultIndex.Add((ulong)row, vector);
        }

        _query = new float[Dimension];
        FillVector(random, _query);
        _scalarResults = new SearchResult[ResultCount];
        _publicDefaultResults = new SearchResult[ResultCount];
    }

    [Benchmark(Baseline = true)]
    public int ScalarReferenceSearch()
    {
        return _scalarIndex.Search(_query, _scalarResults);
    }

    [Benchmark]
    public int PublicDefaultSearch()
    {
        return _publicDefaultIndex.Search(_query, _publicDefaultResults);
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }
}
