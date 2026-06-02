using System.Numerics;
using System.Globalization;
using BenchmarkDotNet.Attributes;

namespace VecNet.Benchmarks;

[MemoryDiagnoser]
public class SquaredEuclideanSimdKernelBenchmarks
{
    private float[] _left = null!;
    private float[] _right = null!;

    [Params(32, 128, 384, 768)]
    public int Dimension { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(0x5EED004);

        _left = new float[Dimension];
        _right = new float[Dimension];

        FillVector(random, _left);
        FillVector(random, _right);

        float scalarDouble = ScalarDoubleAccumulation(_left, _right);
        float scalarFloat = ScalarFloatAccumulation(_left, _right);
        float vectorFloat = VectorFloatSquaredL2(_left, _right);

        float scalarFloatDelta = scalarFloat - scalarDouble;
        float vectorFloatDelta = vectorFloat - scalarDouble;

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"VEC-004 setup: Dimension={Dimension}, VectorWidth={Vector<float>.Count}, " +
            $"ScalarDouble={scalarDouble:R}, " +
            $"ScalarFloat={scalarFloat:R}, ScalarFloatDelta={scalarFloatDelta:R}, ScalarFloatAbsDelta={MathF.Abs(scalarFloatDelta):R}, " +
            $"VectorFloat={vectorFloat:R}, VectorFloatDelta={vectorFloatDelta:R}, VectorFloatAbsDelta={MathF.Abs(vectorFloatDelta):R}"));
    }

    [Benchmark(Baseline = true)]
    public float ScalarDoubleAccumulation()
    {
        return ScalarDoubleAccumulation(_left, _right);
    }

    [Benchmark]
    public float ScalarFloatAccumulation()
    {
        return ScalarFloatAccumulation(_left, _right);
    }

    [Benchmark]
    public float VectorFloatSquaredL2()
    {
        return VectorFloatSquaredL2(_left, _right);
    }

    private static float ScalarDoubleAccumulation(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        double sum = 0;

        for (int i = 0; i < left.Length; i++)
        {
            float difference = left[i] - right[i];
            sum += (double)difference * difference;
        }

        return (float)sum;
    }

    private static float ScalarFloatAccumulation(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        float sum = 0;

        for (int i = 0; i < left.Length; i++)
        {
            float difference = left[i] - right[i];
            sum += difference * difference;
        }

        return sum;
    }

    private static float VectorFloatSquaredL2(float[] left, float[] right)
    {
        Vector<float> vectorSum = Vector<float>.Zero;
        int vectorWidth = Vector<float>.Count;
        int i = 0;

        for (; i <= left.Length - vectorWidth; i += vectorWidth)
        {
            var difference = new Vector<float>(left, i) - new Vector<float>(right, i);
            vectorSum += difference * difference;
        }

        float sum = 0;
        for (int lane = 0; lane < vectorWidth; lane++)
        {
            sum += vectorSum[lane];
        }

        for (; i < left.Length; i++)
        {
            float difference = left[i] - right[i];
            sum += difference * difference;
        }

        return sum;
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }
}
