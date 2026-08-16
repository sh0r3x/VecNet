using System.Numerics;

namespace VecNet.BenchmarkRunner;

public static class InnerProductHotPathPrimitives
{
    public const string CurrentScalarName = "current-scalar-negative-dot";
    public const string CandidateSharedDotName = "candidate-shared-negative-dot";

    public static float CurrentScalarDistance(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        double dotProduct = 0;
        for (int i = 0; i < left.Length; i++)
        {
            dotProduct += (double)left[i] * right[i];
        }

        return (float)-dotProduct;
    }

    public static float CandidateSharedDotDistance(ReadOnlySpan<float> left, ReadOnlySpan<float> right) =>
        (float)-CandidateSharedDot(left, right);

    public static double CandidateSharedDot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        double dotProduct = 0;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            int vectorWidth = Vector<float>.Count;
            for (; i <= left.Length - vectorWidth; i += vectorWidth)
            {
                var leftVector = new Vector<float>(left.Slice(i, vectorWidth));
                var rightVector = new Vector<float>(right.Slice(i, vectorWidth));
                Vector.Widen(leftVector, out Vector<double> leftLower, out Vector<double> leftUpper);
                Vector.Widen(rightVector, out Vector<double> rightLower, out Vector<double> rightUpper);

                Vector<double> lowerProduct = leftLower * rightLower;
                Vector<double> upperProduct = leftUpper * rightUpper;
                for (int lane = 0; lane < Vector<double>.Count; lane++)
                {
                    dotProduct += lowerProduct[lane] + upperProduct[lane];
                }
            }
        }

        for (; i < left.Length; i++)
        {
            dotProduct += (double)left[i] * right[i];
        }

        return dotProduct;
    }

    public static string Category(float value)
    {
        if (float.IsNaN(value))
        {
            return "nan";
        }

        if (float.IsPositiveInfinity(value))
        {
            return "positiveInfinity";
        }

        if (float.IsNegativeInfinity(value))
        {
            return "negativeInfinity";
        }

        return "finite";
    }
}
