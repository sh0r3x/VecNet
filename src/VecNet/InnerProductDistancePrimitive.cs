using System.Numerics;

namespace VecNet;

internal static class InnerProductDistancePrimitive
{
    internal static float Distance(ReadOnlySpan<float> left, ReadOnlySpan<float> right) =>
        (float)-Dot(left, right);

    internal static double Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
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
}
