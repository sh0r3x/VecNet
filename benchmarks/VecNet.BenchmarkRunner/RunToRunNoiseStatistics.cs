namespace VecNet.BenchmarkRunner;

public static class RunToRunNoiseStatistics
{
    public static DescriptiveStatistics Calculate(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("At least one value is required.", nameof(values));
        }

        double sum = 0;
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        bool allFinite = true;

        foreach (double value in values)
        {
            sum += value;
            min = Math.Min(min, value);
            max = Math.Max(max, value);
            allFinite &= double.IsFinite(value);
        }

        double mean = sum / values.Length;
        double? sampleStandardDeviation = null;
        if (values.Length > 1 && allFinite && double.IsFinite(mean))
        {
            double squaredDifferenceSum = 0;
            foreach (double value in values)
            {
                double difference = value - mean;
                squaredDifferenceSum += difference * difference;
            }

            sampleStandardDeviation = Math.Sqrt(squaredDifferenceSum / (values.Length - 1));
        }

        double? coefficientOfVariation =
            sampleStandardDeviation.HasValue &&
            double.IsFinite(mean) &&
            mean != 0
                ? sampleStandardDeviation.Value / Math.Abs(mean)
                : null;

        return new DescriptiveStatistics(
            values.Length,
            mean,
            sampleStandardDeviation,
            coefficientOfVariation,
            min,
            max,
            max - min);
    }
}

public sealed record DescriptiveStatistics(
    int Count,
    double Mean,
    double? SampleStandardDeviation,
    double? CoefficientOfVariation,
    double Min,
    double Max,
    double Spread);
