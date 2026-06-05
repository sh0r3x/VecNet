namespace VecNet.BenchmarkRunner;

public static class LatencyPercentiles
{
    public static double NearestRankMilliseconds(
        ReadOnlySpan<long> sortedSampleTicks,
        double percentile,
        long ticksPerSecond)
    {
        if (ticksPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ticksPerSecond), "Tick frequency must be positive.");
        }

        if (!double.IsFinite(percentile) || percentile < 0 || percentile > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(percentile), "Percentile must be in the inclusive range [0, 1].");
        }

        if (sortedSampleTicks.Length == 0)
        {
            return 0;
        }

        int index = (int)Math.Ceiling(sortedSampleTicks.Length * percentile) - 1;
        index = Math.Clamp(index, 0, sortedSampleTicks.Length - 1);
        return sortedSampleTicks[index] * 1000.0 / ticksPerSecond;
    }
}
