namespace VecNet.BenchmarkRunner;

public sealed record ResultComparison(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount);

public static class ResultComparer
{
    public static ResultComparison Compare(
        TruthSet truth,
        SearchResult[][] actual,
        int topK,
        int dimension,
        VectorMetric metric)
    {
        if (truth.Results.Length != actual.Length)
        {
            throw new ArgumentException("Truth and actual result query counts differ.", nameof(actual));
        }

        int totalDenominator = 0;
        int setMatches = 0;
        int orderedMatches = 0;
        int distanceMismatches = 0;
        int missingResults = 0;

        for (int queryRow = 0; queryRow < truth.Results.Length; queryRow++)
        {
            TruthItem[] expected = truth.Results[queryRow];
            SearchResult[] returned = actual[queryRow];
            int denominator = Math.Min(topK, expected.Length);
            totalDenominator += denominator;

            var returnedIds = new HashSet<ulong>();
            for (int i = 0; i < Math.Min(denominator, returned.Length); i++)
            {
                returnedIds.Add(returned[i].Id);
            }

            for (int i = 0; i < denominator; i++)
            {
                if (returnedIds.Contains(expected[i].Id))
                {
                    setMatches++;
                }

                if (i >= returned.Length)
                {
                    missingResults++;
                    continue;
                }

                if (returned[i].Id == expected[i].Id)
                {
                    orderedMatches++;
                    if (!DistanceMatches(expected[i].Distance, returned[i].Distance, dimension, metric))
                    {
                        distanceMismatches++;
                    }
                }
            }
        }

        return new ResultComparison(
            totalDenominator == 0 ? 1 : (double)setMatches / totalDenominator,
            totalDenominator == 0 ? 1 : (double)orderedMatches / totalDenominator,
            distanceMismatches == 0 && missingResults == 0 ? "passed" : "failed",
            distanceMismatches,
            missingResults);
    }

    internal static bool DistanceMatches(float expected, float actual, int dimension, VectorMetric metric)
    {
        if (!float.IsFinite(actual) || float.IsNaN(expected))
        {
            return false;
        }

        float tolerance = CalculateDistanceTolerance(dimension, metric, expected);
        return MathF.Abs(expected - actual) <= tolerance;
    }

    internal static float CalculateDistanceTolerance(int dimension, VectorMetric metric, float scalarReference) =>
        metric == VectorMetric.SquaredEuclidean
            ? CalculateD026Tolerance(dimension, scalarReference)
            : 1e-5f * MathF.Max(1f, MathF.Abs(scalarReference));

    private static float CalculateD026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }
}
