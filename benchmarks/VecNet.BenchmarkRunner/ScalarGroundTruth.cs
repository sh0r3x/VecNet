namespace VecNet.BenchmarkRunner;

public sealed record TruthItem(ulong Id, float Distance);

public sealed class TruthSet
{
    public TruthSet(TruthItem[][] results, int depth)
    {
        Results = results;
        Depth = depth;
    }

    public TruthItem[][] Results { get; }

    public int Depth { get; }
}

public static class ScalarGroundTruth
{
    public const string Kind = "scalar-reference-generated";
    public const string TiePolicy = "ascending scalar-reference canonical distance, then ascending external id";

    public static TruthSet Generate(GeneratedDataset dataset, VectorMetric metric, int depth)
    {
        if (depth <= 0 || depth > dataset.VectorCount)
        {
            throw new ArgumentOutOfRangeException(nameof(depth), "Truth depth must be positive and no larger than vector count.");
        }

        var results = new TruthItem[dataset.QueryCount][];
        double[]? vectorMagnitudes = metric == VectorMetric.Cosine
            ? CalculateVectorMagnitudes(dataset)
            : null;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            double queryMagnitude = metric == VectorMetric.Cosine ? CalculateMagnitude(query) : 0;
            var candidates = new TruthItem[dataset.VectorCount];

            for (int vectorRow = 0; vectorRow < dataset.VectorCount; vectorRow++)
            {
                ReadOnlySpan<float> vector = dataset.GetVector(vectorRow);
                float distance = CalculateDistance(
                    query,
                    vector,
                    metric,
                    queryMagnitude,
                    vectorMagnitudes is null ? 0 : vectorMagnitudes[vectorRow]);
                candidates[vectorRow] = new TruthItem((ulong)vectorRow, distance);
            }

            Array.Sort(candidates, CompareTruthItems);
            var top = new TruthItem[depth];
            Array.Copy(candidates, top, depth);
            results[queryRow] = top;
        }

        return new TruthSet(results, depth);
    }

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static double[] CalculateVectorMagnitudes(GeneratedDataset dataset)
    {
        var magnitudes = new double[dataset.VectorCount];
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            magnitudes[row] = CalculateMagnitude(dataset.GetVector(row));
        }

        return magnitudes;
    }

    private static double CalculateMagnitude(ReadOnlySpan<float> values)
    {
        double squaredMagnitude = 0;
        foreach (float value in values)
        {
            squaredMagnitude += (double)value * value;
        }

        return Math.Sqrt(squaredMagnitude);
    }

    private static float CalculateDistance(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        VectorMetric metric,
        double queryMagnitude,
        double vectorMagnitude) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
            VectorMetric.InnerProduct => InnerProduct(query, vector),
            VectorMetric.Cosine => Cosine(query, vector, queryMagnitude, vectorMagnitude),
            _ => throw new ArgumentOutOfRangeException(nameof(metric), "Metric is not supported.")
        };

    internal static float CalculateDistance(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        VectorMetric metric)
    {
        double queryMagnitude = metric == VectorMetric.Cosine ? CalculateMagnitude(query) : 0;
        double vectorMagnitude = metric == VectorMetric.Cosine ? CalculateMagnitude(vector) : 0;
        return CalculateDistance(query, vector, metric, queryMagnitude, vectorMagnitude);
    }

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static float InnerProduct(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (double)query[i] * vector[i];
        }

        return (float)-dotProduct;
    }

    private static float Cosine(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        double queryMagnitude,
        double vectorMagnitude)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (query[i] / queryMagnitude) * (vector[i] / vectorMagnitude);
        }

        return (float)(1 - dotProduct);
    }
}
