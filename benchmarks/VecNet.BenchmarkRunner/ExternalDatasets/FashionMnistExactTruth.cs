namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExactTruth
{
    public const string Kind = "vecnet-scalar-reference-squared-l2";
    public const string TiePolicy = "ascending scalar-reference squared distance, then ascending base ordinal";

    public static TruthSet Generate(
        ReadOnlySpan<float> baseVectors,
        int baseCount,
        ReadOnlySpan<float> queryVectors,
        int queryCount,
        int dimension,
        int querySubsetCount,
        int depth)
    {
        if (baseVectors.Length != checked(baseCount * dimension))
        {
            throw new ArgumentException("Base vector payload length does not match count and dimension.", nameof(baseVectors));
        }

        if (queryVectors.Length != checked(queryCount * dimension))
        {
            throw new ArgumentException("Query vector payload length does not match count and dimension.", nameof(queryVectors));
        }

        if (querySubsetCount <= 0 || querySubsetCount > queryCount)
        {
            throw new ArgumentOutOfRangeException(nameof(querySubsetCount), "Query subset count must be positive and no larger than query count.");
        }

        if (depth <= 0 || depth > baseCount)
        {
            throw new ArgumentOutOfRangeException(nameof(depth), "Truth depth must be positive and no larger than base count.");
        }

        var results = new TruthItem[querySubsetCount][];
        for (int queryRow = 0; queryRow < querySubsetCount; queryRow++)
        {
            ReadOnlySpan<float> query = queryVectors.Slice(queryRow * dimension, dimension);
            var candidates = new TruthItem[baseCount];
            for (int baseRow = 0; baseRow < baseCount; baseRow++)
            {
                ReadOnlySpan<float> vector = baseVectors.Slice(baseRow * dimension, dimension);
                candidates[baseRow] = new TruthItem((ulong)baseRow, SquaredEuclidean(query, vector));
            }

            Array.Sort(candidates, Compare);
            var top = new TruthItem[depth];
            Array.Copy(candidates, top, depth);
            results[queryRow] = top;
        }

        return new TruthSet(results, depth);
    }

    public static ExternalExactTruthArtifact CreateArtifact(
        string datasetId,
        TruthSet truth,
        int baseCount,
        int querySubsetCount,
        int dimension,
        string[] rawSha256,
        string converterIdentity)
    {
        var queries = new ExternalTruthQuery[truth.Results.Length];
        for (int queryIndex = 0; queryIndex < truth.Results.Length; queryIndex++)
        {
            queries[queryIndex] = new ExternalTruthQuery(
                queryIndex,
                truth.Results[queryIndex]
                    .Select(item => new ExternalTruthNeighbor(item.Id, item.Distance))
                    .ToArray());
        }

        return new ExternalExactTruthArtifact(
            "VecNet.ExternalExactTruth",
            "0.1",
            datasetId,
            "VEC-023",
            baseCount,
            querySubsetCount,
            dimension,
            nameof(VectorMetric.SquaredEuclidean),
            truth.Depth,
            TiePolicy,
            rawSha256,
            converterIdentity,
            queries);
    }

    private static int Compare(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
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
}
