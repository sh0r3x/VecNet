namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExactTruth
{
    public const string SquaredEuclideanKind = "vecnet-scalar-reference-squared-l2";
    public const string CosineKind = "vecnet-scalar-reference-cosine";
    public const string SquaredEuclideanTiePolicy = "ascending scalar-reference squared distance, then ascending base ordinal";
    public const string CosineTiePolicy = "ascending scalar-reference canonical cosine distance, then ascending base ordinal";

    public static string Kind(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclideanKind,
            VectorMetric.Cosine => CosineKind,
            _ => throw new ArgumentException("Fashion-MNIST truth supports only SquaredEuclidean and Cosine.", nameof(metric))
        };

    public static string TiePolicy(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclideanTiePolicy,
            VectorMetric.Cosine => CosineTiePolicy,
            _ => throw new ArgumentException("Fashion-MNIST truth supports only SquaredEuclidean and Cosine.", nameof(metric))
        };

    public static string DistanceSemantics(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => "VecNet canonical squared distances for the external Euclidean ranking convention",
            VectorMetric.Cosine => "VecNet canonical cosine distances: 1 - dot(normalizedQuery, normalizedBase)",
            _ => throw new ArgumentException("Fashion-MNIST truth supports only SquaredEuclidean and Cosine.", nameof(metric))
        };

    public static TruthSet Generate(
        ReadOnlySpan<float> baseVectors,
        int baseCount,
        ReadOnlySpan<float> queryVectors,
        int queryCount,
        int dimension,
        int querySubsetCount,
        int depth,
        VectorMetric metric = VectorMetric.SquaredEuclidean)
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

        if (metric is not (VectorMetric.SquaredEuclidean or VectorMetric.Cosine))
        {
            throw new ArgumentException("Fashion-MNIST truth supports only SquaredEuclidean and Cosine.", nameof(metric));
        }

        if (metric == VectorMetric.Cosine)
        {
            ValidateNonZeroRows(baseVectors, baseCount, dimension, "base");
            ValidateNonZeroRows(queryVectors, querySubsetCount, dimension, "query");
        }

        var results = new TruthItem[querySubsetCount][];
        for (int queryRow = 0; queryRow < querySubsetCount; queryRow++)
        {
            ReadOnlySpan<float> query = queryVectors.Slice(queryRow * dimension, dimension);
            var candidates = new TruthItem[baseCount];
            for (int baseRow = 0; baseRow < baseCount; baseRow++)
            {
                ReadOnlySpan<float> vector = baseVectors.Slice(baseRow * dimension, dimension);
                candidates[baseRow] = new TruthItem((ulong)baseRow, ScalarGroundTruth.CalculateDistance(query, vector, metric));
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
        string converterIdentity,
        VectorMetric metric = VectorMetric.SquaredEuclidean,
        string taskId = "VEC-023")
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
            taskId,
            baseCount,
            querySubsetCount,
            dimension,
            metric.ToString(),
            truth.Depth,
            TiePolicy(metric),
            rawSha256,
            converterIdentity,
            queries);
    }

    private static int Compare(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    internal static void ValidateNonZeroRows(ReadOnlySpan<float> vectors, int rowCount, int dimension, string role)
    {
        for (int row = 0; row < rowCount; row++)
        {
            ReadOnlySpan<float> vector = vectors.Slice(row * dimension, dimension);
            double squaredMagnitude = 0;
            for (int column = 0; column < vector.Length; column++)
            {
                squaredMagnitude += (double)vector[column] * vector[column];
            }

            if (squaredMagnitude == 0)
            {
                throw new InvalidDataException($"Fashion-MNIST cosine evidence requires nonzero {role} rows; {role} row {row} is zero.");
            }
        }
    }
}
