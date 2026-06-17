using System.Numerics;

namespace VecNet;

/// <summary>
/// An in-memory exhaustive index using canonical distance calculation.
/// </summary>
public sealed partial class ExactFlatIndex
{
    private const int InitialCapacity = 4;

    private readonly ExactFlatIndexDistanceMode _distanceMode;
    private ulong[] _ids = [];
    private float[] _vectors = [];
    private Dictionary<ulong, int> _idToOrdinal = new();
    private int _count;
    private bool _isReadOnly;

    /// <summary>
    /// Initializes a new exact flat index with a fixed dimension and metric.
    /// </summary>
    /// <param name="dimension">The required positive vector dimension.</param>
    /// <param name="metric">The canonical distance metric.</param>
    public ExactFlatIndex(int dimension, VectorMetric metric)
        : this(dimension, metric, GetDefaultDistanceMode(metric))
    {
    }

    internal ExactFlatIndex(int dimension, VectorMetric metric, ExactFlatIndexDistanceMode distanceMode)
    {
        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive.");
        }

        if (!Enum.IsDefined(metric))
        {
            throw new ArgumentOutOfRangeException(nameof(metric), "Metric is not supported.");
        }

        if (!Enum.IsDefined(distanceMode))
        {
            throw new ArgumentOutOfRangeException(nameof(distanceMode), "Distance mode is not supported.");
        }

        if (distanceMode == ExactFlatIndexDistanceMode.VectorFloatSquaredL2 &&
            metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException(
                "The vector float distance mode is available only for squared Euclidean search.",
                nameof(distanceMode));
        }

        Dimension = dimension;
        Metric = metric;
        _distanceMode = distanceMode;
    }

    /// <summary>
    /// Gets the fixed dimension accepted by this index.
    /// </summary>
    public int Dimension { get; }

    /// <summary>
    /// Gets the canonical distance metric used by this index.
    /// </summary>
    public VectorMetric Metric { get; }

    /// <summary>
    /// Gets the current number of vectors stored by this index.
    /// </summary>
    public int VectorCount => _count;

    /// <summary>
    /// Inserts a vector associated with a caller-provided external identifier.
    /// </summary>
    /// <param name="id">The opaque external vector identifier.</param>
    /// <param name="vector">
    /// The vector values to copy into index storage. Cosine vectors are normalized during insertion.
    /// </param>
    public void Add(ulong id, ReadOnlySpan<float> vector)
    {
        if (_isReadOnly)
        {
            throw new InvalidOperationException("This exact flat index was opened read-only and cannot be modified.");
        }

        double magnitude = ValidateVector(vector, nameof(vector));

        if (_idToOrdinal.ContainsKey(id))
        {
            throw new ArgumentException("An item with the same identifier already exists.", nameof(id));
        }

        EnsureCapacity(_count + 1);

        int offset = _count * Dimension;
        if (Metric == VectorMetric.Cosine)
        {
            StoreNormalizedVector(vector, magnitude, offset);
        }
        else
        {
            vector.CopyTo(_vectors.AsSpan(offset, Dimension));
        }

        _ids[_count] = id;
        _idToOrdinal.Add(id, _count);
        _count++;
    }

    /// <summary>
    /// Searches all inserted vectors and writes the nearest results in ascending distance order.
    /// </summary>
    /// <param name="query">The query vector. Cosine queries are normalized during search.</param>
    /// <param name="results">
    /// The caller-owned destination buffer. Its length specifies the requested result count.
    /// </param>
    /// <returns>The number of results written.</returns>
    public int Search(ReadOnlySpan<float> query, Span<SearchResult> results)
    {
        double queryMagnitude = ValidateVector(query, nameof(query));

        if (_count == 0 || results.IsEmpty)
        {
            return 0;
        }

        int written = 0;
        for (int row = 0; row < _count; row++)
        {
            var candidate = new SearchResult(_ids[row], CalculateDistance(row, query, queryMagnitude));
            written = InsertCandidate(results, written, candidate);
        }

        return written;
    }

    /// <summary>
    /// Searches only vectors whose external identifiers are present in the caller-supplied allowlist.
    /// </summary>
    /// <param name="query">The query vector. Cosine queries are normalized during search.</param>
    /// <param name="allowedIds">
    /// Caller-supplied external identifiers allowed for this search. Unknown identifiers are ignored and duplicates
    /// are coalesced.
    /// </param>
    /// <param name="results">
    /// The caller-owned destination buffer. Its length specifies the requested result count.
    /// </param>
    /// <param name="workspace">The caller-owned reusable exact-flat filter workspace.</param>
    /// <returns>The number of filtered results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        ReadOnlySpan<ulong> allowedIds,
        Span<SearchResult> results,
        ExactFlatSearchFilterWorkspace workspace)
    {
        double queryMagnitude = ValidateVector(query, nameof(query));
        ArgumentNullException.ThrowIfNull(workspace);
        if (workspace.MaxVectorCount < _count)
        {
            throw new ArgumentException(
                "Filter workspace capacity must be at least the current vector count.",
                nameof(workspace));
        }

        if (_count == 0 || allowedIds.IsEmpty || results.IsEmpty)
        {
            return 0;
        }

        int searchMark = workspace.BeginSearch();
        int[] rowMarks = workspace.RowMarks;
        for (int allowIndex = 0; allowIndex < allowedIds.Length; allowIndex++)
        {
            if (_idToOrdinal.TryGetValue(allowedIds[allowIndex], out int row))
            {
                rowMarks[row] = searchMark;
            }
        }

        int written = 0;
        for (int row = 0; row < _count; row++)
        {
            if (rowMarks[row] != searchMark)
            {
                continue;
            }

            var candidate = new SearchResult(_ids[row], CalculateDistance(row, query, queryMagnitude));
            written = InsertCandidate(results, written, candidate);
        }

        return written;
    }

    private double ValidateVector(ReadOnlySpan<float> vector, string parameterName)
    {
        if (vector.Length != Dimension)
        {
            throw new ArgumentException($"Vector dimension must be {Dimension}.", parameterName);
        }

        double squaredMagnitude = 0;
        foreach (float component in vector)
        {
            if (!float.IsFinite(component))
            {
                throw new ArgumentException("Vector components must be finite.", parameterName);
            }

            if (Metric == VectorMetric.Cosine)
            {
                squaredMagnitude += (double)component * component;
            }
        }

        if (Metric == VectorMetric.Cosine && squaredMagnitude == 0)
        {
            throw new ArgumentException("Cosine distance does not accept a zero vector.", parameterName);
        }

        return Metric == VectorMetric.Cosine ? Math.Sqrt(squaredMagnitude) : 0;
    }

    private void EnsureCapacity(int requiredCount)
    {
        if (_ids.Length >= requiredCount)
        {
            return;
        }

        int newCapacity = _ids.Length == 0 ? InitialCapacity : checked(_ids.Length * 2);
        if (newCapacity < requiredCount)
        {
            newCapacity = requiredCount;
        }

        var newIds = new ulong[newCapacity];
        var newVectors = new float[checked(newCapacity * Dimension)];
        _ids.AsSpan(0, _count).CopyTo(newIds);
        _vectors.AsSpan(0, _count * Dimension).CopyTo(newVectors);

        _ids = newIds;
        _vectors = newVectors;
    }

    private void StoreNormalizedVector(ReadOnlySpan<float> vector, double magnitude, int offset)
    {
        for (int i = 0; i < Dimension; i++)
        {
            _vectors[offset + i] = (float)(vector[i] / magnitude);
        }
    }

    private float CalculateDistance(int row, ReadOnlySpan<float> query, double queryMagnitude)
    {
        int offset = row * Dimension;
        return Metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclideanDistance(query, offset),
            VectorMetric.InnerProduct => InnerProductDistance(query, offset),
            VectorMetric.Cosine => CosineDistance(query, queryMagnitude, offset),
            _ => throw new InvalidOperationException("Index metric is not supported.")
        };
    }

    private float SquaredEuclideanDistance(ReadOnlySpan<float> query, int offset)
    {
        if (_distanceMode == ExactFlatIndexDistanceMode.VectorFloatSquaredL2)
        {
            return VectorFloatSquaredEuclideanDistance(query, offset);
        }

        double sum = 0;
        for (int i = 0; i < Dimension; i++)
        {
            double difference = query[i] - _vectors[offset + i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private float VectorFloatSquaredEuclideanDistance(ReadOnlySpan<float> query, int offset)
    {
        Vector<float> vectorSum = Vector<float>.Zero;
        int vectorWidth = Vector<float>.Count;
        int i = 0;

        for (; i <= Dimension - vectorWidth; i += vectorWidth)
        {
            var difference =
                new Vector<float>(query.Slice(i, vectorWidth)) -
                new Vector<float>(_vectors.AsSpan(offset + i, vectorWidth));
            vectorSum += difference * difference;
        }

        float sum = 0;
        for (int lane = 0; lane < vectorWidth; lane++)
        {
            sum += vectorSum[lane];
        }

        for (; i < Dimension; i++)
        {
            float difference = query[i] - _vectors[offset + i];
            sum += difference * difference;
        }

        return sum;
    }

    private static ExactFlatIndexDistanceMode GetDefaultDistanceMode(VectorMetric metric) =>
        metric == VectorMetric.SquaredEuclidean
            ? ExactFlatIndexDistanceMode.VectorFloatSquaredL2
            : ExactFlatIndexDistanceMode.ScalarDouble;

    internal static ExactFlatIndex HydrateReadOnly(
        int dimension,
        VectorMetric metric,
        ulong[] ids,
        float[] vectors)
    {
        var index = new ExactFlatIndex(dimension, metric)
        {
            _ids = ids,
            _vectors = vectors,
            _idToOrdinal = BuildIdToOrdinalMap(ids),
            _count = ids.Length,
            _isReadOnly = true
        };

        return index;
    }

    private static Dictionary<ulong, int> BuildIdToOrdinalMap(ReadOnlySpan<ulong> ids)
    {
        var idToOrdinal = new Dictionary<ulong, int>(ids.Length);
        for (int row = 0; row < ids.Length; row++)
        {
            idToOrdinal.Add(ids[row], row);
        }

        return idToOrdinal;
    }

    private float InnerProductDistance(ReadOnlySpan<float> query, int offset)
    {
        double dotProduct = 0;
        for (int i = 0; i < Dimension; i++)
        {
            dotProduct += (double)query[i] * _vectors[offset + i];
        }

        return (float)-dotProduct;
    }

    private float CosineDistance(ReadOnlySpan<float> query, double queryMagnitude, int offset)
    {
        double dotProduct = 0;
        for (int i = 0; i < Dimension; i++)
        {
            dotProduct += _vectors[offset + i] * (query[i] / queryMagnitude);
        }

        return (float)(1 - dotProduct);
    }

    private static int FindInsertionIndex(ReadOnlySpan<SearchResult> results, SearchResult candidate)
    {
        for (int i = 0; i < results.Length; i++)
        {
            SearchResult current = results[i];
            if (candidate.Distance < current.Distance ||
                (candidate.Distance == current.Distance && candidate.Id < current.Id))
            {
                return i;
            }
        }

        return results.Length;
    }

    private static int InsertCandidate(Span<SearchResult> results, int written, SearchResult candidate)
    {
        int insertionIndex = FindInsertionIndex(results[..written], candidate);
        if (insertionIndex >= results.Length)
        {
            return written;
        }

        int valuesToShift = Math.Min(written, results.Length - 1) - insertionIndex;
        if (valuesToShift > 0)
        {
            results.Slice(insertionIndex, valuesToShift)
                .CopyTo(results.Slice(insertionIndex + 1));
        }

        results[insertionIndex] = candidate;
        return written < results.Length ? written + 1 : written;
    }
}
