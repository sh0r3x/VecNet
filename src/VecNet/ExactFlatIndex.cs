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
    private HashSet<ulong> _visibilityTombstoneIds = [];
    private HashSet<ulong> _deletedReservedIds = [];
    private int _count;
    private int _baseRowCount = int.MaxValue;
    private long _generation;
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
    /// Gets the physical stored-row count used to size exact-flat row workspaces.
    /// </summary>
    /// <remarks>
    /// This count includes live visible rows and rows hidden by tombstones. Use
    /// <see cref="LiveVectorCount"/> when caller-visible search cardinality is required.
    /// </remarks>
    public int VectorCount => PhysicalVectorCount;

    /// <summary>
    /// Gets the physical stored-row count used to size exact-flat row workspaces.
    /// </summary>
    /// <remarks>
    /// This count includes live visible rows and rows hidden by tombstones. Raw allowlist
    /// searches require an <see cref="ExactFlatSearchFilterWorkspace"/> whose
    /// <see cref="ExactFlatSearchFilterWorkspace.MaxVectorCount"/> is at least this value.
    /// Checkpoint publication compacts this value to the current live view.
    /// </remarks>
    public int PhysicalVectorCount => _count;

    /// <summary>
    /// Gets the current number of live visible vectors returned by search.
    /// </summary>
    public int LiveVectorCount => _count - _visibilityTombstoneIds.Count;

    /// <summary>
    /// Gets the current live base vector count.
    /// </summary>
    public int BaseVectorCount
    {
        get
        {
            int baseLimit = _baseRowCount > _count ? _count : _baseRowCount;
            int baseCount = 0;
            for (int row = 0; row < baseLimit; row++)
            {
                if (!IsDeleted(row))
                {
                    baseCount++;
                }
            }

            return baseCount;
        }
    }

    /// <summary>
    /// Gets the current live in-memory delta vector count.
    /// </summary>
    public int DeltaVectorCount
    {
        get
        {
            if (_baseRowCount >= _count)
            {
                return 0;
            }

            int deltaCount = 0;
            for (int row = _baseRowCount; row < _count; row++)
            {
                if (!IsDeleted(row))
                {
                    deltaCount++;
                }
            }

            return deltaCount;
        }
    }

    /// <summary>
    /// Gets the current visibility tombstone count.
    /// </summary>
    public int TombstoneCount => _visibilityTombstoneIds.Count;

    /// <summary>
    /// Gets the current deleted or otherwise reserved external identifier count.
    /// </summary>
    public int DeletedReservedIdCount => _deletedReservedIds.Count;

    /// <summary>
    /// Gets the current opaque instance-local generation stamp.
    /// </summary>
    /// <remarks>
    /// The generation is for detecting stale reusable exact-flat candidate sets created by this
    /// process and index instance. It is not a durable version, timestamp, ordering token across
    /// index instances, or cross-process coordination mechanism.
    /// </remarks>
    public long Generation => _generation;

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

        if (IsKnownOrReserved(id))
        {
            throw new ArgumentException("An item with the same identifier already exists.", nameof(id));
        }

        AddValidated(id, vector, magnitude);
        _generation++;
    }

    /// <summary>
    /// Attempts to insert a vector associated with a caller-provided external identifier.
    /// </summary>
    /// <param name="id">The opaque external vector identifier.</param>
    /// <param name="vector">
    /// The vector values to copy into index storage. Cosine vectors are normalized during insertion.
    /// </param>
    /// <returns>
    /// A mutation result describing the broad preview mutation status and current live/count
    /// inspection values. The status values are not a complete stable exception taxonomy.
    /// </returns>
    public VectorMutationResult TryAdd(ulong id, ReadOnlySpan<float> vector)
    {
        if (_isReadOnly)
        {
            return CreateMutationResult(VectorMutationStatus.ReadOnly);
        }

        double magnitude = ValidateVector(vector, nameof(vector));
        if (IsKnownOrReserved(id))
        {
            return CreateMutationResult(VectorMutationStatus.DuplicateId);
        }

        EnsureDeltaBoundary();
        AddValidated(id, vector, magnitude);
        _generation++;
        return CreateMutationResult(VectorMutationStatus.Committed);
    }

    /// <summary>
    /// Attempts to delete a visible vector by external identifier.
    /// </summary>
    /// <param name="id">The opaque external vector identifier to delete.</param>
    /// <returns>
    /// A mutation result describing the broad preview mutation status and current live/count
    /// inspection values. The status values are not a complete stable exception taxonomy.
    /// </returns>
    public VectorMutationResult TryDelete(ulong id)
    {
        if (_isReadOnly)
        {
            return CreateMutationResult(VectorMutationStatus.ReadOnly);
        }

        if (_deletedReservedIds.Contains(id))
        {
            return CreateMutationResult(VectorMutationStatus.AlreadyDeleted);
        }

        if (!_idToOrdinal.ContainsKey(id))
        {
            return CreateMutationResult(VectorMutationStatus.UnknownId);
        }

        EnsureDeltaBoundary();
        _visibilityTombstoneIds.Add(id);
        _deletedReservedIds.Add(id);
        _generation++;
        return CreateMutationResult(VectorMutationStatus.Committed);
    }

    private void AddValidated(ulong id, ReadOnlySpan<float> vector, double magnitude)
    {
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
            if (IsDeleted(row))
            {
                continue;
            }

            var candidate = new SearchResult(_ids[row], CalculateDistance(row, query, queryMagnitude));
            written = SelectCandidate(results, written, candidate);
        }

        SortSelectedResults(results, written);
        return written;
    }

    /// <summary>
    /// Creates a reusable opaque candidate set from caller-supplied external identifiers.
    /// </summary>
    /// <param name="allowedIds">
    /// Caller-supplied external identifiers to compile. Unknown identifiers are ignored and duplicates are coalesced.
    /// </param>
    /// <returns>
    /// An opaque candidate set bound to this index instance and its current generation. It must be
    /// rebuilt after a committed mutation or checkpoint publication.
    /// </returns>
    public ExactFlatCandidateSet CreateCandidateSet(ReadOnlySpan<ulong> allowedIds)
    {
        if (_count == 0 || allowedIds.IsEmpty)
        {
            return new ExactFlatCandidateSet(this, _generation, []);
        }

        var rowOrdinals = new int[allowedIds.Length];
        int matched = 0;
        for (int allowIndex = 0; allowIndex < allowedIds.Length; allowIndex++)
        {
            if (_idToOrdinal.TryGetValue(allowedIds[allowIndex], out int row) && !IsDeleted(row))
            {
                rowOrdinals[matched++] = row;
            }
        }

        if (matched == 0)
        {
            return new ExactFlatCandidateSet(this, _generation, []);
        }

        Array.Sort(rowOrdinals, 0, matched);

        int uniqueCount = 1;
        for (int i = 1; i < matched; i++)
        {
            if (rowOrdinals[i] != rowOrdinals[uniqueCount - 1])
            {
                rowOrdinals[uniqueCount++] = rowOrdinals[i];
            }
        }

        if (uniqueCount != rowOrdinals.Length)
        {
            Array.Resize(ref rowOrdinals, uniqueCount);
        }

        return new ExactFlatCandidateSet(this, _generation, rowOrdinals);
    }

    /// <summary>
    /// Searches only vectors present in a reusable exact-flat candidate set.
    /// </summary>
    /// <param name="query">The query vector. Cosine queries are normalized during search.</param>
    /// <param name="candidates">
    /// The candidate set created by this exact-flat index for its current generation.
    /// Candidate sets created by another index or an older generation are rejected before results
    /// are written.
    /// </param>
    /// <param name="results">
    /// The caller-owned destination buffer. Its length specifies the requested result count.
    /// </param>
    /// <returns>The number of candidate-set filtered results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        ExactFlatCandidateSet candidates,
        Span<SearchResult> results)
    {
        double queryMagnitude = ValidateVector(query, nameof(query));
        ValidateCandidateSet(candidates);

        if (_count == 0 || candidates.Count == 0 || results.IsEmpty)
        {
            return 0;
        }

        int written = 0;
        ReadOnlySpan<int> rowOrdinals = candidates.RowOrdinals;
        for (int i = 0; i < rowOrdinals.Length; i++)
        {
            int row = rowOrdinals[i];
            if (IsDeleted(row))
            {
                continue;
            }

            var candidate = new SearchResult(_ids[row], CalculateDistance(row, query, queryMagnitude));
            written = SelectCandidate(results, written, candidate);
        }

        SortSelectedResults(results, written);
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
    /// <param name="workspace">
    /// The caller-owned reusable exact-flat filter workspace. It is temporary caller storage and
    /// may be reused by the caller after the method returns. Its capacity must be at least
    /// <see cref="PhysicalVectorCount"/>, not merely <see cref="LiveVectorCount"/>.
    /// </param>
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
                if (IsDeleted(row))
                {
                    continue;
                }

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

            if (IsDeleted(row))
            {
                continue;
            }

            var candidate = new SearchResult(_ids[row], CalculateDistance(row, query, queryMagnitude));
            written = SelectCandidate(results, written, candidate);
        }

        SortSelectedResults(results, written);
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

    private void ValidateCandidateSet(ExactFlatCandidateSet candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);

        if (!ReferenceEquals(candidates.Owner, this))
        {
            throw new InvalidOperationException(
                "Candidate set was created by a different exact flat index.");
        }

        if (candidates.Generation != _generation)
        {
            throw new InvalidOperationException(
                "Candidate set was created for an older exact flat index generation and must be rebuilt.");
        }
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

    private void EnsureDeltaBoundary()
    {
        if (_baseRowCount > _count)
        {
            _baseRowCount = _count;
        }
    }

    private bool IsKnownOrReserved(ulong id) =>
        _idToOrdinal.ContainsKey(id) || _deletedReservedIds.Contains(id);

    private bool IsDeleted(int row) =>
        _visibilityTombstoneIds.Count != 0 && _visibilityTombstoneIds.Contains(_ids[row]);

    private VectorMutationResult CreateMutationResult(VectorMutationStatus status) =>
        new(status, _generation, LiveVectorCount, DeltaVectorCount, TombstoneCount);

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
            _baseRowCount = ids.Length,
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

    private static int SelectCandidate(Span<SearchResult> results, int written, SearchResult candidate)
    {
        if (results.Length < 10)
        {
            return InsertCandidate(results, written, candidate);
        }

        if (written < results.Length)
        {
            results[written] = candidate;
            SiftUpWorstFirst(results, written);
            return written + 1;
        }

        if (CompareSearchResult(candidate, results[0]) >= 0)
        {
            return written;
        }

        results[0] = candidate;
        SiftDownWorstFirst(results[..written], 0);
        return written;
    }

    private static void SortSelectedResults(Span<SearchResult> results, int written)
    {
        if (written <= 1 || results.Length < 10)
        {
            return;
        }

        Span<SearchResult> heap = results[..written];
        for (int end = heap.Length - 1; end > 0; end--)
        {
            (heap[0], heap[end]) = (heap[end], heap[0]);
            SiftDownWorstFirst(heap[..end], 0);
        }
    }

    private static void SiftUpWorstFirst(Span<SearchResult> heap, int index)
    {
        while (index > 0)
        {
            int parent = (index - 1) / 2;
            if (CompareSearchResult(heap[parent], heap[index]) >= 0)
            {
                return;
            }

            (heap[parent], heap[index]) = (heap[index], heap[parent]);
            index = parent;
        }
    }

    private static void SiftDownWorstFirst(Span<SearchResult> heap, int index)
    {
        while (true)
        {
            int left = (index * 2) + 1;
            if (left >= heap.Length)
            {
                return;
            }

            int right = left + 1;
            int worse = right < heap.Length && CompareSearchResult(heap[right], heap[left]) > 0
                ? right
                : left;

            if (CompareSearchResult(heap[index], heap[worse]) >= 0)
            {
                return;
            }

            (heap[index], heap[worse]) = (heap[worse], heap[index]);
            index = worse;
        }
    }

    private static int CompareSearchResult(SearchResult left, SearchResult right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0
            ? distanceComparison
            : left.Id.CompareTo(right.Id);
    }
}
