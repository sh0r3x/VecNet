using System.Numerics;

namespace VecNet;

internal sealed class HnswBasePlusExactDeltaIndex
{
    private const int InitialDeltaCapacity = 4;

    private readonly HnswIndex _baseIndex;
    private readonly int _basePhysicalVectorCount;
    private readonly HashSet<ulong> _baseIds = [];
    private readonly HashSet<ulong> _baseTombstoneIds = [];
    private readonly HashSet<ulong> _deltaTombstoneIds = [];
    private readonly HashSet<ulong> _deletedReservedIds = [];
    private readonly Dictionary<ulong, int> _deltaIdToOrdinal = [];
    private readonly bool _isReadOnly;

    private ulong[] _deltaIds = [];
    private float[] _deltaVectors = [];
    private int _deltaPhysicalVectorCount;
    private long _generation;

    internal HnswBasePlusExactDeltaIndex(HnswIndex baseIndex, bool isReadOnly = false)
    {
        ArgumentNullException.ThrowIfNull(baseIndex);
        if (baseIndex.Metric != VectorMetric.SquaredEuclidean)
        {
            throw new NotSupportedException("HNSW base-plus-exact-delta search currently supports only squared Euclidean distance.");
        }

        _baseIndex = baseIndex;
        _basePhysicalVectorCount = baseIndex.Count;
        foreach (ulong id in baseIndex.InternalIds)
        {
            _baseIds.Add(id);
        }

        Dimension = baseIndex.Dimension;
        Metric = baseIndex.Metric;
        Options = baseIndex.Options;
        _isReadOnly = isReadOnly;
    }

    internal int Dimension { get; }

    internal VectorMetric Metric { get; }

    internal HnswIndexOptions Options { get; }

    internal int BasePhysicalVectorCount => _basePhysicalVectorCount;

    internal int BaseLiveVectorCount => _basePhysicalVectorCount - _baseTombstoneIds.Count;

    internal int DeltaPhysicalVectorCount => _deltaPhysicalVectorCount;

    internal int DeltaLiveVectorCount => _deltaPhysicalVectorCount - _deltaTombstoneIds.Count;

    internal int TombstoneCount => BaseTombstoneCount + DeltaTombstoneCount;

    internal int BaseTombstoneCount => _baseTombstoneIds.Count;

    internal int DeltaTombstoneCount => _deltaTombstoneIds.Count;

    internal int LiveVectorCount => BaseLiveVectorCount + DeltaLiveVectorCount;

    internal int DeletedReservedIdCount => _deletedReservedIds.Count;

    internal long Generation => _generation;

    internal VectorMutationResult TryAdd(ulong id, ReadOnlySpan<float> vector)
    {
        if (_isReadOnly)
        {
            return CreateMutationResult(VectorMutationStatus.ReadOnly);
        }

        ValidateVector(vector, nameof(vector));
        if (IsKnownOrReserved(id))
        {
            return CreateMutationResult(VectorMutationStatus.DuplicateId);
        }

        EnsureDeltaCapacity(checked(_deltaPhysicalVectorCount + 1));
        int offset = _deltaPhysicalVectorCount * Dimension;
        vector.CopyTo(_deltaVectors.AsSpan(offset, Dimension));
        _deltaIds[_deltaPhysicalVectorCount] = id;
        _deltaIdToOrdinal.Add(id, _deltaPhysicalVectorCount);
        _deltaPhysicalVectorCount++;
        _generation++;

        return CreateMutationResult(VectorMutationStatus.Committed);
    }

    internal VectorMutationResult TryDelete(ulong id)
    {
        if (_isReadOnly)
        {
            return CreateMutationResult(VectorMutationStatus.ReadOnly);
        }

        if (_deletedReservedIds.Contains(id))
        {
            return CreateMutationResult(VectorMutationStatus.AlreadyDeleted);
        }

        if (_baseIds.Contains(id))
        {
            _baseTombstoneIds.Add(id);
            _deletedReservedIds.Add(id);
            _generation++;
            return CreateMutationResult(VectorMutationStatus.Committed);
        }

        if (_deltaIdToOrdinal.ContainsKey(id))
        {
            _deltaTombstoneIds.Add(id);
            _deletedReservedIds.Add(id);
            _generation++;
            return CreateMutationResult(VectorMutationStatus.Committed);
        }

        return CreateMutationResult(VectorMutationStatus.UnknownId);
    }

    internal int Search(
        ReadOnlySpan<float> query,
        Span<SearchResult> results,
        HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ValidateBaseUnchanged();
        ValidateVector(query, nameof(query));
        ValidateWorkspace(results.Length, workspace);

        if (results.IsEmpty || LiveVectorCount == 0)
        {
            return 0;
        }

        if (Options.EfSearch < results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(results), "EfSearch must be at least the requested result count.");
        }

        int baseCandidateCount = SearchBaseCandidates(query, workspace);
        int deltaCandidateCount = SearchDeltaCandidates(query, results.Length, workspace);
        return MergeCandidates(
            workspace.BaseCandidates.AsSpan(0, baseCandidateCount),
            workspace.DeltaCandidates.AsSpan(0, deltaCandidateCount),
            results);
    }

    private int SearchBaseCandidates(ReadOnlySpan<float> query, HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        if (_basePhysicalVectorCount == 0 || BaseLiveVectorCount == 0)
        {
            return 0;
        }

        int requestedBaseCandidates = Math.Min(_basePhysicalVectorCount, Options.EfSearch);
        Span<SearchResult> baseCandidates = workspace.BaseCandidates.AsSpan(0, requestedBaseCandidates);
        int rawCount = _baseIndex.Search(query, baseCandidates, workspace.HnswWorkspace);

        int liveCount = 0;
        for (int i = 0; i < rawCount; i++)
        {
            SearchResult candidate = baseCandidates[i];
            if (_baseTombstoneIds.Contains(candidate.Id))
            {
                continue;
            }

            baseCandidates[liveCount++] = candidate;
        }

        return liveCount;
    }

    private int SearchDeltaCandidates(
        ReadOnlySpan<float> query,
        int requestedResultCount,
        HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        if (requestedResultCount == 0 || DeltaLiveVectorCount == 0)
        {
            return 0;
        }

        Span<SearchResult> deltaCandidates = workspace.DeltaCandidates.AsSpan(0, requestedResultCount);
        int written = 0;
        for (int ordinal = 0; ordinal < _deltaPhysicalVectorCount; ordinal++)
        {
            ulong id = _deltaIds[ordinal];
            if (_deltaTombstoneIds.Contains(id))
            {
                continue;
            }

            var candidate = new SearchResult(id, SquaredEuclideanDistance(query, ordinal));
            written = InsertCandidate(deltaCandidates, written, candidate);
        }

        return written;
    }

    private static int MergeCandidates(
        ReadOnlySpan<SearchResult> baseCandidates,
        ReadOnlySpan<SearchResult> deltaCandidates,
        Span<SearchResult> results)
    {
        int baseIndex = 0;
        int deltaIndex = 0;
        int written = 0;

        while (written < results.Length &&
               (baseIndex < baseCandidates.Length || deltaIndex < deltaCandidates.Length))
        {
            bool takeBase;
            if (baseIndex >= baseCandidates.Length)
            {
                takeBase = false;
            }
            else if (deltaIndex >= deltaCandidates.Length)
            {
                takeBase = true;
            }
            else
            {
                takeBase = Compare(baseCandidates[baseIndex], deltaCandidates[deltaIndex]) <= 0;
            }

            results[written++] = takeBase ? baseCandidates[baseIndex++] : deltaCandidates[deltaIndex++];
        }

        return written;
    }

    private void ValidateWorkspace(int requestedResultCount, HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        int requestedBaseCandidates = Math.Min(_basePhysicalVectorCount, Options.EfSearch);
        if (workspace.HnswWorkspace.MaxElements < _basePhysicalVectorCount)
        {
            throw new ArgumentException("Workspace base element capacity is smaller than the immutable HNSW base count.", nameof(workspace));
        }

        if (workspace.HnswWorkspace.MaxEf < Options.EfSearch)
        {
            throw new ArgumentException("Workspace HNSW ef capacity is smaller than EfSearch.", nameof(workspace));
        }

        if (workspace.BaseCandidates.Length < requestedBaseCandidates)
        {
            throw new ArgumentException("Workspace base candidate capacity is smaller than the requested HNSW base overfetch count.", nameof(workspace));
        }

        if (workspace.DeltaCandidates.Length < requestedResultCount)
        {
            throw new ArgumentException("Workspace delta candidate capacity is smaller than the requested result count.", nameof(workspace));
        }
    }

    private void ValidateBaseUnchanged()
    {
        if (_baseIndex.Count != _basePhysicalVectorCount)
        {
            throw new InvalidOperationException("The HNSW base changed after composite construction.");
        }
    }

    private void ValidateVector(ReadOnlySpan<float> vector, string parameterName)
    {
        if (vector.Length != Dimension)
        {
            throw new ArgumentException($"Vector dimension must be {Dimension}.", parameterName);
        }

        foreach (float component in vector)
        {
            if (!float.IsFinite(component))
            {
                throw new ArgumentException("Vector components must be finite.", parameterName);
            }
        }
    }

    private bool IsKnownOrReserved(ulong id) =>
        _baseIds.Contains(id) || _deltaIdToOrdinal.ContainsKey(id) || _deletedReservedIds.Contains(id);

    private VectorMutationResult CreateMutationResult(VectorMutationStatus status) =>
        new(status, _generation, LiveVectorCount, DeltaLiveVectorCount, TombstoneCount);

    private void EnsureDeltaCapacity(int requiredCount)
    {
        if (_deltaIds.Length >= requiredCount)
        {
            return;
        }

        int newCapacity = _deltaIds.Length == 0 ? InitialDeltaCapacity : checked(_deltaIds.Length * 2);
        if (newCapacity < requiredCount)
        {
            newCapacity = requiredCount;
        }

        var ids = new ulong[newCapacity];
        var vectors = new float[checked(newCapacity * Dimension)];
        _deltaIds.AsSpan(0, _deltaPhysicalVectorCount).CopyTo(ids);
        _deltaVectors.AsSpan(0, _deltaPhysicalVectorCount * Dimension).CopyTo(vectors);
        _deltaIds = ids;
        _deltaVectors = vectors;
    }

    private float SquaredEuclideanDistance(ReadOnlySpan<float> query, int deltaOrdinal)
    {
        int offset = deltaOrdinal * Dimension;
        Vector<float> vectorSum = Vector<float>.Zero;
        int vectorWidth = Vector<float>.Count;
        int i = 0;

        for (; i <= Dimension - vectorWidth; i += vectorWidth)
        {
            var difference =
                new Vector<float>(query.Slice(i, vectorWidth)) -
                new Vector<float>(_deltaVectors.AsSpan(offset + i, vectorWidth));
            vectorSum += difference * difference;
        }

        float sum = 0;
        for (int lane = 0; lane < vectorWidth; lane++)
        {
            sum += vectorSum[lane];
        }

        for (; i < Dimension; i++)
        {
            float difference = query[i] - _deltaVectors[offset + i];
            sum += difference * difference;
        }

        return sum;
    }

    private static int InsertCandidate(Span<SearchResult> results, int written, SearchResult candidate)
    {
        int insertionIndex = written;
        for (int i = 0; i < written; i++)
        {
            if (Compare(candidate, results[i]) < 0)
            {
                insertionIndex = i;
                break;
            }
        }

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

    private static int Compare(SearchResult left, SearchResult right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        if (distanceComparison != 0)
        {
            return distanceComparison;
        }

        return left.Id.CompareTo(right.Id);
    }
}
