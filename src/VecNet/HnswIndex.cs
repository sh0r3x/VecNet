using System.Numerics;

namespace VecNet;

internal sealed class HnswIndex
{
    private const int InitialCapacity = 4;
    private const int MinM = 2;
    private const int MaxM = 64;
    private const int MaxEf = 4096;

    private readonly HnswIndexOptions _options;
    private readonly int _mMax;
    private readonly int _mMax0;
    private readonly double _levelMultiplier;
    private readonly Func<int>? _levelProvider;
    private readonly Dictionary<ulong, int> _idToOrdinal = new();

    private ulong[] _ids = [];
    private float[] _vectors = [];
    private int[] _levels = [];
    private HnswGraphLayer[] _layers = [];
    private ulong _randomState;
    private int _count;
    private int _entryPoint = -1;
    private int _maxLayer = -1;

    internal HnswIndex(int dimension, VectorMetric metric)
        : this(dimension, metric, HnswIndexOptions.Default)
    {
    }

    internal HnswIndex(int dimension, VectorMetric metric, HnswIndexOptions options, Func<int>? levelProvider = null)
    {
        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive.");
        }

        if (!Enum.IsDefined(metric))
        {
            throw new ArgumentOutOfRangeException(nameof(metric), "Metric is not supported.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new NotSupportedException("HNSW currently supports only squared Euclidean distance.");
        }

        ValidateOptions(options);

        Dimension = dimension;
        Metric = metric;
        _options = options;
        _mMax = options.M;
        _mMax0 = checked(options.M * 2);
        _levelMultiplier = 1.0 / Math.Log(options.M);
        _randomState = options.RandomSeed;
        _levelProvider = levelProvider;
    }

    internal int Dimension { get; }

    internal VectorMetric Metric { get; }

    internal int Count => _count;

    internal int EntryPoint => _entryPoint;

    internal int MaxLayer => _maxLayer;

    internal HnswIndexOptions Options => _options;

    internal void Add(ulong id, ReadOnlySpan<float> vector)
    {
        ValidateVector(vector, nameof(vector));
        if (_idToOrdinal.ContainsKey(id))
        {
            throw new ArgumentException("An item with the same identifier already exists.", nameof(id));
        }

        int level = GenerateLevel();
        int ordinal = _count;
        EnsureCapacity(checked(_count + 1), level);

        _ids[ordinal] = id;
        vector.CopyTo(_vectors.AsSpan(ordinal * Dimension, Dimension));
        _levels[ordinal] = level;

        if (_count == 0)
        {
            _entryPoint = ordinal;
            _maxLayer = level;
            _count++;
            _idToOrdinal.Add(id, ordinal);
            return;
        }

        var workspace = new HnswSearchWorkspace(_count, Math.Max(_options.EfConstruction, 1));
        int entryPoint = _entryPoint;
        int previousMaxLayer = _maxLayer;

        for (int layer = previousMaxLayer; layer > level; layer--)
        {
            int found = SearchLayer(vector, entryPoint, 1, layer, workspace, _count);
            if (found > 0)
            {
                entryPoint = workspace.ResultOrdinals[0];
            }
        }

        for (int layer = Math.Min(previousMaxLayer, level); layer >= 0; layer--)
        {
            int found = SearchLayer(vector, entryPoint, _options.EfConstruction, layer, workspace, _count);
            int[] candidates = new int[found];
            Array.Copy(workspace.ResultOrdinals, candidates, found);

            int selectedCount = SelectNeighbors(ordinal, candidates, candidates.Length, _options.M, layer, _count, out int[] selected);
            SetNeighbors(ordinal, layer, selected, selectedCount);
            for (int i = 0; i < selectedCount; i++)
            {
                AddOrPruneNeighbor(selected[i], ordinal, layer);
            }

            if (found > 0)
            {
                entryPoint = workspace.ResultOrdinals[0];
            }
        }

        if (level > previousMaxLayer)
        {
            _entryPoint = ordinal;
            _maxLayer = level;
        }

        _count++;
        _idToOrdinal.Add(id, ordinal);
    }

    internal int Search(ReadOnlySpan<float> query, Span<SearchResult> results, HnswSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ValidateVector(query, nameof(query));
        ValidateWorkspace(workspace);

        if (results.IsEmpty || _count == 0)
        {
            return 0;
        }

        if (_options.EfSearch < results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(results), "EfSearch must be at least the requested result count.");
        }

        int entryPoint = _entryPoint;
        for (int layer = _maxLayer; layer > 0; layer--)
        {
            int found = SearchLayer(query, entryPoint, 1, layer, workspace, _count);
            if (found > 0)
            {
                entryPoint = workspace.ResultOrdinals[0];
            }
        }

        int candidateCount = SearchLayer(query, entryPoint, _options.EfSearch, 0, workspace, _count);
        int written = Math.Min(results.Length, candidateCount);
        for (int i = 0; i < written; i++)
        {
            results[i] = new SearchResult(_ids[workspace.ResultOrdinals[i]], workspace.ResultDistances[i]);
        }

        return written;
    }

    internal int DebugGetLevel(int ordinal)
    {
        if ((uint)ordinal >= (uint)_count)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal));
        }

        return _levels[ordinal];
    }

    internal int DebugGetNeighbors(int layer, int ordinal, Span<int> destination)
    {
        if (layer < 0 || layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        if ((uint)ordinal >= (uint)_count)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal));
        }

        HnswGraphLayer graphLayer = _layers[layer];
        int count = graphLayer.Counts[ordinal];
        graphLayer.Neighbors.AsSpan(ordinal * graphLayer.Stride, Math.Min(count, destination.Length)).CopyTo(destination);
        return count;
    }

    internal int DebugSelectNeighbors(int baseOrdinal, ReadOnlySpan<int> candidates, Span<int> selected, int layer)
    {
        if ((uint)baseOrdinal >= (uint)_count)
        {
            throw new ArgumentOutOfRangeException(nameof(baseOrdinal));
        }

        int[] candidateArray = candidates.ToArray();
        int selectedCount = SelectNeighbors(baseOrdinal, candidateArray, candidateArray.Length, selected.Length, layer, _count, out int[] selectedArray);
        selectedArray.AsSpan(0, selectedCount).CopyTo(selected);
        return selectedCount;
    }

    private static void ValidateOptions(HnswIndexOptions options)
    {
        if (options.M is < MinM or > MaxM)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "M must be in the range [2, 64].");
        }

        if (options.EfConstruction < options.M || options.EfConstruction > MaxEf)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "EfConstruction must be at least M and no more than 4096.");
        }

        if (options.EfSearch is < 1 or > MaxEf)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "EfSearch must be in the range [1, 4096].");
        }
    }

    private void ValidateWorkspace(HnswSearchWorkspace workspace)
    {
        if (workspace.MaxElements < _count)
        {
            throw new ArgumentException("Workspace element capacity is smaller than the index count.", nameof(workspace));
        }

        if (workspace.MaxEf < _options.EfSearch)
        {
            throw new ArgumentException("Workspace ef capacity is smaller than EfSearch.", nameof(workspace));
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

    private void EnsureCapacity(int requiredCount, int requiredMaxLayer)
    {
        int oldCapacity = _ids.Length;
        if (oldCapacity < requiredCount)
        {
            int newCapacity = oldCapacity == 0 ? InitialCapacity : checked(oldCapacity * 2);
            if (newCapacity < requiredCount)
            {
                newCapacity = requiredCount;
            }

            var ids = new ulong[newCapacity];
            var vectors = new float[checked(newCapacity * Dimension)];
            var levels = new int[newCapacity];
            _ids.AsSpan(0, _count).CopyTo(ids);
            _vectors.AsSpan(0, _count * Dimension).CopyTo(vectors);
            _levels.AsSpan(0, _count).CopyTo(levels);
            _ids = ids;
            _vectors = vectors;
            _levels = levels;

            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].EnsureCapacity(newCapacity);
            }
        }

        if (_layers.Length <= requiredMaxLayer)
        {
            int oldLayerCount = _layers.Length;
            Array.Resize(ref _layers, requiredMaxLayer + 1);
            int capacity = _ids.Length;
            for (int layer = oldLayerCount; layer < _layers.Length; layer++)
            {
                _layers[layer] = new HnswGraphLayer(layer == 0 ? _mMax0 : _mMax, capacity);
            }
        }
    }

    private int GenerateLevel()
    {
        if (_levelProvider is not null)
        {
            int level = _levelProvider();
            if (level < 0)
            {
                throw new InvalidOperationException("Deterministic HNSW level provider returned a negative level.");
            }

            return level;
        }

        double u = NextUnitIntervalInclusive();
        return (int)Math.Floor(-Math.Log(u) * _levelMultiplier);
    }

    private double NextUnitIntervalInclusive()
    {
        _randomState += 0x9E3779B97F4A7C15UL;
        ulong value = _randomState;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        value ^= value >> 31;

        ulong mantissa = (value >> 11) + 1UL;
        return mantissa * (1.0 / 9007199254740992.0);
    }

    private int SearchLayer(
        ReadOnlySpan<float> query,
        int entryPoint,
        int ef,
        int layer,
        HnswSearchWorkspace workspace,
        int validCount)
    {
        int visitMark = workspace.BeginSearch();
        int candidateCount = 0;
        int bestCount = 0;

        float entryDistance = SquaredEuclideanDistance(query, entryPoint);
        workspace.VisitMarks[entryPoint] = visitMark;
        AddCandidate(workspace, ref candidateCount, entryPoint, entryDistance);
        AddBest(workspace, ref bestCount, ef, entryPoint, entryDistance);

        HnswGraphLayer graphLayer = _layers[layer];
        while (candidateCount > 0)
        {
            int candidateIndex = FindNearestCandidateIndex(workspace, candidateCount);
            int current = workspace.CandidateOrdinals[candidateIndex];
            float currentDistance = workspace.CandidateDistances[candidateIndex];
            RemoveAt(workspace.CandidateOrdinals, workspace.CandidateDistances, ref candidateCount, candidateIndex);

            int worstIndex = FindWorstBestIndex(workspace, bestCount);
            if (currentDistance > workspace.BestDistances[worstIndex])
            {
                break;
            }

            int neighborCount = graphLayer.Counts[current];
            int offset = current * graphLayer.Stride;
            for (int i = 0; i < neighborCount; i++)
            {
                int neighbor = graphLayer.Neighbors[offset + i];
                if ((uint)neighbor >= (uint)validCount || workspace.VisitMarks[neighbor] == visitMark)
                {
                    continue;
                }

                workspace.VisitMarks[neighbor] = visitMark;
                float distance = SquaredEuclideanDistance(query, neighbor);
                worstIndex = FindWorstBestIndex(workspace, bestCount);
                bool shouldAdd = bestCount < ef ||
                    CompareNearest(
                        distance,
                        _ids[neighbor],
                        neighbor,
                        workspace.BestDistances[worstIndex],
                        _ids[workspace.BestOrdinals[worstIndex]],
                        workspace.BestOrdinals[worstIndex]) < 0;

                if (!shouldAdd)
                {
                    continue;
                }

                AddCandidate(workspace, ref candidateCount, neighbor, distance);
                AddBest(workspace, ref bestCount, ef, neighbor, distance);
            }
        }

        for (int i = 0; i < bestCount; i++)
        {
            workspace.ResultOrdinals[i] = workspace.BestOrdinals[i];
            workspace.ResultDistances[i] = workspace.BestDistances[i];
        }

        SortNearest(workspace.ResultOrdinals, workspace.ResultDistances, bestCount);
        return bestCount;
    }

    private void AddCandidate(HnswSearchWorkspace workspace, ref int count, int ordinal, float distance)
    {
        workspace.CandidateOrdinals[count] = ordinal;
        workspace.CandidateDistances[count] = distance;
        count++;
    }

    private void AddBest(HnswSearchWorkspace workspace, ref int count, int ef, int ordinal, float distance)
    {
        if (count < ef)
        {
            workspace.BestOrdinals[count] = ordinal;
            workspace.BestDistances[count] = distance;
            count++;
            return;
        }

        int worstIndex = FindWorstBestIndex(workspace, count);
        workspace.BestOrdinals[worstIndex] = ordinal;
        workspace.BestDistances[worstIndex] = distance;
    }

    private int SelectNeighbors(
        int baseOrdinal,
        int[] candidates,
        int candidateCount,
        int maxSelected,
        int layer,
        int validExistingCount,
        out int[] selected)
    {
        var unique = new NeighborCandidate[candidateCount];
        int uniqueCount = 0;

        for (int i = 0; i < candidateCount; i++)
        {
            int candidate = candidates[i];
            int validExclusive = baseOrdinal == _count ? _count : validExistingCount;
            if (candidate == baseOrdinal || candidate < 0 || candidate >= validExclusive)
            {
                continue;
            }

            if (_levels[candidate] < layer)
            {
                continue;
            }

            bool duplicate = false;
            for (int existing = 0; existing < uniqueCount; existing++)
            {
                if (unique[existing].Ordinal == candidate)
                {
                    duplicate = true;
                    break;
                }
            }

            if (duplicate)
            {
                continue;
            }

            unique[uniqueCount++] = new NeighborCandidate(
                candidate,
                DistanceBetween(baseOrdinal, candidate),
                _ids[candidate]);
        }

        Array.Sort(unique, 0, uniqueCount);
        selected = new int[Math.Min(maxSelected, uniqueCount)];
        int selectedCount = 0;

        for (int i = 0; i < uniqueCount && selectedCount < maxSelected; i++)
        {
            int candidate = unique[i].Ordinal;
            float distanceToBase = unique[i].Distance;
            bool accepted = true;
            for (int j = 0; j < selectedCount; j++)
            {
                float distanceToSelected = DistanceBetween(candidate, selected[j]);
                if (distanceToBase >= distanceToSelected)
                {
                    accepted = false;
                    break;
                }
            }

            if (accepted)
            {
                selected[selectedCount++] = candidate;
            }
        }

        return selectedCount;
    }

    private void SetNeighbors(int ordinal, int layer, int[] selected, int selectedCount)
    {
        HnswGraphLayer graphLayer = _layers[layer];
        graphLayer.Counts[ordinal] = selectedCount;
        selected.AsSpan(0, selectedCount).CopyTo(
            graphLayer.Neighbors.AsSpan(ordinal * graphLayer.Stride, selectedCount));
    }

    private void AddOrPruneNeighbor(int baseOrdinal, int newNeighbor, int layer)
    {
        HnswGraphLayer graphLayer = _layers[layer];
        int offset = baseOrdinal * graphLayer.Stride;
        int count = graphLayer.Counts[baseOrdinal];

        for (int i = 0; i < count; i++)
        {
            if (graphLayer.Neighbors[offset + i] == newNeighbor)
            {
                return;
            }
        }

        if (count < graphLayer.Stride)
        {
            graphLayer.Neighbors[offset + count] = newNeighbor;
            graphLayer.Counts[baseOrdinal] = count + 1;
            return;
        }

        int[] candidates = new int[count + 1];
        graphLayer.Neighbors.AsSpan(offset, count).CopyTo(candidates);
        candidates[count] = newNeighbor;
        int selectedCount = SelectNeighbors(baseOrdinal, candidates, candidates.Length, graphLayer.Stride, layer, _count + 1, out int[] selected);
        SetNeighbors(baseOrdinal, layer, selected, selectedCount);
    }

    private float SquaredEuclideanDistance(ReadOnlySpan<float> query, int ordinal)
    {
        int offset = ordinal * Dimension;
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

    private float DistanceBetween(int leftOrdinal, int rightOrdinal)
    {
        int leftOffset = leftOrdinal * Dimension;
        int rightOffset = rightOrdinal * Dimension;
        Vector<float> vectorSum = Vector<float>.Zero;
        int vectorWidth = Vector<float>.Count;
        int i = 0;

        for (; i <= Dimension - vectorWidth; i += vectorWidth)
        {
            var difference =
                new Vector<float>(_vectors.AsSpan(leftOffset + i, vectorWidth)) -
                new Vector<float>(_vectors.AsSpan(rightOffset + i, vectorWidth));
            vectorSum += difference * difference;
        }

        float sum = 0;
        for (int lane = 0; lane < vectorWidth; lane++)
        {
            sum += vectorSum[lane];
        }

        for (; i < Dimension; i++)
        {
            float difference = _vectors[leftOffset + i] - _vectors[rightOffset + i];
            sum += difference * difference;
        }

        return sum;
    }

    private int FindNearestCandidateIndex(HnswSearchWorkspace workspace, int count)
    {
        int bestIndex = 0;
        for (int i = 1; i < count; i++)
        {
            if (CompareNearest(
                    workspace.CandidateDistances[i],
                    _ids[workspace.CandidateOrdinals[i]],
                    workspace.CandidateOrdinals[i],
                    workspace.CandidateDistances[bestIndex],
                    _ids[workspace.CandidateOrdinals[bestIndex]],
                    workspace.CandidateOrdinals[bestIndex]) < 0)
            {
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private int FindWorstBestIndex(HnswSearchWorkspace workspace, int count)
    {
        int worstIndex = 0;
        for (int i = 1; i < count; i++)
        {
            if (CompareWorst(
                    workspace.BestDistances[i],
                    _ids[workspace.BestOrdinals[i]],
                    workspace.BestOrdinals[i],
                    workspace.BestDistances[worstIndex],
                    _ids[workspace.BestOrdinals[worstIndex]],
                    workspace.BestOrdinals[worstIndex]) > 0)
            {
                worstIndex = i;
            }
        }

        return worstIndex;
    }

    private void SortNearest(int[] ordinals, float[] distances, int count)
    {
        for (int i = 1; i < count; i++)
        {
            int ordinal = ordinals[i];
            float distance = distances[i];
            int j = i - 1;
            while (j >= 0 &&
                   CompareNearest(distance, _ids[ordinal], ordinal, distances[j], _ids[ordinals[j]], ordinals[j]) < 0)
            {
                ordinals[j + 1] = ordinals[j];
                distances[j + 1] = distances[j];
                j--;
            }

            ordinals[j + 1] = ordinal;
            distances[j + 1] = distance;
        }
    }

    private static void RemoveAt(int[] ordinals, float[] distances, ref int count, int index)
    {
        int itemsToMove = count - index - 1;
        if (itemsToMove > 0)
        {
            ordinals.AsSpan(index + 1, itemsToMove).CopyTo(ordinals.AsSpan(index, itemsToMove));
            distances.AsSpan(index + 1, itemsToMove).CopyTo(distances.AsSpan(index, itemsToMove));
        }

        count--;
    }

    private static int CompareNearest(
        float leftDistance,
        ulong leftId,
        int leftOrdinal,
        float rightDistance,
        ulong rightId,
        int rightOrdinal)
    {
        int distanceComparison = leftDistance.CompareTo(rightDistance);
        if (distanceComparison != 0)
        {
            return distanceComparison;
        }

        int idComparison = leftId.CompareTo(rightId);
        return idComparison != 0 ? idComparison : leftOrdinal.CompareTo(rightOrdinal);
    }

    private static int CompareWorst(
        float leftDistance,
        ulong leftId,
        int leftOrdinal,
        float rightDistance,
        ulong rightId,
        int rightOrdinal)
    {
        int distanceComparison = leftDistance.CompareTo(rightDistance);
        if (distanceComparison != 0)
        {
            return distanceComparison;
        }

        int idComparison = leftId.CompareTo(rightId);
        return idComparison != 0 ? idComparison : leftOrdinal.CompareTo(rightOrdinal);
    }

    private sealed class HnswGraphLayer
    {
        internal HnswGraphLayer(int stride, int capacity)
        {
            Stride = stride;
            Neighbors = new int[checked(capacity * stride)];
            Counts = new int[capacity];
        }

        internal int Stride { get; }

        internal int[] Neighbors { get; private set; }

        internal int[] Counts { get; private set; }

        internal void EnsureCapacity(int capacity)
        {
            if (Counts.Length >= capacity)
            {
                return;
            }

            var neighbors = new int[checked(capacity * Stride)];
            var counts = new int[capacity];
            Counts.AsSpan().CopyTo(counts);

            int oldCapacity = Counts.Length;
            for (int ordinal = 0; ordinal < oldCapacity; ordinal++)
            {
                Neighbors.AsSpan(ordinal * Stride, Counts[ordinal])
                    .CopyTo(neighbors.AsSpan(ordinal * Stride, Counts[ordinal]));
            }

            Neighbors = neighbors;
            Counts = counts;
        }
    }

    private readonly struct NeighborCandidate : IComparable<NeighborCandidate>
    {
        internal NeighborCandidate(int ordinal, float distance, ulong id)
        {
            Ordinal = ordinal;
            Distance = distance;
            Id = id;
        }

        internal int Ordinal { get; }

        internal float Distance { get; }

        private ulong Id { get; }

        public int CompareTo(NeighborCandidate other)
        {
            int distanceComparison = Distance.CompareTo(other.Distance);
            if (distanceComparison != 0)
            {
                return distanceComparison;
            }

            int idComparison = Id.CompareTo(other.Id);
            return idComparison != 0 ? idComparison : Ordinal.CompareTo(other.Ordinal);
        }
    }
}
