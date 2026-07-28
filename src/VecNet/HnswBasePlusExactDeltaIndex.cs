using System.Diagnostics;
using System.Numerics;

namespace VecNet;

internal sealed class HnswBasePlusExactDeltaIndex
{
    private const int InitialDeltaCapacity = 4;

    private HnswIndex _baseIndex;
    private int _basePhysicalVectorCount;
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
        if (baseIndex.Metric == VectorMetric.InnerProduct)
        {
            throw new NotSupportedException("HNSW base-plus-exact-delta search does not support inner product distance.");
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

        double magnitude = ValidateVector(vector, nameof(vector));
        if (IsKnownOrReserved(id))
        {
            return CreateMutationResult(VectorMutationStatus.DuplicateId);
        }

        EnsureDeltaCapacity(checked(_deltaPhysicalVectorCount + 1));
        StoreDeltaVector(vector, magnitude, _deltaPhysicalVectorCount);
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

    internal HnswBasePlusExactDeltaCheckpointResult Checkpoint(string directoryPath)
    {
        return CheckpointWithDiagnostics(directoryPath).Result;
    }

    internal HnswBasePlusExactDeltaCheckpointDiagnosticResult CheckpointWithDiagnostics(string directoryPath)
    {
        if (_isReadOnly)
        {
            throw new InvalidOperationException("This HNSW base-plus-exact-delta index is read-only and cannot be checkpointed.");
        }

        ValidateNewOrEmptyDirectoryPath(directoryPath);
        ValidateBaseUnchanged();

        int foldedDeltaVectorCount = DeltaLiveVectorCount;
        int foldedBaseTombstoneCount = BaseTombstoneCount;
        int foldedDeltaTombstoneCount = DeltaTombstoneCount;
        if (foldedDeltaVectorCount == 0 &&
            foldedBaseTombstoneCount == 0 &&
            foldedDeltaTombstoneCount == 0)
        {
            HnswBasePlusExactDeltaCheckpointResult noChangesResult = CreateCheckpointResult(
                HnswBasePlusExactDeltaCheckpointStatus.NoChanges,
                foldedDeltaVectorCount,
                foldedBaseTombstoneCount,
                foldedDeltaTombstoneCount);
            return new HnswBasePlusExactDeltaCheckpointDiagnosticResult(
                noChangesResult,
                HnswBasePlusExactDeltaCheckpointDiagnostics.None);
        }

        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics liveSnapshotDiagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted;
        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics rebuildBuildDiagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted;
        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics saveDiagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted;
        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics openValidationDiagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted;
        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics publicationDiagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted;

        int liveVectorCount = MeasurePhase(
            ValidateLiveRowsForCheckpoint,
            ref liveSnapshotDiagnostics);
        HnswIndex rebuilt = MeasurePhase(
            () => BuildBaseIndexFromLiveRows(liveVectorCount),
            ref rebuildBuildDiagnostics);
        MeasurePhase(
            () => rebuilt.Save(directoryPath),
            ref saveDiagnostics);

        MeasurePhase(
            () => HnswIndexStorage.ValidateSavedIndex(directoryPath, rebuilt),
            ref openValidationDiagnostics);

        MeasurePhase(
            () =>
            {
                PublishRebuiltBase(rebuilt);
                _generation++;
            },
            ref publicationDiagnostics);

        HnswBasePlusExactDeltaCheckpointResult publishedResult = CreateCheckpointResult(
            HnswBasePlusExactDeltaCheckpointStatus.Published,
            foldedDeltaVectorCount,
            foldedBaseTombstoneCount,
            foldedDeltaTombstoneCount);
        return new HnswBasePlusExactDeltaCheckpointDiagnosticResult(
            publishedResult,
            new HnswBasePlusExactDeltaCheckpointDiagnostics(
                liveSnapshotDiagnostics,
                rebuildBuildDiagnostics,
                saveDiagnostics,
                openValidationDiagnostics,
                publicationDiagnostics));
    }

    internal int Search(
        ReadOnlySpan<float> query,
        Span<SearchResult> results,
        HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ValidateBaseUnchanged();
        double queryMagnitude = ValidateVector(query, nameof(query));
        ValidateWorkspace(results.Length, workspace);
        workspace.ObservedGeneration = _generation;

        if (results.IsEmpty || LiveVectorCount == 0)
        {
            return 0;
        }

        if (Options.EfSearch < results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(results), "EfSearch must be at least the requested result count.");
        }

        int baseCandidateCount = SearchBaseCandidates(query, workspace);
        int deltaCandidateCount = SearchDeltaCandidates(query, queryMagnitude, results.Length, workspace);
        return MergeCandidates(
            workspace.BaseCandidates.AsSpan(0, baseCandidateCount),
            workspace.DeltaCandidates.AsSpan(0, deltaCandidateCount),
            results);
    }

    internal int Search(
        ReadOnlySpan<float> query,
        ReadOnlySpan<ulong> allowedIds,
        Span<SearchResult> results,
        HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ValidateBaseUnchanged();
        double queryMagnitude = ValidateVector(query, nameof(query));
        ValidateFilteredWorkspace(results.Length, workspace);
        workspace.ObservedGeneration = _generation;

        if (results.IsEmpty || LiveVectorCount == 0 || allowedIds.IsEmpty)
        {
            return 0;
        }

        if (Options.EfSearch < results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(results), "EfSearch must be at least the requested result count.");
        }

        int baseFilterMark = workspace.HnswWorkspace.BeginFilter();
        int deltaFilterMark = workspace.BeginDeltaFilter();
        (int baseAllowedCount, int deltaAllowedCount) = MarkAllowedCandidates(
            allowedIds,
            workspace,
            baseFilterMark,
            deltaFilterMark);
        int allowedCount = checked(baseAllowedCount + deltaAllowedCount);
        if (allowedCount == 0)
        {
            return 0;
        }

        if (allowedCount <= Options.EfSearch)
        {
            return SearchAllowedExact(query, queryMagnitude, results, workspace, baseFilterMark, deltaFilterMark);
        }

        int baseCandidateCount = SearchBaseAllowedCandidates(query, workspace, baseFilterMark);
        int deltaCandidateCount = SearchDeltaAllowedCandidates(query, queryMagnitude, results.Length, workspace, deltaFilterMark);
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
        double queryMagnitude,
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

            var candidate = new SearchResult(id, DistanceToDelta(query, queryMagnitude, ordinal));
            written = InsertCandidate(deltaCandidates, written, candidate);
        }

        return written;
    }

    private (int BaseAllowedCount, int DeltaAllowedCount) MarkAllowedCandidates(
        ReadOnlySpan<ulong> allowedIds,
        HnswBasePlusExactDeltaSearchWorkspace workspace,
        int baseFilterMark,
        int deltaFilterMark)
    {
        int baseAllowedCount = 0;
        int deltaAllowedCount = 0;

        for (int allowIndex = 0; allowIndex < allowedIds.Length; allowIndex++)
        {
            ulong id = allowedIds[allowIndex];
            if (_baseIndex.TryGetOrdinal(id, out int baseOrdinal) &&
                baseOrdinal < _basePhysicalVectorCount)
            {
                if (_baseTombstoneIds.Contains(id))
                {
                    continue;
                }

                if (workspace.HnswWorkspace.FilterMarks[baseOrdinal] == baseFilterMark)
                {
                    continue;
                }

                workspace.HnswWorkspace.FilterMarks[baseOrdinal] = baseFilterMark;
                baseAllowedCount++;
                continue;
            }

            if (!_deltaIdToOrdinal.TryGetValue(id, out int deltaOrdinal))
            {
                continue;
            }

            if (_deltaTombstoneIds.Contains(id))
            {
                continue;
            }

            if (workspace.DeltaFilterMarks[deltaOrdinal] == deltaFilterMark)
            {
                continue;
            }

            workspace.DeltaFilterMarks[deltaOrdinal] = deltaFilterMark;
            deltaAllowedCount++;
        }

        return (baseAllowedCount, deltaAllowedCount);
    }

    private int SearchAllowedExact(
        ReadOnlySpan<float> query,
        double queryMagnitude,
        Span<SearchResult> results,
        HnswBasePlusExactDeltaSearchWorkspace workspace,
        int baseFilterMark,
        int deltaFilterMark)
    {
        int written = 0;
        ReadOnlySpan<ulong> baseIds = _baseIndex.InternalIds;
        for (int ordinal = 0; ordinal < _basePhysicalVectorCount; ordinal++)
        {
            if (workspace.HnswWorkspace.FilterMarks[ordinal] != baseFilterMark)
            {
                continue;
            }

            ulong id = baseIds[ordinal];
            if (_baseTombstoneIds.Contains(id))
            {
                continue;
            }

            var candidate = new SearchResult(id, _baseIndex.CalculateDistance(query, queryMagnitude, ordinal));
            written = InsertCandidate(results, written, candidate);
        }

        for (int ordinal = 0; ordinal < _deltaPhysicalVectorCount; ordinal++)
        {
            if (workspace.DeltaFilterMarks[ordinal] != deltaFilterMark)
            {
                continue;
            }

            ulong id = _deltaIds[ordinal];
            if (_deltaTombstoneIds.Contains(id))
            {
                continue;
            }

            var candidate = new SearchResult(id, DistanceToDelta(query, queryMagnitude, ordinal));
            written = InsertCandidate(results, written, candidate);
        }

        return written;
    }

    private int SearchBaseAllowedCandidates(
        ReadOnlySpan<float> query,
        HnswBasePlusExactDeltaSearchWorkspace workspace,
        int baseFilterMark)
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

            if (!_baseIndex.TryGetOrdinal(candidate.Id, out int ordinal) ||
                workspace.HnswWorkspace.FilterMarks[ordinal] != baseFilterMark)
            {
                continue;
            }

            baseCandidates[liveCount++] = candidate;
        }

        return liveCount;
    }

    private int SearchDeltaAllowedCandidates(
        ReadOnlySpan<float> query,
        double queryMagnitude,
        int requestedResultCount,
        HnswBasePlusExactDeltaSearchWorkspace workspace,
        int deltaFilterMark)
    {
        if (requestedResultCount == 0 || DeltaLiveVectorCount == 0)
        {
            return 0;
        }

        Span<SearchResult> deltaCandidates = workspace.DeltaCandidates.AsSpan(0, requestedResultCount);
        int written = 0;
        for (int ordinal = 0; ordinal < _deltaPhysicalVectorCount; ordinal++)
        {
            if (workspace.DeltaFilterMarks[ordinal] != deltaFilterMark)
            {
                continue;
            }

            ulong id = _deltaIds[ordinal];
            if (_deltaTombstoneIds.Contains(id))
            {
                continue;
            }

            var candidate = new SearchResult(id, DistanceToDelta(query, queryMagnitude, ordinal));
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
        if (workspace.ObservedGeneration != long.MinValue && workspace.ObservedGeneration != _generation)
        {
            throw new InvalidOperationException("Workspace generation is stale for this HNSW base-plus-exact-delta index.");
        }

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

    private void ValidateFilteredWorkspace(int requestedResultCount, HnswBasePlusExactDeltaSearchWorkspace workspace)
    {
        ValidateWorkspace(requestedResultCount, workspace);

        if (workspace.DeltaFilterMarks.Length < _deltaPhysicalVectorCount)
        {
            throw new ArgumentException("Workspace delta filter capacity is smaller than the physical delta count.", nameof(workspace));
        }
    }

    private void ValidateBaseUnchanged()
    {
        if (_baseIndex.Count != _basePhysicalVectorCount)
        {
            throw new InvalidOperationException("The HNSW base changed after composite construction.");
        }
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

    private void StoreDeltaVector(ReadOnlySpan<float> vector, double magnitude, int ordinal)
    {
        int offset = ordinal * Dimension;
        if (Metric == VectorMetric.Cosine)
        {
            for (int i = 0; i < Dimension; i++)
            {
                _deltaVectors[offset + i] = (float)(vector[i] / magnitude);
            }

            return;
        }

        vector.CopyTo(_deltaVectors.AsSpan(offset, Dimension));
    }

    private bool IsKnownOrReserved(ulong id) =>
        _baseIds.Contains(id) || _deltaIdToOrdinal.ContainsKey(id) || _deletedReservedIds.Contains(id);

    private VectorMutationResult CreateMutationResult(VectorMutationStatus status) =>
        new(status, _generation, LiveVectorCount, DeltaLiveVectorCount, TombstoneCount);

    private HnswBasePlusExactDeltaCheckpointResult CreateCheckpointResult(
        HnswBasePlusExactDeltaCheckpointStatus status,
        int foldedDeltaVectorCount,
        int foldedBaseTombstoneCount,
        int foldedDeltaTombstoneCount) =>
        new(
            status,
            _generation,
            BasePhysicalVectorCount,
            LiveVectorCount,
            BasePhysicalVectorCount,
            BaseLiveVectorCount,
            DeltaPhysicalVectorCount,
            DeltaLiveVectorCount,
            BaseTombstoneCount,
            DeltaTombstoneCount,
            TombstoneCount,
            DeletedReservedIdCount,
            foldedDeltaVectorCount,
            foldedBaseTombstoneCount,
            foldedDeltaTombstoneCount);

    private int ValidateLiveRowsForCheckpoint()
    {
        int liveCount = 0;
        ReadOnlySpan<ulong> baseIds = _baseIndex.InternalIds;
        for (int sourceOrdinal = 0; sourceOrdinal < _basePhysicalVectorCount; sourceOrdinal++)
        {
            if (_baseTombstoneIds.Contains(baseIds[sourceOrdinal]))
            {
                continue;
            }

            ValidateStoredVectorForCheckpoint(
                _baseIndex.InternalVectors.Slice(sourceOrdinal * Dimension, Dimension));
            liveCount++;
        }

        for (int sourceOrdinal = 0; sourceOrdinal < _deltaPhysicalVectorCount; sourceOrdinal++)
        {
            if (_deltaTombstoneIds.Contains(_deltaIds[sourceOrdinal]))
            {
                continue;
            }

            ValidateStoredVectorForCheckpoint(
                _deltaVectors.AsSpan(sourceOrdinal * Dimension, Dimension));
            liveCount++;
        }

        if (liveCount != LiveVectorCount)
        {
            throw new InvalidOperationException("HNSW base-plus-exact-delta live row accounting is inconsistent.");
        }

        return liveCount;
    }

    private void ValidateStoredVectorForCheckpoint(ReadOnlySpan<float> vector)
    {
        double squaredMagnitude = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            float value = vector[i];
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("HNSW checkpoint source vector contains a non-finite component.");
            }

            if (Metric == VectorMetric.Cosine)
            {
                squaredMagnitude += (double)value * value;
            }
        }

        if (Metric == VectorMetric.Cosine &&
            (squaredMagnitude == 0 ||
             Math.Abs(squaredMagnitude - 1) > HnswIndexStorage.CosineStoredRowSquaredLengthTolerance))
        {
            throw new InvalidOperationException("HNSW checkpoint cosine source vector is not within the unit-length tolerance.");
        }
    }

    private HnswIndex BuildBaseIndexFromLiveRows(int liveCount)
    {
        var rebuilt = new HnswIndex(Dimension, Metric, Options, liveCount);
        ReadOnlySpan<ulong> baseIds = _baseIndex.InternalIds;
        ReadOnlySpan<float> baseVectors = _baseIndex.InternalVectors;

        for (int sourceOrdinal = 0; sourceOrdinal < _basePhysicalVectorCount; sourceOrdinal++)
        {
            ulong id = baseIds[sourceOrdinal];
            if (_baseTombstoneIds.Contains(id))
            {
                continue;
            }

            rebuilt.Add(id, baseVectors.Slice(sourceOrdinal * Dimension, Dimension));
        }

        for (int sourceOrdinal = 0; sourceOrdinal < _deltaPhysicalVectorCount; sourceOrdinal++)
        {
            ulong id = _deltaIds[sourceOrdinal];
            if (_deltaTombstoneIds.Contains(id))
            {
                continue;
            }

            rebuilt.Add(id, _deltaVectors.AsSpan(sourceOrdinal * Dimension, Dimension));
        }

        if (rebuilt.Count != liveCount)
        {
            throw new InvalidOperationException("HNSW checkpoint rebuilt base count is inconsistent.");
        }

        return rebuilt;
    }

    private void PublishRebuiltBase(HnswIndex rebuilt)
    {
        _baseIndex = rebuilt;
        _basePhysicalVectorCount = rebuilt.Count;
        _baseIds.Clear();
        foreach (ulong id in rebuilt.InternalIds)
        {
            _baseIds.Add(id);
        }

        _baseTombstoneIds.Clear();
        _deltaTombstoneIds.Clear();
        _deltaIdToOrdinal.Clear();
        _deltaIds = [];
        _deltaVectors = [];
        _deltaPhysicalVectorCount = 0;
    }

    private static void ValidateNewOrEmptyDirectoryPath(string directoryPath)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        string directory = Path.GetFullPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("HNSW index save path is an existing file, not a directory.");
        }

        if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
        {
            throw new IOException("HNSW index save directory must be empty.");
        }
    }

    private static T MeasurePhase<T>(
        Func<T> action,
        ref HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        long timestamp = Stopwatch.GetTimestamp();
        try
        {
            T result = action();
            diagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.Measured(
                Stopwatch.GetElapsedTime(timestamp).Ticks,
                GC.GetAllocatedBytesForCurrentThread() - allocatedBefore);
            return result;
        }
        catch
        {
            diagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.Failed(
                Stopwatch.GetElapsedTime(timestamp).Ticks,
                GC.GetAllocatedBytesForCurrentThread() - allocatedBefore);
            throw;
        }
    }

    private static void MeasurePhase(
        Action action,
        ref HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        long timestamp = Stopwatch.GetTimestamp();
        try
        {
            action();
            diagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.Measured(
                Stopwatch.GetElapsedTime(timestamp).Ticks,
                GC.GetAllocatedBytesForCurrentThread() - allocatedBefore);
        }
        catch
        {
            diagnostics = HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.Failed(
                Stopwatch.GetElapsedTime(timestamp).Ticks,
                GC.GetAllocatedBytesForCurrentThread() - allocatedBefore);
            throw;
        }
    }

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

    private float DistanceToDelta(ReadOnlySpan<float> query, double queryMagnitude, int deltaOrdinal) =>
        Metric == VectorMetric.Cosine
            ? CosineDistance(query, queryMagnitude, deltaOrdinal)
            : SquaredEuclideanDistance(query, deltaOrdinal);

    private float CosineDistance(ReadOnlySpan<float> query, double queryMagnitude, int deltaOrdinal)
    {
        int offset = deltaOrdinal * Dimension;
        double dotProduct = 0;
        for (int i = 0; i < Dimension; i++)
        {
            dotProduct += (double)query[i] * _deltaVectors[offset + i];
        }

        return (float)(1 - (dotProduct / queryMagnitude));
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
