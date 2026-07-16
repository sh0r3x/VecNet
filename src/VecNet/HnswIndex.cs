using System.Numerics;

namespace VecNet;

/// <summary>
/// Approximate HNSW index for squared Euclidean distance.
/// </summary>
/// <remarks>
/// This API supports build ingestion with <see cref="Add"/>, caller-owned workspace
/// search with <see cref="Search(ReadOnlySpan{float}, Span{SearchResult}, HnswSearchWorkspace)"/>,
/// caller-owned external-ID allowlist filtering, and durable round trips with
/// <see cref="Save"/> and <see cref="OpenReadOnly"/>. Read-only searches may overlap only when each
/// caller uses an independent result buffer and independent workspace. It currently supports only
/// <see cref="VectorMetric.SquaredEuclidean"/>. Cosine HNSW, inner-product HNSW, stored labels,
/// durable graph-aware filtering metadata, public ordinal filters, full filter-aware graph
/// traversal, update/delete, replacement, repair, direct graph mutation, and stable file-format
/// compatibility are not supported by this API.
/// </remarks>
public sealed partial class HnswIndex
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
    private bool _isReadOnly;
    private HnswBuildScratch? _buildScratch;

    /// <summary>
    /// Initializes an HNSW index with <see cref="HnswIndexOptions.Default"/>.
    /// </summary>
    /// <param name="dimension">The required positive vector dimension.</param>
    /// <param name="metric">The canonical distance metric. Only <see cref="VectorMetric.SquaredEuclidean"/> is supported.</param>
    public HnswIndex(int dimension, VectorMetric metric)
        : this(dimension, metric, HnswIndexOptions.Default)
    {
    }

    /// <summary>
    /// Initializes an HNSW index with <see cref="HnswIndexOptions.Default"/> and preallocated mutable row capacity.
    /// </summary>
    /// <param name="dimension">The required positive vector dimension.</param>
    /// <param name="metric">The canonical distance metric. Only <see cref="VectorMetric.SquaredEuclidean"/> is supported.</param>
    /// <param name="initialCapacity">The non-negative number of vector rows to reserve in contiguous HNSW storage.</param>
    public HnswIndex(int dimension, VectorMetric metric, int initialCapacity)
        : this(dimension, metric, HnswIndexOptions.Default, initialCapacity)
    {
    }

    /// <summary>
    /// Initializes an HNSW index with explicit options.
    /// </summary>
    /// <param name="dimension">The required positive vector dimension.</param>
    /// <param name="metric">The canonical distance metric. Only <see cref="VectorMetric.SquaredEuclidean"/> is supported.</param>
    /// <param name="options">The HNSW build and search options.</param>
    public HnswIndex(int dimension, VectorMetric metric, HnswIndexOptions options)
        : this(dimension, metric, options, levelProvider: null)
    {
    }

    /// <summary>
    /// Initializes an HNSW index with explicit options and preallocated mutable row capacity.
    /// </summary>
    /// <param name="dimension">The required positive vector dimension.</param>
    /// <param name="metric">The canonical distance metric. Only <see cref="VectorMetric.SquaredEuclidean"/> is supported.</param>
    /// <param name="options">The HNSW build and search options.</param>
    /// <param name="initialCapacity">The non-negative number of vector rows to reserve in contiguous HNSW storage.</param>
    public HnswIndex(int dimension, VectorMetric metric, HnswIndexOptions options, int initialCapacity)
        : this(dimension, metric, options, initialCapacity, levelProvider: null)
    {
    }

    internal HnswIndex(int dimension, VectorMetric metric, HnswIndexOptions options, Func<int>? levelProvider)
        : this(dimension, metric, options, initialCapacity: 0, levelProvider)
    {
    }

    internal HnswIndex(
        int dimension,
        VectorMetric metric,
        HnswIndexOptions options,
        int initialCapacity,
        Func<int>? levelProvider)
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

        ValidateVectorCapacity(initialCapacity, nameof(initialCapacity));
        if (initialCapacity > 0)
        {
            _idToOrdinal.EnsureCapacity(initialCapacity);
            EnsureStorageCapacity(initialCapacity, requiredMaxLayer: -1, nameof(initialCapacity), growForAppend: false);
        }
    }

    /// <summary>
    /// Gets the fixed dimension accepted by this HNSW index.
    /// </summary>
    public int Dimension { get; }

    /// <summary>
    /// Gets the canonical distance metric used by this HNSW index.
    /// </summary>
    /// <remarks>Only <see cref="VectorMetric.SquaredEuclidean"/> is supported.</remarks>
    public VectorMetric Metric { get; }

    /// <summary>
    /// Gets the number of vectors ingested into this HNSW index.
    /// </summary>
    /// <remarks>This count is useful for sizing caller-owned <see cref="HnswSearchWorkspace"/> instances.</remarks>
    public int Count => _count;

    /// <summary>
    /// Gets the current allocated vector-row capacity of this HNSW index.
    /// </summary>
    /// <remarks>
    /// Capacity is storage reservation, not vector cardinality. Use <see cref="Count"/>
    /// for the number of ingested vectors.
    /// </remarks>
    public int Capacity => _ids.Length;

    internal int EntryPoint => _entryPoint;

    internal int MaxLayer => _maxLayer;

    internal ReadOnlySpan<ulong> InternalIds => _ids.AsSpan(0, _count);

    internal ReadOnlySpan<float> InternalVectors => _vectors.AsSpan(0, checked(_count * Dimension));

    internal ReadOnlySpan<int> InternalLevels => _levels.AsSpan(0, _count);

    internal int InternalMMax => _mMax;

    internal int InternalMMax0 => _mMax0;

    internal double InternalLevelMultiplier => _levelMultiplier;

    /// <summary>
    /// Gets the HNSW options used by this index.
    /// </summary>
    /// <remarks>The configured options are not public performance, recall, memory, allocation, capacity, or storage-size claims.</remarks>
    public HnswIndexOptions Options => _options;

    /// <summary>
    /// Ensures this mutable HNSW index can store at least the requested number of vector rows.
    /// </summary>
    /// <param name="vectorCapacity">The non-negative vector-row capacity to reserve.</param>
    /// <remarks>
    /// This method may allocate and copy existing row, graph, and build-scratch storage. It does not
    /// change <see cref="Count"/>, graph contents, insertion order, search results, or durable output.
    /// HNSW indexes opened with <see cref="OpenReadOnly(string)"/> reject this method.
    /// </remarks>
    public void EnsureCapacity(int vectorCapacity)
    {
        if (_isReadOnly)
        {
            throw new InvalidOperationException(
                "This HNSW index was opened read-only and cannot be capacity planned.");
        }

        EnsureStorageCapacity(vectorCapacity, requiredMaxLayer: -1, nameof(vectorCapacity), growForAppend: false);
        _idToOrdinal.EnsureCapacity(vectorCapacity);
        EnsurePlannedBuildScratch();
    }

    /// <summary>
    /// Saves this HNSW index to a new or empty durable HNSW directory.
    /// </summary>
    /// <remarks>
    /// Save writes HNSW round-trip files. It requires a new or empty target location
    /// and does not replace an active index directory, coordinate with other processes, provide
    /// caller-level crash recovery for directory swaps, or establish a stable file-format
    /// compatibility promise.
    /// </remarks>
    /// <param name="directoryPath">
    /// The target directory path. It must not be null or whitespace, must not name an existing file,
    /// and must either not exist or name an empty directory. Existing index directories are not overwritten.
    /// </param>
    public void Save(string directoryPath)
    {
        HnswIndexStorage.Save(directoryPath, this);
    }

    /// <summary>
    /// Opens a durable HNSW directory as an immutable read-only index.
    /// </summary>
    /// <remarks>
    /// Open validates the manifest and binary files using broad failure categories such as
    /// invalid data, missing files, unsupported format, or I/O errors. It does not establish a
    /// stable complete exception taxonomy, does not open the index for mutation, and does not make
    /// a stable file-format compatibility promise.
    /// </remarks>
    /// <param name="directoryPath">
    /// The HNSW index directory path. It must not be null or whitespace and must name an existing
    /// HNSW directory containing a valid manifest and binary files.
    /// </param>
    /// <returns>A searchable read-only HNSW index.</returns>
    public static HnswIndex OpenReadOnly(string directoryPath) =>
        HnswIndexStorage.OpenReadOnly(directoryPath);

    /// <summary>
    /// Inserts a vector associated with a caller-provided external identifier during HNSW build ingestion.
    /// </summary>
    /// <remarks>
    /// This is build ingestion for an immutable HNSW index, not an upsert, delete, repair,
    /// replacement, direct graph mutation, or live-update contract. Indexes opened with
    /// <see cref="OpenReadOnly"/> reject this operation.
    /// </remarks>
    /// <param name="id">The opaque external vector identifier.</param>
    /// <param name="vector">The finite squared-L2 vector values to copy into index storage.</param>
    public void Add(ulong id, ReadOnlySpan<float> vector)
    {
        if (_isReadOnly)
        {
            throw new InvalidOperationException("This HNSW index was opened read-only and cannot be modified.");
        }

        ValidateVector(vector, nameof(vector));
        if (_idToOrdinal.ContainsKey(id))
        {
            throw new ArgumentException("An item with the same identifier already exists.", nameof(id));
        }

        int level = GenerateLevel();
        int ordinal = _count;
        EnsureStorageCapacity(checked(_count + 1), level, nameof(vector), growForAppend: true);
        _idToOrdinal.EnsureCapacity(checked(_count + 1));

        HnswBuildScratch? scratch = null;
        if (_count > 0)
        {
            scratch = EnsureBuildScratch(
                _ids.Length,
                Math.Max(_options.EfConstruction, _mMax0 + 1),
                _mMax0,
                Math.Max(_options.EfConstruction, _mMax0 + 1));
        }

        _ids[ordinal] = id;
        vector.CopyTo(_vectors.AsSpan(ordinal * Dimension, Dimension));
        _levels[ordinal] = level;
        for (int layer = 1; layer <= level; layer++)
        {
            _layers[layer].EnsureParticipantRow(ordinal);
        }

        if (_count == 0)
        {
            _entryPoint = ordinal;
            _maxLayer = level;
            _count++;
            _idToOrdinal.Add(id, ordinal);
            return;
        }

        HnswSearchWorkspace workspace = scratch!.Workspace;
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
            int selectedCount = SelectNeighbors(
                ordinal,
                workspace.ResultOrdinals.AsSpan(0, found),
                scratch.SelectedOrdinals.AsSpan(0, _options.M),
                layer,
                _count,
                scratch.NeighborCandidates);
            SetNeighbors(ordinal, layer, scratch.SelectedOrdinals, selectedCount);

            ReadOnlySpan<int> selectedNeighbors = _layers[layer].GetNeighbors(ordinal);
            for (int i = 0; i < selectedCount; i++)
            {
                AddOrPruneNeighbor(selectedNeighbors[i], ordinal, layer, scratch);
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

    /// <summary>
    /// Searches this HNSW index and writes approximate nearest results in ascending distance order.
    /// </summary>
    /// <remarks>
    /// Results are ordered by the executing squared-L2 distance, with external ID breaking equal
    /// computed-distance ties. The caller owns the result buffer and workspace. Do not share one
    /// workspace or one result buffer across overlapping searches.
    /// </remarks>
    /// <param name="query">The finite squared-L2 query vector.</param>
    /// <param name="results">
    /// The caller-owned destination buffer. Its length specifies the requested result count.
    /// </param>
    /// <param name="workspace">
    /// The caller-owned reusable HNSW workspace. It must have <see cref="HnswSearchWorkspace.MaxElements"/>
    /// at least <see cref="Count"/> and <see cref="HnswSearchWorkspace.MaxEf"/> at least
    /// <see cref="HnswIndexOptions.EfSearch"/> for this index. Workspaces must not be shared by
    /// overlapping searches.
    /// </param>
    /// <returns>The number of results written.</returns>
    public int Search(ReadOnlySpan<float> query, Span<SearchResult> results, HnswSearchWorkspace workspace)
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

    /// <summary>
    /// Searches this HNSW index while emitting only vectors whose external identifiers are present in an allowlist.
    /// </summary>
    /// <remarks>
    /// The allowlist is caller-owned query input. Unknown identifiers are ignored and duplicates are
    /// coalesced. When the known allowed count is no greater than <see cref="HnswIndexOptions.EfSearch"/>,
    /// this method uses an exact filtered scan over the known allowed vectors. For broader allowlists,
    /// HNSW traversal remains unfiltered and non-allowed candidates are suppressed at emission, so the
    /// result may underfill even when exact filtered truth has at least the requested number of results.
    /// </remarks>
    /// <param name="query">The finite squared-L2 query vector.</param>
    /// <param name="allowedIds">Caller-supplied external identifiers allowed for this search.</param>
    /// <param name="results">The caller-owned destination buffer. Its length specifies the requested result count.</param>
    /// <param name="workspace">
    /// The caller-owned reusable HNSW workspace. It must satisfy the same sizing rules as unfiltered
    /// <see cref="Search(ReadOnlySpan{float}, Span{SearchResult}, HnswSearchWorkspace)"/>.
    /// </param>
    /// <returns>The number of filtered results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        ReadOnlySpan<ulong> allowedIds,
        Span<SearchResult> results,
        HnswSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ValidateVector(query, nameof(query));
        ValidateWorkspace(workspace);

        if (results.IsEmpty || _count == 0 || allowedIds.IsEmpty)
        {
            return 0;
        }

        if (_options.EfSearch < results.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(results), "EfSearch must be at least the requested result count.");
        }

        int filterMark = workspace.BeginFilter();
        int allowedCount = MarkAllowedOrdinals(allowedIds, workspace.FilterMarks, filterMark);
        if (allowedCount == 0)
        {
            return 0;
        }

        if (allowedCount <= _options.EfSearch)
        {
            return SearchAllowedExact(query, results, workspace.FilterMarks, filterMark);
        }

        return SearchAllowedByEmission(query, results, workspace, filterMark);
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
        ReadOnlySpan<int> neighbors = graphLayer.GetNeighbors(ordinal);
        neighbors[..Math.Min(neighbors.Length, destination.Length)].CopyTo(destination);
        return neighbors.Length;
    }

    internal int InternalGetGraphStride(int layer) => _layers[layer].Stride;

    internal ReadOnlySpan<int> InternalGetGraphNeighbors(int layer, int ordinal) =>
        _layers[layer].GetNeighbors(ordinal);

    internal int DebugSelectNeighbors(int baseOrdinal, ReadOnlySpan<int> candidates, Span<int> selected, int layer)
    {
        if ((uint)baseOrdinal >= (uint)_count)
        {
            throw new ArgumentOutOfRangeException(nameof(baseOrdinal));
        }

        HnswBuildScratch scratch = EnsureBuildScratch(
            Math.Max(_ids.Length, _count),
            Math.Max(candidates.Length, _mMax0 + 1),
            Math.Max(selected.Length, _mMax0),
            Math.Max(candidates.Length, _mMax0 + 1));
        return SelectNeighbors(baseOrdinal, candidates, selected, layer, _count, scratch.NeighborCandidates);
    }

    internal HnswSearchWorkspace? DebugBuildSearchWorkspace => _buildScratch?.Workspace;

    internal int DebugGetLayerCapacity(int layer)
    {
        if (layer < 0 || layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer].Capacity;
    }

    internal int DebugGetLayerCompactRowCount(int layer)
    {
        if (layer < 0 || layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer].CompactRowCount;
    }

    internal int DebugGetLayerCountStorageLength(int layer)
    {
        if (layer < 0 || layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer].CountStorageLength;
    }

    internal int DebugGetLayerNeighborStorageLength(int layer)
    {
        if (layer < 0 || layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer].NeighborStorageLength;
    }

    internal bool TryGetOrdinal(ulong id, out int ordinal) => _idToOrdinal.TryGetValue(id, out ordinal);

    internal float CalculateSquaredEuclideanDistance(ReadOnlySpan<float> query, int ordinal) =>
        SquaredEuclideanDistance(query, ordinal);

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

    private void EnsureStorageCapacity(int requiredCount, int requiredMaxLayer, string parameterName, bool growForAppend)
    {
        ValidateVectorCapacity(requiredCount, parameterName);

        if (_ids.Length < requiredCount)
        {
            int newCapacity = growForAppend ? CalculateGrowthCapacity(requiredCount) : requiredCount;
            AllocateCapacity(newCapacity);
        }

        if (_layers.Length <= requiredMaxLayer)
        {
            int oldLayerCount = _layers.Length;
            Array.Resize(ref _layers, requiredMaxLayer + 1);
            int capacity = _ids.Length;
            for (int layer = oldLayerCount; layer < _layers.Length; layer++)
            {
                _layers[layer] = new HnswGraphLayer(layer == 0 ? _mMax0 : _mMax, capacity, sparse: layer > 0);
            }
        }
    }

    private int CalculateGrowthCapacity(int requiredCount)
    {
        int maxCapacity = GetMaximumVectorCapacity();
        int newCapacity;
        if (_ids.Length == 0)
        {
            newCapacity = Math.Min(InitialCapacity, maxCapacity);
        }
        else
        {
            newCapacity = _ids.Length <= maxCapacity / 2
                ? _ids.Length * 2
                : maxCapacity;
        }

        if (newCapacity < requiredCount)
        {
            newCapacity = requiredCount;
        }

        ValidateVectorCapacity(newCapacity, nameof(requiredCount));
        return newCapacity;
    }

    private void AllocateCapacity(int newCapacity)
    {
        ValidateVectorCapacity(newCapacity, nameof(newCapacity));

        var ids = new ulong[newCapacity];
        var vectors = new float[newCapacity * Dimension];
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

    private void ValidateVectorCapacity(int vectorCapacity, string parameterName)
    {
        if (vectorCapacity < 0)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                vectorCapacity,
                "Vector capacity must be non-negative.");
        }

        if (vectorCapacity > int.MaxValue / Dimension)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                vectorCapacity,
                "Vector capacity times dimension exceeds the supported element count.");
        }

        if (vectorCapacity > int.MaxValue / _mMax0)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                vectorCapacity,
                "Vector capacity times HNSW graph stride exceeds the supported element count.");
        }

        if (vectorCapacity > GetMaximumVectorCapacity())
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                vectorCapacity,
                "Vector capacity exceeds the maximum contiguous HNSW row or graph capacity.");
        }
    }

    private int GetMaximumVectorCapacity() =>
        Math.Min(Array.MaxLength / Dimension, Array.MaxLength / _mMax0);

    private void EnsurePlannedBuildScratch()
    {
        if (_buildScratch is null)
        {
            return;
        }

        EnsureBuildScratch(
            _ids.Length,
            Math.Max(_options.EfConstruction, _mMax0 + 1),
            _mMax0,
            Math.Max(_options.EfConstruction, _mMax0 + 1));
    }

    private HnswBuildScratch EnsureBuildScratch(
        int maxElements,
        int candidateOrdinalCapacity,
        int selectedCapacity,
        int neighborCandidateCapacity)
    {
        HnswBuildScratch? scratch = _buildScratch;
        if (scratch is null ||
            scratch.Workspace.MaxElements < maxElements ||
            scratch.Workspace.MaxEf < _options.EfConstruction ||
            scratch.CandidateOrdinals.Length < candidateOrdinalCapacity ||
            scratch.SelectedOrdinals.Length < selectedCapacity ||
            scratch.NeighborCandidates.Length < neighborCandidateCapacity)
        {
            scratch = new HnswBuildScratch(
                maxElements,
                _options.EfConstruction,
                candidateOrdinalCapacity,
                selectedCapacity,
                neighborCandidateCapacity);
            _buildScratch = scratch;
        }

        return scratch;
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
        HnswPriorityQueues.PushNearest(
            workspace.CandidateOrdinals,
            workspace.CandidateDistances,
            _ids,
            ref candidateCount,
            entryPoint,
            entryDistance);
        HnswPriorityQueues.AddBoundedNearest(
            workspace.BestOrdinals,
            workspace.BestDistances,
            _ids,
            ref bestCount,
            ef,
            entryPoint,
            entryDistance);

        HnswGraphLayer graphLayer = _layers[layer];
        while (candidateCount > 0)
        {
            HnswQueueItem currentCandidate = HnswPriorityQueues.PopNearest(
                workspace.CandidateOrdinals,
                workspace.CandidateDistances,
                _ids,
                ref candidateCount);
            int current = currentCandidate.Ordinal;
            float currentDistance = currentCandidate.Distance;

            HnswQueueItem worst = HnswPriorityQueues.PeekWorst(workspace.BestOrdinals, workspace.BestDistances, bestCount);
            if (currentDistance > worst.Distance)
            {
                break;
            }

            ReadOnlySpan<int> neighbors = graphLayer.GetNeighbors(current);
            for (int i = 0; i < neighbors.Length; i++)
            {
                int neighbor = neighbors[i];
                if ((uint)neighbor >= (uint)validCount || workspace.VisitMarks[neighbor] == visitMark)
                {
                    continue;
                }

                workspace.VisitMarks[neighbor] = visitMark;
                float distance = SquaredEuclideanDistance(query, neighbor);
                bool accepted = HnswPriorityQueues.AddBoundedNearest(
                    workspace.BestOrdinals,
                    workspace.BestDistances,
                    _ids,
                    ref bestCount,
                    ef,
                    neighbor,
                    distance);
                if (!accepted)
                {
                    continue;
                }

                HnswPriorityQueues.PushNearest(
                    workspace.CandidateOrdinals,
                    workspace.CandidateDistances,
                    _ids,
                    ref candidateCount,
                    neighbor,
                    distance);
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

    private int MarkAllowedOrdinals(ReadOnlySpan<ulong> allowedIds, int[] filterMarks, int filterMark)
    {
        int allowedCount = 0;
        for (int allowIndex = 0; allowIndex < allowedIds.Length; allowIndex++)
        {
            if (!_idToOrdinal.TryGetValue(allowedIds[allowIndex], out int ordinal))
            {
                continue;
            }

            if (filterMarks[ordinal] == filterMark)
            {
                continue;
            }

            filterMarks[ordinal] = filterMark;
            allowedCount++;
        }

        return allowedCount;
    }

    private int SearchAllowedExact(
        ReadOnlySpan<float> query,
        Span<SearchResult> results,
        int[] filterMarks,
        int filterMark)
    {
        int written = 0;
        for (int ordinal = 0; ordinal < _count; ordinal++)
        {
            if (filterMarks[ordinal] != filterMark)
            {
                continue;
            }

            var candidate = new SearchResult(_ids[ordinal], SquaredEuclideanDistance(query, ordinal));
            written = InsertCandidate(results, written, candidate);
        }

        return written;
    }

    private int SearchAllowedByEmission(
        ReadOnlySpan<float> query,
        Span<SearchResult> results,
        HnswSearchWorkspace workspace,
        int filterMark)
    {
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
        int written = 0;
        for (int i = 0; i < candidateCount && written < results.Length; i++)
        {
            int ordinal = workspace.ResultOrdinals[i];
            if (workspace.FilterMarks[ordinal] != filterMark)
            {
                continue;
            }

            results[written++] = new SearchResult(_ids[ordinal], workspace.ResultDistances[i]);
        }

        return written;
    }

    private int SelectNeighbors(
        int baseOrdinal,
        ReadOnlySpan<int> candidates,
        Span<int> selected,
        int layer,
        int validExistingCount,
        NeighborCandidate[] unique)
    {
        int uniqueCount = 0;

        for (int i = 0; i < candidates.Length; i++)
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
        int selectedCount = 0;
        int maxSelected = selected.Length;

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

    private void SetNeighbors(int ordinal, int layer, ReadOnlySpan<int> selected, int selectedCount)
    {
        HnswGraphLayer graphLayer = _layers[layer];
        graphLayer.SetNeighbors(ordinal, selected, selectedCount);
    }

    private void AddOrPruneNeighbor(int baseOrdinal, int newNeighbor, int layer, HnswBuildScratch scratch)
    {
        HnswGraphLayer graphLayer = _layers[layer];
        ReadOnlySpan<int> existingNeighbors = graphLayer.GetNeighbors(baseOrdinal);
        int count = existingNeighbors.Length;

        for (int i = 0; i < count; i++)
        {
            if (existingNeighbors[i] == newNeighbor)
            {
                return;
            }
        }

        if (count < graphLayer.Stride)
        {
            graphLayer.AppendNeighbor(baseOrdinal, newNeighbor);
            return;
        }

        Span<int> candidates = scratch.CandidateOrdinals.AsSpan(0, count + 1);
        existingNeighbors.CopyTo(candidates);
        candidates[count] = newNeighbor;
        int selectedCount = SelectNeighbors(
            baseOrdinal,
            candidates,
            scratch.SelectedOrdinals.AsSpan(0, graphLayer.Stride),
            layer,
            _count + 1,
            scratch.NeighborCandidates);
        SetNeighbors(baseOrdinal, layer, scratch.SelectedOrdinals, selectedCount);
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

    private void SortNearest(int[] ordinals, float[] distances, int count)
    {
        for (int i = 1; i < count; i++)
        {
            int ordinal = ordinals[i];
            float distance = distances[i];
            int j = i - 1;
            while (j >= 0 &&
                   HnswPriorityQueues.CompareNearest(distance, _ids[ordinal], ordinal, distances[j], _ids[ordinals[j]], ordinals[j]) < 0)
            {
                ordinals[j + 1] = ordinals[j];
                distances[j + 1] = distances[j];
                j--;
            }

            ordinals[j + 1] = ordinal;
            distances[j + 1] = distance;
        }
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

    internal static HnswIndex HydrateReadOnly(HnswStorageSnapshot snapshot)
    {
        var index = new HnswIndex(snapshot.Dimension, snapshot.Metric, snapshot.Options)
        {
            _ids = snapshot.Ids,
            _vectors = snapshot.Vectors,
            _levels = snapshot.Levels,
            _layers = new HnswGraphLayer[snapshot.Layers.Length],
            _count = snapshot.Ids.Length,
            _entryPoint = snapshot.EntryPoint,
            _maxLayer = snapshot.MaxLayer,
            _isReadOnly = true
        };

        for (int layer = 0; layer < snapshot.Layers.Length; layer++)
        {
            HnswLayerSnapshot source = snapshot.Layers[layer];
            index._layers[layer] = layer == 0
                ? HnswGraphLayer.CreateDenseFromStorage(
                    source.Stride,
                    snapshot.Ids.Length,
                    source.Counts,
                    source.Neighbors)
                : HnswGraphLayer.CreateSparseFromStorage(
                    source.Stride,
                    snapshot.Ids.Length,
                    source.RowOrdinals,
                    source.Counts,
                    source.Neighbors);
        }

        for (int ordinal = 0; ordinal < snapshot.Ids.Length; ordinal++)
        {
            index._idToOrdinal.Add(snapshot.Ids[ordinal], ordinal);
        }

        return index;
    }

    internal sealed record HnswStorageSnapshot(
        int Dimension,
        VectorMetric Metric,
        HnswIndexOptions Options,
        int MMax,
        int MMax0,
        double LevelMultiplier,
        int EntryPoint,
        int MaxLayer,
        ulong[] Ids,
        float[] Vectors,
        int[] Levels,
        HnswLayerSnapshot[] Layers);

    internal sealed record HnswLayerSnapshot(
        int Stride,
        int[] RowOrdinals,
        int[] Counts,
        int[] Neighbors);

    private sealed class HnswGraphLayer
    {
        private int[] _counts;
        private int[] _neighbors;
        private int[] _ordinalToCompactRow;
        private int[] _rowOrdinals;
        private int _compactRowCount;

        internal HnswGraphLayer(int stride, int capacity, bool sparse)
        {
            Stride = stride;
            IsSparse = sparse;
            if (sparse)
            {
                _ordinalToCompactRow = new int[capacity];
                Array.Fill(_ordinalToCompactRow, -1);
                _rowOrdinals = [];
                _counts = [];
                _neighbors = [];
            }
            else
            {
                _ordinalToCompactRow = [];
                _rowOrdinals = [];
                _counts = new int[capacity];
                _neighbors = new int[checked(capacity * stride)];
            }
        }

        internal int Stride { get; }

        internal bool IsSparse { get; }

        internal int Capacity => IsSparse ? _ordinalToCompactRow.Length : _counts.Length;

        internal int CompactRowCount => IsSparse ? _compactRowCount : _counts.Length;

        internal int CountStorageLength => _counts.Length;

        internal int NeighborStorageLength => _neighbors.Length;

        internal void EnsureCapacity(int capacity)
        {
            if (Capacity >= capacity)
            {
                return;
            }

            if (IsSparse)
            {
                int oldLength = _ordinalToCompactRow.Length;
                Array.Resize(ref _ordinalToCompactRow, capacity);
                _ordinalToCompactRow.AsSpan(oldLength).Fill(-1);
                return;
            }

            var neighbors = new int[checked(capacity * Stride)];
            var counts = new int[capacity];
            _counts.AsSpan().CopyTo(counts);

            int oldCapacity = _counts.Length;
            for (int ordinal = 0; ordinal < oldCapacity; ordinal++)
            {
                _neighbors.AsSpan(ordinal * Stride, _counts[ordinal])
                    .CopyTo(neighbors.AsSpan(ordinal * Stride, _counts[ordinal]));
            }

            _neighbors = neighbors;
            _counts = counts;
        }

        internal ReadOnlySpan<int> GetNeighbors(int ordinal)
        {
            int row = GetCompactRowOrMissing(ordinal);
            if (row < 0)
            {
                return [];
            }

            return _neighbors.AsSpan(row * Stride, _counts[row]);
        }

        internal void SetNeighbors(int ordinal, ReadOnlySpan<int> selected, int selectedCount)
        {
            int row = EnsureRow(ordinal);
            _counts[row] = selectedCount;
            selected[..selectedCount].CopyTo(_neighbors.AsSpan(row * Stride, selectedCount));
        }

        internal void AppendNeighbor(int ordinal, int neighbor)
        {
            int row = EnsureRow(ordinal);
            int count = _counts[row];
            _neighbors[(row * Stride) + count] = neighbor;
            _counts[row] = count + 1;
        }

        internal void EnsureParticipantRow(int ordinal)
        {
            if (!IsSparse)
            {
                return;
            }

            _ = EnsureRow(ordinal);
        }

        internal void LoadDenseRows(int[] counts, int[] neighbors)
        {
            if (IsSparse ||
                counts.Length != _counts.Length ||
                neighbors.Length != _neighbors.Length)
            {
                throw new InvalidOperationException("HNSW graph layer dense payload shape is invalid.");
            }

            counts.CopyTo(_counts, 0);
            neighbors.CopyTo(_neighbors, 0);
        }

        internal static HnswGraphLayer CreateDenseFromStorage(
            int stride,
            int capacity,
            int[] counts,
            int[] neighbors)
        {
            if (counts.Length != capacity || neighbors.Length != checked(capacity * stride))
            {
                throw new InvalidOperationException("HNSW graph layer dense payload shape is invalid.");
            }

            return new HnswGraphLayer(stride, 0, sparse: false)
            {
                _counts = counts,
                _neighbors = neighbors
            };
        }

        internal static HnswGraphLayer CreateSparseFromStorage(
            int stride,
            int capacity,
            int[] rowOrdinals,
            int[] counts,
            int[] neighbors)
        {
            if (rowOrdinals.Length != counts.Length || neighbors.Length != checked(counts.Length * stride))
            {
                throw new InvalidOperationException("HNSW graph layer sparse payload shape is invalid.");
            }

            var graphLayer = new HnswGraphLayer(stride, capacity, sparse: true)
            {
                _rowOrdinals = rowOrdinals,
                _counts = counts,
                _neighbors = neighbors,
                _compactRowCount = rowOrdinals.Length
            };

            for (int compactRow = 0; compactRow < rowOrdinals.Length; compactRow++)
            {
                int ordinal = rowOrdinals[compactRow];
                if ((uint)ordinal >= (uint)capacity || graphLayer._ordinalToCompactRow[ordinal] >= 0)
                {
                    throw new InvalidOperationException("HNSW graph layer compact row ordinals are invalid.");
                }

                graphLayer._ordinalToCompactRow[ordinal] = compactRow;
            }

            return graphLayer;
        }

        private int GetCompactRowOrMissing(int ordinal)
        {
            if (!IsSparse)
            {
                return ordinal;
            }

            return (uint)ordinal < (uint)_ordinalToCompactRow.Length
                ? _ordinalToCompactRow[ordinal]
                : -1;
        }

        private int EnsureRow(int ordinal)
        {
            if (!IsSparse)
            {
                return ordinal;
            }

            if ((uint)ordinal >= (uint)_ordinalToCompactRow.Length)
            {
                EnsureCapacity(ordinal + 1);
            }

            int existingRow = _ordinalToCompactRow[ordinal];
            if (existingRow >= 0)
            {
                return existingRow;
            }

            EnsureCompactCapacity(_compactRowCount + 1);
            int row = _compactRowCount++;
            _ordinalToCompactRow[ordinal] = row;
            _rowOrdinals[row] = ordinal;
            return row;
        }

        private void EnsureCompactCapacity(int requiredRows)
        {
            if (_counts.Length >= requiredRows)
            {
                return;
            }

            int maxRows = Array.MaxLength / Stride;
            int newCapacity = _counts.Length == 0
                ? 4
                : _counts.Length <= maxRows / 2
                    ? _counts.Length * 2
                    : maxRows;
            if (newCapacity < requiredRows)
            {
                newCapacity = requiredRows;
            }

            Array.Resize(ref _rowOrdinals, newCapacity);
            Array.Resize(ref _counts, newCapacity);
            var neighbors = new int[checked(newCapacity * Stride)];
            for (int row = 0; row < _compactRowCount; row++)
            {
                _neighbors.AsSpan(row * Stride, _counts[row])
                    .CopyTo(neighbors.AsSpan(row * Stride, _counts[row]));
            }

            _neighbors = neighbors;
        }
    }

    private sealed class HnswBuildScratch
    {
        internal HnswBuildScratch(
            int maxElements,
            int maxEf,
            int candidateOrdinalCapacity,
            int selectedCapacity,
            int neighborCandidateCapacity)
        {
            Workspace = new HnswSearchWorkspace(maxElements, maxEf);
            CandidateOrdinals = new int[candidateOrdinalCapacity];
            SelectedOrdinals = new int[selectedCapacity];
            NeighborCandidates = new NeighborCandidate[neighborCandidateCapacity];
        }

        internal HnswSearchWorkspace Workspace { get; }

        internal int[] CandidateOrdinals { get; }

        internal int[] SelectedOrdinals { get; }

        internal NeighborCandidate[] NeighborCandidates { get; }
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
            return HnswPriorityQueues.CompareNearest(Distance, Id, Ordinal, other.Distance, other.Id, other.Ordinal);
        }
    }
}
