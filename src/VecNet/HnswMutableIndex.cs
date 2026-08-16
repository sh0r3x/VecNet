namespace VecNet;

/// <summary>
/// Update-oriented approximate HNSW index for squared Euclidean, inner product, and cosine distance.
/// </summary>
/// <remarks>
/// This wrapper searches an immutable HNSW base plus exact in-memory delta rows. Deletes are
/// tombstones over base or delta external IDs. <see cref="Checkpoint"/> rebuilds the live view into
/// a new immutable HNSW snapshot and publishes that rebuilt base in the current instance after
/// validation. This API does not mutate the HNSW graph in place, reopen a durable mutable
/// overlay, expose checkpoint diagnostics, support upsert/replacement/repair, or support
/// concurrent mutation/search or concurrent checkpoint/search.
/// To grow a saved HNSW index, open it with <see cref="HnswIndex.OpenReadOnly(string)"/>, create a
/// mutable wrapper, apply <see cref="TryAdd"/> or <see cref="TryDelete"/>, checkpoint to a new or
/// empty directory, and reopen the checkpoint output as a read-only <see cref="HnswIndex"/>.
/// Checkpoint output is an immutable HNSW snapshot in the current supported format, not durable
/// mutable overlay storage.
/// </remarks>
public sealed class HnswMutableIndex
{
    private readonly HnswBasePlusExactDeltaIndex _inner;

    /// <summary>
    /// Initializes a mutable HNSW wrapper over an existing immutable HNSW base.
    /// </summary>
    /// <param name="baseIndex">
    /// The immutable HNSW base generation. The base must not be modified after constructing this
    /// wrapper.
    /// </param>
    public HnswMutableIndex(HnswIndex baseIndex)
    {
        ArgumentNullException.ThrowIfNull(baseIndex);
        _inner = new HnswBasePlusExactDeltaIndex(baseIndex);
    }

    /// <summary>
    /// Gets the fixed vector dimension accepted by this mutable HNSW index.
    /// </summary>
    public int Dimension => _inner.Dimension;

    /// <summary>
    /// Gets the supported metric. Mutable HNSW supports squared Euclidean, inner product, and cosine distance.
    /// </summary>
    public VectorMetric Metric => _inner.Metric;

    /// <summary>
    /// Gets the HNSW options used for the immutable base and checkpoint rebuilds.
    /// </summary>
    public HnswIndexOptions Options => _inner.Options;

    /// <summary>
    /// Gets the current live visible vector count.
    /// </summary>
    /// <remarks>
    /// Compatibility count name for <see cref="LiveVectorCount"/>. This excludes base and delta
    /// rows hidden by tombstones.
    /// </remarks>
    public int Count => _inner.LiveVectorCount;

    /// <summary>
    /// Gets the current physical vector count in the immutable HNSW base.
    /// </summary>
    /// <remarks>
    /// This includes base rows hidden by tombstones and is used for base-workspace sizing.
    /// </remarks>
    public int BasePhysicalVectorCount => _inner.BasePhysicalVectorCount;

    /// <summary>
    /// Gets the current live vector count in the immutable HNSW base after tombstones.
    /// </summary>
    public int BaseLiveVectorCount => _inner.BaseLiveVectorCount;

    /// <summary>
    /// Gets the current physical in-memory exact delta row count.
    /// </summary>
    /// <remarks>
    /// This includes exact delta rows hidden by tombstones and is used for delta-filter workspace
    /// sizing.
    /// </remarks>
    public int DeltaPhysicalVectorCount => _inner.DeltaPhysicalVectorCount;

    /// <summary>
    /// Gets the current live in-memory exact delta row count.
    /// </summary>
    public int DeltaLiveVectorCount => _inner.DeltaLiveVectorCount;

    /// <summary>
    /// Gets the current live visible vector count.
    /// </summary>
    /// <remarks>
    /// This excludes base and delta rows hidden by tombstones.
    /// </remarks>
    public int LiveVectorCount => _inner.LiveVectorCount;

    /// <summary>
    /// Gets the current base-row tombstone count.
    /// </summary>
    /// <remarks>
    /// Base tombstones hide rows from the immutable HNSW base without mutating the graph in place.
    /// </remarks>
    public int BaseTombstoneCount => _inner.BaseTombstoneCount;

    /// <summary>
    /// Gets the current delta-row tombstone count.
    /// </summary>
    /// <remarks>
    /// Delta tombstones hide rows from the exact in-memory delta.
    /// </remarks>
    public int DeltaTombstoneCount => _inner.DeltaTombstoneCount;

    /// <summary>
    /// Gets the current total tombstone count.
    /// </summary>
    /// <remarks>
    /// This is the sum of base-row and delta-row tombstones.
    /// </remarks>
    public int TombstoneCount => _inner.TombstoneCount;

    /// <summary>
    /// Gets the count of deleted external IDs reserved by this mutable instance.
    /// </summary>
    /// <remarks>
    /// Deleted external IDs remain unavailable for reuse in this mutable instance, including after
    /// checkpoint publication.
    /// </remarks>
    public int DeletedReservedIdCount => _inner.DeletedReservedIdCount;

    /// <summary>
    /// Gets the current opaque instance-local generation stamp.
    /// </summary>
    public long Generation => _inner.Generation;

    /// <summary>
    /// Adds an exact in-memory delta row.
    /// </summary>
    /// <remarks>
    /// This does not insert into or repair the HNSW graph. Duplicate visible IDs and IDs reserved
    /// by prior deletes are reported through <see cref="VectorMutationResult.Status"/>.
    /// </remarks>
    /// <param name="id">The caller-owned external vector identifier.</param>
    /// <param name="vector">The finite vector values to copy into exact delta storage. Inner-product vectors are stored as supplied. Cosine vectors are normalized during insertion.</param>
    /// <returns>A status-reporting mutation result.</returns>
    public VectorMutationResult TryAdd(ulong id, ReadOnlySpan<float> vector) => _inner.TryAdd(id, vector);

    /// <summary>
    /// Adds a tombstone for a base or delta external ID.
    /// </summary>
    /// <remarks>
    /// Deleted IDs remain reserved for this mutable instance. This does not delete or repair HNSW
    /// graph nodes in place.
    /// </remarks>
    /// <param name="id">The caller-owned external vector identifier to hide from search.</param>
    /// <returns>A status-reporting mutation result.</returns>
    public VectorMutationResult TryDelete(ulong id) => _inner.TryDelete(id);

    /// <summary>
    /// Creates a mutable HNSW search workspace sized for the current generation and configured search width.
    /// </summary>
    /// <param name="maxResults">The maximum result buffer length this workspace can support.</param>
    /// <returns>A caller-owned mutable HNSW workspace for the current generation.</returns>
    /// <remarks>
    /// This factory is equivalent to <see cref="HnswMutableSearchWorkspace(HnswMutableIndex, int)"/>.
    /// Recreate the workspace after a committed mutation or published checkpoint.
    /// </remarks>
    public HnswMutableSearchWorkspace CreateSearchWorkspace(int maxResults) =>
        new(this, maxResults);

    /// <summary>
    /// Creates a mutable HNSW search workspace sized for the current generation and a caller-selected maximum search width.
    /// </summary>
    /// <param name="maxResults">The maximum result buffer length this workspace can support.</param>
    /// <param name="maxEfSearch">The maximum per-search HNSW base candidate width this workspace can support.</param>
    /// <returns>A caller-owned mutable HNSW workspace for the current generation.</returns>
    /// <remarks>
    /// <paramref name="maxEfSearch"/> must be in the range [1, 4096]. Recreate the workspace after
    /// a committed mutation or published checkpoint.
    /// </remarks>
    public HnswMutableSearchWorkspace CreateSearchWorkspace(int maxResults, int maxEfSearch) =>
        new(this, maxResults, maxEfSearch);

    /// <summary>
    /// Searches the immutable HNSW base plus exact in-memory delta rows.
    /// </summary>
    /// <remarks>
    /// Base candidates are approximate HNSW candidates with tombstones suppressed. Delta rows are
    /// searched exactly, then base and delta candidates are merged by canonical distance and
    /// external-ID ties. The caller owns the result buffer and workspace.
    /// </remarks>
    /// <param name="query">The finite query vector. Inner-product queries are used as supplied. Cosine queries are normalized during search.</param>
    /// <param name="results">The caller-owned destination buffer. Its length is the requested result count.</param>
    /// <param name="workspace">The caller-owned mutable HNSW workspace for this generation.</param>
    /// <returns>The number of results written.</returns>
    public int Search(ReadOnlySpan<float> query, Span<SearchResult> results, HnswMutableSearchWorkspace workspace)
    {
        ValidateWorkspace(results.Length, workspace);
        return _inner.Search(query, results, workspace.Inner);
    }

    /// <summary>
    /// Searches the immutable HNSW base plus exact in-memory delta rows with a caller-selected HNSW base search width.
    /// </summary>
    /// <remarks>
    /// <paramref name="efSearch"/> must be in the range [1, 4096] and must be at least the
    /// requested result count. This per-search width controls the HNSW base traversal for the
    /// current query and does not change <see cref="Options"/>. Exact delta search remains bounded
    /// by the requested result count.
    /// </remarks>
    /// <param name="query">The finite query vector. Inner-product queries are used as supplied. Cosine queries are normalized during search.</param>
    /// <param name="results">The caller-owned destination buffer. Its length is the requested result count.</param>
    /// <param name="workspace">The caller-owned mutable HNSW workspace for this generation.</param>
    /// <param name="efSearch">The HNSW base candidate width for this search.</param>
    /// <returns>The number of results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        Span<SearchResult> results,
        HnswMutableSearchWorkspace workspace,
        int efSearch)
    {
        ValidateEfSearch(efSearch, nameof(efSearch));
        ValidateWorkspace(results.Length, workspace, efSearch, nameof(results), requireDeltaFilterCapacity: false);
        return _inner.Search(query, results, workspace.Inner, efSearch);
    }

    /// <summary>
    /// Searches the immutable HNSW base plus exact delta rows while emitting only allowed external IDs.
    /// </summary>
    /// <remarks>
    /// The allowlist is caller-owned query input. Unknown IDs are ignored and duplicates are
    /// coalesced. When the known live allowed count is no greater than <see cref="HnswIndexOptions.EfSearch"/>,
    /// this method uses exact filtered fallback over base and delta rows. For broader allowlists,
    /// HNSW base traversal remains unfiltered with emission suppression and may underfill; live
    /// allowed delta rows are still searched exactly.
    /// </remarks>
    /// <param name="query">The finite query vector. Inner-product queries are used as supplied. Cosine queries are normalized during search.</param>
    /// <param name="allowedIds">Caller-supplied external identifiers allowed for this search.</param>
    /// <param name="results">The caller-owned destination buffer. Its length is the requested result count.</param>
    /// <param name="workspace">The caller-owned mutable HNSW workspace for this generation.</param>
    /// <returns>The number of filtered results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        ReadOnlySpan<ulong> allowedIds,
        Span<SearchResult> results,
        HnswMutableSearchWorkspace workspace)
    {
        ValidateWorkspace(results.Length, workspace);
        return _inner.Search(query, allowedIds, results, workspace.Inner);
    }

    /// <summary>
    /// Searches the immutable HNSW base plus exact delta rows with a caller-selected HNSW base search width while emitting only allowed external IDs.
    /// </summary>
    /// <remarks>
    /// <paramref name="efSearch"/> must be in the range [1, 4096] and must be at least the
    /// requested result count. Unknown IDs, duplicates and tombstoned IDs are ignored. When the
    /// known live allowed count is no greater than <paramref name="efSearch"/>, this method uses
    /// exact filtered fallback over live base and delta rows. For broader allowlists, HNSW base
    /// traversal remains unfiltered with emission suppression and may underfill; live allowed delta
    /// rows are still searched exactly.
    /// </remarks>
    /// <param name="query">The finite query vector. Inner-product queries are used as supplied. Cosine queries are normalized during search.</param>
    /// <param name="allowedIds">Caller-supplied external identifiers allowed for this search.</param>
    /// <param name="results">The caller-owned destination buffer. Its length is the requested result count.</param>
    /// <param name="workspace">The caller-owned mutable HNSW workspace for this generation.</param>
    /// <param name="efSearch">The HNSW base candidate width for this search.</param>
    /// <returns>The number of filtered results written.</returns>
    public int Search(
        ReadOnlySpan<float> query,
        ReadOnlySpan<ulong> allowedIds,
        Span<SearchResult> results,
        HnswMutableSearchWorkspace workspace,
        int efSearch)
    {
        ValidateEfSearch(efSearch, nameof(efSearch));
        ValidateWorkspace(results.Length, workspace, efSearch, nameof(results), requireDeltaFilterCapacity: true);
        return _inner.Search(query, allowedIds, results, workspace.Inner, efSearch);
    }

    /// <summary>
    /// Rebuilds the current live view into a new immutable HNSW snapshot.
    /// </summary>
    /// <remarks>
    /// Checkpoint writes an immutable HNSW snapshot to a new or empty directory, validates the
    /// opened output, and publishes the rebuilt immutable HNSW base in this mutable instance. It
    /// folds exact delta rows and tombstones into the rebuilt base. Mutable overlay state is not
    /// durable mutable storage and is not durably reopened, and
    /// diagnostic timing/allocation phases are not part of this public API. It does not edit the
    /// original base directory or any existing durable directory in place.
    /// </remarks>
    /// <param name="directoryPath">The new or empty output directory path.</param>
    /// <returns>A checkpoint result.</returns>
    public HnswMutableCheckpointResult Checkpoint(string directoryPath) =>
        FromInternal(_inner.Checkpoint(directoryPath));

    private void ValidateWorkspace(int requestedResultCount, HnswMutableSearchWorkspace workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        if (workspace.Generation != Generation)
        {
            throw new InvalidOperationException("Workspace generation is stale for this mutable HNSW index.");
        }

        if (workspace.MaxDeltaCandidates < requestedResultCount)
        {
            throw new ArgumentException("Workspace result capacity is smaller than the requested result count.", nameof(workspace));
        }
    }

    private void ValidateWorkspace(
        int requestedResultCount,
        HnswMutableSearchWorkspace workspace,
        int effectiveEfSearch,
        string resultsParameterName,
        bool requireDeltaFilterCapacity)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        if (workspace.Generation != Generation)
        {
            throw new InvalidOperationException("Workspace generation is stale for this mutable HNSW index.");
        }

        if (effectiveEfSearch < requestedResultCount)
        {
            throw new ArgumentOutOfRangeException(resultsParameterName, "EfSearch must be at least the requested result count.");
        }

        if (workspace.MaxDeltaCandidates < requestedResultCount)
        {
            throw new ArgumentException("Workspace result capacity is smaller than the requested result count.", nameof(workspace));
        }

        if (workspace.MaxBaseElements < BasePhysicalVectorCount)
        {
            throw new ArgumentException("Workspace base element capacity is smaller than the immutable HNSW base count.", nameof(workspace));
        }

        if (workspace.MaxEfSearch < effectiveEfSearch)
        {
            throw new ArgumentException("Workspace HNSW ef capacity is smaller than EfSearch.", nameof(workspace));
        }

        if (workspace.MaxBaseCandidates < Math.Min(BasePhysicalVectorCount, effectiveEfSearch))
        {
            throw new ArgumentException("Workspace base candidate capacity is smaller than the requested HNSW base overfetch count.", nameof(workspace));
        }

        if (requireDeltaFilterCapacity && workspace.MaxDeltaFilterElements < DeltaPhysicalVectorCount)
        {
            throw new ArgumentException("Workspace delta filter capacity is smaller than the physical delta count.", nameof(workspace));
        }
    }

    private static void ValidateEfSearch(int efSearch, string parameterName)
    {
        if (efSearch is < 1 or > 4096)
        {
            throw new ArgumentOutOfRangeException(parameterName, "EfSearch must be in the range [1, 4096].");
        }
    }

    private static HnswMutableCheckpointResult FromInternal(HnswBasePlusExactDeltaCheckpointResult result) =>
        new(
            result.Status == HnswBasePlusExactDeltaCheckpointStatus.Published
                ? HnswMutableCheckpointStatus.Published
                : HnswMutableCheckpointStatus.NoChanges,
            result.Generation,
            result.RebuiltBaseVectorCount,
            result.LiveVectorCount,
            result.BasePhysicalVectorCount,
            result.BaseLiveVectorCount,
            result.DeltaPhysicalVectorCount,
            result.DeltaLiveVectorCount,
            result.BaseTombstoneCount,
            result.DeltaTombstoneCount,
            result.TombstoneCount,
            result.DeletedReservedIdCount,
            result.FoldedDeltaVectorCount,
            result.FoldedBaseTombstoneCount,
            result.FoldedDeltaTombstoneCount);
}
