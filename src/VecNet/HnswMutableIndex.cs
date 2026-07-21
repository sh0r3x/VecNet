namespace VecNet;

/// <summary>
/// Update-oriented approximate HNSW index for squared Euclidean distance.
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
/// </remarks>
public sealed class HnswMutableIndex
{
    private readonly HnswBasePlusExactDeltaIndex _inner;

    /// <summary>
    /// Initializes a mutable HNSW wrapper over an existing immutable squared-L2 HNSW base.
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
    /// Gets the supported metric. Mutable HNSW supports only squared Euclidean distance.
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
    /// <param name="vector">The finite squared-L2 vector values to copy into exact delta storage.</param>
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
    /// Searches the immutable HNSW base plus exact in-memory delta rows.
    /// </summary>
    /// <remarks>
    /// Base candidates are approximate HNSW candidates with tombstones suppressed. Delta rows are
    /// searched exactly, then base and delta candidates are merged by squared-L2 distance and
    /// external-ID ties. The caller owns the result buffer and workspace.
    /// </remarks>
    /// <param name="query">The finite squared-L2 query vector.</param>
    /// <param name="results">The caller-owned destination buffer. Its length is the requested result count.</param>
    /// <param name="workspace">The caller-owned mutable HNSW workspace for this generation.</param>
    /// <returns>The number of results written.</returns>
    public int Search(ReadOnlySpan<float> query, Span<SearchResult> results, HnswMutableSearchWorkspace workspace)
    {
        ValidateWorkspace(results.Length, workspace);
        return _inner.Search(query, results, workspace.Inner);
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
    /// <param name="query">The finite squared-L2 query vector.</param>
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
    /// Rebuilds the current live view into a new immutable HNSW snapshot.
    /// </summary>
    /// <remarks>
    /// Checkpoint writes to a new or empty directory, validates the opened output, and publishes the
    /// rebuilt immutable HNSW base in this mutable instance. It folds exact delta rows and
    /// tombstones into the rebuilt base. Mutable overlay state is not durably reopened, and
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
