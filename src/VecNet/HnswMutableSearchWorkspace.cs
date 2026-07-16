namespace VecNet;

/// <summary>
/// Caller-owned reusable workspace for mutable HNSW search.
/// </summary>
/// <remarks>
/// A workspace stores transient HNSW and exact-delta search state for one search at a time. It is
/// bound to the <see cref="HnswMutableIndex.Generation"/> observed when it is created. Recreate it
/// after a committed <see cref="HnswMutableIndex.TryAdd"/>, committed
/// <see cref="HnswMutableIndex.TryDelete"/>, or published <see cref="HnswMutableIndex.Checkpoint"/>.
/// Do not share one workspace across overlapping searches.
/// </remarks>
public sealed class HnswMutableSearchWorkspace
{
    /// <summary>
    /// Initializes a workspace from the current mutable HNSW index shape.
    /// </summary>
    /// <param name="index">The mutable HNSW index this workspace will be used with.</param>
    /// <param name="maxResults">The maximum result buffer length this workspace can support.</param>
    public HnswMutableSearchWorkspace(HnswMutableIndex index, int maxResults)
    {
        ArgumentNullException.ThrowIfNull(index);
        if (maxResults < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxResults), "Workspace result capacity must not be negative.");
        }

        Generation = index.Generation;
        MaxBaseElements = index.BasePhysicalVectorCount;
        MaxEfSearch = index.Options.EfSearch;
        MaxBaseCandidates = Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch);
        MaxDeltaCandidates = maxResults;
        MaxDeltaFilterElements = index.DeltaPhysicalVectorCount;
        Inner = new HnswBasePlusExactDeltaSearchWorkspace(
            MaxBaseElements,
            MaxEfSearch,
            MaxBaseCandidates,
            MaxDeltaCandidates,
            MaxDeltaFilterElements);
    }

    /// <summary>
    /// Gets the mutable HNSW generation this workspace was created for.
    /// </summary>
    public long Generation { get; }

    /// <summary>
    /// Gets the maximum immutable HNSW base count supported by this workspace.
    /// </summary>
    public int MaxBaseElements { get; }

    /// <summary>
    /// Gets the maximum HNSW search candidate width supported by this workspace.
    /// </summary>
    public int MaxEfSearch { get; }

    /// <summary>
    /// Gets the maximum base-candidate buffer length supported by this workspace.
    /// </summary>
    public int MaxBaseCandidates { get; }

    /// <summary>
    /// Gets the maximum exact-delta candidate count supported by this workspace.
    /// </summary>
    public int MaxDeltaCandidates { get; }

    /// <summary>
    /// Gets the maximum physical delta-row count supported for allowlist filtering.
    /// </summary>
    public int MaxDeltaFilterElements { get; }

    internal HnswBasePlusExactDeltaSearchWorkspace Inner { get; }
}
