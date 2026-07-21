namespace VecNet;

/// <summary>
/// Result returned by <see cref="HnswMutableIndex.Checkpoint"/>.
/// </summary>
/// <param name="Status">The checkpoint status.</param>
/// <param name="Generation">The current opaque instance-local generation stamp.</param>
/// <param name="RebuiltBaseVectorCount">The physical/live vector count in the rebuilt immutable HNSW base.</param>
/// <param name="LiveVectorCount">The current searchable live visible vector count after this result was produced.</param>
/// <param name="BasePhysicalVectorCount">The physical vector count in the immutable HNSW base after this result was produced.</param>
/// <param name="BaseLiveVectorCount">The searchable live vector count in the immutable HNSW base after tombstones.</param>
/// <param name="DeltaPhysicalVectorCount">The physical in-memory exact delta row count after this result was produced.</param>
/// <param name="DeltaLiveVectorCount">The searchable live in-memory exact delta row count after this result was produced.</param>
/// <param name="BaseTombstoneCount">The number of base-row tombstones after this result was produced.</param>
/// <param name="DeltaTombstoneCount">The number of delta-row tombstones after this result was produced.</param>
/// <param name="TombstoneCount">The total deleted-row tombstone count hidden from search after this result was produced.</param>
/// <param name="DeletedReservedIdCount">The count of deleted external IDs reserved by this mutable instance.</param>
/// <param name="FoldedDeltaVectorCount">The live exact delta rows folded into the rebuilt base.</param>
/// <param name="FoldedBaseTombstoneCount">The base tombstones folded into the rebuilt base.</param>
/// <param name="FoldedDeltaTombstoneCount">The delta tombstones folded into the rebuilt base.</param>
/// <remarks>
/// These counts describe the mutable HNSW checkpoint contract. They are not public
/// performance, memory, allocation, capacity, storage-size, or stable file-format claims.
/// </remarks>
public readonly record struct HnswMutableCheckpointResult(
    HnswMutableCheckpointStatus Status,
    long Generation,
    int RebuiltBaseVectorCount,
    int LiveVectorCount,
    int BasePhysicalVectorCount,
    int BaseLiveVectorCount,
    int DeltaPhysicalVectorCount,
    int DeltaLiveVectorCount,
    int BaseTombstoneCount,
    int DeltaTombstoneCount,
    int TombstoneCount,
    int DeletedReservedIdCount,
    int FoldedDeltaVectorCount,
    int FoldedBaseTombstoneCount,
    int FoldedDeltaTombstoneCount);
