namespace VecNet;

/// <summary>
/// Result returned by an exact-flat checkpoint operation.
/// </summary>
/// <param name="Status">The checkpoint status.</param>
/// <param name="Generation">The current opaque instance-local generation stamp.</param>
/// <param name="PhysicalVectorCount">The current physical stored-row count after this result was produced.</param>
/// <param name="LiveVectorCount">The current searchable live visible vector count after this result was produced.</param>
/// <param name="BaseVectorCount">The current searchable live base vector count after this result was produced.</param>
/// <param name="DeltaVectorCount">The current searchable live in-memory delta vector count after this result was produced.</param>
/// <param name="TombstoneCount">The current deleted-row tombstone count hidden from search after this result was produced.</param>
/// <param name="DeletedReservedIdCount">The current deleted external ID reservation count after this result was produced.</param>
/// <param name="FoldedDeltaVectorCount">The live delta vector count folded by this checkpoint attempt.</param>
/// <param name="FoldedTombstoneCount">The visibility tombstone count folded by this checkpoint attempt.</param>
public readonly record struct ExactFlatCheckpointResult(
    ExactFlatCheckpointStatus Status,
    long Generation,
    int PhysicalVectorCount,
    int LiveVectorCount,
    int BaseVectorCount,
    int DeltaVectorCount,
    int TombstoneCount,
    int DeletedReservedIdCount,
    int FoldedDeltaVectorCount,
    int FoldedTombstoneCount);
