namespace VecNet;

/// <summary>
/// Result returned by an exact-flat checkpoint operation.
/// </summary>
/// <param name="Status">The checkpoint status.</param>
/// <param name="Generation">The current opaque instance-local generation stamp.</param>
/// <param name="PhysicalVectorCount">The current physical stored-row count.</param>
/// <param name="LiveVectorCount">The current live visible vector count.</param>
/// <param name="BaseVectorCount">The current live base vector count.</param>
/// <param name="DeltaVectorCount">The current live in-memory delta vector count.</param>
/// <param name="TombstoneCount">The current visibility tombstone count.</param>
/// <param name="DeletedReservedIdCount">The current deleted/reserved ID count.</param>
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
