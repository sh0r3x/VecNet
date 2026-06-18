namespace VecNet;

/// <summary>
/// Result returned by a status-reporting vector mutation operation.
/// </summary>
/// <param name="Status">The mutation status.</param>
/// <param name="Generation">The current opaque instance-local generation stamp.</param>
/// <param name="VectorCount">The current live vector count.</param>
/// <param name="DeltaCount">The current live in-memory delta vector count.</param>
/// <param name="TombstoneCount">The current tombstone count.</param>
public readonly record struct VectorMutationResult(
    VectorMutationStatus Status,
    long Generation,
    int VectorCount,
    int DeltaCount,
    int TombstoneCount);
