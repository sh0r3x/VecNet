using System.ComponentModel;

namespace VecNet;

/// <summary>
/// Result returned by a status-reporting exact vector mutation operation.
/// </summary>
/// <param name="Status">The broad mutation status.</param>
/// <param name="Generation">The current opaque instance-local generation stamp.</param>
/// <param name="LiveVectorCount">The current live visible vector count.</param>
/// <param name="DeltaVectorCount">The current live in-memory delta vector count.</param>
/// <param name="TombstoneCount">The current tombstone count.</param>
public readonly record struct VectorMutationResult(
    VectorMutationStatus Status,
    long Generation,
    int LiveVectorCount,
    int DeltaVectorCount,
    int TombstoneCount)
{
    /// <summary>
    /// Gets the current live visible vector count.
    /// </summary>
    /// <remarks>
    /// Compatibility alias for <see cref="LiveVectorCount"/>. This value does not have the same
    /// meaning as <see cref="ExactFlatIndex.VectorCount"/>, which reports physical stored rows.
    /// </remarks>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public int VectorCount => LiveVectorCount;

    /// <summary>
    /// Gets the current live in-memory delta vector count.
    /// </summary>
    /// <remarks>
    /// Compatibility alias for <see cref="DeltaVectorCount"/>.
    /// </remarks>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public int DeltaCount => DeltaVectorCount;
}
