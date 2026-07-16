namespace VecNet;

/// <summary>
/// Status returned by <see cref="HnswMutableIndex.Checkpoint"/>.
/// </summary>
public enum HnswMutableCheckpointStatus
{
    /// <summary>
    /// Durable output was written and validated, then a rebuilt immutable HNSW base was published.
    /// </summary>
    Published,

    /// <summary>
    /// No exact delta rows or tombstones needed to be folded into a rebuilt HNSW base.
    /// </summary>
    NoChanges
}
