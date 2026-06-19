namespace VecNet;

/// <summary>
/// Status returned by an exact-flat checkpoint operation.
/// </summary>
public enum ExactFlatCheckpointStatus
{
    /// <summary>
    /// Durable output was written and validated, then a compact in-memory generation was published.
    /// </summary>
    Published,

    /// <summary>
    /// There were no live delta rows or visibility tombstones to fold.
    /// </summary>
    NoChanges
}
