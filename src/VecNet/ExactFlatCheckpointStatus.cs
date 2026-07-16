namespace VecNet;

/// <summary>
/// Broad status returned by an exact-flat checkpoint operation.
/// </summary>
/// <remarks>
/// These values describe current exact-flat checkpoint outcomes and are not a complete stable
/// exception or failure taxonomy.
/// </remarks>
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
