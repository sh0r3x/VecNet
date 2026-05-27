namespace VecNet;

/// <summary>
/// A nearest-neighbor result using canonical ascending distance semantics.
/// </summary>
/// <param name="Id">The caller-provided external vector identifier.</param>
/// <param name="Distance">The canonical distance; lower values rank first.</param>
public readonly record struct SearchResult(ulong Id, float Distance);
