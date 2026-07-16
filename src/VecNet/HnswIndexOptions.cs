namespace VecNet;

/// <summary>
/// Configuration values for a squared-L2 <see cref="HnswIndex"/>.
/// </summary>
/// <param name="M">
/// The maximum upper-layer neighbor count. The supported validation range is 2 through 64.
/// Layer zero uses twice this value internally.
/// </param>
/// <param name="EfConstruction">
/// The build-time candidate width. It must be at least <paramref name="M"/> and no more than 4096.
/// Larger values can change build cost and approximate result quality.
/// </param>
/// <param name="EfSearch">
/// The search-time candidate width. It must be 1 through 4096 and at least the requested result
/// count for each search.
/// </param>
/// <param name="RandomSeed">
/// The deterministic level-generation seed. The seed and insertion order affect graph shape and
/// approximate result ordering; graph identity is not a stable compatibility contract.
/// </param>
public readonly record struct HnswIndexOptions(
    int M,
    int EfConstruction,
    int EfSearch,
    ulong RandomSeed)
{
    /// <summary>
    /// Gets the current default HNSW options.
    /// </summary>
    /// <remarks>
    /// These defaults are current release defaults, not a stable tuning promise or a public performance,
    /// recall, memory, allocation, capacity, or storage-size claim.
    /// </remarks>
    public static HnswIndexOptions Default { get; } = new(16, 200, 50, 0x564543_034UL);
}
