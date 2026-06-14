namespace VecNet;

internal readonly record struct HnswIndexOptions(
    int M,
    int EfConstruction,
    int EfSearch,
    ulong RandomSeed)
{
    internal static HnswIndexOptions Default { get; } = new(16, 200, 50, 0x564543_034UL);
}
