namespace VecNet;

/// <summary>
/// Caller-owned reusable workspace for exact-flat allowlist filtered search.
/// </summary>
public sealed class ExactFlatSearchFilterWorkspace
{
    private int _searchMark;

    /// <summary>
    /// Initializes a reusable filter workspace sized for an exact-flat index vector count.
    /// </summary>
    /// <param name="maxVectorCount">The maximum vector count this workspace can support.</param>
    public ExactFlatSearchFilterWorkspace(int maxVectorCount)
    {
        if (maxVectorCount < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxVectorCount),
                "Workspace vector capacity must not be negative.");
        }

        MaxVectorCount = maxVectorCount;
        RowMarks = new int[maxVectorCount];
    }

    /// <summary>
    /// Gets the maximum exact-flat vector count supported by this workspace.
    /// </summary>
    public int MaxVectorCount { get; }

    internal int[] RowMarks { get; }

    internal int BeginSearch()
    {
        if (_searchMark == int.MaxValue)
        {
            Array.Clear(RowMarks);
            _searchMark = 0;
        }

        _searchMark++;
        return _searchMark;
    }
}
