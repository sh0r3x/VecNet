namespace VecNet;

/// <summary>
/// Caller-owned reusable workspace for exact-flat allowlist filtered search.
/// </summary>
/// <remarks>
/// A workspace stores transient row marks for one exact-flat search at a time. The caller owns
/// its lifetime and must provide separate workspace instances for concurrent searches. Prefer
/// <see cref="ExactFlatIndex.CreateSearchFilterWorkspace"/> when the workspace is tied to a
/// specific index instance.
/// </remarks>
public sealed class ExactFlatSearchFilterWorkspace
{
    private int _searchMark;

    /// <summary>
    /// Initializes a reusable filter workspace sized for an exact-flat physical vector count.
    /// </summary>
    /// <param name="maxVectorCount">
    /// The maximum physical stored-row count this workspace can support. Use
    /// <see cref="ExactFlatIndex.PhysicalVectorCount"/> or <see cref="ExactFlatIndex.VectorCount"/>
    /// for the index being searched, not <see cref="ExactFlatIndex.LiveVectorCount"/>.
    /// </param>
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
    /// Gets the maximum exact-flat physical stored-row count supported by this workspace.
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
