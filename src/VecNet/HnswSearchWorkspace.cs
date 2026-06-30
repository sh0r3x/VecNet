namespace VecNet;

/// <summary>
/// Caller-owned reusable workspace for preview HNSW search.
/// </summary>
/// <remarks>
/// A workspace stores transient visited-set and candidate/result queue state for one search at a
/// time. The caller owns its lifetime and must provide separate workspace instances for
/// overlapping searches.
/// </remarks>
public sealed class HnswSearchWorkspace
{
    private int _visitMark;

    /// <summary>
    /// Initializes a reusable HNSW search workspace.
    /// </summary>
    /// <param name="maxElements">
    /// The maximum HNSW index <see cref="HnswIndex.Count"/> this workspace can support.
    /// </param>
    /// <param name="maxEf">
    /// The maximum search candidate width this workspace can support. Use at least
    /// <see cref="HnswIndexOptions.EfSearch"/> for the index being searched.
    /// </param>
    public HnswSearchWorkspace(int maxElements, int maxEf)
    {
        if (maxElements < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxElements), "Workspace element capacity must not be negative.");
        }

        if (maxEf <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxEf), "Workspace ef capacity must be positive.");
        }

        MaxElements = maxElements;
        MaxEf = maxEf;
        VisitMarks = new int[maxElements];
        CandidateOrdinals = new int[maxElements];
        CandidateDistances = new float[maxElements];
        BestOrdinals = new int[maxEf];
        BestDistances = new float[maxEf];
        ResultOrdinals = new int[maxEf];
        ResultDistances = new float[maxEf];
    }

    /// <summary>
    /// Gets the maximum HNSW index count supported by this workspace.
    /// </summary>
    public int MaxElements { get; }

    /// <summary>
    /// Gets the maximum search candidate width supported by this workspace.
    /// </summary>
    public int MaxEf { get; }

    internal int CurrentVisitMark => _visitMark;

    internal int[] VisitMarks { get; }

    internal int[] CandidateOrdinals { get; }

    internal float[] CandidateDistances { get; }

    internal int[] BestOrdinals { get; }

    internal float[] BestDistances { get; }

    internal int[] ResultOrdinals { get; }

    internal float[] ResultDistances { get; }

    internal int BeginSearch()
    {
        if (_visitMark == int.MaxValue)
        {
            Array.Clear(VisitMarks);
            _visitMark = 0;
        }

        _visitMark++;
        return _visitMark;
    }
}
