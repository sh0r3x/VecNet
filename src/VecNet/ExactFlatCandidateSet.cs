using System.Diagnostics;

namespace VecNet;

/// <summary>
/// Opaque reusable candidate set for exact-flat filtered search.
/// </summary>
/// <remarks>
/// Candidate sets are caller-owned immutable search inputs created by
/// <see cref="ExactFlatIndex.CreateCandidateSet(ReadOnlySpan{ulong})"/>. They are bound to the
/// creating index instance and its generation; using a candidate set with the wrong index or after
/// the creating index changes is rejected before results are written.
/// </remarks>
[DebuggerDisplay("Count = {Count}")]
public sealed class ExactFlatCandidateSet
{
    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    private readonly ExactFlatIndex _owner;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    private readonly long _generation;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    private readonly int[] _rowOrdinals;

    internal ExactFlatCandidateSet(ExactFlatIndex owner, long generation, int[] rowOrdinals)
    {
        _owner = owner;
        _generation = generation;
        _rowOrdinals = rowOrdinals;
    }

    /// <summary>
    /// Gets the number of known, distinct external IDs represented by this candidate set.
    /// </summary>
    public int Count => _rowOrdinals.Length;

    internal ExactFlatIndex Owner => _owner;

    internal long Generation => _generation;

    internal ReadOnlySpan<int> RowOrdinals => _rowOrdinals;
}
