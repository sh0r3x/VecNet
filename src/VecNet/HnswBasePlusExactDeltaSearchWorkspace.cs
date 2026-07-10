namespace VecNet;

internal sealed class HnswBasePlusExactDeltaSearchWorkspace
{
    internal HnswBasePlusExactDeltaSearchWorkspace(
        int maxBaseElements,
        int maxEfSearch,
        int maxBaseCandidates,
        int maxDeltaCandidates,
        int maxDeltaFilterElements = 0)
    {
        if (maxBaseElements < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBaseElements), "Workspace base element capacity must not be negative.");
        }

        if (maxEfSearch <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxEfSearch), "Workspace ef capacity must be positive.");
        }

        if (maxBaseCandidates < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBaseCandidates), "Workspace base candidate capacity must not be negative.");
        }

        if (maxDeltaCandidates < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDeltaCandidates), "Workspace delta candidate capacity must not be negative.");
        }

        if (maxDeltaFilterElements < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDeltaFilterElements), "Workspace delta filter capacity must not be negative.");
        }

        HnswWorkspace = new HnswSearchWorkspace(maxBaseElements, maxEfSearch);
        BaseCandidates = new SearchResult[maxBaseCandidates];
        DeltaCandidates = new SearchResult[maxDeltaCandidates];
        DeltaFilterMarks = new int[maxDeltaFilterElements];
    }

    internal HnswSearchWorkspace HnswWorkspace { get; }

    internal SearchResult[] BaseCandidates { get; }

    internal SearchResult[] DeltaCandidates { get; }

    internal int[] DeltaFilterMarks { get; }

    internal long ObservedGeneration { get; set; } = long.MinValue;

    private int _deltaFilterMark;

    internal int BeginDeltaFilter()
    {
        if (_deltaFilterMark == int.MaxValue)
        {
            Array.Clear(DeltaFilterMarks);
            _deltaFilterMark = 0;
        }

        _deltaFilterMark++;
        return _deltaFilterMark;
    }
}
