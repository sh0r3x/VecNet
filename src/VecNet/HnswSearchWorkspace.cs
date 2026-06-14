namespace VecNet;

internal sealed class HnswSearchWorkspace
{
    private int _visitMark;

    internal HnswSearchWorkspace(int maxElements, int maxEf)
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

    internal int MaxElements { get; }

    internal int MaxEf { get; }

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
