using System.Globalization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace VecNet.Benchmarks;

public enum ExactTopKCandidateStream
{
    AlreadySorted,
    ReverseSorted,
    Random,
    DuplicateDistance,
    EqualDistanceWithIdTie
}

public enum ExactTopKSelectionStrategy
{
    SortedSpanInsertion,
    BoundedMaxHeap,
    PartialSelection
}

[MemoryDiagnoser]
[ShortRunJob]
public class ExactTopKStrategyBenchmarks
{
    private static readonly Comparison<SearchResult> s_comparison = Compare;

    private SearchResult[] _candidates = null!;
    private SearchResult[] _reference = null!;
    private SearchResult[] _results = null!;
    private SearchResult[] _heap = null!;
    private SearchResult[] _partialWorkspace = null!;

    [Params(1, 10, 100, 1000)]
    public int ResultCount { get; set; }

    [Params(1024, 10000, 100000)]
    public int CandidateCount { get; set; }

    [Params(
        ExactTopKCandidateStream.AlreadySorted,
        ExactTopKCandidateStream.ReverseSorted,
        ExactTopKCandidateStream.Random,
        ExactTopKCandidateStream.DuplicateDistance,
        ExactTopKCandidateStream.EqualDistanceWithIdTie)]
    public ExactTopKCandidateStream Stream { get; set; }

    [Params(
        ExactTopKSelectionStrategy.SortedSpanInsertion,
        ExactTopKSelectionStrategy.BoundedMaxHeap,
        ExactTopKSelectionStrategy.PartialSelection)]
    public ExactTopKSelectionStrategy Strategy { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        if (ResultCount > CandidateCount)
        {
            throw new InvalidOperationException("Top-k must not exceed candidate count.");
        }

        _candidates = CreateCandidates(CandidateCount, Stream);
        _reference = CreateReference(_candidates, ResultCount);
        _results = new SearchResult[ResultCount];
        _heap = new SearchResult[ResultCount];
        _partialWorkspace = new SearchResult[CandidateCount];

        ValidateStrategy(ExactTopKSelectionStrategy.SortedSpanInsertion);
        ValidateStrategy(ExactTopKSelectionStrategy.BoundedMaxHeap);
        ValidateStrategy(ExactTopKSelectionStrategy.PartialSelection);

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"VEC-162 setup: Candidates={CandidateCount}, TopK={ResultCount}, Stream={Stream}, " +
            $"Strategy={Strategy}, Validation=passed"));
    }

    [Benchmark]
    public int SelectTopK()
    {
        return Strategy switch
        {
            ExactTopKSelectionStrategy.SortedSpanInsertion =>
                SelectSortedSpanInsertion(_candidates, _results),
            ExactTopKSelectionStrategy.BoundedMaxHeap =>
                SelectBoundedMaxHeap(_candidates, _results, _heap),
            ExactTopKSelectionStrategy.PartialSelection =>
                SelectPartialSelection(_candidates, _results, _partialWorkspace),
            _ => throw new InvalidOperationException("Unknown top-k selection strategy.")
        };
    }

    private void ValidateStrategy(ExactTopKSelectionStrategy strategy)
    {
        Array.Clear(_results);
        Array.Clear(_heap);
        Array.Clear(_partialWorkspace);

        int written = strategy switch
        {
            ExactTopKSelectionStrategy.SortedSpanInsertion =>
                SelectSortedSpanInsertion(_candidates, _results),
            ExactTopKSelectionStrategy.BoundedMaxHeap =>
                SelectBoundedMaxHeap(_candidates, _results, _heap),
            ExactTopKSelectionStrategy.PartialSelection =>
                SelectPartialSelection(_candidates, _results, _partialWorkspace),
            _ => throw new InvalidOperationException("Unknown top-k selection strategy.")
        };

        if (written != _reference.Length)
        {
            throw new InvalidOperationException(
                $"{strategy} wrote {written} results; expected {_reference.Length}.");
        }

        for (int i = 0; i < _reference.Length; i++)
        {
            if (_results[i].Id != _reference[i].Id ||
                _results[i].Distance != _reference[i].Distance)
            {
                throw new InvalidOperationException(
                    $"{strategy} mismatch at {i}: got ({_results[i].Id}, {_results[i].Distance:R}), " +
                    $"expected ({_reference[i].Id}, {_reference[i].Distance:R}).");
            }
        }
    }

    internal static SearchResult[] CreateCandidates(int count, ExactTopKCandidateStream stream)
    {
        var candidates = new SearchResult[count];
        switch (stream)
        {
            case ExactTopKCandidateStream.AlreadySorted:
                for (int i = 0; i < count; i++)
                {
                    candidates[i] = new SearchResult((ulong)i, i);
                }

                break;

            case ExactTopKCandidateStream.ReverseSorted:
                for (int i = 0; i < count; i++)
                {
                    int rank = count - i - 1;
                    candidates[i] = new SearchResult((ulong)rank, rank);
                }

                break;

            case ExactTopKCandidateStream.Random:
                FillRandomCandidates(candidates, duplicateDistanceBucketCount: 0);
                break;

            case ExactTopKCandidateStream.DuplicateDistance:
                FillRandomCandidates(candidates, duplicateDistanceBucketCount: 64);
                break;

            case ExactTopKCandidateStream.EqualDistanceWithIdTie:
                for (int i = 0; i < count; i++)
                {
                    candidates[i] = new SearchResult((ulong)(count - i), 1);
                }

                Shuffle(candidates, new Random(0x5EED162));
                break;

            default:
                throw new InvalidOperationException("Unknown candidate stream.");
        }

        return candidates;
    }

    private static void FillRandomCandidates(SearchResult[] candidates, int duplicateDistanceBucketCount)
    {
        var random = new Random(0x5EED162);
        for (int i = 0; i < candidates.Length; i++)
        {
            float distance = duplicateDistanceBucketCount == 0
                ? random.NextSingle() * candidates.Length
                : random.Next(duplicateDistanceBucketCount);
            candidates[i] = new SearchResult((ulong)(i + 1), distance);
        }

        Shuffle(candidates, random);
    }

    private static void Shuffle(SearchResult[] candidates, Random random)
    {
        for (int i = candidates.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
        }
    }

    internal static SearchResult[] CreateReference(SearchResult[] candidates, int resultCount)
    {
        var reference = new SearchResult[candidates.Length];
        candidates.AsSpan().CopyTo(reference);
        Array.Sort(reference, s_comparison);

        if (reference.Length == resultCount)
        {
            return reference;
        }

        var topK = new SearchResult[resultCount];
        reference.AsSpan(0, resultCount).CopyTo(topK);
        return topK;
    }

    internal static int SelectSortedSpanInsertion(
        ReadOnlySpan<SearchResult> candidates,
        Span<SearchResult> results)
    {
        int written = 0;
        for (int i = 0; i < candidates.Length; i++)
        {
            written = InsertCandidate(results, written, candidates[i]);
        }

        return written;
    }

    private static int InsertCandidate(Span<SearchResult> results, int written, SearchResult candidate)
    {
        int insertionIndex = FindInsertionIndex(results[..written], candidate);
        if (insertionIndex >= results.Length)
        {
            return written;
        }

        int valuesToShift = Math.Min(written, results.Length - 1) - insertionIndex;
        if (valuesToShift > 0)
        {
            results.Slice(insertionIndex, valuesToShift)
                .CopyTo(results.Slice(insertionIndex + 1));
        }

        results[insertionIndex] = candidate;
        return written < results.Length ? written + 1 : written;
    }

    private static int FindInsertionIndex(ReadOnlySpan<SearchResult> results, SearchResult candidate)
    {
        for (int i = 0; i < results.Length; i++)
        {
            if (Compare(candidate, results[i]) < 0)
            {
                return i;
            }
        }

        return results.Length;
    }

    internal static int SelectBoundedMaxHeap(
        ReadOnlySpan<SearchResult> candidates,
        Span<SearchResult> results,
        Span<SearchResult> heap)
    {
        int heapCount = 0;
        for (int i = 0; i < candidates.Length; i++)
        {
            SearchResult candidate = candidates[i];
            if (heapCount < results.Length)
            {
                heap[heapCount] = candidate;
                SiftUp(heap, heapCount);
                heapCount++;
            }
            else if (Compare(candidate, heap[0]) < 0)
            {
                heap[0] = candidate;
                SiftDown(heap[..heapCount], 0);
            }
        }

        heap[..heapCount].CopyTo(results);
        results[..heapCount].Sort(s_comparison);
        return heapCount;
    }

    private static void SiftUp(Span<SearchResult> heap, int index)
    {
        while (index > 0)
        {
            int parent = (index - 1) / 2;
            if (Compare(heap[parent], heap[index]) >= 0)
            {
                return;
            }

            (heap[parent], heap[index]) = (heap[index], heap[parent]);
            index = parent;
        }
    }

    private static void SiftDown(Span<SearchResult> heap, int index)
    {
        while (true)
        {
            int left = (index * 2) + 1;
            if (left >= heap.Length)
            {
                return;
            }

            int right = left + 1;
            int worse = right < heap.Length && Compare(heap[right], heap[left]) > 0
                ? right
                : left;

            if (Compare(heap[index], heap[worse]) >= 0)
            {
                return;
            }

            (heap[index], heap[worse]) = (heap[worse], heap[index]);
            index = worse;
        }
    }

    internal static int SelectPartialSelection(
        ReadOnlySpan<SearchResult> candidates,
        Span<SearchResult> results,
        Span<SearchResult> workspace)
    {
        candidates.CopyTo(workspace);
        int resultCount = results.Length;
        QuickSelect(workspace[..candidates.Length], resultCount);
        workspace[..resultCount].Sort(s_comparison);
        workspace[..resultCount].CopyTo(results);
        return resultCount;
    }

    private static void QuickSelect(Span<SearchResult> values, int take)
    {
        if (take >= values.Length)
        {
            return;
        }

        int left = 0;
        int right = values.Length - 1;
        int target = take - 1;

        while (left < right)
        {
            int pivotIndex = left + ((right - left) / 2);
            pivotIndex = Partition(values, left, right, pivotIndex);

            if (pivotIndex == target)
            {
                return;
            }

            if (target < pivotIndex)
            {
                right = pivotIndex - 1;
            }
            else
            {
                left = pivotIndex + 1;
            }
        }
    }

    private static int Partition(Span<SearchResult> values, int left, int right, int pivotIndex)
    {
        SearchResult pivot = values[pivotIndex];
        (values[pivotIndex], values[right]) = (values[right], values[pivotIndex]);

        int storeIndex = left;
        for (int i = left; i < right; i++)
        {
            if (Compare(values[i], pivot) < 0)
            {
                (values[storeIndex], values[i]) = (values[i], values[storeIndex]);
                storeIndex++;
            }
        }

        (values[right], values[storeIndex]) = (values[storeIndex], values[right]);
        return storeIndex;
    }

    internal static int Compare(SearchResult left, SearchResult right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0
            ? distanceComparison
            : left.Id.CompareTo(right.Id);
    }
}

[MemoryDiagnoser]
[ShortRunJob]
public class ExactTopKPublicSearchBenchmarks
{
    private ExactFlatIndex _index = null!;
    private float[] _query = null!;
    private SearchResult[] _results = null!;

    [Params(1, 10, 100, 1000)]
    public int ResultCount { get; set; }

    [Params(1024, 10000, 100000)]
    public int VectorCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        const int dimension = 32;
        var random = new Random(0x5EED162);
        var vector = new float[dimension];

        _index = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        for (int row = 0; row < VectorCount; row++)
        {
            FillVector(random, vector);
            _index.Add((ulong)(row + 1), vector);
        }

        _query = new float[dimension];
        FillVector(random, _query);
        _results = new SearchResult[ResultCount];

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"VEC-162 public-search setup: Dimension={dimension}, Vectors={VectorCount}, TopK={ResultCount}"));
    }

    [Benchmark]
    public int PublicDefaultSearch()
    {
        return _index.Search(_query, _results);
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }
}
