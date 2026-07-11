using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VecNet.Benchmarks;

internal static class Vec162TopKReportCommand
{
    private static readonly int[] s_resultCounts = [1, 10, 100, 1000];
    private static readonly int[] s_candidateCounts = [1024, 10000, 100000];
    private static readonly ExactTopKCandidateStream[] s_streams =
    [
        ExactTopKCandidateStream.AlreadySorted,
        ExactTopKCandidateStream.ReverseSorted,
        ExactTopKCandidateStream.Random,
        ExactTopKCandidateStream.DuplicateDistance,
        ExactTopKCandidateStream.EqualDistanceWithIdTie
    ];
    private static readonly ExactTopKSelectionStrategy[] s_strategies =
    [
        ExactTopKSelectionStrategy.SortedSpanInsertion,
        ExactTopKSelectionStrategy.BoundedMaxHeap,
        ExactTopKSelectionStrategy.PartialSelection
    ];

    public static bool TryRun(string[] args)
    {
        if (args.Length == 0 || !string.Equals(args[0], "vec162-topk-report", StringComparison.Ordinal))
        {
            return false;
        }

        string outputPath = args.Length >= 2
            ? args[1]
            : Path.Combine("BenchmarkDotNet.Artifacts", "vec-162-topk-strategy-report.json");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        Vec162TopKReport report = RunReport();

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };
        File.WriteAllText(outputPath, JsonSerializer.Serialize(report, options));
        Console.WriteLine($"VEC-162 report written: {outputPath}");
        return true;
    }

    private static Vec162TopKReport RunReport()
    {
        var selectionMeasurements = new List<Vec162SelectionMeasurement>();
        foreach (int candidateCount in s_candidateCounts)
        {
            foreach (ExactTopKCandidateStream stream in s_streams)
            {
                SearchResult[] candidates = ExactTopKStrategyBenchmarks.CreateCandidates(candidateCount, stream);
                foreach (int resultCount in s_resultCounts)
                {
                    SearchResult[] reference = ExactTopKStrategyBenchmarks.CreateReference(candidates, resultCount);
                    foreach (ExactTopKSelectionStrategy strategy in s_strategies)
                    {
                        selectionMeasurements.Add(MeasureSelection(
                            candidates,
                            reference,
                            candidateCount,
                            resultCount,
                            stream,
                            strategy));
                    }
                }
            }
        }

        var publicSearchMeasurements = new List<Vec162PublicSearchMeasurement>();
        foreach (int vectorCount in s_candidateCounts)
        {
            foreach (int resultCount in s_resultCounts)
            {
                publicSearchMeasurements.Add(MeasurePublicSearch(vectorCount, resultCount));
            }
        }

        return new Vec162TopKReport(
            "VEC-162",
            DateTimeOffset.UtcNow,
            CreateEnvironment(),
            selectionMeasurements,
            publicSearchMeasurements);
    }

    private static Vec162SelectionMeasurement MeasureSelection(
        SearchResult[] candidates,
        SearchResult[] reference,
        int candidateCount,
        int resultCount,
        ExactTopKCandidateStream stream,
        ExactTopKSelectionStrategy strategy)
    {
        var results = new SearchResult[resultCount];
        var heap = new SearchResult[resultCount];
        var partialWorkspace = new SearchResult[candidateCount];

        ValidateSelection(candidates, reference, results, heap, partialWorkspace, strategy);

        int iterations = GetSelectionIterations(candidateCount, resultCount, stream, strategy);
        WarmupSelection(candidates, results, heap, partialWorkspace, strategy);

        MeasurementRuns runs = MeasureRuns(iterations, () =>
        {
            int written = Select(candidates, results, heap, partialWorkspace, strategy);
            if (written != resultCount)
            {
                throw new InvalidOperationException($"{strategy} wrote {written} results; expected {resultCount}.");
            }
        });

        return new Vec162SelectionMeasurement(
            strategy,
            stream,
            candidateCount,
            resultCount,
            iterations,
            runs.RunCount,
            runs.MeanMillisecondsPerOperation,
            runs.MinMillisecondsPerOperation,
            runs.MaxMillisecondsPerOperation,
            runs.StandardDeviationMillisecondsPerOperation,
            runs.RelativeStandardDeviation,
            runs.ManagedAllocatedBytesPerOperation,
            "passed");
    }

    private static Vec162PublicSearchMeasurement MeasurePublicSearch(int vectorCount, int resultCount)
    {
        const int dimension = 32;
        var random = new Random(0x5EED162);
        var vector = new float[dimension];
        var index = new ExactFlatIndex(dimension, VectorMetric.SquaredEuclidean);
        for (int row = 0; row < vectorCount; row++)
        {
            FillVector(random, vector);
            index.Add((ulong)(row + 1), vector);
        }

        var query = new float[dimension];
        FillVector(random, query);
        var results = new SearchResult[resultCount];

        int written = index.Search(query, results);
        if (written != Math.Min(resultCount, vectorCount))
        {
            throw new InvalidOperationException(
                $"Public search wrote {written} results; expected {Math.Min(resultCount, vectorCount)}.");
        }

        ValidateOrdered(results.AsSpan(0, written));

        int iterations = GetPublicSearchIterations(vectorCount, resultCount);
        index.Search(query, results);

        MeasurementRuns runs = MeasureRuns(iterations, () =>
        {
            int measuredWritten = index.Search(query, results);
            if (measuredWritten != written)
            {
                throw new InvalidOperationException(
                    $"Public search wrote {measuredWritten} results during measurement; expected {written}.");
            }
        });

        return new Vec162PublicSearchMeasurement(
            vectorCount,
            dimension,
            resultCount,
            iterations,
            runs.RunCount,
            runs.MeanMillisecondsPerOperation,
            runs.MinMillisecondsPerOperation,
            runs.MaxMillisecondsPerOperation,
            runs.StandardDeviationMillisecondsPerOperation,
            runs.RelativeStandardDeviation,
            runs.ManagedAllocatedBytesPerOperation,
            "passed");
    }

    private static MeasurementRuns MeasureRuns(int iterations, Action operation)
    {
        const int runCount = 5;
        var millisecondsPerOperation = new double[runCount];
        long totalAllocatedBytes = 0;

        for (int run = 0; run < runCount; run++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            long timestampBefore = Stopwatch.GetTimestamp();
            for (int i = 0; i < iterations; i++)
            {
                operation();
            }

            long timestampAfter = Stopwatch.GetTimestamp();
            long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();

            double elapsedMilliseconds = Stopwatch.GetElapsedTime(timestampBefore, timestampAfter).TotalMilliseconds;
            millisecondsPerOperation[run] = elapsedMilliseconds / iterations;
            totalAllocatedBytes += allocatedAfter - allocatedBefore;
        }

        double mean = Mean(millisecondsPerOperation);
        return new MeasurementRuns(
            runCount,
            mean,
            millisecondsPerOperation.Min(),
            millisecondsPerOperation.Max(),
            StandardDeviation(millisecondsPerOperation, mean),
            mean == 0 ? 0 : StandardDeviation(millisecondsPerOperation, mean) / mean,
            (double)totalAllocatedBytes / (runCount * iterations));
    }

    private static void ValidateSelection(
        SearchResult[] candidates,
        SearchResult[] reference,
        SearchResult[] results,
        SearchResult[] heap,
        SearchResult[] partialWorkspace,
        ExactTopKSelectionStrategy strategy)
    {
        Array.Clear(results);
        Array.Clear(heap);
        Array.Clear(partialWorkspace);

        int written = Select(candidates, results, heap, partialWorkspace, strategy);
        if (written != reference.Length)
        {
            throw new InvalidOperationException($"{strategy} wrote {written} results; expected {reference.Length}.");
        }

        for (int i = 0; i < reference.Length; i++)
        {
            if (results[i].Id != reference[i].Id || results[i].Distance != reference[i].Distance)
            {
                throw new InvalidOperationException(
                    $"{strategy} mismatch at result {i}: got ({results[i].Id}, {results[i].Distance:R}), " +
                    $"expected ({reference[i].Id}, {reference[i].Distance:R}).");
            }
        }
    }

    private static void ValidateOrdered(ReadOnlySpan<SearchResult> results)
    {
        for (int i = 1; i < results.Length; i++)
        {
            if (ExactTopKStrategyBenchmarks.Compare(results[i - 1], results[i]) > 0)
            {
                throw new InvalidOperationException($"Public search result order failed at result {i}.");
            }
        }
    }

    private static void WarmupSelection(
        SearchResult[] candidates,
        SearchResult[] results,
        SearchResult[] heap,
        SearchResult[] partialWorkspace,
        ExactTopKSelectionStrategy strategy)
    {
        for (int i = 0; i < 2; i++)
        {
            Select(candidates, results, heap, partialWorkspace, strategy);
        }
    }

    private static int Select(
        SearchResult[] candidates,
        SearchResult[] results,
        SearchResult[] heap,
        SearchResult[] partialWorkspace,
        ExactTopKSelectionStrategy strategy)
    {
        return strategy switch
        {
            ExactTopKSelectionStrategy.SortedSpanInsertion =>
                ExactTopKStrategyBenchmarks.SelectSortedSpanInsertion(candidates, results),
            ExactTopKSelectionStrategy.BoundedMaxHeap =>
                ExactTopKStrategyBenchmarks.SelectBoundedMaxHeap(candidates, results, heap),
            ExactTopKSelectionStrategy.PartialSelection =>
                ExactTopKStrategyBenchmarks.SelectPartialSelection(candidates, results, partialWorkspace),
            _ => throw new InvalidOperationException("Unknown top-k selection strategy.")
        };
    }

    private static int GetSelectionIterations(
        int candidateCount,
        int resultCount,
        ExactTopKCandidateStream stream,
        ExactTopKSelectionStrategy strategy)
    {
        if (candidateCount >= 100000)
        {
            if (strategy == ExactTopKSelectionStrategy.SortedSpanInsertion &&
                resultCount >= 1000 &&
                stream == ExactTopKCandidateStream.ReverseSorted)
            {
                return 1;
            }

            return resultCount >= 100 ? 2 : 5;
        }

        if (candidateCount >= 10000)
        {
            return resultCount >= 100 ? 10 : 25;
        }

        return resultCount >= 100 ? 50 : 100;
    }

    private static int GetPublicSearchIterations(int vectorCount, int resultCount)
    {
        if (vectorCount >= 100000)
        {
            return resultCount >= 100 ? 1 : 3;
        }

        if (vectorCount >= 10000)
        {
            return resultCount >= 100 ? 3 : 10;
        }

        return resultCount >= 100 ? 10 : 30;
    }

    private static Vec162Environment CreateEnvironment()
    {
        return new Vec162Environment(
            RuntimeInformation.OSDescription,
            RuntimeInformation.OSArchitecture.ToString(),
            RuntimeInformation.ProcessArchitecture.ToString(),
            RuntimeInformation.RuntimeIdentifier,
            Environment.Version.ToString(),
            Environment.ProcessorCount,
            Vector.IsHardwareAccelerated,
            Vector<float>.Count,
            GCSettingsIsServerGc(),
            Stopwatch.Frequency);
    }

    private static bool GCSettingsIsServerGc() => System.Runtime.GCSettings.IsServerGC;

    private static double Mean(double[] values) => values.Sum() / values.Length;

    private static double StandardDeviation(double[] values, double mean)
    {
        if (values.Length <= 1)
        {
            return 0;
        }

        double sumSquares = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double difference = values[i] - mean;
            sumSquares += difference * difference;
        }

        return Math.Sqrt(sumSquares / (values.Length - 1));
    }

    private static void FillVector(Random random, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextSingle();
        }
    }

    private sealed record MeasurementRuns(
        int RunCount,
        double MeanMillisecondsPerOperation,
        double MinMillisecondsPerOperation,
        double MaxMillisecondsPerOperation,
        double StandardDeviationMillisecondsPerOperation,
        double RelativeStandardDeviation,
        double ManagedAllocatedBytesPerOperation);

    private sealed record Vec162TopKReport(
        string TaskId,
        DateTimeOffset CreatedUtc,
        Vec162Environment Environment,
        IReadOnlyList<Vec162SelectionMeasurement> SelectionMeasurements,
        IReadOnlyList<Vec162PublicSearchMeasurement> PublicSearchMeasurements);

    private sealed record Vec162Environment(
        string OsDescription,
        string OsArchitecture,
        string ProcessArchitecture,
        string RuntimeIdentifier,
        string DotNetVersion,
        int ProcessorCount,
        bool VectorHardwareAccelerated,
        int VectorFloatCount,
        bool ServerGc,
        long StopwatchFrequency);

    private sealed record Vec162SelectionMeasurement(
        ExactTopKSelectionStrategy Strategy,
        ExactTopKCandidateStream Stream,
        int CandidateCount,
        int ResultCount,
        int IterationsPerRun,
        int RunCount,
        double MeanMillisecondsPerOperation,
        double MinMillisecondsPerOperation,
        double MaxMillisecondsPerOperation,
        double StandardDeviationMillisecondsPerOperation,
        double RelativeStandardDeviation,
        double ManagedAllocatedBytesPerOperation,
        string OrderingValidation);

    private sealed record Vec162PublicSearchMeasurement(
        int VectorCount,
        int Dimension,
        int ResultCount,
        int IterationsPerRun,
        int RunCount,
        double MeanMillisecondsPerOperation,
        double MinMillisecondsPerOperation,
        double MaxMillisecondsPerOperation,
        double StandardDeviationMillisecondsPerOperation,
        double RelativeStandardDeviation,
        double ManagedAllocatedBytesPerOperation,
        string OrderingValidation);
}
