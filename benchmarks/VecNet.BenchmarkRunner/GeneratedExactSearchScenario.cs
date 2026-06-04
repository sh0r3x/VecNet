using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactSearchScenario
{
    private const string TaskId = "VEC-012";

    public static BenchmarkReport Run(GeneratedExactSearchOptions options, IReadOnlyList<string> commandArguments)
    {
        GeneratedDataset dataset = GeneratedDatasetFactory.Create(options);
        ValidateFinite(dataset);

        TruthSet truth = ScalarGroundTruth.Generate(dataset, options.Metric, options.TopK);
        ExactFlatIndex index = BuildIndex(options, dataset);

        WarmupSearch(options, dataset, index);
        SearchMeasurement measurement = MeasureSearch(options, dataset, index);
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            measurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);

        RepositoryInfo repository = RepositoryInfo.Create();

        return new BenchmarkReport(
            SchemaName: "VecNet.BenchmarkReport",
            SchemaVersion: "0.1",
            ReportId: CreateReportId(repository.Commit, options),
            GeneratedAtUtc: DateTimeOffset.UtcNow,
            TaskId: TaskId,
            ClaimClass: "local-evidence",
            PrivacyClass: "private-raw",
            Evidence: new EvidenceInfo(
                "smoke",
                "local-evidence",
                false,
                "Private generated-data runner output lacks allocation and memory measurement and is not reviewed public evidence.",
                [
                    "Generated data only; no external dataset source, license, version or checksum applies.",
                    "Repeated measured runs are private local evidence only and do not implement regression comparison math.",
                    "Allocation and memory values are explicitly not measured.",
                    "Not eligible for public performance, scale, ANN, real-dataset or concurrency claims."
                ]),
            Repository: repository,
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            Command: new CommandInfo(GeneratedExactSearchOptions.ScenarioName, commandArguments.ToArray()),
            Environment: new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            Dataset: new DatasetInfo(
                GeneratedDataset.Kind,
                "generated-no-external-source",
                GeneratedDataset.Distribution,
                dataset.SeedText,
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount),
            Truth: new TruthInfo(
                ScalarGroundTruth.Kind,
                truth.Depth,
                ScalarGroundTruth.TiePolicy),
            Scenario: new ScenarioInfo(
                GeneratedExactSearchOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "setup, index build, truth generation, warmup queries, result comparison and report writing are excluded from search timing"),
            Index: new IndexInfo(
                "Exact",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "public ExactFlatIndex constructor; no persistence, filtering, updates or concurrency"),
            Search: new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            Measurement: new MeasurementInfo(
                ManagedAllocations: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytesPerOperation",
                    "The current runner does not use allocation instrumentation or BenchmarkDotNet MemoryDiagnoser."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "The current runner does not measure resident, managed heap or process memory."),
                RepeatedRuns: new RepeatedRunInfo(
                    options.Runs > 1 ? "measured" : "singleRun",
                    options.Runs,
                    options.Runs > 1,
                    options.Runs > 1
                        ? "Multiple measured runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                        : "Only one measured run executed, so cross-run variance/noise is not measured."),
                Warmup: new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed before measured runs and excluded from measured timing totals."
                        : "No warmup queries were requested.")),
            Metrics: new MetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.DistanceToleranceStatus,
                comparison.DistanceMismatchCount,
                comparison.MissingResultCount),
            Baseline: new BaselineInfo(
                options.BaselineReportId,
                "smoke",
                false,
                false,
                "Baseline comparison math, regression decisions and variance thresholds are not implemented."),
            Validation: new ValidationInfo(
                comparison.RecallAtK == 1 &&
                comparison.OrderedAgreement == 1 &&
                comparison.DistanceToleranceStatus == "passed"
                    ? "passed"
                    : "failed",
                "smoke",
                true,
                true,
                false,
                false,
                true),
            Notes:
            [
                "Private generated-data smoke evidence only; not a public benchmark claim.",
                "Allocation and memory values are not measured.",
                "Warmup query timings are deliberately excluded from measured timing totals.",
                "External datasets, ANN, persistence, filtering, updates and concurrency are out of scope for VEC-012."
            ]);
    }

    private static ExactFlatIndex BuildIndex(GeneratedExactSearchOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static SearchMeasurement MeasureSearch(
        GeneratedExactSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, index, captureResults);
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(
            capturedResults ?? [],
            runs,
            AggregateRuns(runs, options.QueryCount));
    }

    private static void WarmupSearch(
        GeneratedExactSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(i % dataset.QueryCount);
            index.Search(query, results);
        }
    }

    private static SingleRunMeasurement MeasureSingleRun(
        GeneratedExactSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, results);
            long elapsed = Stopwatch.GetTimestamp() - start;

            latencyTicks[queryRow] = elapsed;
            totalTicks += elapsed;

            if (captureResults)
            {
                var queryResults = new SearchResult[written];
                results.AsSpan(0, written).CopyTo(queryResults);
                allResults![queryRow] = queryResults;
            }
        }

        Array.Sort(latencyTicks);
        double elapsedSeconds = (double)totalTicks / Stopwatch.Frequency;
        return new SingleRunMeasurement(
            new SearchRunInfo(
                RunNumber: 0,
                options.QueryCount,
                elapsedSeconds * 1000,
                PercentileMilliseconds(latencyTicks, 0.50),
                PercentileMilliseconds(latencyTicks, 0.95),
                PercentileMilliseconds(latencyTicks, 0.99),
                elapsedSeconds == 0 ? double.PositiveInfinity : options.QueryCount / elapsedSeconds),
            allResults);
    }

    private static AggregateTimingInfo AggregateRuns(SearchRunInfo[] runs, int measuredQueryCountPerRun)
    {
        return new AggregateTimingInfo(
            runs.Length,
            measuredQueryCountPerRun,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds),
            runs.Average(run => run.LatencyP50Milliseconds),
            runs.Average(run => run.LatencyP95Milliseconds),
            runs.Average(run => run.LatencyP99Milliseconds),
            runs.Average(run => run.Qps),
            runs.Min(run => run.Qps),
            runs.Max(run => run.Qps));
    }

    private sealed record SingleRunMeasurement(
        SearchRunInfo Summary,
        SearchResult[][]? Results);

    private sealed record SearchMeasurement(
        SearchResult[][] Results,
        SearchRunInfo[] Runs,
        AggregateTimingInfo Aggregate);

    private static double PercentileMilliseconds(long[] sortedTicks, double percentile)
    {
        if (sortedTicks.Length == 0)
        {
            return 0;
        }

        int index = (int)Math.Ceiling(sortedTicks.Length * percentile) - 1;
        index = Math.Clamp(index, 0, sortedTicks.Length - 1);
        return sortedTicks[index] * 1000.0 / Stopwatch.Frequency;
    }

    private static void ValidateFinite(GeneratedDataset dataset)
    {
        foreach (float value in dataset.Vectors)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Generated vector data must be finite.");
            }
        }

        foreach (float value in dataset.Queries)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Generated query data must be finite.");
            }
        }
    }

    private static string CreateReportId(string? commit, GeneratedExactSearchOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactSearchOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-{options.Seed:X8}");
    }
}
