using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactSearchScenario
{
    private const string TaskId = "VEC-014";

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
        ExactGeneratedPublicEvidenceValidationInfo publicEvidenceValidation =
            ExactGeneratedPublicEvidencePolicy.Evaluate(
                truth,
                measurement.Results,
                dataset,
                options.Metric,
                options.TopK,
                options.Dimension,
                comparison);

        RepositoryInfo repository = RepositoryInfo.Create();

        var report = new BenchmarkReport(
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
                "Private generated-data runner output measures managed allocations but lacks resident/process memory measurement and is not reviewed public evidence.",
                [
                    "Generated data only; no external dataset source, license, version or checksum applies.",
                    "Run-to-run noise summaries are private local descriptive statistics only and do not implement regression comparison math or acceptable-noise thresholds.",
                    "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                    "Managed allocations are measured for public ExactFlatIndex.Search calls only; resident/process memory is explicitly not measured.",
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
                "setup, index build, scalar-reference truth generation, warmup queries, result capture/comparison and report writing are excluded from search timing"),
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
                Latency: new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perMeasuredQuery",
                    "public ExactFlatIndex.Search(query, results)",
                    "setup, index build, scalar-reference truth generation, warmup queries, result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search.aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, results) call; setup, index build, scalar-reference truth generation, warmup, result capture/comparison and report writing are excluded."),
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
                RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
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
                true,
                publicEvidenceValidation),
            Notes:
            [
                "Private generated-data smoke evidence only; not a public benchmark claim.",
                "For exact-generated public evidence, validation.exactGeneratedPublicEvidence records the VEC-215 strict/near-tie policy; orderedAgreement remains a reported diagnostic metric and private baseline eligibility still requires perfect ordering.",
                "Latency samples are per measured query around public ExactFlatIndex.Search(query, results); setup, index build, scalar-reference truth, warmup, result capture/comparison and report writing are excluded.",
                "Latency p50/p95/p99 use nearest-rank per-run samples, with top-level and aggregate fields reported as means across per-run percentiles rather than BenchmarkDotNet statistics.",
                "Run-to-run noise metadata uses simple mean, sample standard deviation, coefficient of variation where available, and min/max spread across measured runs; it is not BenchmarkDotNet statistics, a threshold policy or a regression decision.",
                "Managed allocations are measured only for the public ExactFlatIndex.Search operation inside measured runs.",
                "Resident/process memory values are not measured.",
                "Warmup query timings are deliberately excluded from measured timing totals.",
                "External datasets, ANN, persistence, filtering, updates and concurrency are out of scope for generated exact runner reports."
            ]);

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
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
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, results);
            long elapsed = Stopwatch.GetTimestamp() - start;
            long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;

            latencyTicks[queryRow] = elapsed;
            totalTicks += elapsed;
            totalAllocatedBytes += allocatedBytes;

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
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.50, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.95, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.99, Stopwatch.Frequency),
                elapsedSeconds == 0 ? double.PositiveInfinity : options.QueryCount / elapsedSeconds,
                totalAllocatedBytes,
                options.QueryCount == 0 ? 0 : (double)totalAllocatedBytes / options.QueryCount),
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
            runs.Max(run => run.Qps),
            runs.Average(run => run.ManagedAllocatedBytes),
            runs.Min(run => run.ManagedAllocatedBytes),
            runs.Max(run => run.ManagedAllocatedBytes),
            runs.Average(run => run.ManagedAllocatedBytesPerQuery),
            runs.Min(run => run.ManagedAllocatedBytesPerQuery),
            runs.Max(run => run.ManagedAllocatedBytesPerQuery));
    }

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string status = measured ? "measured" : "notMeasured";
        string reason = measured
            ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured generated exact-search runs for public ExactFlatIndex.Search(query, results); warmup, setup, index build, scalar-reference truth, result capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            reason,
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateMetricNoise(
                runs,
                "milliseconds",
                run => run.ElapsedMilliseconds,
                measured,
                unavailableReason),
            CreateMetricNoise(
                runs,
                "queriesPerSecond",
                run => run.Qps,
                measured,
                unavailableReason),
            CreateMetricNoise(
                runs,
                "milliseconds",
                run => run.LatencyP50Milliseconds,
                measured,
                unavailableReason),
            CreateMetricNoise(
                runs,
                "milliseconds",
                run => run.LatencyP95Milliseconds,
                measured,
                unavailableReason),
            CreateMetricNoise(
                runs,
                "milliseconds",
                run => run.LatencyP99Milliseconds,
                measured,
                unavailableReason),
            CreateMetricNoise(
                runs,
                "bytesPerQuery",
                run => run.ManagedAllocatedBytesPerQuery,
                measured,
                unavailableReason));
    }

    private static RunToRunMetricNoiseInfo CreateMetricNoise(
        SearchRunInfo[] runs,
        string unit,
        Func<SearchRunInfo, double> valueSelector,
        bool measured,
        string unavailableReason)
    {
        if (!measured)
        {
            return new RunToRunMetricNoiseInfo(
                "notMeasured",
                unit,
                Mean: null,
                SampleStandardDeviation: null,
                CoefficientOfVariation: null,
                Min: null,
                Max: null,
                Spread: null,
                unavailableReason);
        }

        double[] values = runs.Select(valueSelector).ToArray();
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate(values);
        return new RunToRunMetricNoiseInfo(
            "measured",
            unit,
            FiniteOrNull(statistics.Mean),
            statistics.SampleStandardDeviation,
            statistics.CoefficientOfVariation,
            FiniteOrNull(statistics.Min),
            FiniteOrNull(statistics.Max),
            FiniteOrNull(statistics.Spread),
            "Computed across measured runs using the documented private descriptive-statistics formula.");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record SingleRunMeasurement(
        SearchRunInfo Summary,
        SearchResult[][]? Results);

    private sealed record SearchMeasurement(
        SearchResult[][] Results,
        SearchRunInfo[] Runs,
        AggregateTimingInfo Aggregate);

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
