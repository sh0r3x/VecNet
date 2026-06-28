using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactOpenedSearchScenario
{
    private const string TaskId = "VEC-092";
    private const string SchemaName = "VecNet.ExactOpenedReadOnlySearchAllocationReport";
    private const string SchemaVersion = "0.1";

    public static GeneratedExactOpenedSearchBenchmarkReport Run(
        GeneratedExactOpenedSearchOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        TruthSet truth = ScalarGroundTruth.Generate(dataset, options.Metric, options.TopK);

        ExactFlatIndex sourceIndex = BuildIndex(options, dataset);
        sourceIndex.Save(options.IndexDirectory);
        ExactFlatIndex openedIndex = ExactFlatIndex.OpenReadOnly(options.IndexDirectory);

        WarmupSearch(options, dataset, openedIndex);
        SearchMeasurement measurement = MeasureSearch(options, dataset, openedIndex);
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            measurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);

        bool validationPassed =
            comparison.RecallAtK == 1 &&
            comparison.OrderedAgreement == 1 &&
            comparison.DistanceToleranceStatus == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        return new GeneratedExactOpenedSearchBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactOpenedSearchOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactOpenedSearchOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new DatasetInfo(
                GeneratedDataset.Kind,
                "generated-no-external-source",
                GeneratedDataset.Distribution,
                dataset.SeedText,
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount),
            new TruthInfo(
                ScalarGroundTruth.Kind,
                truth.Depth,
                ScalarGroundTruth.TiePolicy),
            new ScenarioInfo(
                GeneratedExactOpenedSearchOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data creation, source index construction, save, OpenReadOnly, warmup, scalar-reference truth construction, result capture/comparison, validation and report writing are excluded from opened read-only search allocation samples"),
            new IndexInfo(
                "ExactOpenedReadOnly",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "source ExactFlatIndex is built and saved before measurement; ExactFlatIndex.OpenReadOnly(directoryPath) is completed before measurement; measured operation is public Search(query, results) on the opened read-only instance only"),
            new GeneratedExactOpenedSearchLifecycleInfo(
                "public ExactFlatIndex constructor plus Add calls completed before measurement",
                "public ExactFlatIndex.Save(directoryPath) completed before measurement",
                "public ExactFlatIndex.OpenReadOnly(directoryPath) completed before warmup and measurement",
                options.IndexDirectory,
                "private ignored artifact directory supplied by --index-directory or defaulted under VecNet.BenchmarkRunner.Artifacts",
                SourceIndexBuiltBeforeMeasurement: true,
                SavedBeforeMeasurement: true,
                OpenedReadOnlyBeforeMeasurement: true,
                CallerOwnedResultBuffers: true,
                "Only repeated public ExactFlatIndex.Search(query, results) calls on the opened read-only index are sampled for managed allocation."),
            new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            CreateMeasurement(options, measurement),
            new MetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.DistanceToleranceStatus,
                comparison.DistanceMismatchCount,
                comparison.MissingResultCount),
            new GeneratedExactOpenedSearchValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-exact-opened-read-only-search-allocation-smoke",
                FiniteVectors: true,
                TruthGenerated: true,
                SourceIndexSaved: true,
                OpenedReadOnlyIndexCreated: true,
                OpenedReadOnlySearchComparedToTruth: validationPassed,
                CallerOwnedResultBuffers: true,
                SaveOpenSetupExcludedFromMeasurement: true,
                WarmupExcludedFromMeasurement: true,
                ResultCaptureComparisonExcludedFromMeasurement: true,
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private generated exact opened read-only search allocation smoke evidence only; not a public benchmark claim.",
                "The measured operation is only public ExactFlatIndex.Search(query, results) on an index returned by ExactFlatIndex.OpenReadOnly(directoryPath).",
                "Caller-owned result buffers are allocated before measured search calls and reused within each run.",
                "Generated data creation, source index construction, Save, OpenReadOnly, warmup, truth construction, validation, result capture/comparison and report writing are excluded from managed allocation samples.",
                "Save/open timings and allocations are not measured by this report.",
                "Resident/process/GC/private/peak memory and storage-size claims are out of scope.",
                "Public, preview, baseline, comparison and regression eligibility are false."
            ]);
    }

    public static void Write(GeneratedExactOpenedSearchBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactOpenedSearchOptions options) =>
        new(
            options.Metric,
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            options.Runs,
            options.WarmupQueries);

    private static ExactFlatIndex BuildIndex(GeneratedExactOpenedSearchOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static void WarmupSearch(
        GeneratedExactOpenedSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex openedIndex)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            openedIndex.Search(dataset.GetQuery(i % dataset.QueryCount), results);
        }
    }

    private static SearchMeasurement MeasureSearch(
        GeneratedExactOpenedSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex openedIndex)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, openedIndex, captureResults);
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

    private static SingleRunMeasurement MeasureSingleRun(
        GeneratedExactOpenedSearchOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex openedIndex,
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
            int written = openedIndex.Search(query, results);
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
                (double)totalAllocatedBytes / options.QueryCount),
            allResults);
    }

    private static AggregateTimingInfo AggregateRuns(SearchRunInfo[] runs, int measuredQueryCountPerRun) =>
        new(
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

    private static GeneratedExactOpenedSearchMeasurementInfo CreateMeasurement(
        GeneratedExactOpenedSearchOptions options,
        SearchMeasurement measurement) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perOpenedReadOnlySearchCall",
                "public ExactFlatIndex.Search(query, results) on an index returned by ExactFlatIndex.OpenReadOnly(directoryPath)",
                "generated data creation, source index construction/build, Save, OpenReadOnly, warmup, scalar-reference truth construction, validation, result capture/comparison and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Top-level openedReadOnlySearch latency percentile fields and aggregate fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                "bytesPerOpenedReadOnlySearchCall",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, results) call on the opened read-only index; caller-owned result buffer allocation, source index construction/build, Save, OpenReadOnly, warmup, truth construction, validation, result capture/comparison and report writing are excluded."),
            NotMeasured("notApplicable", "Source index construction/build is setup and explicitly excluded from opened-search allocation samples."),
            NotMeasured("notApplicable", "Save is setup and explicitly excluded from opened-search allocation samples."),
            NotMeasured("notApplicable", "OpenReadOnly is setup and explicitly excluded from opened-search allocation samples."),
            NotMeasured("notApplicable", "Scalar-reference truth construction is validation setup and explicitly excluded from opened-search allocation samples."),
            NotMeasured("notApplicable", "Validation is executed after measured search and explicitly excluded from opened-search allocation samples."),
            NotMeasured("notApplicable", "Result capture/copying and comparison are outside the per-call allocation sample boundary."),
            NotMeasured("notApplicable", "Report serialization and writing occur after measurement and are excluded from opened-search allocation samples."),
            NotMeasured("bytes", "Resident/process memory, working set, private bytes, managed heap and peak memory are not measured."),
            new RepeatedRunInfo(
                options.Runs > 1 ? "measured" : "singleRun",
                options.Runs,
                options.Runs > 1,
                options.Runs > 1
                    ? "Multiple measured opened read-only search runs executed; aggregate mean/min/max timing and allocation metadata is recorded without regression thresholds."
                    : "Only one measured opened read-only search run executed, so cross-run variance/noise is not measured."),
            CreateRunToRunNoise(measurement.Runs),
            new WarmupInfo(
                options.WarmupQueries > 0 ? "executed" : "absent",
                options.WarmupQueries,
                options.WarmupQueries > 0
                    ? "Warmup queries execute against the opened read-only index before measured runs and are excluded from measured timing/allocation totals."
                    : "No warmup queries were requested."),
            "Generated data creation, source index construction/build, Save, OpenReadOnly, warmup, scalar-reference truth construction, validation, result capture/comparison and report writing are excluded from opened read-only search timing and allocation samples.");

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string status = measured ? "measured" : "notMeasured";
        string reason = measured
            ? "Multiple measured opened read-only search runs executed; simple descriptive run-to-run statistics are recorded for private local noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured generated exact opened read-only search runs for public ExactFlatIndex.Search(query, results); setup, Save, OpenReadOnly, warmup, truth, validation, capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            reason,
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateMetricNoise(runs, "milliseconds", run => run.ElapsedMilliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "queriesPerSecond", run => run.Qps, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP50Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP95Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP99Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "bytesPerOpenedReadOnlySearchCall", run => run.ManagedAllocatedBytesPerQuery, measured, unavailableReason));
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

    private static GeneratedExactOpenedSearchEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-exact-opened-read-only-search-allocation-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated opened read-only exact search allocation smoke evidence is not reviewed public evidence and has no public reporting policy.",
            "This report is not preview-readiness evidence; actual memory, platform, package and admission gates remain outside scope.",
            "No opened read-only exact search baseline-candidate policy is accepted for this smoke report.",
            "Opened read-only exact search allocation smoke reports are not accepted comparison artifacts.",
            "No opened read-only exact search regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated exact opened read-only search allocation smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Measured allocation samples wrap only public ExactFlatIndex.Search(query, results) on the opened read-only index.",
                "Source index construction/build, Save, OpenReadOnly, warmup, truth construction, validation, result capture/comparison and report writing are excluded from allocation samples.",
                "Save/open timings and allocations are not measured.",
                "Resident/process memory, managed heap, private bytes, working set and peak memory are not measured.",
                "Not a public claim, baseline candidate, comparison artifact, regression gate, preview-readiness result, Linux x64 validation or BenchmarkDotNet-grade evidence."
            ]);

    private static GeneratedExactOpenedSearchEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated opened read-only exact search allocation smoke evidence is not reviewed public evidence.",
            "One local generated allocation smoke report does not establish preview readiness.",
            "No opened read-only exact search baseline-candidate policy is accepted for this smoke report.",
            "No opened read-only exact search comparison-artifact policy is accepted.",
            "No opened read-only exact search regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static void ValidateOptions(GeneratedExactOpenedSearchOptions options)
    {
        if (options.Runs is <= 0 or > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.TopK > options.VectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.", nameof(options));
        }
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

    private static string CreateReportId(string? commit, GeneratedExactOpenedSearchOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactOpenedSearchOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-{options.Seed:X8}");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record SingleRunMeasurement(
        SearchRunInfo Summary,
        SearchResult[][]? Results);

    private sealed record SearchMeasurement(
        SearchResult[][] Results,
        SearchRunInfo[] Runs,
        AggregateTimingInfo Aggregate);
}
