using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactFilteredScenario
{
    private const string TaskId = "VEC-046";
    private const string SchemaName = "VecNet.ExactFilteredBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static GeneratedExactFilteredBenchmarkReport Run(
        GeneratedExactFilteredOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        GeneratedFilterSet filters = GenerateFilters(options);
        TruthSet truth = GenerateFilteredTruth(dataset, options.Metric, options.TopK, filters);
        ExactFlatIndex index = BuildIndex(options, dataset);

        WarmupSearch(options, dataset, index, filters);
        SearchMeasurement measurement = MeasureSearch(options, dataset, index, filters);
        GeneratedExactFilteredResultComparison comparison = ValidateFilteredResults(
            truth,
            measurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);

        RepositoryInfo repository = RepositoryInfo.Create();
        string validationStatus = comparison.Integrity.Status == "passed" ? "passed" : "failed";

        return new GeneratedExactFilteredBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactFilteredOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            new GeneratedExactFilteredEvidenceInfo(
                "smoke",
                "generated-exact-filtered-smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private generated exact-filter runner output is not reviewed public evidence.",
                "No generated exact-filter baseline-candidate policy is accepted in VEC-046.",
                "No generated exact-filter regression-gate policy is accepted in VEC-046.",
                [
                    "Generated exact-filter smoke evidence only; no external dataset source, license, version or checksum applies.",
                    "Generated data setup, index build, allowlist generation, workspace construction, warmup queries, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                    "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                    "Managed allocations are measured for public ExactFlatIndex.Search(query, allowedIds, results, workspace) calls only; resident/process memory is explicitly not measured.",
                    "Filter allowlists are synthetic and deterministic; stored labels, persisted filters, HNSW filtering, external dataset filters, baselines, comparisons and regression gates are out of scope."
                ]),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactFilteredOptions.ScenarioName, commandArguments.ToArray()),
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
                "scalar-reference-generated-filtered",
                options.TopK,
                "allowlist is coalesced to known generated ids, then results are ordered by ascending scalar-reference canonical distance and ascending external id"),
            new ScenarioInfo(
                GeneratedExactFilteredOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, exact-flat index build, synthetic allowlist generation, filter workspace construction, scalar-reference filtered truth generation, warmup queries, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "ExactFiltered",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "public ExactFlatIndex constructor and public ExactFlatIndex.Search(query, allowedIds, results, workspace); no persistence, stored labels, retained ID-to-ordinal map, HNSW filtering, updates or external dataset mode"),
            filters.Info,
            new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            new MeasurementInfo(
                Latency: new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perMeasuredQuery",
                    "public ExactFlatIndex.Search(query, allowedIds, results, workspace)",
                    "generated data setup, exact-flat index build, synthetic allowlist generation, filter workspace construction, scalar-reference filtered truth generation, warmup queries, final-run result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search.aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, allowedIds, results, workspace) call using caller-owned SearchResult[], per-query generated allowlists and caller-owned ExactFlatSearchFilterWorkspace; setup, exact-flat index build, allowlist generation, workspace construction, filtered truth generation, warmup, result capture/comparison and report writing are excluded."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, private bytes, managed heap size and peak memory are not measured in VEC-046."),
                RepeatedRuns: new RepeatedRunInfo(
                    options.Runs > 1 ? "measured" : "singleRun",
                    options.Runs,
                    options.Runs > 1,
                    options.Runs > 1
                        ? "Multiple measured exact-filter search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                        : "Only one measured run executed, so cross-run variance/noise is not measured."),
                RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
                Warmup: new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed before measured runs using caller-owned results/workspace and excluded from measured timing and allocation totals."
                        : "No warmup queries were requested.")),
            new GeneratedExactFilteredMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.Integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
                comparison.Integrity.DistanceMismatchCount,
                comparison.Integrity.MissingResultCount,
                comparison.Integrity.ExtraResultCount,
                comparison.Integrity,
                "set recall@k = returned ids intersect exact filtered top-k ids divided by exact filtered result count, summed across measured queries; empty exact filtered truth contributes a perfect denominator-free query",
                "Final measured run is compared against independently generated scalar-reference filtered truth; IDs, order, result count and distances are all validation failures when incorrect."),
            new GeneratedExactFilteredValidationInfo(
                validationStatus,
                "generated-exact-filtered-smoke",
                FiniteVectors: true,
                TruthGenerated: true,
                FinalRunComparedToTruth: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new GeneratedExactFilteredEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated exact-filter reports are private local smoke evidence only until a reviewed public summary policy exists.",
                "No generated exact-filter baseline-candidate policy is accepted in VEC-046.",
                "No generated exact-filter regression-gate policy is accepted in VEC-046."),
            [
                "Private generated exact-filter smoke evidence only; not a public benchmark claim.",
                "Latency and QPS time only public ExactFlatIndex.Search(query, allowedIds, results, workspace) calls.",
                "Generated data, exact-flat index build, allowlist generation, workspace construction and exact filtered truth generation are setup work and excluded from measured search timing.",
                "Managed allocations are measured only for the public filtered-search call boundary.",
                "Filter selectivity is synthetic and deterministic according to the filter.generationFormula metadata.",
                "Baseline candidacy, comparison artifacts, regression gates, matrix presets, external dataset filters, HNSW filtering, stored labels, persistence, updates and public claims are out of scope."
            ]);
    }

    public static void Write(GeneratedExactFilteredBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    public static GeneratedExactFilteredResultComparison ValidateFilteredResults(
        TruthSet truth,
        SearchResult[][] actual,
        int topK,
        int dimension,
        VectorMetric metric)
    {
        int queryCountMismatchCount = truth.Results.Length == actual.Length ? 0 : 1;
        int checkedResultCount = 0;
        int missingResultCount = 0;
        int extraResultCount = 0;
        int wrongIdCount = 0;
        int orderMismatchCount = 0;
        int nonFiniteDistanceCount = 0;
        int distanceMismatchCount = 0;
        int denominator = 0;
        int setMatches = 0;
        int orderedMatches = 0;
        int queryCount = Math.Min(truth.Results.Length, actual.Length);

        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            TruthItem[] expected = truth.Results[queryRow];
            SearchResult[] returned = actual[queryRow];
            int expectedCount = Math.Min(topK, expected.Length);
            denominator += expectedCount;
            missingResultCount += Math.Max(0, expectedCount - returned.Length);
            extraResultCount += Math.Max(0, returned.Length - expectedCount);

            var returnedIds = new HashSet<ulong>();
            int comparableCount = Math.Min(expectedCount, returned.Length);
            for (int i = 0; i < Math.Min(expectedCount, returned.Length); i++)
            {
                returnedIds.Add(returned[i].Id);
            }

            for (int i = 0; i < expectedCount; i++)
            {
                if (returnedIds.Contains(expected[i].Id))
                {
                    setMatches++;
                }

                if (i >= returned.Length)
                {
                    continue;
                }

                SearchResult result = returned[i];
                checkedResultCount++;
                if (!float.IsFinite(result.Distance))
                {
                    nonFiniteDistanceCount++;
                }

                if (result.Id == expected[i].Id)
                {
                    orderedMatches++;
                }
                else
                {
                    wrongIdCount++;
                    orderMismatchCount++;
                }

                if (!DistanceMatches(expected[i].Distance, result.Distance, dimension, metric))
                {
                    distanceMismatchCount++;
                }
            }

            for (int i = comparableCount; i < returned.Length; i++)
            {
                checkedResultCount++;
                if (!float.IsFinite(returned[i].Distance))
                {
                    nonFiniteDistanceCount++;
                }
            }
        }

        bool passed = queryCountMismatchCount == 0 &&
            missingResultCount == 0 &&
            extraResultCount == 0 &&
            wrongIdCount == 0 &&
            orderMismatchCount == 0 &&
            nonFiniteDistanceCount == 0 &&
            distanceMismatchCount == 0;

        var integrity = new GeneratedExactFilteredResultIntegrityInfo(
            passed ? "passed" : "failed",
            queryCountMismatchCount,
            checkedResultCount,
            missingResultCount,
            extraResultCount,
            wrongIdCount,
            orderMismatchCount,
            nonFiniteDistanceCount,
            distanceMismatchCount,
            "Filtered exact results must match independently generated exact filtered truth for query count, result count, IDs, order and distances within the accepted metric tolerance.",
            passed
                ? "All filtered exact results matched independent filtered truth."
                : "One or more filtered exact results failed count, ID, order, finite-distance or distance-integrity validation.");

        return new GeneratedExactFilteredResultComparison(
            denominator == 0 ? 1 : (double)setMatches / denominator,
            denominator == 0 ? 1 : (double)orderedMatches / denominator,
            integrity);
    }

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactFilteredOptions options) =>
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

    private static ExactFlatIndex BuildIndex(GeneratedExactFilteredOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static GeneratedFilterSet GenerateFilters(GeneratedExactFilteredOptions options)
    {
        int visibleCount = GetVisibleCount(options);
        var allowlists = new ulong[options.QueryCount][];
        int knownPerQuery = visibleCount;
        int allowlistLength = checked(knownPerQuery + options.DuplicateIdsPerQuery + options.UnknownIdsPerQuery);
        int minVisible = visibleCount;
        int maxVisible = visibleCount;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            var allowlist = new ulong[allowlistLength];
            int write = 0;
            int start = options.VectorCount == 0
                ? 0
                : (int)(((ulong)options.Seed + ((ulong)queryRow * 2_654_435_761UL)) % (ulong)options.VectorCount);
            for (int i = 0; i < visibleCount; i++)
            {
                allowlist[write++] = (ulong)((start + i) % options.VectorCount);
            }

            for (int i = 0; i < options.DuplicateIdsPerQuery; i++)
            {
                allowlist[write++] = visibleCount == 0
                    ? (ulong)options.VectorCount + 1UL
                    : allowlist[i % visibleCount];
            }

            ulong firstUnknown = (ulong)options.VectorCount + 1UL + ((ulong)queryRow * (ulong)Math.Max(1, options.UnknownIdsPerQuery));
            for (int i = 0; i < options.UnknownIdsPerQuery; i++)
            {
                allowlist[write++] = firstUnknown + (ulong)i;
            }

            allowlists[queryRow] = allowlist;
        }

        var info = new GeneratedExactFilterInfo(
            options.FilterKind,
            GetSelectivityTarget(options.FilterKind),
            options.VectorCount == 0 ? 0 : (double)visibleCount / options.VectorCount,
            visibleCount,
            knownPerQuery,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            allowlistLength,
            minVisible,
            maxVisible,
            visibleCount,
            checked(knownPerQuery * options.QueryCount),
            checked(options.DuplicateIdsPerQuery * options.QueryCount),
            checked(options.UnknownIdsPerQuery * options.QueryCount),
            "deterministic query-rotated known IDs followed by requested duplicate known IDs and requested unknown IDs",
            "visibleCount = all: vectorCount; broad: ceiling(vectorCount * 0.50); selective: ceiling(vectorCount * 0.10); very-selective: min(vectorCount, topK - 1); empty: 0. For query q, known IDs start at (seed + q * 2654435761) mod vectorCount and advance by one modulo vectorCount. Duplicate IDs repeat earlier known IDs when visibleCount is greater than zero; empty filters duplicate an unknown ID so no indexed row becomes visible. Unknown IDs are greater than or equal to vectorCount + 1.",
            "Duplicate allowlist IDs are deliberately admitted as caller input and coalesced by ExactFlatIndex.Search; for empty filters, duplicate inputs are duplicate unknown IDs.",
            "Unknown allowlist IDs are deliberately admitted as caller input and ignored by ExactFlatIndex.Search.");

        return new GeneratedFilterSet(allowlists, info);
    }

    private static int GetVisibleCount(GeneratedExactFilteredOptions options) =>
        options.FilterKind switch
        {
            "all" => options.VectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(options.VectorCount * 0.50), 1, options.VectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(options.VectorCount * 0.10), 1, options.VectorCount),
            "very-selective" => Math.Min(options.VectorCount, options.TopK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact-filter kind.", nameof(options))
        };

    private static string GetSelectivityTarget(string filterKind) =>
        filterKind switch
        {
            "all" => "100% of indexed rows visible",
            "broad" => "approximately 50% of indexed rows visible",
            "selective" => "approximately 10% of indexed rows visible",
            "very-selective" => "fewer than top-k visible rows",
            "empty" => "0% of indexed rows visible",
            _ => "unknown"
        };

    private static TruthSet GenerateFilteredTruth(
        GeneratedDataset dataset,
        VectorMetric metric,
        int depth,
        GeneratedFilterSet filters)
    {
        var results = new TruthItem[dataset.QueryCount][];
        double[]? vectorMagnitudes = metric == VectorMetric.Cosine ? CalculateVectorMagnitudes(dataset) : null;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            bool[] visibleRows = BuildVisibleRows(filters.Allowlists[queryRow], dataset.VectorCount);
            int visibleCount = 0;
            foreach (bool visible in visibleRows)
            {
                if (visible)
                {
                    visibleCount++;
                }
            }

            if (visibleCount == 0)
            {
                results[queryRow] = [];
                continue;
            }

            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            double queryMagnitude = metric == VectorMetric.Cosine ? CalculateMagnitude(query) : 0;
            var candidates = new TruthItem[visibleCount];
            int write = 0;
            for (int vectorRow = 0; vectorRow < dataset.VectorCount; vectorRow++)
            {
                if (!visibleRows[vectorRow])
                {
                    continue;
                }

                float distance = CalculateDistance(
                    query,
                    dataset.GetVector(vectorRow),
                    metric,
                    queryMagnitude,
                    vectorMagnitudes is null ? 0 : vectorMagnitudes[vectorRow]);
                candidates[write++] = new TruthItem((ulong)vectorRow, distance);
            }

            Array.Sort(candidates, CompareTruthItems);
            int resultCount = Math.Min(depth, candidates.Length);
            var top = new TruthItem[resultCount];
            Array.Copy(candidates, top, resultCount);
            results[queryRow] = top;
        }

        return new TruthSet(results, depth);
    }

    private static bool[] BuildVisibleRows(ulong[] allowlist, int vectorCount)
    {
        var visible = new bool[vectorCount];
        foreach (ulong id in allowlist)
        {
            if (id < (ulong)vectorCount)
            {
                visible[checked((int)id)] = true;
            }
        }

        return visible;
    }

    private static void WarmupSearch(
        GeneratedExactFilteredOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        GeneratedFilterSet filters)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new ExactFlatSearchFilterWorkspace(options.VectorCount);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int queryRow = i % dataset.QueryCount;
            index.Search(dataset.GetQuery(queryRow), filters.Allowlists[queryRow], results, workspace);
        }
    }

    private static SearchMeasurement MeasureSearch(
        GeneratedExactFilteredOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        GeneratedFilterSet filters)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, index, filters, captureResults);
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureSingleRun(
        GeneratedExactFilteredOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        GeneratedFilterSet filters,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        var workspace = new ExactFlatSearchFilterWorkspace(options.VectorCount);
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            ulong[] allowlist = filters.Allowlists[queryRow];
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, allowlist, results, workspace);
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

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string status = measured ? "measured" : "notMeasured";
        string reason = measured
            ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local exact-filter noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured generated exact-filter runs for public ExactFlatIndex.Search(query, allowedIds, results, workspace); warmup, setup, index build, allowlist generation, workspace construction, filtered truth, result capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            reason,
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateMetricNoise(runs, "milliseconds", run => run.ElapsedMilliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "queriesPerSecond", run => run.Qps, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP50Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP95Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP99Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "bytesPerQuery", run => run.ManagedAllocatedBytesPerQuery, measured, unavailableReason));
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
            return new RunToRunMetricNoiseInfo("notMeasured", unit, null, null, null, null, null, null, unavailableReason);
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

    private static double[] CalculateVectorMagnitudes(GeneratedDataset dataset)
    {
        var magnitudes = new double[dataset.VectorCount];
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            magnitudes[row] = CalculateMagnitude(dataset.GetVector(row));
        }

        return magnitudes;
    }

    private static double CalculateMagnitude(ReadOnlySpan<float> values)
    {
        double squaredMagnitude = 0;
        foreach (float value in values)
        {
            squaredMagnitude += (double)value * value;
        }

        return Math.Sqrt(squaredMagnitude);
    }

    private static float CalculateDistance(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        VectorMetric metric,
        double queryMagnitude,
        double vectorMagnitude) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
            VectorMetric.InnerProduct => InnerProduct(query, vector),
            VectorMetric.Cosine => Cosine(query, vector, queryMagnitude, vectorMagnitude),
            _ => throw new ArgumentOutOfRangeException(nameof(metric), "Metric is not supported.")
        };

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static float InnerProduct(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (double)query[i] * vector[i];
        }

        return (float)-dotProduct;
    }

    private static float Cosine(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        double queryMagnitude,
        double vectorMagnitude)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (query[i] / queryMagnitude) * (vector[i] / vectorMagnitude);
        }

        return (float)(1 - dotProduct);
    }

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static bool DistanceMatches(float expected, float actual, int dimension, VectorMetric metric)
    {
        if (!float.IsFinite(actual))
        {
            return false;
        }

        float tolerance = metric == VectorMetric.SquaredEuclidean
            ? CalculateD026Tolerance(dimension, expected)
            : 1e-5f * MathF.Max(1f, MathF.Abs(expected));
        return MathF.Abs(expected - actual) <= tolerance;
    }

    private static float CalculateD026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static void ValidateOptions(GeneratedExactFilteredOptions options)
    {
        if (options.TopK > options.VectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.DuplicateIdsPerQuery < 0)
        {
            throw new ArgumentException("duplicate id count must be non-negative.", nameof(options));
        }

        if (options.UnknownIdsPerQuery < 0)
        {
            throw new ArgumentException("unknown id count must be non-negative.", nameof(options));
        }

        if (options.FilterKind == "very-selective" && options.TopK <= 1)
        {
            throw new ArgumentException("very-selective filters require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactFilteredOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactFilteredOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.FilterKind}-{options.Runs}r-{options.WarmupQueries}w-{options.Seed:X8}");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record GeneratedFilterSet(ulong[][] Allowlists, GeneratedExactFilterInfo Info);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);
}

public sealed record GeneratedExactFilteredResultComparison(
    double RecallAtK,
    double OrderedAgreement,
    GeneratedExactFilteredResultIntegrityInfo Integrity);
