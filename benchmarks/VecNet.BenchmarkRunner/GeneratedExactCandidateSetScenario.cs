using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactCandidateSetScenario
{
    private const string TaskId = "VEC-053";
    private const string SchemaName = "VecNet.ExactCandidateSetBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static GeneratedExactCandidateSetBenchmarkReport Run(
        GeneratedExactCandidateSetOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        GeneratedCandidateInputSet candidateInputs = GenerateCandidateInputs(options);
        TruthSet truth = GenerateCandidateSetTruth(dataset, options.Metric, options.TopK, candidateInputs);
        ExactFlatIndex index = BuildIndex(options, dataset);
        ExactFlatCandidateSet[] candidateSets = BuildCandidateSets(index, candidateInputs);

        WarmupSearch(options, dataset, index, candidateSets);
        SearchMeasurement measurement = MeasureSearch(options, dataset, index, candidateSets);
        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            measurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);

        RepositoryInfo repository = RepositoryInfo.Create();
        string validationStatus = comparison.Integrity.Status == "passed" ? "passed" : "failed";

        return new GeneratedExactCandidateSetBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactCandidateSetOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            new GeneratedExactCandidateSetEvidenceInfo(
                "smoke",
                "generated-exact-candidate-set-smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private generated exact candidate-set runner output is not reviewed public evidence.",
                "No generated exact candidate-set baseline-candidate policy is accepted in VEC-053.",
                "No generated exact candidate-set regression-gate policy is accepted in VEC-053.",
                [
                    "Generated exact candidate-set smoke evidence only; no external dataset source, license, version or checksum applies.",
                    "Application filter evaluation, authorization decisions, record storage and record hydration are outside VecNet and outside this report.",
                    "Generated data setup, finite validation, index build, candidate ID generation, candidate-set construction, warmup queries, scalar-reference filtered truth, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                    "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                    "Managed allocations are measured for public ExactFlatIndex.Search(query, candidateSet, results) calls only; candidate-set construction and resident/process memory are explicitly not measured.",
                    "Stored labels, persisted filters, HNSW/ANN filtering, matrix presets, baselines, comparisons, regression gates and public benchmark claims are out of scope."
                ]),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactCandidateSetOptions.ScenarioName, commandArguments.ToArray()),
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
                "scalar-reference-generated-candidate-set-filtered",
                options.TopK,
                "candidate ID input is coalesced to known generated IDs, then results are ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                GeneratedExactCandidateSetOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, exact-flat index build, candidate ID generation, candidate-set construction, scalar-reference filtered truth generation, warmup queries, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "ExactCandidateSet",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "public ExactFlatIndex constructor, public ExactFlatIndex.CreateCandidateSet(allowedIds) during setup and public ExactFlatIndex.Search(query, candidateSet, results) during measured search; no persistence, stored labels, HNSW/ANN filtering, updates, external dataset mode or public row ordinals"),
            candidateInputs.Info,
            CreateCandidateSetInfo(candidateSets, candidateInputs),
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
                    "public ExactFlatIndex.Search(query, candidateSet, results)",
                    "generated data setup, exact-flat index build, candidate ID generation, candidate-set construction through ExactFlatIndex.CreateCandidateSet, scalar-reference filtered truth generation, warmup queries, final-run result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search.aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, candidateSet, results) call using caller-owned SearchResult[] and prebuilt ExactFlatCandidateSet instances; generated data setup, exact-flat index build, candidate ID generation, candidate-set construction, filtered truth generation, warmup, result capture/comparison and report writing are excluded."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, private bytes, managed heap size, peak memory and candidate-set retained memory are not measured in VEC-053."),
                RepeatedRuns: new RepeatedRunInfo(
                    options.Runs > 1 ? "measured" : "singleRun",
                    options.Runs,
                    options.Runs > 1,
                    options.Runs > 1
                        ? "Multiple measured exact candidate-set search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                        : "Only one measured run executed, so cross-run variance/noise is not measured."),
                RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
                Warmup: new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed before measured runs using caller-owned results and prebuilt candidate sets, and excluded from measured timing and allocation totals."
                        : "No warmup queries were requested.")),
            new GeneratedExactFilteredMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.Integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
                comparison.Integrity.DistanceMismatchCount,
                comparison.Integrity.MissingResultCount,
                comparison.Integrity.ExtraResultCount,
                comparison.Integrity,
                "set recall@k = returned IDs intersect exact candidate-set filtered top-k IDs divided by exact filtered result count, summed across measured queries; empty exact filtered truth contributes a perfect denominator-free query",
                "Final measured run is compared against independently generated scalar-reference filtered truth from candidate ID inputs; result count, set membership, finite distances and distance tolerance are hard validation failures. Squared-L2 positional order differences are accepted only when the returned IDs are still in the scalar top-k set and the involved scalar-reference distances are inside the D-026 near-tie tolerance envelope."),
            new GeneratedExactCandidateSetValidationInfo(
                validationStatus,
                "generated-exact-candidate-set-smoke",
                FiniteVectors: true,
                TruthGenerated: true,
                CandidateSetsConstructed: true,
                FinalRunComparedToTruth: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new GeneratedExactCandidateSetEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated exact candidate-set reports are private local smoke evidence only until a reviewed public summary policy exists.",
                "No generated exact candidate-set baseline-candidate policy is accepted in VEC-053.",
                "No generated exact candidate-set regression-gate policy is accepted in VEC-053."),
            [
                "Private generated exact candidate-set smoke evidence only; not a public benchmark claim.",
                "Latency and QPS time only public ExactFlatIndex.Search(query, candidateSet, results) calls.",
                "Candidate ID generation and ExactFlatIndex.CreateCandidateSet construction are setup work and excluded from measured search timing and allocation samples.",
                "Application filter evaluation, authorization and record storage remain outside VecNet scope.",
                "Managed allocations are measured only for the public candidate-set search call boundary.",
                "Candidate-set selectivity is synthetic and deterministic according to candidateInput.generationFormula metadata.",
                "Baseline candidacy, comparison artifacts, regression gates, matrix presets, external dataset filters, HNSW/ANN filtering, stored labels, persistence, updates and public claims are out of scope."
            ]);
    }

    public static void Write(GeneratedExactCandidateSetBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactCandidateSetOptions options) =>
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

    private static ExactFlatIndex BuildIndex(GeneratedExactCandidateSetOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static GeneratedCandidateInputSet GenerateCandidateInputs(GeneratedExactCandidateSetOptions options)
    {
        int knownPerQuery = GetKnownCount(options);
        int inputLength = checked(knownPerQuery + options.DuplicateIdsPerQuery + options.UnknownIdsPerQuery);
        var inputs = new ulong[options.QueryCount][];
        int minVisible = knownPerQuery;
        int maxVisible = knownPerQuery;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            var input = new ulong[inputLength];
            int write = 0;
            ulong start = options.VectorCount == 0
                ? 0
                : (options.Seed + ((ulong)queryRow * 2_654_435_761UL)) % (ulong)options.VectorCount;
            for (int i = 0; i < knownPerQuery; i++)
            {
                input[write++] = (ulong)((start + (ulong)i) % (ulong)options.VectorCount);
            }

            for (int i = 0; i < options.DuplicateIdsPerQuery; i++)
            {
                input[write++] = knownPerQuery == 0
                    ? (ulong)options.VectorCount + 1UL
                    : input[i % knownPerQuery];
            }

            ulong firstUnknown = (ulong)options.VectorCount + 1UL + ((ulong)queryRow * (ulong)Math.Max(1, options.UnknownIdsPerQuery));
            for (int i = 0; i < options.UnknownIdsPerQuery; i++)
            {
                input[write++] = firstUnknown + (ulong)i;
            }

            inputs[queryRow] = input;
        }

        var info = new GeneratedExactCandidateInputInfo(
            options.CandidateSetKind,
            GetSelectivityTarget(options.CandidateSetKind),
            options.VectorCount == 0 ? 0 : (double)knownPerQuery / options.VectorCount,
            knownPerQuery,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            inputLength,
            minVisible,
            maxVisible,
            knownPerQuery,
            checked(knownPerQuery * options.QueryCount),
            checked(options.DuplicateIdsPerQuery * options.QueryCount),
            checked(options.UnknownIdsPerQuery * options.QueryCount),
            "deterministic query-rotated known IDs followed by requested duplicate known IDs and requested unknown IDs",
            "knownCount = all: vectorCount; broad: ceiling(vectorCount * 0.50); selective: ceiling(vectorCount * 0.10); very-selective: min(vectorCount, topK - 1); empty: 0. For query q, known IDs start at (seed + q * 2654435761) mod vectorCount and advance by one modulo vectorCount. Duplicate IDs repeat earlier known IDs when knownCount is greater than zero; empty candidate inputs duplicate an unknown ID so no indexed row becomes visible. Unknown IDs are greater than or equal to vectorCount + 1.",
            "Duplicate candidate input IDs are deliberately admitted as caller input and coalesced during ExactFlatIndex.CreateCandidateSet; for empty candidate sets, duplicate inputs are duplicate unknown IDs.",
            "Unknown candidate input IDs are deliberately admitted as caller input and ignored during ExactFlatIndex.CreateCandidateSet.",
            "Application filter evaluation, authorization decisions, source records and record hydration are outside VecNet; this runner only receives generated application-owned candidate IDs.");

        return new GeneratedCandidateInputSet(inputs, info);
    }

    private static GeneratedExactCandidateSetInfo CreateCandidateSetInfo(
        ExactFlatCandidateSet[] candidateSets,
        GeneratedCandidateInputSet candidateInputs)
    {
        int minCount = candidateSets.Length == 0 ? 0 : candidateSets.Min(item => item.Count);
        int maxCount = candidateSets.Length == 0 ? 0 : candidateSets.Max(item => item.Count);
        double meanCount = candidateSets.Length == 0 ? 0 : candidateSets.Average(item => item.Count);
        int totalCount = candidateSets.Sum(item => item.Count);

        return new GeneratedExactCandidateSetInfo(
            "constructedOutsideMeasuredSearch",
            "public ExactFlatIndex.CreateCandidateSet(allowedIds)",
            "Candidate-set construction is completed after index build and before warmup/measured search; it is excluded from latency samples and QPS.",
            "Candidate-set construction may allocate and is excluded from measured search allocation samples.",
            ConstructedBeforeMeasuredSearch: true,
            candidateSets.Length,
            candidateInputs.Info.KnownIdCountPerQuery,
            minCount,
            maxCount,
            meanCount,
            totalCount,
            "Candidate sets are opaque, exact-flat index-bound, generation-bound runtime objects; row ordinals are not exposed in this report.",
            "Duplicate candidate input IDs are coalesced into one candidate-set entry.",
            "Unknown candidate input IDs are ignored and do not create candidate-set entries.",
            "Candidate sets are transient setup artifacts, not persisted filters or public row-ordinal sidecars.");
    }

    private static ExactFlatCandidateSet[] BuildCandidateSets(
        ExactFlatIndex index,
        GeneratedCandidateInputSet candidateInputs)
    {
        var candidateSets = new ExactFlatCandidateSet[candidateInputs.InputIds.Length];
        for (int queryRow = 0; queryRow < candidateInputs.InputIds.Length; queryRow++)
        {
            candidateSets[queryRow] = index.CreateCandidateSet(candidateInputs.InputIds[queryRow]);
        }

        return candidateSets;
    }

    private static int GetKnownCount(GeneratedExactCandidateSetOptions options) =>
        options.CandidateSetKind switch
        {
            "all" => options.VectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(options.VectorCount * 0.50), 1, options.VectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(options.VectorCount * 0.10), 1, options.VectorCount),
            "very-selective" => Math.Min(options.VectorCount, options.TopK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact candidate-set kind.", nameof(options))
        };

    private static string GetSelectivityTarget(string candidateSetKind) =>
        candidateSetKind switch
        {
            "all" => "100% of indexed rows visible",
            "broad" => "approximately 50% of indexed rows visible",
            "selective" => "approximately 10% of indexed rows visible",
            "very-selective" => "fewer than top-k visible rows",
            "empty" => "0% of indexed rows visible",
            _ => "unknown"
        };

    private static TruthSet GenerateCandidateSetTruth(
        GeneratedDataset dataset,
        VectorMetric metric,
        int depth,
        GeneratedCandidateInputSet candidateInputs)
    {
        var results = new TruthItem[dataset.QueryCount][];
        double[]? vectorMagnitudes = metric == VectorMetric.Cosine ? CalculateVectorMagnitudes(dataset) : null;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            bool[] visibleRows = BuildVisibleRows(candidateInputs.InputIds[queryRow], dataset.VectorCount);
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

    private static bool[] BuildVisibleRows(ulong[] inputIds, int vectorCount)
    {
        var visible = new bool[vectorCount];
        foreach (ulong id in inputIds)
        {
            if (id < (ulong)vectorCount)
            {
                visible[checked((int)id)] = true;
            }
        }

        return visible;
    }

    private static void WarmupSearch(
        GeneratedExactCandidateSetOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ExactFlatCandidateSet[] candidateSets)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int queryRow = i % dataset.QueryCount;
            index.Search(dataset.GetQuery(queryRow), candidateSets[queryRow], results);
        }
    }

    private static SearchMeasurement MeasureSearch(
        GeneratedExactCandidateSetOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ExactFlatCandidateSet[] candidateSets)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, index, candidateSets, captureResults);
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureSingleRun(
        GeneratedExactCandidateSetOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ExactFlatCandidateSet[] candidateSets,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            ExactFlatCandidateSet candidateSet = candidateSets[queryRow];
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, candidateSet, results);
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
            ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local exact candidate-set noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured generated exact candidate-set runs for public ExactFlatIndex.Search(query, candidateSet, results); warmup, setup, index build, candidate ID generation, candidate-set construction, filtered truth, result capture/comparison and report writing are excluded.",
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

    private static void ValidateOptions(GeneratedExactCandidateSetOptions options)
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

        if (options.CandidateSetKind == "very-selective" && options.TopK <= 1)
        {
            throw new ArgumentException("very-selective candidate sets require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactCandidateSetOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactCandidateSetOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.CandidateSetKind}-{options.Runs}r-{options.WarmupQueries}w-{options.Seed:X8}");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record GeneratedCandidateInputSet(ulong[][] InputIds, GeneratedExactCandidateInputInfo Info);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);
}
