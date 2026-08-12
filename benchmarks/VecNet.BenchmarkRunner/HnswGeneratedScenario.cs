using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class HnswGeneratedScenario
{
    private const string TaskId = "VEC-036";
    private const string SchemaName = "VecNet.HnswBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static HnswBenchmarkReport Run(HnswGeneratedOptions options, IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateDataset(dataset, options.Metric);
        TruthSet truth = ScalarGroundTruth.Generate(dataset, options.Metric, options.TopK);

        BuildMeasurement build = BuildIndex(options, dataset);
        WarmupSearch(options, dataset, build.Index);
        SearchMeasurement measurement = MeasureSearch(options, dataset, build.Index);

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            measurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);
        HnswReturnedResultIntegrityInfo returnedIntegrity = ValidateReturnedResults(dataset, options.Metric, measurement.Results, options.TopK);
        int extraResultCount = CountExtraResults(truth, measurement.Results, options.TopK);
        string validationStatus = comparison.MissingResultCount == 0 &&
            extraResultCount == 0 &&
            returnedIntegrity.Status == "passed"
                ? "passed"
                : "failed";

        RepositoryInfo repository = RepositoryInfo.Create();
        HnswMemoryEstimateInfo memoryEstimate = EstimateMemory(options, build.Index);

        return new HnswBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswGeneratedOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            new HnswEvidenceInfo(
                "smoke",
                "generated-hnsw-smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private generated HNSW runner output is not reviewed public evidence.",
                "Generated HNSW baseline-candidate policy has not been accepted.",
                "Generated HNSW regression-gate policy has not been accepted.",
                [
                    $"Generated {options.Metric} HNSW smoke evidence only; no external dataset source, license, version or checksum applies.",
                    "HNSW build, exact truth generation, warmup queries, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                    "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                    "Managed allocations are measured for the internal HnswIndex.Search(query, results, workspace) call boundary only; resident/process memory is explicitly not measured.",
                    "Graph and workspace memory values are estimates from the current VEC-035 layout, not resident, working-set, GC-heap or process memory measurements.",
                    "Not eligible for public performance, recall, memory, allocation, scale, baseline, regression-gate, external-dataset, ANN-Benchmarks or concurrency claims."
                ]),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswGeneratedOptions.ScenarioName, commandArguments.ToArray()),
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
                HnswGeneratedOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, exact scalar-reference truth generation, HNSW index build, warmup queries, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "InternalHnswEvaluation",
                nameof(HnswIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "internal/evaluation-only HnswIndex; built from generated vectors outside measured search timing; no public API, persistence, filtering, updates, external dataset mode or parameter matrix"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated vector row order, external ids 0..vectorCount-1",
                $"{options.Metric} private generated HNSW metric"),
            new HnswBuildInfo(
                "measured",
                build.ElapsedMilliseconds,
                new MeasurementStatusInfo(
                    "measured",
                    build.ManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                    "bytes",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around internal HnswIndex construction and Add calls for generated base vectors only; generated data setup and exact truth generation are excluded."),
                options.VectorCount,
                options.Dimension,
                "internal HnswIndex construction and generated base-vector Add calls",
                "generated data setup, exact scalar-reference truth generation, warmup, measured search, result comparison and report writing"),
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
                    "internal HnswIndex.Search(query, results, workspace)",
                    "generated data setup, exact scalar-reference truth generation, HNSW build, warmup queries, final-run result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search.aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each internal HnswIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswSearchWorkspace; setup, exact truth, HNSW build, warmup, result capture/comparison and report writing are excluded."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, private bytes, managed heap size and peak memory are not measured in VEC-036; see memoryEstimate for layout-derived estimates only."),
                RepeatedRuns: new RepeatedRunInfo(
                    options.Runs > 1 ? "measured" : "singleRun",
                    options.Runs,
                    options.Runs > 1,
                    options.Runs > 1
                        ? "Multiple measured HNSW search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                        : "Only one measured run executed, so cross-run variance/noise is not measured."),
                RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
                Warmup: new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed before measured runs using caller-owned results/workspace and excluded from measured timing and allocation totals."
                        : "No warmup queries were requested.")),
            memoryEstimate,
            new HnswMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                returnedIntegrity.Status,
                returnedIntegrity.DistanceMismatchCount,
                comparison.MissingResultCount,
                extraResultCount,
                returnedIntegrity,
                "set recall@k = returned ids intersect exact top-k ids divided by min(k, vectorCount), summed across measured queries",
                $"Every returned HNSW result is checked for finite distance, no duplicate ID within its query, generated-index ID membership, and {options.Metric} distance matching recomputation for that returned ID/query within the accepted runner tolerance. HNSW is approximate and exact top-k recall/order are recorded, not required."),
            new HnswValidationInfo(
                validationStatus,
                "generated-hnsw-smoke",
                FiniteVectors: true,
                TruthGenerated: true,
                FinalRunComparedToTruth: true,
                AllowsApproximateRecallBelowOne: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new HnswEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated HNSW reports are private local evidence only until a reviewed public summary policy exists.",
                "No generated HNSW baseline-candidate policy is accepted in VEC-036.",
                "No generated HNSW regression-gate policy is accepted in VEC-036."),
            [
                "Private generated HNSW smoke evidence only; not a public benchmark claim.",
                "This report exercises internal/evaluation-only HnswIndex and does not add or imply a public HNSW API.",
                $"HNSW metric in this report is {options.Metric}.",
                "Latency and QPS time only internal HnswIndex.Search(query, results, workspace) calls.",
                "HNSW build and exact scalar-reference truth generation are setup work and are excluded from measured search timing.",
                "Managed allocations are measured only for the internal HNSW search-call boundary.",
                "Approximate recall below 1.0 is allowed and recorded; exact recall/order are not required for validation.",
                "Memory fields are layout-derived estimates, not resident/process memory measurements.",
                "Baseline candidacy, comparison artifacts, regression gates, external dataset HNSW, parameter matrices, persistence, filtering, updates and public claims are out of scope."
            ]);
    }

    public static void Write(HnswBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    public static HnswReturnedResultIntegrityInfo ValidateReturnedResults(
        GeneratedDataset dataset,
        VectorMetric metric,
        SearchResult[][] actual,
        int topK)
    {
        int checkedResultCount = 0;
        int queryCountMismatchCount = actual.Length == dataset.QueryCount ? 0 : 1;
        int resultCountViolationCount = 0;
        int nonFiniteDistanceCount = 0;
        int duplicateIdCount = 0;
        int unknownIdCount = 0;
        int distanceMismatchCount = 0;
        int queryCount = Math.Min(dataset.QueryCount, actual.Length);
        int maxExpectedResults = Math.Min(topK, dataset.VectorCount);

        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            SearchResult[] returned = actual[queryRow];
            if (returned.Length > maxExpectedResults)
            {
                resultCountViolationCount++;
            }

            var seen = new HashSet<ulong>();
            for (int i = 0; i < returned.Length; i++)
            {
                SearchResult result = returned[i];
                checkedResultCount++;

                if (!float.IsFinite(result.Distance))
                {
                    nonFiniteDistanceCount++;
                }

                if (!seen.Add(result.Id))
                {
                    duplicateIdCount++;
                }

                if (result.Id >= (ulong)dataset.VectorCount)
                {
                    unknownIdCount++;
                    continue;
                }

                float expectedDistance = ScalarGroundTruth.CalculateDistance(
                    dataset.GetQuery(queryRow),
                    dataset.GetVector(checked((int)result.Id)),
                    metric);
                if (!ResultComparer.DistanceMatches(expectedDistance, result.Distance, dataset.Dimension, metric))
                {
                    distanceMismatchCount++;
                }
            }
        }

        bool passed = queryCountMismatchCount == 0 &&
            resultCountViolationCount == 0 &&
            nonFiniteDistanceCount == 0 &&
            duplicateIdCount == 0 &&
            unknownIdCount == 0 &&
            distanceMismatchCount == 0;

        return new HnswReturnedResultIntegrityInfo(
            passed ? "passed" : "failed",
            checkedResultCount,
            queryCountMismatchCount,
            resultCountViolationCount,
            nonFiniteDistanceCount,
            duplicateIdCount,
            unknownIdCount,
            distanceMismatchCount,
            $"For every returned approximate HNSW result: distance must be finite; IDs must be unique within a query; ID must be one of the generated index IDs; and reported distance must match recomputed {metric} distance for that query and returned ID within the accepted runner tolerance.",
            passed
                ? "All returned approximate HNSW results are well formed and distance-integrity checked."
                : "One or more returned approximate HNSW results failed well-formedness or distance-integrity checks.");
    }

    public static HnswReturnedResultIntegrityInfo ValidateReturnedResults(
        GeneratedDataset dataset,
        SearchResult[][] actual,
        int topK) =>
        ValidateReturnedResults(dataset, VectorMetric.SquaredEuclidean, actual, topK);

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswGeneratedOptions options) =>
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

    private static BuildMeasurement BuildIndex(HnswGeneratedOptions options, GeneratedDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var index = new HnswIndex(options.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        long elapsed = Stopwatch.GetTimestamp() - start;
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;
        return new BuildMeasurement(index, (double)elapsed / Stopwatch.Frequency * 1000, allocatedBytes);
    }

    private static void WarmupSearch(HnswGeneratedOptions options, GeneratedDataset dataset, HnswIndex index)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            index.Search(dataset.GetQuery(i % dataset.QueryCount), results, workspace);
        }
    }

    private static SearchMeasurement MeasureSearch(HnswGeneratedOptions options, GeneratedDataset dataset, HnswIndex index)
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

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureSingleRun(
        HnswGeneratedOptions options,
        GeneratedDataset dataset,
        HnswIndex index,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, results, workspace);
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

    private static HnswMemoryEstimateInfo EstimateMemory(HnswGeneratedOptions options, HnswIndex index)
    {
        int layerCount = index.MaxLayer + 1;
        var layers = new HnswLayerMemoryEstimateInfo[Math.Max(0, layerCount)];
        long adjacencyBytes = 0;
        long countBytes = 0;
        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = layer == 0 ? checked(options.M * 2) : options.M;
            long layerNeighborBytes = checked((long)options.VectorCount * stride * sizeof(int));
            long layerCountBytes = checked((long)options.VectorCount * sizeof(int));
            layers[layer] = new HnswLayerMemoryEstimateInfo(layer, stride, layerNeighborBytes, layerCountBytes);
            adjacencyBytes = checked(adjacencyBytes + layerNeighborBytes);
            countBytes = checked(countBytes + layerCountBytes);
        }

        long vectorBytes = checked((long)options.VectorCount * options.Dimension * sizeof(float));
        long idBytes = checked((long)options.VectorCount * sizeof(ulong));
        long levelBytes = checked((long)options.VectorCount * sizeof(int));
        long workspaceBytes = EstimateWorkspaceBytes(options.VectorCount, options.EfSearch);
        long total = checked(vectorBytes + idBytes + levelBytes + adjacencyBytes + countBytes + workspaceBytes);

        return new HnswMemoryEstimateInfo(
            "estimated",
            "layout-derived logical estimate for current VEC-035 arrays at vectorCount rows plus one search workspace",
            "bytes",
            total,
            vectorBytes,
            idBytes,
            levelBytes,
            adjacencyBytes,
            countBytes,
            workspaceBytes,
            index.MaxLayer,
            layerCount,
            layers,
            "Estimates row-major vector, id, level, fixed-stride adjacency/count arrays and caller-owned HnswSearchWorkspace from known element sizes; this is not a resident/process/GC-heap measurement.",
            [
                "Managed object headers, array alignment and Dictionary<ulong,int> duplicate-map overhead are excluded.",
                "Backing-array capacity slack from growth is excluded because capacity is not exposed by the internal HNSW type.",
                "Build-time temporary arrays and per-insertion workspaces are excluded from retained search memory estimates.",
                "Resident memory, working set, private bytes, GC heap size and peak process memory are not measured."
            ]);
    }

    private static long EstimateWorkspaceBytes(int maxElements, int maxEf) =>
        checked(
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)));

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string status = measured ? "measured" : "notMeasured";
        string reason = measured
            ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local HNSW noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured generated HNSW runs for internal HnswIndex.Search(query, results, workspace); warmup, setup, exact truth, HNSW build, result capture/comparison and report writing are excluded.",
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

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static void ValidateOptions(HnswGeneratedOptions options)
    {
        if (!IsSupportedMetric(options.Metric))
        {
            throw new ArgumentException("hnsw-generated supports SquaredEuclidean, InnerProduct and Cosine only.", nameof(options));
        }

        if (options.TopK > options.VectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.", nameof(options));
        }

        if (options.EfSearch < options.TopK)
        {
            throw new ArgumentException("ef-search must be greater than or equal to top-k.", nameof(options));
        }

        if (options.M is < 2 or > 64)
        {
            throw new ArgumentException("m must be in the range 2..64.", nameof(options));
        }

        if (options.EfConstruction < options.M || options.EfConstruction > 4096)
        {
            throw new ArgumentException("ef-construction must be at least m and no more than 4096.", nameof(options));
        }

        if (options.EfSearch > 4096)
        {
            throw new ArgumentException("ef-search must be in the range 1..4096.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }
    }

    private static bool IsSupportedMetric(VectorMetric metric) =>
        metric is VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine;

    private static void ValidateDataset(GeneratedDataset dataset, VectorMetric metric)
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

        if (metric == VectorMetric.Cosine)
        {
            ValidateNonZeroRows(dataset.Vectors, dataset.VectorCount, dataset.Dimension, "vector");
            ValidateNonZeroRows(dataset.Queries, dataset.QueryCount, dataset.Dimension, "query");
        }
    }

    private static void ValidateNonZeroRows(float[] values, int rowCount, int dimension, string rowKind)
    {
        for (int row = 0; row < rowCount; row++)
        {
            double magnitudeSquared = 0;
            int offset = checked(row * dimension);
            for (int i = 0; i < dimension; i++)
            {
                float value = values[offset + i];
                magnitudeSquared += (double)value * value;
            }

            if (magnitudeSquared == 0)
            {
                throw new InvalidOperationException($"Generated cosine {rowKind} data must not contain zero rows.");
            }
        }
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string CreateReportId(string? commit, HnswGeneratedOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswGeneratedOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private sealed record BuildMeasurement(HnswIndex Index, double ElapsedMilliseconds, long ManagedAllocatedBytes);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);
}
