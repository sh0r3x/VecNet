using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalExactBenchmarkScenario
{
    private const string TaskId = "VEC-026";
    private const string SchemaName = "VecNet.ExternalBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static ExternalBenchmarkReport Run(
        FashionMnistExternalExactBenchmarkOptions options,
        IReadOnlyList<string> commandArguments)
    {
        LoadedExternalDataset dataset = LoadAndValidateDataset(options);
        ExactFlatIndex index = BuildIndex(dataset);

        WarmupSearch(options, dataset, index);
        SearchMeasurement measurement = MeasureSearch(options, dataset, index);

        TruthSet truth = CreateTruthSet(dataset.Truth, options.QueryCount);
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            measurement.Results,
            options.TopK,
            dataset.Dimension,
            options.Metric);
        int extraResultCount = CountExtraResults(truth, measurement.Results, options.TopK);
        string validationStatus = comparison.RecallAtK == 1 &&
            comparison.OrderedAgreement == 1 &&
            comparison.DistanceToleranceStatus == "passed" &&
            comparison.MissingResultCount == 0 &&
            extraResultCount == 0
                ? "passed"
                : "failed";

        RepositoryInfo repository = RepositoryInfo.Create();
        string[] sanitizedCommandArguments = SanitizeCommandArguments(commandArguments);
        return new ExternalBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            FashionMnistExternalExactBenchmarkOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            new ExternalBenchmarkEvidenceInfo(
                "smoke",
                "external-exact-smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private external exact-search runner output is not reviewed public evidence.",
                "External baseline-candidate policy has not been accepted.",
                "External regression-gate policy has not been accepted.",
                [
                    "External Fashion-MNIST exact-search smoke evidence only; not ANN-Benchmarks leaderboard evidence.",
                    "Dataset admission, checksum validation, matrix/truth loading, index build, warmup, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                    "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                    "Managed allocations are measured for public ExactFlatIndex.Search calls only; resident/process memory is explicitly not measured.",
                    "Not eligible for public performance, scale, baseline, regression-gate, resident-memory, ANN or concurrency claims."
                ]),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", sanitizedCommandArguments),
            new CommandInfo(FashionMnistExternalExactBenchmarkOptions.ScenarioName, sanitizedCommandArguments),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new ExternalBenchmarkDatasetInfo(
                dataset.Manifest.DatasetId,
                dataset.Manifest.Source,
                dataset.Manifest.License,
                dataset.Manifest.Privacy,
                dataset.Manifest.Shape,
                dataset.Manifest.Metric,
                new ExternalBenchmarkAdmissionManifestInfo(
                    dataset.Manifest.SchemaName,
                    dataset.Manifest.SchemaVersion,
                    dataset.Paths.RelativeManifestPath,
                    dataset.ManifestSha256),
                dataset.Manifest.RawFiles,
                dataset.Manifest.Conversion.OutputFiles,
                dataset.Manifest.Conversion,
                dataset.Manifest.Labels),
            new ExternalBenchmarkWorkloadInfo(
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                options.QueryCount,
                "first N query vectors from the admitted query matrix and existing truth artifact",
                dataset.Dimension,
                dataset.Manifest.Shape.SourceDataType,
                dataset.Manifest.Shape.ConvertedDataType,
                dataset.Manifest.Metric.UpstreamName,
                options.Metric.ToString(),
                dataset.Manifest.Metric.RankingNote,
                options.TopK,
                dataset.Truth.TruthDepth,
                dataset.Truth.TiePolicy),
            new ExternalBenchmarkTruthInfo(
                dataset.Truth.SchemaName,
                dataset.Truth.SchemaVersion,
                dataset.Manifest.Truth.Kind,
                dataset.Manifest.Truth.RelativePath,
                dataset.TruthSha256,
                "first N query vectors from the admitted query matrix",
                dataset.Truth.QuerySubsetCount,
                dataset.Truth.TruthDepth,
                options.TopK,
                dataset.Truth.TiePolicy,
                FashionMnistExactTruth.DistanceSemantics(options.Metric),
                dataset.Truth.SourceRawSha256),
            new ScenarioInfo(
                FashionMnistExternalExactBenchmarkOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "cache checks, manifest checksum validation, matrix load, truth load, index build, warmup queries, result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "Exact",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                dataset.Dimension,
                dataset.BaseCount,
                "public ExactFlatIndex constructor; built from admitted converted base matrix outside measured search timing; no persistence, filtering, updates or concurrency"),
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
                    "public ExactFlatIndex.Search(query, results)",
                    "download, checksum verification, IDX parsing, conversion, matrix load, truth load, index build, warmup queries, final-run result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search.aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, results) call; setup, matrix/truth load, index build, warmup, result capture/comparison and report writing are excluded."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, managed heap size and peak memory are not measured in VEC-026."),
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
                        ? "Warmup queries executed before measured runs and excluded from measured timing totals and allocation totals."
                        : "No warmup queries were requested.")),
            new ExternalBenchmarkMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.DistanceToleranceStatus,
                comparison.DistanceMismatchCount,
                comparison.MissingResultCount,
                extraResultCount),
            new ExternalBenchmarkValidationInfo(
                validationStatus,
                "external-exact-smoke",
                LoadedExistingTruth: true,
                FinalRunComparedToTruth: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new ExternalBenchmarkEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "External benchmark reports are private local evidence only until a reviewed public summary policy exists.",
                "No external baseline-candidate policy is accepted for VEC-026.",
                "No external regression-gate policy is accepted for VEC-026."),
            [
                "Private external exact-search smoke evidence only; not a public benchmark claim.",
                "This report loads an already admitted Fashion-MNIST cache and existing truth artifact; it does not download, parse IDX raw files, convert vectors or generate truth.",
                "Latency and QPS time only public ExactFlatIndex.Search(query, results) calls.",
                "ExactFlatIndex construction and vector insertion are setup work and are excluded from measured search timing.",
                "Managed allocations are measured only for the public search call boundary.",
                "Resident/process memory values are not measured.",
                "Baseline candidacy, comparison artifacts, regression gates, ANN work and ANN-Benchmarks HDF5 import are out of scope."
            ]);
    }

    public static void Write(ExternalBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static LoadedExternalDataset LoadAndValidateDataset(FashionMnistExternalExactBenchmarkOptions options)
    {
        ValidateOptions(options);
        DatasetPaths paths = DatasetPaths.Create(options.CacheRoot, FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
        string manifestPath = paths.ManifestPath;
        if (!File.Exists(manifestPath))
        {
            throw new FileNotFoundException("External benchmark requires an existing admitted Fashion-MNIST dataset manifest. Run the admission command separately; this benchmark command does not download, convert or generate truth.", manifestPath);
        }

        string manifestJson = File.ReadAllText(manifestPath);
        ExternalDatasetManifest manifest = ReportWriter.Deserialize<ExternalDatasetManifest>(manifestJson)
            ?? throw new InvalidDataException("External dataset manifest JSON could not be deserialized.");
        string manifestSha256 = FileChecksum.ComputeSha256(manifestPath);
        ValidateManifest(manifest, options.Metric);

        string conversionManifestPath = ResolveCacheRelativePath(options.CacheRoot, manifest.Conversion.ManifestRelativePath);
        RequireExistingFile(conversionManifestPath, "conversion manifest");
        string conversionManifestSha256 = FileChecksum.ComputeSha256(conversionManifestPath);
        if (!string.Equals(conversionManifestSha256, manifest.Conversion.ManifestSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Conversion manifest SHA-256 does not match the admitted dataset manifest.");
        }

        ExternalConvertedMatrixEntry baseEntry = GetMatrixEntry(manifest, "base");
        ExternalConvertedMatrixEntry queryEntry = GetMatrixEntry(manifest, "query");
        string baseMatrixPath = ResolveCacheRelativePath(options.CacheRoot, baseEntry.RelativePath);
        string queryMatrixPath = ResolveCacheRelativePath(options.CacheRoot, queryEntry.RelativePath);
        ValidateMatrixEntry(baseEntry, manifest.Shape.BaseCount, manifest.Shape.Dimension, "base");
        ValidateMatrixEntry(queryEntry, manifest.Shape.QueryCount, manifest.Shape.Dimension, "query");
        ValidateFileSha256(baseMatrixPath, baseEntry.Sha256, "base matrix");
        ValidateFileSha256(queryMatrixPath, queryEntry.Sha256, "query matrix");

        float[] baseVectors = DenseFloat32Matrix.Read(baseMatrixPath, (ulong)baseEntry.RowCount, (uint)baseEntry.Dimension);
        float[] queryVectors = DenseFloat32Matrix.Read(queryMatrixPath, (ulong)queryEntry.RowCount, (uint)queryEntry.Dimension);
        ValidateCosineSelectedRows(options.Metric, baseVectors, queryVectors, baseEntry.RowCount, queryEntry.RowCount, baseEntry.Dimension, options.QueryCount, options.WarmupQueries);

        string truthPath = ResolveCacheRelativePath(options.CacheRoot, manifest.Truth.RelativePath);
        ValidateFileSha256(truthPath, manifest.Truth.Sha256, "truth artifact");
        ExternalExactTruthArtifact truth = ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(truthPath))
            ?? throw new InvalidDataException("External exact truth JSON could not be deserialized.");
        ValidateTruth(manifest, truth, options);

        return new LoadedExternalDataset(
            paths,
            manifest,
            manifestSha256,
            truth,
            manifest.Truth.Sha256,
            baseVectors,
            queryVectors,
            baseEntry.RowCount,
            queryEntry.RowCount,
            baseEntry.Dimension);
    }

    private static void ValidateOptions(FashionMnistExternalExactBenchmarkOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.CacheRoot))
        {
            throw new ArgumentException("Cache root must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(options));
        }

        if (options.QueryCount <= 0)
        {
            throw new ArgumentException("Query count must be positive.", nameof(options));
        }

        if (options.TopK <= 0)
        {
            throw new ArgumentException("top-k must be positive.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.Cosine))
        {
            throw new ArgumentException("The Fashion-MNIST external exact benchmark supports only squared-euclidean and cosine metric mapping.", nameof(options));
        }
    }

    private static void ValidateManifest(ExternalDatasetManifest manifest, VectorMetric metric)
    {
        Require(manifest.SchemaName == "VecNet.ExternalDatasetManifest", "External dataset manifest schemaName must be VecNet.ExternalDatasetManifest.");
        Require(manifest.SchemaVersion == "0.1", "External dataset manifest schemaVersion must be 0.1.");
        string expectedDatasetId = FashionMnistDatasetSpecification.GetDatasetId(metric);
        Require(manifest.DatasetId == expectedDatasetId, $"External dataset manifest datasetId must be {expectedDatasetId}.");
        Require(manifest.Metric.VecNetMetric == metric.ToString(), $"External dataset manifest VecNet metric must be {metric}.");
        Require(!manifest.Privacy.PublicClaimEligible, "External dataset manifest public-claim eligibility must be false.");
        Require(!manifest.Privacy.BaselineCandidateEligible, "External dataset manifest baseline-candidate eligibility must be false.");
        Require(!manifest.Privacy.RegressionGateEligible, "External dataset manifest regression-gate eligibility must be false.");
        Require(manifest.Shape.BaseCount > 0, "External dataset manifest base count must be positive.");
        Require(manifest.Shape.QueryCount > 0, "External dataset manifest query count must be positive.");
        Require(manifest.Shape.Dimension > 0, "External dataset manifest dimension must be positive.");
        Require(manifest.RawFiles.Length == 4, "External dataset manifest must contain the four Fashion-MNIST raw file entries.");
        Require(!string.IsNullOrWhiteSpace(manifest.Conversion.ManifestRelativePath), "External dataset manifest must record the conversion manifest relative path.");
        Require(!string.IsNullOrWhiteSpace(manifest.Conversion.ManifestSha256), "External dataset manifest must record the conversion manifest SHA-256.");
        Require(!string.IsNullOrWhiteSpace(manifest.Truth.RelativePath), "External dataset manifest must record the truth artifact relative path.");
        Require(!string.IsNullOrWhiteSpace(manifest.Truth.Sha256), "External dataset manifest must record the truth artifact SHA-256.");
    }

    private static ExternalConvertedMatrixEntry GetMatrixEntry(ExternalDatasetManifest manifest, string role)
    {
        ExternalConvertedMatrixEntry[] entries = manifest.Conversion.OutputFiles
            .Where(entry => string.Equals(entry.Role, role, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        if (entries.Length != 1)
        {
            throw new InvalidDataException($"External dataset manifest must contain exactly one converted '{role}' matrix entry.");
        }

        return entries[0];
    }

    private static void ValidateMatrixEntry(ExternalConvertedMatrixEntry entry, int expectedRowCount, int expectedDimension, string role)
    {
        Require(entry.Format == DenseFloat32Matrix.SchemaName, $"Converted {role} matrix format must be {DenseFloat32Matrix.SchemaName}.");
        Require(entry.SchemaVersion == DenseFloat32Matrix.SchemaVersion, $"Converted {role} matrix schemaVersion must be {DenseFloat32Matrix.SchemaVersion}.");
        Require(entry.RowCount == expectedRowCount, $"Converted {role} matrix row count must match the manifest shape.");
        Require(entry.Dimension == expectedDimension, $"Converted {role} matrix dimension must match the manifest shape.");
        Require(!string.IsNullOrWhiteSpace(entry.RelativePath), $"Converted {role} matrix relative path must be present.");
        Require(!string.IsNullOrWhiteSpace(entry.Sha256), $"Converted {role} matrix SHA-256 must be present.");
    }

    private static void ValidateTruth(
        ExternalDatasetManifest manifest,
        ExternalExactTruthArtifact truth,
        FashionMnistExternalExactBenchmarkOptions options)
    {
        Require(truth.SchemaName == "VecNet.ExternalExactTruth", "External truth schemaName must be VecNet.ExternalExactTruth.");
        Require(truth.SchemaVersion == "0.1", "External truth schemaVersion must be 0.1.");
        Require(truth.DatasetId == manifest.DatasetId, "External truth datasetId must match the manifest.");
        Require(truth.BaseCount == manifest.Shape.BaseCount, "External truth base count must match the manifest shape.");
        Require(truth.Dimension == manifest.Shape.Dimension, "External truth dimension must match the manifest shape.");
        Require(truth.Metric == options.Metric.ToString(), $"External truth metric must be {options.Metric}.");
        Require(truth.QuerySubsetCount == manifest.Truth.QuerySubsetCount, "External truth query subset count must match the manifest truth summary.");
        Require(truth.TruthDepth == manifest.Truth.TruthDepth, "External truth depth must match the manifest truth summary.");
        Require(options.QueryCount <= manifest.Shape.QueryCount, "Requested query count must not exceed the admitted query matrix count.");
        Require(options.QueryCount <= truth.QuerySubsetCount, "Requested query count must not exceed the existing truth query subset count.");
        Require(options.TopK <= truth.TruthDepth, "Requested top-k must not exceed existing truth depth.");
        Require(options.WarmupQueries == 0 || options.WarmupQueries <= manifest.Shape.QueryCount, "Warmup query count must not exceed admitted query matrix count.");
        Require(truth.Queries.Length >= truth.QuerySubsetCount, "External truth query array must contain the declared query subset.");
        string[] manifestRawSha256 = manifest.RawFiles.Select(file => file.ComputedSha256).ToArray();
        Require(truth.SourceRawSha256.SequenceEqual(manifestRawSha256, StringComparer.OrdinalIgnoreCase), "External truth raw SHA-256 values must match the admitted manifest raw file values.");
        Require(truth.ConverterIdentity == manifest.Conversion.ConverterIdentity, "External truth converter identity must match the admitted manifest conversion identity.");

        for (int i = 0; i < options.QueryCount; i++)
        {
            ExternalTruthQuery query = truth.Queries[i];
            Require(query.QueryOrdinal == i, "External truth query ordinals must match the first-N query subset policy.");
            Require(query.Neighbors.Length >= options.TopK, "External truth query depth must cover requested top-k.");
        }
    }

    private static void ValidateCosineSelectedRows(
        VectorMetric metric,
        ReadOnlySpan<float> baseVectors,
        ReadOnlySpan<float> queryVectors,
        int baseCount,
        int queryMatrixCount,
        int dimension,
        int queryCount,
        int warmupQueries)
    {
        if (metric != VectorMetric.Cosine)
        {
            return;
        }

        int warmupSelectedQueryCount = warmupQueries == 0 ? 0 : Math.Min(warmupQueries, queryMatrixCount);
        int selectedQueryCount = Math.Max(queryCount, warmupSelectedQueryCount);
        FashionMnistExactTruth.ValidateNonZeroRows(baseVectors, baseCount, dimension, "base");
        FashionMnistExactTruth.ValidateNonZeroRows(queryVectors, selectedQueryCount, dimension, "query");
    }

    private static ExactFlatIndex BuildIndex(LoadedExternalDataset dataset)
    {
        var index = new ExactFlatIndex(dataset.Dimension, dataset.Manifest.Metric.VecNetMetric == nameof(VectorMetric.Cosine) ? VectorMetric.Cosine : VectorMetric.SquaredEuclidean);
        for (int row = 0; row < dataset.BaseCount; row++)
        {
            index.Add((ulong)row, dataset.GetBaseVector(row));
        }

        return index;
    }

    private static void WarmupSearch(
        FashionMnistExternalExactBenchmarkOptions options,
        LoadedExternalDataset dataset,
        ExactFlatIndex index)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            index.Search(dataset.GetQueryVector(i % dataset.QueryMatrixCount), results);
        }
    }

    private static SearchMeasurement MeasureSearch(
        FashionMnistExternalExactBenchmarkOptions options,
        LoadedExternalDataset dataset,
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

    private static SingleRunMeasurement MeasureSingleRun(
        FashionMnistExternalExactBenchmarkOptions options,
        LoadedExternalDataset dataset,
        ExactFlatIndex index,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQueryVector(queryRow);
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
            ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local external exact-search noise inspection."
            : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            "Across measured external exact-search runs for public ExactFlatIndex.Search(query, results); warmup, setup, matrix/truth loading, index build, result capture/comparison and report writing are excluded.",
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

    private static TruthSet CreateTruthSet(ExternalExactTruthArtifact artifact, int queryCount)
    {
        var results = new TruthItem[queryCount][];
        for (int i = 0; i < queryCount; i++)
        {
            results[i] = artifact.Queries[i].Neighbors
                .Select(neighbor => new TruthItem(neighbor.Id, neighbor.SquaredDistance))
                .ToArray();
        }

        return new TruthSet(results, artifact.TruthDepth);
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

    private static void ValidateFileSha256(string path, string expectedSha256, string description)
    {
        RequireExistingFile(path, description);
        string actual = FileChecksum.ComputeSha256(path);
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"{description} SHA-256 mismatch. Expected {expectedSha256}, got {actual}.");
        }
    }

    private static void RequireExistingFile(string path, string description)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Required existing {description} is missing. The external exact benchmark command does not download, convert or generate truth.", path);
        }
    }

    private static string ResolveCacheRelativePath(string cacheRoot, string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
        {
            throw new InvalidDataException("A cache artifact relative path is missing.");
        }

        if (Path.IsPathRooted(relativePath))
        {
            throw new InvalidDataException("Cache artifact paths in the external dataset manifest must be relative.");
        }

        string[] parts = relativePath.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries);
        if (parts.Any(part => part == "." || part == ".."))
        {
            throw new InvalidDataException("Cache artifact relative paths must not contain dot segments.");
        }

        return Path.Combine([cacheRoot, .. parts]);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidDataException(message);
        }
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string[] SanitizeCommandArguments(IReadOnlyList<string> commandArguments)
    {
        var sanitized = new string[commandArguments.Count];
        for (int i = 0; i < commandArguments.Count; i++)
        {
            string argument = commandArguments[i];
            sanitized[i] = argument;

            if (IsSensitivePathOption(argument))
            {
                if (i + 1 < commandArguments.Count)
                {
                    sanitized[++i] = "<redacted-local-path>";
                }

                continue;
            }

            int equalsIndex = argument.IndexOf('=');
            if (equalsIndex > 0 && IsSensitivePathOption(argument[..equalsIndex]))
            {
                sanitized[i] = argument[..(equalsIndex + 1)] + "<redacted-local-path>";
            }
        }

        return sanitized;
    }

    private static bool IsSensitivePathOption(string argument) =>
        string.Equals(argument, "--cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(argument, "--output", StringComparison.OrdinalIgnoreCase);

    private static string CreateReportId(string? commit, FashionMnistExternalExactBenchmarkOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistExternalExactBenchmarkOptions.ScenarioName}-{commitPart}-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w");
    }

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);

    private sealed record LoadedExternalDataset(
        DatasetPaths Paths,
        ExternalDatasetManifest Manifest,
        string ManifestSha256,
        ExternalExactTruthArtifact Truth,
        string TruthSha256,
        float[] BaseVectors,
        float[] QueryVectors,
        int BaseCount,
        int QueryMatrixCount,
        int Dimension)
    {
        public ReadOnlySpan<float> GetBaseVector(int row) => BaseVectors.AsSpan(row * Dimension, Dimension);

        public ReadOnlySpan<float> GetQueryVector(int row) => QueryVectors.AsSpan(row * Dimension, Dimension);
    }

    private sealed record DatasetPaths(string CacheRoot, string DatasetId, string ManifestPath)
    {
        public string RelativeManifestPath => string.Join('/', "manifests", DatasetId, "dataset-manifest.json");

        public static DatasetPaths Create(string cacheRoot, string datasetId) =>
            new(cacheRoot, datasetId, Path.Combine(cacheRoot, "manifests", datasetId, "dataset-manifest.json"));
    }
}
