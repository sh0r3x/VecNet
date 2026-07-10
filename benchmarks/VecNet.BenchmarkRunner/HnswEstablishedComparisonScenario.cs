using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace VecNet.BenchmarkRunner;

public static class HnswEstablishedComparisonScenario
{
    private const string TaskId = "VEC-118";
    private const string SchemaName = "VecNet.HnswEstablishedComparisonReport";
    private const string SchemaVersion = "0.1";
    private const string VecNetImplementationName = "VecNet";
    private const string HnswlibImplementationName = "hnswlib";

    public static HnswEstablishedComparisonReport Run(HnswEstablishedComparisonOptions options, IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);
        ValidateExternalTool(options.HnswlibPythonPath);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        TruthSet truth = ScalarGroundTruth.Generate(dataset, VectorMetric.SquaredEuclidean, options.TopK);

        VecNetMeasurement vecNet = MeasureVecNet(options, dataset);
        HnswlibMeasurement hnswlib = RunHnswlib(options, dataset);

        HnswEstablishedComparisonMetricsInfo vecNetMetrics = CreateMetrics(dataset, truth, vecNet.Results, options.TopK);
        HnswEstablishedComparisonMetricsInfo hnswlibMetrics = CreateMetrics(dataset, truth, hnswlib.Results, options.TopK);
        bool validationPassed =
            vecNetMetrics.ReturnedResultIntegrity.Status == "passed" &&
            hnswlibMetrics.ReturnedResultIntegrity.Status == "passed" &&
            vecNetMetrics.MissingResultCount == 0 &&
            hnswlibMetrics.MissingResultCount == 0;

        RepositoryInfo repository = RepositoryInfo.Create();

        return new HnswEstablishedComparisonReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswEstablishedComparisonOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswEstablishedComparisonOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new HnswEstablishedComparisonSourcePinningInfo(
                HnswlibImplementationName,
                HnswEstablishedComparisonOptions.HnswlibPackageName,
                HnswEstablishedComparisonOptions.HnswlibPackageSource,
                HnswEstablishedComparisonOptions.HnswlibVersion,
                HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256,
                HnswEstablishedComparisonOptions.HnswlibLicense,
                "Apache-2.0 dependency is used only by private, non-shipping comparison tooling and is not distributed with VecNet.",
                "hnswlib executes as Python/native external tooling through a private ignored virtual environment; VecNet remains managed .NET in-process.",
                "No hnswlib, Python or native asset is referenced by src/VecNet or included in the VecNet package."),
            CreateDesign(options),
            new DatasetInfo(
                GeneratedDataset.Kind,
                "generated-no-external-source",
                GeneratedDataset.Distribution,
                dataset.SeedText,
                VectorMetric.SquaredEuclidean.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount),
            new TruthInfo(ScalarGroundTruth.Kind, truth.Depth, ScalarGroundTruth.TiePolicy),
            new ScenarioInfo(
                HnswEstablishedComparisonOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "Generated data setup, exact scalar-reference truth generation, external process startup, binary interchange, warmup, final-run comparison, persistence/file scans and report writing are excluded from measured search latency and QPS."),
            CreateMethodology(),
            new HnswEstablishedComparisonParametersInfo(
                VectorMetric.SquaredEuclidean.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount,
                options.TopK,
                options.Runs,
                options.WarmupQueries,
                options.M,
                options.EfConstruction,
                options.EfConstruction,
                options.EfSearch,
                options.EfSearch,
                dataset.SeedText,
                FormatHex(options.HnswSeed),
                "generated vector row order, external ids 0..vectorCount-1",
                1),
            new HnswEstablishedComparisonImplementationResult(
                VecNetImplementationName,
                "pure-managed .NET public-preview squared-L2 HNSW",
                GetVecNetVersion(),
                "in-process managed .NET API call",
                typeof(HnswIndex).FullName ?? nameof(HnswIndex),
                vecNet.Build,
                vecNet.Search,
                vecNetMetrics,
                NotMeasured("bytes", "Whole-process, resident, GC and index-only memory attribution is not measured by VEC-118 comparison foundation."),
                FileFacts(vecNet.PersistedBytes, "VecNet preview HNSW Save output scanned after build/search timing and outside measured search latency.")),
            new HnswEstablishedComparisonImplementationResult(
                HnswlibImplementationName,
                "native hnswlib 0.8.0 through Python bindings",
                hnswlib.Version,
                "out-of-process Python/native hnswlib API call",
                hnswlib.Identity,
                hnswlib.Build,
                hnswlib.Search,
                hnswlibMetrics,
                NotMeasured("bytes", "hnswlib native allocation and resident/index-only memory are not measured by this runner."),
                FileFacts(hnswlib.PersistedBytes, "hnswlib save_index output scanned after build/search timing and outside measured search latency.")),
            new HnswEstablishedComparisonValidationInfo(
                validationPassed ? "passed" : "failed",
                "private-hnswlib-generated-comparison",
                FiniteVectors: true,
                TruthGenerated: true,
                IdenticalVectorsQueriesIdsAndParameters: true,
                VecNetComparedToTruth: true,
                HnswlibComparedToTruth: true,
                VecNetReturnedResultIntegrityPassed: vecNetMetrics.ReturnedResultIntegrity.Status == "passed",
                HnswlibReturnedResultIntegrityPassed: hnswlibMetrics.ReturnedResultIntegrity.Status == "passed",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                ComparisonPublicationEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private generated hnswlib comparison foundation only; not a public performance, recall, memory, capacity, storage-size, baseline, publication or regression-gate claim.",
                "Both implementations receive identical generated float32 vectors, query vectors, external IDs, metric, top-k, M, efConstruction/ef_construction, efSearch/ef and comparable seed value.",
                "VecNet is measured as in-process managed .NET; hnswlib is measured through a Python process and native extension. The report discloses this boundary instead of treating it as equivalent deployment.",
                "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                "Memory fields are intentionally notMeasured because reliable index-only native-versus-managed memory attribution is not implemented in this foundation.",
                "Persisted bytes are file facts from private ignored outputs and are not storage-size claims."
            ]);
    }

    public static void Write(HnswEstablishedComparisonReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static HnswEstablishedComparisonMetricsInfo CreateMetrics(
        GeneratedDataset dataset,
        TruthSet truth,
        SearchResult[][] results,
        int topK)
    {
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            results,
            topK,
            dataset.Dimension,
            VectorMetric.SquaredEuclidean);
        HnswReturnedResultIntegrityInfo integrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, results, topK);
        int extraResultCount = CountExtraResults(truth, results, topK);
        return new HnswEstablishedComparisonMetricsInfo(
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.DistanceToleranceStatus,
            comparison.DistanceMismatchCount,
            comparison.MissingResultCount,
            extraResultCount,
            integrity,
            "set recall@k = returned ids intersect exact top-k ids divided by min(k, vectorCount), summed across measured queries",
            "Ordered agreement is diagnostic only because HNSW is approximate and near-tie/order differences are expected across implementations.");
    }

    private static VecNetMeasurement MeasureVecNet(HnswEstablishedComparisonOptions options, GeneratedDataset dataset)
    {
        HnswIndex index = null!;
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long buildStart = Stopwatch.GetTimestamp();
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        index = new HnswIndex(options.Dimension, VectorMetric.SquaredEuclidean, hnswOptions);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        long buildTicks = Stopwatch.GetTimestamp() - buildStart;
        long buildAllocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;

        WarmupVecNet(options, dataset, index);
        SearchMeasurement search = MeasureVecNetSearch(options, dataset, index);
        index.Save(options.VecNetSnapshotDirectory);
        long persistedBytes = SumFiles(options.VecNetSnapshotDirectory);

        return new VecNetMeasurement(
            new HnswEstablishedComparisonBuildInfo(
                "measured",
                TicksToMilliseconds(buildTicks),
                FileMeasured(buildAllocatedBytes, "bytes", "Measured with GC.GetAllocatedBytesForCurrentThread around public-preview HnswIndex construction and Add calls only."),
                "new HnswIndex(...) plus generated base-vector Add calls",
                "generated data creation, exact truth generation, warmup, measured search, save, external hnswlib execution and report writing"),
            search.Search,
            search.Results,
            persistedBytes);
    }

    private static void WarmupVecNet(HnswEstablishedComparisonOptions options, GeneratedDataset dataset, HnswIndex index)
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

    private static SearchMeasurement MeasureVecNetSearch(HnswEstablishedComparisonOptions options, GeneratedDataset dataset, HnswIndex index)
    {
        var runs = new HnswEstablishedComparisonSearchRunInfo[options.Runs];
        SearchResult[][]? captured = null;

        for (int run = 0; run < options.Runs; run++)
        {
            bool capture = run == options.Runs - 1;
            SingleRunMeasurement measurement = MeasureVecNetSingleRun(options, dataset, index, capture);
            runs[run] = measurement.Run with { RunNumber = run + 1 };
            if (capture)
            {
                captured = measurement.Results;
            }
        }

        HnswEstablishedComparisonAggregateTimingInfo aggregate = AggregateRuns(runs);
        var search = new HnswEstablishedComparisonSearchInfo(
            "measured",
            options.QueryCount,
            aggregate.MeanElapsedMilliseconds,
            aggregate.MeanLatencyP50Milliseconds,
            aggregate.MeanLatencyP95Milliseconds,
            aggregate.MeanLatencyP99Milliseconds,
            aggregate.MeanQps,
            runs,
            aggregate,
            FileMeasured(
                aggregate.MeanManagedAllocatedBytesPerQuery ?? 0,
                "bytesPerQuery",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each public-preview HnswIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswSearchWorkspace."));

        return new SearchMeasurement(search, captured ?? []);
    }

    private static SingleRunMeasurement MeasureVecNetSingleRun(
        HnswEstablishedComparisonOptions options,
        GeneratedDataset dataset,
        HnswIndex index,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        SearchResult[][]? captured = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int query = 0; query < options.QueryCount; query++)
        {
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(dataset.GetQuery(query), results, workspace);
            long elapsed = Stopwatch.GetTimestamp() - start;
            totalAllocatedBytes += GC.GetAllocatedBytesForCurrentThread() - allocationStart;
            latencyTicks[query] = elapsed;
            totalTicks += elapsed;

            if (captureResults)
            {
                var queryResults = new SearchResult[written];
                results.AsSpan(0, written).CopyTo(queryResults);
                captured![query] = queryResults;
            }
        }

        Array.Sort(latencyTicks);
        double elapsedMilliseconds = TicksToMilliseconds(totalTicks);
        double elapsedSeconds = (double)totalTicks / Stopwatch.Frequency;
        return new SingleRunMeasurement(
            new HnswEstablishedComparisonSearchRunInfo(
                RunNumber: 0,
                options.QueryCount,
                elapsedMilliseconds,
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.50, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.95, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.99, Stopwatch.Frequency),
                elapsedSeconds == 0 ? double.PositiveInfinity : options.QueryCount / elapsedSeconds,
                totalAllocatedBytes,
                (double)totalAllocatedBytes / options.QueryCount),
            captured);
    }

    private static HnswlibMeasurement RunHnswlib(HnswEstablishedComparisonOptions options, GeneratedDataset dataset)
    {
        Directory.CreateDirectory(options.WorkDirectory);
        string vectorsPath = Path.Combine(options.WorkDirectory, "vectors.f32");
        string queriesPath = Path.Combine(options.WorkDirectory, "queries.f32");
        string idsPath = Path.Combine(options.WorkDirectory, "ids.u64");
        string configPath = Path.Combine(options.WorkDirectory, "hnswlib-config.json");
        string outputPath = Path.Combine(options.WorkDirectory, "hnswlib-results.json");
        string scriptPath = Path.Combine(options.WorkDirectory, "run-hnswlib.py");

        WriteFloat32(vectorsPath, dataset.Vectors);
        WriteFloat32(queriesPath, dataset.Queries);
        WriteIds(idsPath, dataset.VectorCount);
        File.WriteAllText(scriptPath, PythonDriver);
        File.WriteAllText(configPath, ReportWriter.Serialize(new HnswlibDriverConfig(
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            options.M,
            options.EfConstruction,
            options.EfSearch,
            unchecked((int)(options.HnswSeed & 0x7FFF_FFFF)),
            vectorsPath,
            queriesPath,
            idsPath,
            options.HnswlibIndexPath,
            outputPath)));

        var startInfo = new ProcessStartInfo(options.HnswlibPythonPath)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add(configPath);

        using Process process = Process.Start(startInfo) ?? throw new IOException("Failed to start hnswlib Python process.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new IOException($"hnswlib Python process failed with exit code {process.ExitCode}. stdout: {stdout} stderr: {stderr}");
        }

        if (!File.Exists(outputPath))
        {
            throw new IOException("hnswlib Python process completed without writing its result JSON.");
        }

        return ParseHnswlibResult(File.ReadAllText(outputPath), options);
    }

    private static HnswlibMeasurement ParseHnswlibResult(string json, HnswEstablishedComparisonOptions options)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        string version = root.GetProperty("version").GetString() ?? "unknown";
        if (!string.Equals(version, HnswEstablishedComparisonOptions.HnswlibVersion, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"hnswlib version '{version}' does not match pinned version '{HnswEstablishedComparisonOptions.HnswlibVersion}'.");
        }

        JsonElement build = root.GetProperty("build");
        JsonElement search = root.GetProperty("search");
        HnswEstablishedComparisonSearchRunInfo[] runs = ParseRuns(search.GetProperty("runs"));
        HnswEstablishedComparisonAggregateTimingInfo aggregate = AggregateRuns(runs);
        SearchResult[][] results = ParseResults(root.GetProperty("results"));
        long persistedBytes = root.GetProperty("persistedBytes").GetInt64();

        return new HnswlibMeasurement(
            version,
            root.GetProperty("identity").GetString() ?? "hnswlib Python bindings",
            new HnswEstablishedComparisonBuildInfo(
                "measured",
                build.GetProperty("elapsedMilliseconds").GetDouble(),
                NotMeasured("bytes", "Managed allocations are not meaningful for the native hnswlib Python process from this .NET runner."),
                "hnswlib.Index(space='l2'), init_index(max_elements, M, ef_construction, random_seed) and add_items(vectors, ids, num_threads=1)",
                "binary input loading, Python process startup, warmup, measured search, save_index and JSON writing"),
            new HnswEstablishedComparisonSearchInfo(
                "measured",
                options.QueryCount,
                aggregate.MeanElapsedMilliseconds,
                aggregate.MeanLatencyP50Milliseconds,
                aggregate.MeanLatencyP95Milliseconds,
                aggregate.MeanLatencyP99Milliseconds,
                aggregate.MeanQps,
                runs,
                aggregate,
                NotMeasured("bytesPerQuery", "hnswlib native/Python allocation is not measured by this .NET runner.")),
            results,
            persistedBytes);
    }

    private static HnswEstablishedComparisonSearchRunInfo[] ParseRuns(JsonElement runsJson)
    {
        var runs = new HnswEstablishedComparisonSearchRunInfo[runsJson.GetArrayLength()];
        int index = 0;
        foreach (JsonElement run in runsJson.EnumerateArray())
        {
            runs[index++] = new HnswEstablishedComparisonSearchRunInfo(
                run.GetProperty("runNumber").GetInt32(),
                run.GetProperty("measuredQueryCount").GetInt32(),
                run.GetProperty("elapsedMilliseconds").GetDouble(),
                run.GetProperty("latencyP50Milliseconds").GetDouble(),
                run.GetProperty("latencyP95Milliseconds").GetDouble(),
                run.GetProperty("latencyP99Milliseconds").GetDouble(),
                run.GetProperty("qps").GetDouble(),
                ManagedAllocatedBytes: null,
                ManagedAllocatedBytesPerQuery: null);
        }

        return runs;
    }

    private static SearchResult[][] ParseResults(JsonElement resultsJson)
    {
        var results = new SearchResult[resultsJson.GetArrayLength()][];
        int query = 0;
        foreach (JsonElement row in resultsJson.EnumerateArray())
        {
            JsonElement labels = row.GetProperty("ids");
            JsonElement distances = row.GetProperty("distances");
            var queryResults = new SearchResult[labels.GetArrayLength()];
            for (int i = 0; i < queryResults.Length; i++)
            {
                queryResults[i] = new SearchResult(labels[i].GetUInt64(), distances[i].GetSingle());
            }

            results[query++] = queryResults;
        }

        return results;
    }

    private static HnswEstablishedComparisonAggregateTimingInfo AggregateRuns(HnswEstablishedComparisonSearchRunInfo[] runs)
    {
        double? meanManagedBytes = runs.All(run => run.ManagedAllocatedBytes.HasValue)
            ? runs.Average(run => (double)run.ManagedAllocatedBytes!.Value)
            : null;
        double? meanManagedBytesPerQuery = runs.All(run => run.ManagedAllocatedBytesPerQuery.HasValue)
            ? runs.Average(run => run.ManagedAllocatedBytesPerQuery!.Value)
            : null;
        return new HnswEstablishedComparisonAggregateTimingInfo(
            runs.Length,
            runs.Length == 0 ? 0 : runs[0].MeasuredQueryCount,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds),
            runs.Average(run => run.LatencyP50Milliseconds),
            runs.Average(run => run.LatencyP95Milliseconds),
            runs.Average(run => run.LatencyP99Milliseconds),
            runs.Average(run => run.Qps),
            runs.Min(run => run.Qps),
            runs.Max(run => run.Qps),
            meanManagedBytes,
            meanManagedBytesPerQuery);
    }

    private static HnswEstablishedComparisonEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "private-hnswlib-generated-comparison",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "Private hnswlib comparison output has not been reviewed for public reporting and is not a public VecNet claim.",
            "Established-implementation comparison reports are not baseline candidates.",
            "No comparison-publication policy is accepted for VEC-118 output.",
            "No regression-gate policy or threshold is accepted for hnswlib comparison output.",
            [
                "Generated squared-L2 comparison foundation only.",
                "hnswlib is native Python tooling and is not a VecNet product dependency.",
                "Python process/API overhead is disclosed and not treated as equivalent to an in-process managed API boundary.",
                "Memory is not measured for either side in this foundation.",
                "Not eligible for public performance, recall, memory, capacity, storage-size, baseline, comparison-publication or regression-gate claims."
            ]);

    private static HnswEstablishedComparisonDesignInfo CreateDesign(HnswEstablishedComparisonOptions options)
    {
        string role = HnswEstablishedComparisonOptions.RepresentativeDimensions.Contains(options.Dimension)
            ? "representative"
            : HnswEstablishedComparisonOptions.OptionalAdversarialDimensions.Contains(options.Dimension)
                ? "optional-adversarial-tail"
                : "custom-smoke";
        return new HnswEstablishedComparisonDesignInfo(
            HnswEstablishedComparisonOptions.RepresentativeDimensions,
            HnswEstablishedComparisonOptions.OptionalAdversarialDimensions,
            options.Dimension,
            role,
            VectorMetric.SquaredEuclidean.ToString(),
            "Accepted comparison design preserves representative generated dimensions 128, 384 and 768; this command may run one bounded smoke case.",
            "Dimension 386 is optional adversarial/tail coverage only and must not replace representative dimension 384.");
    }

    private static HnswEstablishedComparisonMethodologyInfo CreateMethodology() =>
        new(
            "C# generates one deterministic float32 vector/query dataset and external IDs 0..vectorCount-1, then feeds the same binary inputs to VecNet and hnswlib.",
            "Measured build times include only index construction/add operations. Measured search latency sums per-query calls after warmup. QPS is measuredQueryCount divided by summed measured per-query elapsed time.",
            "Dataset generation, exact truth generation, Python process startup, binary interchange, warmup, result conversion/comparison, persistence/file scans and report writing.",
            "Nearest-rank over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to the sample range.",
            "Top-level fields are arithmetic means across per-run percentile/QPS/elapsed values.",
            "Single-threaded first comparison: VecNet uses one caller thread and hnswlib receives num_threads=1 where its Python API accepts it.",
            "hnswlib runs out-of-process through Python/native extension tooling from an ignored private virtual environment; this boundary is explicitly not a product dependency.",
            "Returned IDs are compared to scalar exact truth for recall and ordered agreement; every returned result is checked for known ID, duplicate IDs, finite distance and recomputed squared-L2 distance.");

    private static HnswEstablishedComparisonEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "VEC-118 private raw output is not reviewed public evidence.",
            "Established-comparison output is not a VecNet baseline candidate.",
            "No accepted public comparison-summary policy exists.",
            "No hnswlib comparison regression-gate policy or threshold exists.");

    private static void ValidateOptions(HnswEstablishedComparisonOptions options)
    {
        if (options.Metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("hnswlib-generated-comparison supports only SquaredEuclidean.", nameof(options));
        }

        if (options.TopK > options.VectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.", nameof(options));
        }

        if (options.Runs is < 1 or > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.M is < 2 or > 64)
        {
            throw new ArgumentException("m must be in the range 2..64.", nameof(options));
        }

        if (options.EfConstruction < options.M || options.EfConstruction > 4096)
        {
            throw new ArgumentException("ef-construction must be at least m and no more than 4096.", nameof(options));
        }

        if (options.EfSearch < options.TopK || options.EfSearch > 4096)
        {
            throw new ArgumentException("ef-search must be at least top-k and no more than 4096.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath) ||
            string.IsNullOrWhiteSpace(options.WorkDirectory) ||
            string.IsNullOrWhiteSpace(options.VecNetSnapshotDirectory) ||
            string.IsNullOrWhiteSpace(options.HnswlibIndexPath) ||
            string.IsNullOrWhiteSpace(options.HnswlibPythonPath))
        {
            throw new ArgumentException("output, work directory, snapshot, hnswlib index and hnswlib python paths must not be empty.", nameof(options));
        }
    }

    private static void ValidateExternalTool(string pythonPath)
    {
        if (!File.Exists(pythonPath))
        {
            throw new FileNotFoundException("Pinned hnswlib Python environment is unavailable; comparison evidence was not produced.", pythonPath);
        }
    }

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswEstablishedComparisonOptions options) =>
        new(
            VectorMetric.SquaredEuclidean,
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            options.Runs,
            options.WarmupQueries);

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

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static void WriteFloat32(string path, float[] values)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        foreach (float value in values)
        {
            writer.Write(value);
        }
    }

    private static void WriteIds(string path, int count)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        for (int i = 0; i < count; i++)
        {
            writer.Write((ulong)i);
        }
    }

    private static long SumFiles(string directoryPath) =>
        Directory.Exists(directoryPath)
            ? Directory.EnumerateFiles(directoryPath).Sum(path => new FileInfo(path).Length)
            : 0;

    private static MeasurementStatusInfo FileFacts(long bytes, string reason) =>
        new("fileFacts", bytes.ToString(CultureInfo.InvariantCulture), "bytes", reason);

    private static MeasurementStatusInfo FileMeasured(long bytes, string unit, string reason) =>
        new("measured", bytes.ToString(CultureInfo.InvariantCulture), unit, reason);

    private static MeasurementStatusInfo FileMeasured(double value, string unit, string reason) =>
        new("measured", value.ToString(CultureInfo.InvariantCulture), unit, reason);

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static double TicksToMilliseconds(long ticks) => (double)ticks / Stopwatch.Frequency * 1000;

    private static string GetVecNetVersion() =>
        typeof(HnswIndex).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ??
        typeof(HnswIndex).Assembly.GetName().Version?.ToString() ??
        "unknown";

    private static string CreateReportId(string? commit, HnswEstablishedComparisonOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswEstablishedComparisonOptions.ScenarioName}-{commitPart}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private sealed record VecNetMeasurement(
        HnswEstablishedComparisonBuildInfo Build,
        HnswEstablishedComparisonSearchInfo Search,
        SearchResult[][] Results,
        long PersistedBytes);

    private sealed record HnswlibMeasurement(
        string Version,
        string Identity,
        HnswEstablishedComparisonBuildInfo Build,
        HnswEstablishedComparisonSearchInfo Search,
        SearchResult[][] Results,
        long PersistedBytes);

    private sealed record SingleRunMeasurement(
        HnswEstablishedComparisonSearchRunInfo Run,
        SearchResult[][]? Results);

    private sealed record SearchMeasurement(
        HnswEstablishedComparisonSearchInfo Search,
        SearchResult[][] Results);

    private sealed record HnswlibDriverConfig(
        int Dimension,
        int VectorCount,
        int QueryCount,
        int TopK,
        int Runs,
        int WarmupQueries,
        int M,
        int EfConstruction,
        int EfSearch,
        int RandomSeed,
        string VectorsPath,
        string QueriesPath,
        string IdsPath,
        string IndexPath,
        string OutputPath);

    private const string PythonDriver = """
import json
import importlib.metadata
import math
import os
import platform
import sys
import time

import hnswlib
import numpy as np


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(math.ceil(len(ordered) * p)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index] / 1_000_000.0


with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

try:
    hnswlib_version = importlib.metadata.version("hnswlib")
except importlib.metadata.PackageNotFoundError:
    hnswlib_version = getattr(hnswlib, "__version__", "unknown")

vectors = np.fromfile(cfg["vectorsPath"], dtype=np.float32).reshape(cfg["vectorCount"], cfg["dimension"])
queries = np.fromfile(cfg["queriesPath"], dtype=np.float32).reshape(cfg["queryCount"], cfg["dimension"])
ids = np.fromfile(cfg["idsPath"], dtype=np.uint64)

index = hnswlib.Index(space="l2", dim=cfg["dimension"])
build_start = time.perf_counter_ns()
index.init_index(
    max_elements=cfg["vectorCount"],
    ef_construction=cfg["efConstruction"],
    M=cfg["m"],
    random_seed=cfg["randomSeed"])
index.add_items(vectors, ids, num_threads=1)
build_elapsed_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0
index.set_ef(cfg["efSearch"])
try:
    index.set_num_threads(1)
except AttributeError:
    pass

for i in range(cfg["warmupQueries"]):
    query_index = i % cfg["queryCount"]
    index.knn_query(queries[query_index:query_index + 1], k=cfg["topK"], num_threads=1)

runs = []
last_labels = None
last_distances = None
for run_index in range(cfg["runs"]):
    latencies = []
    labels = []
    distances = []
    total_ns = 0
    for query_index in range(cfg["queryCount"]):
        start = time.perf_counter_ns()
        query_labels, query_distances = index.knn_query(queries[query_index:query_index + 1], k=cfg["topK"], num_threads=1)
        elapsed = time.perf_counter_ns() - start
        total_ns += elapsed
        latencies.append(elapsed)
        labels.append([int(value) for value in query_labels[0].tolist()])
        distances.append([float(value) for value in query_distances[0].tolist()])
    elapsed_ms = total_ns / 1_000_000.0
    elapsed_seconds = total_ns / 1_000_000_000.0
    runs.append({
        "runNumber": run_index + 1,
        "measuredQueryCount": cfg["queryCount"],
        "elapsedMilliseconds": elapsed_ms,
        "latencyP50Milliseconds": percentile(latencies, 0.50),
        "latencyP95Milliseconds": percentile(latencies, 0.95),
        "latencyP99Milliseconds": percentile(latencies, 0.99),
        "qps": float("inf") if elapsed_seconds == 0 else cfg["queryCount"] / elapsed_seconds
    })
    last_labels = labels
    last_distances = distances

index.save_index(cfg["indexPath"])
persisted_bytes = os.path.getsize(cfg["indexPath"])
results = []
for labels, distances in zip(last_labels or [], last_distances or []):
    results.append({"ids": labels, "distances": distances})

output = {
    "version": hnswlib_version,
    "identity": "hnswlib " + hnswlib_version + " on Python " + platform.python_version(),
    "build": {"elapsedMilliseconds": build_elapsed_ms},
    "search": {"runs": runs},
    "persistedBytes": persisted_bytes,
    "results": results
}
with open(cfg["outputPath"], "w", encoding="utf-8") as f:
    json.dump(output, f)
""";
}
