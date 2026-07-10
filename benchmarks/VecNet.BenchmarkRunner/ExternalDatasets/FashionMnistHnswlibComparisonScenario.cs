using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistHnswlibComparisonScenario
{
    private const string TaskId = "VEC-120";
    private const string SchemaName = "VecNet.FashionMnistHnswlibComparisonReport";
    private const string SchemaVersion = "0.1";

    public static FashionMnistHnswlibComparisonReport Run(
        FashionMnistHnswlibComparisonOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);
        ValidateExternalTool(options.HnswlibPythonPath);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadDataset(options);

        TruthSet truth = CreateTruthSet(dataset.Truth, options.QueryCount);
        VecNetMeasurement vecNet = MeasureVecNet(options, dataset);
        HnswlibMeasurement hnswlib = RunHnswlib(options, dataset);

        HnswEstablishedComparisonMetricsInfo vecNetMetrics = CreateMetrics(dataset, truth, vecNet.Results, options.QueryCount, options.TopK);
        HnswEstablishedComparisonMetricsInfo hnswlibMetrics = CreateMetrics(dataset, truth, hnswlib.Results, options.QueryCount, options.TopK);
        bool validationPassed =
            vecNetMetrics.ReturnedResultIntegrity.Status == "passed" &&
            hnswlibMetrics.ReturnedResultIntegrity.Status == "passed" &&
            vecNetMetrics.MissingResultCount == 0 &&
            hnswlibMetrics.MissingResultCount == 0;

        RepositoryInfo repository = RepositoryInfo.Create();
        return new FashionMnistHnswlibComparisonReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            FashionMnistHnswlibComparisonOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistHnswlibComparisonOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            CreateSourcePinning(),
            CreateDatasetInfo(dataset),
            CreateWorkloadInfo(options, dataset),
            CreateTruthInfo(options, dataset),
            CreateMethodology(),
            new FashionMnistHnswlibComparisonParametersInfo(
                VectorMetric.SquaredEuclidean.ToString(),
                dataset.Dimension,
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                options.QueryCount,
                options.TopK,
                options.Runs,
                options.WarmupQueries,
                options.M,
                options.EfConstruction,
                options.EfConstruction,
                options.EfSearch,
                options.EfSearch,
                FormatHex(options.Seed),
                "admitted Fashion-MNIST base matrix row order, external ids 0..baseCount-1",
                1),
            new HnswEstablishedComparisonImplementationResult(
                "VecNet",
                "pure-managed .NET public-preview squared-L2 HNSW",
                GetVecNetVersion(),
                "in-process managed .NET API call",
                typeof(HnswIndex).FullName ?? nameof(HnswIndex),
                vecNet.Build,
                vecNet.Search,
                vecNetMetrics,
                NotMeasured("bytes", "Whole-process, resident, GC and index-only memory attribution is not measured by VEC-120 comparison foundation."),
                FileFacts(vecNet.PersistedBytes, "VecNet preview HNSW Save output scanned after build/search timing and outside measured search latency.")),
            new HnswEstablishedComparisonImplementationResult(
                "hnswlib",
                "native hnswlib 0.8.0 through Python bindings",
                hnswlib.Version,
                "out-of-process Python/native hnswlib API call",
                hnswlib.Identity,
                hnswlib.Build,
                hnswlib.Search,
                hnswlibMetrics,
                NotMeasured("bytes", "hnswlib native allocation and resident/index-only memory are not measured by this runner."),
                FileFacts(hnswlib.PersistedBytes, "hnswlib save_index output scanned after build/search timing and outside measured search latency.")),
            new FashionMnistHnswlibComparisonValidationInfo(
                validationPassed ? "passed" : "failed",
                "private-fashion-mnist-hnswlib-comparison",
                LoadedExistingCache: true,
                LoadedExistingTruth: true,
                FiniteVectors: true,
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
                "Private Fashion-MNIST hnswlib comparison foundation only; not a public performance, recall, memory, allocation, capacity, storage-size, baseline, comparison-publication or regression-gate claim.",
                "This report loads an already admitted local Fashion-MNIST cache and existing exact truth artifact; it does not download data, convert data or refresh truth.",
                "Both implementations receive identical admitted float32 base vectors, query vectors, external IDs, metric, top-k, M, efConstruction/ef_construction, efSearch/ef and comparable seed value.",
                "VecNet is measured as in-process managed .NET; hnswlib is measured through a Python process and native extension. The report discloses this boundary instead of treating it as equivalent deployment.",
                "Memory fields are intentionally notMeasured because reliable index-only native-versus-managed memory attribution is not implemented in this foundation.",
                "Persisted bytes are file facts from private ignored outputs and are not storage-size claims."
            ]);
    }

    public static void Write(FashionMnistHnswlibComparisonReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadDataset(FashionMnistHnswlibComparisonOptions options) =>
        FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(
            new FashionMnistExternalHnswBenchmarkOptions(
                options.CacheRoot,
                options.OutputPath,
                options.QueryCount,
                options.TopK,
                options.Runs,
                options.WarmupQueries,
                VectorMetric.SquaredEuclidean,
                options.M,
                options.EfConstruction,
                options.EfSearch,
                options.Seed));

    private static HnswEstablishedComparisonMetricsInfo CreateMetrics(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        TruthSet truth,
        SearchResult[][] results,
        int expectedQueryCount,
        int topK)
    {
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            results,
            topK,
            dataset.Dimension,
            VectorMetric.SquaredEuclidean);
        HnswReturnedResultIntegrityInfo integrity = FashionMnistExternalHnswBenchmarkScenario.ValidateReturnedResults(
            dataset,
            results,
            expectedQueryCount,
            topK);
        return new HnswEstablishedComparisonMetricsInfo(
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.DistanceToleranceStatus,
            comparison.DistanceMismatchCount,
            comparison.MissingResultCount,
            CountExtraResults(truth, results, topK),
            integrity,
            "set recall@k = returned ids intersect loaded exact truth top-k ids divided by min(k, truth depth), summed across measured queries",
            "Ordered agreement is diagnostic only because HNSW is approximate and near-tie/order differences are expected across implementations.");
    }

    private static VecNetMeasurement MeasureVecNet(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset)
    {
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long buildStart = Stopwatch.GetTimestamp();
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.Seed);
        var index = new HnswIndex(dataset.Dimension, VectorMetric.SquaredEuclidean, hnswOptions);
        for (int row = 0; row < dataset.BaseCount; row++)
        {
            index.Add((ulong)row, dataset.GetBaseVector(row));
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
                Measured(buildAllocatedBytes, "bytes", "Measured with GC.GetAllocatedBytesForCurrentThread around public-preview HnswIndex construction and Add calls for admitted Fashion-MNIST base vectors only."),
                "new HnswIndex(...) plus admitted base-vector Add calls",
                "cache checks, manifest checksum validation, matrix load, truth load, warmup, measured search, save, external hnswlib execution and report writing"),
            search.Search,
            search.Results,
            persistedBytes);
    }

    private static void WarmupVecNet(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswIndex index)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(dataset.BaseCount, options.EfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            index.Search(dataset.GetQueryVector(i % options.QueryCount), results, workspace);
        }
    }

    private static SearchMeasurement MeasureVecNetSearch(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswIndex index)
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
        return new SearchMeasurement(
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
                Measured(
                    aggregate.MeanManagedAllocatedBytesPerQuery ?? 0,
                    "bytesPerQuery",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public-preview HnswIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswSearchWorkspace.")),
            captured ?? []);
    }

    private static SingleRunMeasurement MeasureVecNetSingleRun(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswIndex index,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(dataset.BaseCount, options.EfSearch);
        SearchResult[][]? captured = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int query = 0; query < options.QueryCount; query++)
        {
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = index.Search(dataset.GetQueryVector(query), results, workspace);
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

    private static HnswlibMeasurement RunHnswlib(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset)
    {
        Directory.CreateDirectory(options.WorkDirectory);
        string? indexDirectory = Path.GetDirectoryName(options.HnswlibIndexPath);
        if (!string.IsNullOrEmpty(indexDirectory))
        {
            Directory.CreateDirectory(indexDirectory);
        }

        string vectorsPath = Path.Combine(options.WorkDirectory, "vectors.f32");
        string queriesPath = Path.Combine(options.WorkDirectory, "queries.f32");
        string idsPath = Path.Combine(options.WorkDirectory, "ids.u64");
        string configPath = Path.Combine(options.WorkDirectory, "hnswlib-config.json");
        string outputPath = Path.Combine(options.WorkDirectory, "hnswlib-results.json");
        string scriptPath = Path.Combine(options.WorkDirectory, "run-hnswlib.py");

        WriteFloat32(vectorsPath, dataset.BaseVectors);
        WriteFloat32Rows(queriesPath, dataset.QueryVectors, options.QueryCount, dataset.Dimension);
        WriteIds(idsPath, dataset.BaseCount);
        File.WriteAllText(scriptPath, PythonDriver);
        File.WriteAllText(configPath, ReportWriter.Serialize(new HnswlibDriverConfig(
            dataset.Dimension,
            dataset.BaseCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            options.M,
            options.EfConstruction,
            options.EfSearch,
            unchecked((int)(options.Seed & 0x7FFF_FFFF)),
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

    private static HnswlibMeasurement ParseHnswlibResult(string json, FashionMnistHnswlibComparisonOptions options)
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
            "private-fashion-mnist-hnswlib-comparison",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "Private Fashion-MNIST hnswlib comparison output has not been reviewed for public reporting and is not a public VecNet claim.",
            "Established-implementation external comparison reports are not baseline candidates.",
            "No comparison-publication policy is accepted for VEC-120 output.",
            "No regression-gate policy or threshold is accepted for Fashion-MNIST hnswlib comparison output.",
            [
                "Fashion-MNIST comparison foundation only.",
                "Uses only already admitted local cache and exact truth; no download, conversion or truth refresh is performed.",
                "hnswlib is native Python tooling and is not a VecNet product dependency.",
                "Python process/API overhead is disclosed and not treated as equivalent to an in-process managed API boundary.",
                "Memory is not measured for either side in this foundation.",
                "Not eligible for public performance, recall, memory, allocation, capacity, storage-size, baseline, comparison-publication or regression-gate claims."
            ]);

    private static HnswEstablishedComparisonSourcePinningInfo CreateSourcePinning() =>
        new(
            "hnswlib",
            HnswEstablishedComparisonOptions.HnswlibPackageName,
            HnswEstablishedComparisonOptions.HnswlibPackageSource,
            HnswEstablishedComparisonOptions.HnswlibVersion,
            HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256,
            HnswEstablishedComparisonOptions.HnswlibLicense,
            "Apache-2.0 dependency is used only by private, non-shipping comparison tooling and is not distributed with VecNet.",
            "hnswlib executes as Python/native external tooling through a private ignored environment; VecNet remains managed .NET in-process.",
            "No hnswlib, Python or native asset is referenced by src/VecNet or included in the VecNet package.");

    private static ExternalBenchmarkDatasetInfo CreateDatasetInfo(FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset) =>
        new(
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
            dataset.Manifest.Labels);

    private static ExternalBenchmarkWorkloadInfo CreateWorkloadInfo(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset) =>
        new(
            dataset.BaseCount,
            dataset.QueryMatrixCount,
            options.QueryCount,
            "first N query vectors from the admitted query matrix and existing truth artifact",
            dataset.Dimension,
            dataset.Manifest.Shape.SourceDataType,
            dataset.Manifest.Shape.ConvertedDataType,
            dataset.Manifest.Metric.UpstreamName,
            VectorMetric.SquaredEuclidean.ToString(),
            dataset.Manifest.Metric.RankingNote,
            options.TopK,
            dataset.Truth.TruthDepth,
            dataset.Truth.TiePolicy);

    private static ExternalBenchmarkTruthInfo CreateTruthInfo(
        FashionMnistHnswlibComparisonOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset) =>
        new(
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
            "VecNet canonical squared distances for the external Euclidean ranking convention",
            dataset.Truth.SourceRawSha256);

    private static FashionMnistHnswlibComparisonMethodologyInfo CreateMethodology() =>
        new(
            "The runner loads one admitted local Fashion-MNIST float32 base/query dataset and external IDs 0..baseCount-1, then feeds the same binary inputs to VecNet and hnswlib.",
            "Measured build times include only index construction/add operations. Measured search latency sums per-query calls after warmup. QPS is measuredQueryCount divided by summed measured per-query elapsed time.",
            "Cache checks, manifest checksum validation, matrix load, exact truth load, Python process startup, binary interchange, warmup, result conversion/comparison, persistence/file scans and report writing.",
            "Nearest-rank over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to the sample range.",
            "Top-level fields are arithmetic means across per-run percentile/QPS/elapsed values.",
            "Single-threaded first external comparison: VecNet uses one caller thread and hnswlib receives num_threads=1 where its Python API accepts it.",
            "hnswlib runs out-of-process through Python/native extension tooling from an ignored private environment; this boundary is explicitly not a product dependency.",
            "Returned IDs are compared to existing scalar exact truth for recall and ordered agreement; every returned result is checked for known ID, duplicate IDs, finite distance and recomputed squared-L2 distance.",
            "The scenario refuses missing or invalid admitted cache/truth artifacts and never downloads, converts or refreshes Fashion-MNIST data.");

    private static HnswEstablishedComparisonEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "VEC-120 private raw output is not reviewed public evidence.",
            "Established external-comparison output is not a VecNet baseline candidate.",
            "No accepted public comparison-summary policy exists.",
            "No hnswlib comparison regression-gate policy or threshold exists.");

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

    private static void ValidateOptions(FashionMnistHnswlibComparisonOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.CacheRoot) ||
            string.IsNullOrWhiteSpace(options.OutputPath) ||
            string.IsNullOrWhiteSpace(options.WorkDirectory) ||
            string.IsNullOrWhiteSpace(options.VecNetSnapshotDirectory) ||
            string.IsNullOrWhiteSpace(options.HnswlibIndexPath) ||
            string.IsNullOrWhiteSpace(options.HnswlibPythonPath))
        {
            throw new ArgumentException("cache root, output, work directory, snapshot, hnswlib index and hnswlib python paths must not be empty.", nameof(options));
        }

        if (options.QueryCount <= 0)
        {
            throw new ArgumentException("query count must be positive.", nameof(options));
        }

        if (options.TopK <= 0)
        {
            throw new ArgumentException("top-k must be positive.", nameof(options));
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
    }

    private static void ValidateExternalTool(string pythonPath)
    {
        if (!File.Exists(pythonPath))
        {
            throw new FileNotFoundException("Pinned hnswlib Python environment is unavailable; Fashion-MNIST comparison evidence was not produced.", pythonPath);
        }

        try
        {
            var startInfo = new ProcessStartInfo(pythonPath)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            startInfo.ArgumentList.Add("-c");
            startInfo.ArgumentList.Add("import importlib.metadata; import hnswlib; print(importlib.metadata.version('hnswlib'))");

            using Process process = Process.Start(startInfo) ?? throw new IOException("Failed to start pinned hnswlib Python probe.");
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();
            if (process.ExitCode != 0)
            {
                throw new IOException($"Pinned hnswlib Python probe failed with exit code {process.ExitCode}. stdout: {stdout} stderr: {stderr}");
            }

            string version = stdout.Trim();
            if (!string.Equals(version, HnswEstablishedComparisonOptions.HnswlibVersion, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"hnswlib version '{version}' does not match pinned version '{HnswEstablishedComparisonOptions.HnswlibVersion}'.");
            }
        }
        catch (Exception ex) when (ex is not FileNotFoundException and not InvalidDataException)
        {
            throw new IOException("Pinned hnswlib Python environment is unavailable; Fashion-MNIST comparison evidence was not produced.", ex);
        }
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

    private static void WriteFloat32Rows(string path, float[] values, int rowCount, int dimension)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        int count = checked(rowCount * dimension);
        for (int i = 0; i < count; i++)
        {
            writer.Write(values[i]);
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

    private static MeasurementStatusInfo Measured(long bytes, string unit, string reason) =>
        new("measured", bytes.ToString(CultureInfo.InvariantCulture), unit, reason);

    private static MeasurementStatusInfo Measured(double value, string unit, string reason) =>
        new("measured", value.ToString(CultureInfo.InvariantCulture), unit, reason);

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static double TicksToMilliseconds(long ticks) => (double)ticks / Stopwatch.Frequency * 1000;

    private static string GetVecNetVersion() =>
        typeof(HnswIndex).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ??
        typeof(HnswIndex).Assembly.GetName().Version?.ToString() ??
        "unknown";

    private static string CreateReportId(string? commit, FashionMnistHnswlibComparisonOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistHnswlibComparisonOptions.ScenarioName}-{commitPart}-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X16}");
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
