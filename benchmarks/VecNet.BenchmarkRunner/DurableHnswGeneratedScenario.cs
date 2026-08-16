using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class DurableHnswGeneratedScenario
{
    private const string TaskId = "VEC-074";
    private const string SchemaName = "VecNet.DurableHnswBenchmarkReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "hnsw.manifest.json";
    private const string IdsFileName = "hnsw.ids.u64";
    private const string VectorsFileName = "hnsw.vectors.f32";
    private const string LevelsFileName = "hnsw.levels.i32";
    private const string GraphFileName = "hnsw.graph.bin";

    public static DurableHnswBenchmarkReport Run(DurableHnswGeneratedOptions options, IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options), options.VectorProfile);
        ValidateDataset(dataset, options.Metric);
        TruthSet truth = ScalarGroundTruth.Generate(dataset, options.Metric, options.TopK);

        string snapshotRoot = Path.GetFullPath(options.SnapshotDirectory);
        Directory.CreateDirectory(snapshotRoot);

        var buildRuns = new DurableHnswOperationRunInfo[options.Runs];
        var saveRuns = new DurableHnswOperationRunInfo[options.Runs];
        var openRuns = new DurableHnswOperationRunInfo[options.Runs];
        var openedSearchRuns = new SearchRunInfo[options.Runs];
        long[] buildAllocatedBytes = new long[options.Runs];
        SearchResult[][]? finalSourceResults = null;
        SearchResult[][]? finalOpenedResults = null;
        HnswIndex? finalSourceIndex = null;
        HnswIndex? finalOpenedIndex = null;
        string? finalSnapshotDirectory = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            string runSnapshotDirectory = CreateFreshRunSnapshotDirectory(snapshotRoot, runIndex + 1);

            BuildMeasurement build = MeasureBuild(options, dataset);
            buildRuns[runIndex] = new DurableHnswOperationRunInfo(
                runIndex + 1,
                build.ElapsedMilliseconds,
                "built",
                "notApplicable");
            buildAllocatedBytes[runIndex] = build.ManagedAllocatedBytes;

            SaveMeasurement save = MeasureSave(build.Index, runSnapshotDirectory);
            saveRuns[runIndex] = new DurableHnswOperationRunInfo(
                runIndex + 1,
                save.ElapsedMilliseconds,
                "saved",
                runSnapshotDirectory);

            OpenMeasurement open = MeasureOpen(runSnapshotDirectory);
            openRuns[runIndex] = new DurableHnswOperationRunInfo(
                runIndex + 1,
                open.ElapsedMilliseconds,
                "openedReadOnly",
                runSnapshotDirectory);

            WarmupOpenedSearch(options, dataset, open.Index);
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement openedSearch = MeasureOpenedSearch(options, dataset, open.Index, captureResults);
            openedSearchRuns[runIndex] = openedSearch.Summary with { RunNumber = runIndex + 1 };

            if (captureResults)
            {
                finalSourceIndex = build.Index;
                finalOpenedIndex = open.Index;
                finalSourceResults = CaptureSearch(options, dataset, build.Index);
                finalOpenedResults = openedSearch.Results;
                finalSnapshotDirectory = runSnapshotDirectory;
            }
        }

        if (finalSourceIndex is null || finalOpenedIndex is null || finalSourceResults is null || finalOpenedResults is null || finalSnapshotDirectory is null)
        {
            throw new InvalidOperationException("At least one durable HNSW run is required.");
        }

        ResultComparison sourceComparison = ResultComparer.Compare(
            truth,
            finalSourceResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        ResultComparison openedComparison = ResultComparer.Compare(
            truth,
            finalOpenedResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        HnswReturnedResultIntegrityInfo sourceIntegrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, options.Metric, finalSourceResults, options.TopK);
        HnswReturnedResultIntegrityInfo openedIntegrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, options.Metric, finalOpenedResults, options.TopK);
        DurableHnswParityInfo parity = CompareSavedOpenedParity(finalSourceResults, finalOpenedResults);
        DurableHnswReadOnlyMutationInfo readOnlyMutation = ValidateOpenedReadOnlyMutation(finalOpenedIndex);
        DurableHnswSnapshotOutputInfo snapshotOutput = InspectSnapshotOutput(finalSnapshotDirectory, options.VectorCount);

        int sourceExtraResultCount = CountExtraResults(truth, finalSourceResults, options.TopK);
        int openedExtraResultCount = CountExtraResults(truth, finalOpenedResults, options.TopK);
        bool validationPassed =
            sourceIntegrity.Status == "passed" &&
            openedIntegrity.Status == "passed" &&
            parity.AllResultsMatched &&
            readOnlyMutation.Status == "passed" &&
            snapshotOutput.ValidationOpenStatus == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        DurableHnswOperationAggregateInfo buildAggregate = AggregateOperationRuns(buildRuns);
        DurableHnswOperationAggregateInfo saveAggregate = AggregateOperationRuns(saveRuns);
        DurableHnswOperationAggregateInfo openAggregate = AggregateOperationRuns(openRuns);
        AggregateTimingInfo openedSearchAggregate = AggregateSearchRuns(openedSearchRuns, options.QueryCount);
        DurableHnswEvidenceInfo evidence = CreateEvidence();
        DurableHnswEligibilityInfo eligibility = CreateEligibility();

        return new DurableHnswBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            DurableHnswGeneratedOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            evidence,
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(DurableHnswGeneratedOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new DatasetInfo(
                dataset.DatasetKind,
                "generated-no-external-source",
                dataset.ProfileDistribution,
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
                DurableHnswGeneratedOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, exact scalar-reference truth construction, durable output path creation, source search capture, validation, output-byte scans, cleanup outside save/open and report writing are excluded from measured operation durations"),
            new IndexInfo(
                "DurableInternalHnswEvaluation",
                nameof(HnswIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                "internal/evaluation-only HnswIndex; build, Save, OpenReadOnly and opened Search are timed as separate private smoke operations; no public HNSW API/profile admission, matrix, baseline, comparison, regression or public claim"),
            new DurableHnswWorkloadInfo(
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount,
                options.TopK,
                dataset.SeedText,
                FormatHex(options.HnswSeed),
                options.M,
                options.EfConstruction,
                options.EfSearch,
                options.Runs,
                options.WarmupQueries,
                "generated vector row order, external ids 0..vectorCount-1",
                "immutable source HNSW saved to a fresh ignored per-run snapshot directory and opened read-only",
                "hnsw"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated vector row order, external ids 0..vectorCount-1",
                $"{options.Metric} private generated durable HNSW metric"),
            new DurableHnswOperationsInfo(
                new DurableHnswOperationInfo(
                    "build",
                    "internal HnswIndex construction and Add calls for generated base vectors",
                    buildRuns,
                    buildAggregate),
                new DurableHnswOperationInfo(
                    "save",
                    "internal HnswIndex.Save(directoryPath)",
                    saveRuns,
                    saveAggregate),
                new DurableHnswOperationInfo(
                    "open",
                    "internal HnswIndex.OpenReadOnly(directoryPath)",
                    openRuns,
                    openAggregate),
                new DurableHnswOpenedSearchOperationInfo(
                    "openedSearch",
                    "internal opened HnswIndex.Search(query, results, workspace)",
                    openedSearchRuns,
                    openedSearchAggregate),
                NotMeasured("milliseconds", "source-HNSW search was captured for parity/recall validation only"),
                NotMeasured("bytes", "Resident/process memory, working set, private bytes, managed heap and peak memory are not measured.")),
            new DurableHnswMeasurementInfo(
                CreateOperationMeasurement(
                    "build",
                    "internal HnswIndex construction and Add calls for generated base vectors",
                    "generated data creation, exact truth construction, durable output path creation, save, open, warmup, source/opened search, validation, output-byte scans, cleanup and report writing",
                    buildRuns,
                    new MeasurementStatusInfo(
                        "measured",
                        buildAllocatedBytes.Average().ToString(CultureInfo.InvariantCulture),
                        "bytesPerBuild",
                        "Measured with GC.GetAllocatedBytesForCurrentThread around internal HnswIndex construction and Add calls only; generated data setup and exact truth generation are excluded.")),
                CreateOperationMeasurement(
                    "save",
                    "internal HnswIndex.Save(directoryPath)",
                    "generated data creation, HNSW build, exact truth construction, target directory name generation, validation searches, output-byte scans, OpenReadOnly, opened search, cleanup outside the save call and report writing",
                    saveRuns,
                    NotMeasured("bytesPerSaveCall", "Managed allocation for HnswIndex.Save(directoryPath) is not measured in VEC-074.")),
                CreateOperationMeasurement(
                    "open",
                    "internal HnswIndex.OpenReadOnly(directoryPath)",
                    "HNSW build, save, durable output byte scans, opened-index mutation rejection probes, exact truth construction, source/opened search result comparison, warmup and report writing",
                    openRuns,
                    NotMeasured("bytesPerOpenCall", "Managed allocation for HnswIndex.OpenReadOnly(directoryPath) is not measured in VEC-074.")),
                CreateOpenedSearchMeasurement(options, openedSearchRuns),
                NotMeasured("milliseconds", "source-HNSW search was captured for parity/recall validation only"),
                NotMeasured("bytes", "Process resident memory, GC heap, working set, private bytes and peak memory are not measured."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed against the opened read-only index before measured opened search and excluded from all measured operation timings."
                        : "No warmup queries were requested."),
                "Generated data setup, exact truth construction, durable output path creation, source result capture, validation, saved/opened parity comparison, output-byte scans, cleanup outside save/open and report writing are excluded from the relevant measured operations."),
            new DurableHnswOutputsInfo(
                snapshotOutput,
                NotMeasured("bytes", "Temporary disk and failed-save remnants are not measured in VEC-074."),
                NotMeasured("bytes", "Peak temporary disk usage is not sampled in VEC-074.")),
            new DurableHnswMetricsInfo(
                CreateMetricsInfo(sourceComparison, sourceIntegrity, sourceExtraResultCount, options),
                CreateMetricsInfo(openedComparison, openedIntegrity, openedExtraResultCount, options),
                sourceComparison.RecallAtK == openedComparison.RecallAtK,
                sourceComparison.OrderedAgreement == openedComparison.OrderedAgreement,
                sourceIntegrity.DistanceMismatchCount == openedIntegrity.DistanceMismatchCount,
                parity.AllResultsMatched
                    ? "Saved/opened graph-identity parity preserved source and opened result sets, so recall and distance-integrity are expected to match for the same query set."
                    : "Saved/opened result parity failed; recall-equivalence fields are retained as private smoke diagnostics."),
            new DurableHnswValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-durable-hnsw-smoke",
                FiniteVectors: true,
                ExactTruthGenerated: true,
                SourceHnswBuilt: true,
                SourceHnswSaved: true,
                OpenedHnswOpened: true,
                OpenedIndexReadOnly: readOnlyMutation.Status == "passed",
                SourceHnswComparedToTruth: true,
                OpenedHnswComparedToTruth: true,
                ReturnedResultIntegrityPassedForSource: sourceIntegrity.Status == "passed",
                ReturnedResultIntegrityPassedForOpened: openedIntegrity.Status == "passed",
                parity,
                readOnlyMutation,
                OutputBytesScannedOutsideSaveOpenDuration: true,
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateMemoryEstimates(options, finalSourceIndex, snapshotOutput.TotalBytes),
            eligibility,
            [
                "Private generated durable-HNSW smoke evidence only; not a public benchmark claim.",
                "Build, save, open and opened-search timings are separate operation blocks.",
                "Source-HNSW search is validation-only and is not timed in VEC-074.",
                "Opened-search timing measures only internal opened HnswIndex.Search(query, results, workspace) calls with caller-owned buffers/workspace.",
                "Output bytes are private local file facts scanned outside save/open duration, not public persisted-size claims.",
                "Recall below 1.0 is valid approximate HNSW behavior when returned-result integrity and saved/opened parity pass.",
                "No public HNSW API/profile admission, preview-readiness evidence, baseline candidate, comparison artifact, regression gate, matrix preset, external durable-HNSW evidence or public documentation is included."
            ]);
    }

    public static void Write(DurableHnswBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(DurableHnswGeneratedOptions options) =>
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

    private static BuildMeasurement MeasureBuild(DurableHnswGeneratedOptions options, GeneratedDataset dataset)
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
        return new BuildMeasurement(index, StopwatchTicksToMilliseconds(elapsed), allocatedBytes);
    }

    private static SaveMeasurement MeasureSave(HnswIndex index, string snapshotDirectory)
    {
        long start = Stopwatch.GetTimestamp();
        index.Save(snapshotDirectory);
        long elapsed = Stopwatch.GetTimestamp() - start;
        return new SaveMeasurement(StopwatchTicksToMilliseconds(elapsed));
    }

    private static OpenMeasurement MeasureOpen(string snapshotDirectory)
    {
        long start = Stopwatch.GetTimestamp();
        HnswIndex index = HnswIndex.OpenReadOnly(snapshotDirectory);
        long elapsed = Stopwatch.GetTimestamp() - start;
        return new OpenMeasurement(index, StopwatchTicksToMilliseconds(elapsed));
    }

    private static void WarmupOpenedSearch(DurableHnswGeneratedOptions options, GeneratedDataset dataset, HnswIndex openedIndex)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            openedIndex.Search(dataset.GetQuery(i % dataset.QueryCount), results, workspace);
        }
    }

    private static SingleRunMeasurement MeasureOpenedSearch(
        DurableHnswGeneratedOptions options,
        GeneratedDataset dataset,
        HnswIndex openedIndex,
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
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = openedIndex.Search(dataset.GetQuery(queryRow), results, workspace);
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

    private static SearchResult[][] CaptureSearch(DurableHnswGeneratedOptions options, GeneratedDataset dataset, HnswIndex index)
    {
        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        var allResults = new SearchResult[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), results, workspace);
            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            allResults[queryRow] = queryResults;
        }

        return allResults;
    }

    private static DurableHnswParityInfo CompareSavedOpenedParity(SearchResult[][] sourceResults, SearchResult[][] openedResults)
    {
        int queryCount = Math.Min(sourceResults.Length, openedResults.Length);
        int writtenCountMismatchCount = sourceResults.Length == openedResults.Length ? 0 : 1;
        int idMismatchCount = 0;
        int orderMismatchCount = 0;
        int distanceMismatchCount = 0;

        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            SearchResult[] source = sourceResults[queryRow];
            SearchResult[] opened = openedResults[queryRow];
            if (source.Length != opened.Length)
            {
                writtenCountMismatchCount++;
            }

            int resultCount = Math.Min(source.Length, opened.Length);
            for (int i = 0; i < resultCount; i++)
            {
                if (source[i].Id != opened[i].Id)
                {
                    idMismatchCount++;
                    orderMismatchCount++;
                }

                if (source[i].Distance != opened[i].Distance)
                {
                    distanceMismatchCount++;
                }
            }
        }

        bool passed = writtenCountMismatchCount == 0 &&
            idMismatchCount == 0 &&
            orderMismatchCount == 0 &&
            distanceMismatchCount == 0;
        return new DurableHnswParityInfo(
            sourceResults.Length,
            writtenCountMismatchCount,
            idMismatchCount,
            orderMismatchCount,
            distanceMismatchCount,
            passed,
            "Graph-identity parity requires source and opened HNSW results to match written count, IDs, result order and exact returned distances for every validation query.");
    }

    private static DurableHnswReadOnlyMutationInfo ValidateOpenedReadOnlyMutation(HnswIndex openedIndex)
    {
        try
        {
            openedIndex.Add(0, ReadOnlySpan<float>.Empty);
            return new DurableHnswReadOnlyMutationInfo(
                "failed",
                "none",
                RejectedBeforeVectorValidation: false,
                "opened HnswIndex.Add(duplicateId, emptyVector)",
                "Opened read-only HNSW accepted a mutation probe.");
        }
        catch (InvalidOperationException ex)
        {
            return new DurableHnswReadOnlyMutationInfo(
                "passed",
                ex.GetType().Name,
                RejectedBeforeVectorValidation: true,
                "opened HnswIndex.Add(duplicateId, emptyVector)",
                "Opened read-only HNSW rejected mutation before vector dimension or duplicate-ID validation.");
        }
    }

    private static DurableHnswSnapshotOutputInfo InspectSnapshotOutput(string snapshotDirectory, int vectorCount)
    {
        long manifestBytes = FileLength(snapshotDirectory, ManifestFileName);
        long idsBytes = FileLength(snapshotDirectory, IdsFileName);
        long vectorsBytes = FileLength(snapshotDirectory, VectorsFileName);
        long levelsBytes = FileLength(snapshotDirectory, LevelsFileName);
        long graphBytes = FileLength(snapshotDirectory, GraphFileName);
        long totalBytes = checked(manifestBytes + idsBytes + vectorsBytes + levelsBytes + graphBytes);
        _ = HnswIndex.OpenReadOnly(snapshotDirectory);

        return new DurableHnswSnapshotOutputInfo(
            "written",
            "caller-selected snapshot root with fresh per-run durable HNSW snapshot directory",
            snapshotDirectory,
            FileCount: 5,
            totalBytes,
            manifestBytes,
            idsBytes,
            vectorsBytes,
            levelsBytes,
            graphBytes,
            vectorCount,
            vectorCount == 0 ? 0 : (double)totalBytes / vectorCount,
            "passed",
            "outsideSaveAndOpenDuration");
    }

    private static long FileLength(string directory, string fileName) =>
        new FileInfo(Path.Combine(directory, fileName)).Length;

    private static string CreateFreshRunSnapshotDirectory(string snapshotRoot, int runNumber)
    {
        string directory = Path.Combine(snapshotRoot, string.Create(CultureInfo.InvariantCulture, $"run-{runNumber:000}"));
        if (!Directory.Exists(directory) || Directory.GetFileSystemEntries(directory).Length == 0)
        {
            Directory.CreateDirectory(directory);
            return directory;
        }

        directory = Path.Combine(snapshotRoot, string.Create(CultureInfo.InvariantCulture, $"run-{runNumber:000}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static DurableHnswOperationMetricsInfo CreateMetricsInfo(
        ResultComparison comparison,
        HnswReturnedResultIntegrityInfo integrity,
        int extraResultCount,
        DurableHnswGeneratedOptions options) =>
        new(
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.MissingResultCount,
            extraResultCount,
            integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
            integrity.DistanceMismatchCount,
            integrity,
            "set recall@k = returned IDs intersect exact top-k IDs divided by min(k, vectorCount), summed across measured queries",
            $"Every returned durable-HNSW result is checked for finite distance, no duplicate ID within its query, generated-index ID membership and {options.Metric} distance matching recomputation for that returned ID/query within the accepted runner tolerance; exact top-k recall/order are recorded but not required.");

    private static DurableHnswOperationMeasurementInfo CreateOperationMeasurement(
        string name,
        string timedOperation,
        string excludedOperations,
        DurableHnswOperationRunInfo[] runs,
        MeasurementStatusInfo managedAllocations) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                string.Create(CultureInfo.InvariantCulture, $"per{name}Call"),
                timedOperation,
                excludedOperations,
                "single elapsed Stopwatch sample per measured operation run",
                "Mean/min/max are private local descriptive metadata across independently rebuilt equivalent durable-HNSW runs, not BenchmarkDotNet statistics.",
                "Raw per-run elapsed milliseconds are emitted in operations."),
            managedAllocations,
            CreateRepeatedRunInfo(runs.Length, string.Create(CultureInfo.InvariantCulture, $"{name} operation")),
            CreateOperationRunToRunNoise(runs, string.Create(CultureInfo.InvariantCulture, $"{name} operation")));

    private static DurableHnswSearchMeasurementInfo CreateOpenedSearchMeasurement(DurableHnswGeneratedOptions options, SearchRunInfo[] runs) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredOpenedQuery",
                "internal opened HnswIndex.Search(query, results, workspace)",
                "build, save, open, exact truth construction, warmup queries, source-HNSW result capture, final-run result comparison, saved/opened parity validation, returned-result integrity validation, output-byte scans and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Top-level opened-search aggregate percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                runs.Average(run => run.ManagedAllocatedBytesPerQuery).ToString(CultureInfo.InvariantCulture),
                "bytesPerQuery",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each internal opened HnswIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswSearchWorkspace; setup, build, save, open, warmup, source search, validation and report writing are excluded."),
            CreateRepeatedRunInfo(runs.Length, "opened-search operation"),
            CreateOpenedSearchRunToRunNoise(runs, options.QueryCount));

    private static RepeatedRunInfo CreateRepeatedRunInfo(int runCount, string operationName) =>
        new(
            runCount > 1 ? "measured" : "singleRun",
            runCount,
            runCount > 1,
            runCount > 1
                ? $"Multiple measured {operationName} runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                : $"Only one measured {operationName} run executed, so cross-run variance/noise is not measured.");

    private static RunToRunMetricNoiseInfo CreateOperationRunToRunNoise(DurableHnswOperationRunInfo[] runs, string operationName)
    {
        bool measured = runs.Length > 1;
        if (!measured)
        {
            return new RunToRunMetricNoiseInfo(
                "notMeasured",
                "milliseconds",
                null,
                null,
                null,
                null,
                null,
                null,
                $"Only one measured {operationName} run exists; this field does not establish run-to-run variation.");
        }

        double[] values = runs.Select(run => run.ElapsedMilliseconds).ToArray();
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate(values);
        return new RunToRunMetricNoiseInfo(
            "measured",
            "milliseconds",
            FiniteOrNull(statistics.Mean),
            statistics.SampleStandardDeviation,
            statistics.CoefficientOfVariation,
            FiniteOrNull(statistics.Min),
            FiniteOrNull(statistics.Max),
            FiniteOrNull(statistics.Spread),
            "Computed across measured durable-HNSW operation runs using the documented private descriptive-statistics formula.");
    }

    private static RunToRunNoiseInfo CreateOpenedSearchRunToRunNoise(SearchRunInfo[] runs, int measuredQueryCountPerRun)
    {
        bool measured = runs.Length > 1;
        string status = measured ? "measured" : "notMeasured";
        string reason = measured
            ? "Multiple measured opened-search runs executed; simple descriptive run-to-run statistics are recorded for private local durable-HNSW noise inspection."
            : "Only one measured opened-search run executed, so run-to-run noise is unavailable and cannot be measured.";
        string unavailableReason = "Only one measured opened-search run exists; this field does not establish run-to-run variation.";

        return new RunToRunNoiseInfo(
            status,
            runs.Length,
            measured,
            string.Create(CultureInfo.InvariantCulture, $"Across measured opened HNSW search runs with {measuredQueryCountPerRun} measured queries per run; warmup, build, save, open, validation and report writing are excluded."),
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            reason,
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateSearchMetricNoise(runs, "milliseconds", run => run.ElapsedMilliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "queriesPerSecond", run => run.Qps, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP50Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP95Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP99Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "bytesPerQuery", run => run.ManagedAllocatedBytesPerQuery, measured, unavailableReason));
    }

    private static RunToRunMetricNoiseInfo CreateSearchMetricNoise(
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
            "Computed across measured opened-search runs using the documented private descriptive-statistics formula.");
    }

    private static DurableHnswOperationAggregateInfo AggregateOperationRuns(DurableHnswOperationRunInfo[] runs) =>
        new(
            runs.Length,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds));

    private static AggregateTimingInfo AggregateSearchRuns(SearchRunInfo[] runs, int measuredQueryCountPerRun) =>
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

    private static DurableHnswMemoryEstimateInfo CreateMemoryEstimates(DurableHnswGeneratedOptions options, HnswIndex index, long durableOutputBytes)
    {
        int layerCount = Math.Max(0, index.MaxLayer + 1);
        long graphNeighborBytes = 0;
        long graphCountBytes = 0;
        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = layer == 0 ? checked(options.M * 2) : options.M;
            graphNeighborBytes = checked(graphNeighborBytes + (long)options.VectorCount * stride * sizeof(int));
            graphCountBytes = checked(graphCountBytes + (long)options.VectorCount * sizeof(int));
        }

        return new DurableHnswMemoryEstimateInfo(
            "estimatedPayloadLowerBoundsAndFileFacts",
            "Layout-derived payload lower-bound estimates plus final durable output byte facts only.",
            checked((long)options.VectorCount * options.Dimension * sizeof(float)),
            checked((long)options.VectorCount * sizeof(ulong)),
            checked((long)options.VectorCount * sizeof(int)),
            graphCountBytes,
            graphNeighborBytes,
            EstimateWorkspaceBytes(options.VectorCount, options.EfSearch),
            durableOutputBytes,
            NotMeasured("bytes", "Process resident memory is not measured."),
            NotMeasured("bytes", "GC heap size, GC committed memory and GC fragmented memory are not measured."),
            NotMeasured("bytes", "Working set is OS/cache-sensitive and is not measured."),
            NotMeasured("bytes", "Private bytes are not measured."),
            NotMeasured("bytes", "Peak temporary or process memory is not measured."),
            [
                "Managed object headers, array alignment and backing-array slack capacity are excluded.",
                "Dictionary<ulong,int> capacity and JSON parser/string/object overhead are excluded.",
                "Build-time temporary arrays and temporary save/open allocations are excluded.",
                "Resident/process/GC/private/peak memory fields are explicitly not measured."
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

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static DurableHnswEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-durable-hnsw-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated durable-HNSW smoke output is not reviewed public evidence and has no public reporting policy.",
            "One generated local report does not establish hostile durable-open fuzzing, external data behavior, actual memory, crash, concurrency, Linux, package or public API readiness.",
            "No durable-HNSW baseline-candidate policy is accepted.",
            "No durable-HNSW comparison artifact schema or compatibility policy is accepted.",
            "No durable-HNSW threshold, hard gate or regression decision policy is accepted.",
            [
                "Private generated local durable-HNSW smoke evidence only.",
                "Build, save, open and opened-search operation boundaries are separated.",
                "Storage bytes are private local file facts scanned outside save/open duration.",
                "Not public performance, recall or storage-size evidence; not preview admission; not BenchmarkDotNet-grade timing/allocation evidence."
            ]);

    private static DurableHnswEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated durable-HNSW smoke output is not reviewed public evidence and has no public reporting policy.",
            "One generated local report does not establish hostile durable-open fuzzing, external data behavior, actual memory, crash, concurrency, Linux, package or public API readiness.",
            "No durable-HNSW baseline-candidate policy is accepted.",
            "No durable-HNSW comparison artifact schema or compatibility policy is accepted.",
            "No durable-HNSW threshold, hard gate or regression decision policy is accepted.");

    private static void ValidateOptions(DurableHnswGeneratedOptions options)
    {
        if (!IsSupportedMetric(options.Metric))
        {
            throw new ArgumentException("hnsw-generated-durable supports SquaredEuclidean, InnerProduct and Cosine only.", nameof(options));
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

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("output path must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.SnapshotDirectory))
        {
            throw new ArgumentException("snapshot directory must not be empty.", nameof(options));
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

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static string CreateReportId(string? commit, DurableHnswGeneratedOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{DurableHnswGeneratedOptions.ScenarioName}-{commitPart}-{options.Metric}-{GeneratedDatasetFactory.GetOptionValue(options.VectorProfile)}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static double StopwatchTicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record BuildMeasurement(HnswIndex Index, double ElapsedMilliseconds, long ManagedAllocatedBytes);

    private sealed record SaveMeasurement(double ElapsedMilliseconds);

    private sealed record OpenMeasurement(HnswIndex Index, double ElapsedMilliseconds);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);
}
