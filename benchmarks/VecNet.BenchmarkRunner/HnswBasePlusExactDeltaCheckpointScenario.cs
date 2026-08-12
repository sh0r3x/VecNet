using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class HnswBasePlusExactDeltaCheckpointScenario
{
    private const string TaskId = "VEC-134";
    private const string SchemaName = "VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "hnsw.manifest.json";
    private const string IdsFileName = "hnsw.ids.u64";
    private const string VectorsFileName = "hnsw.vectors.f32";
    private const string LevelsFileName = "hnsw.levels.i32";
    private const string GraphFileName = "hnsw.graph.bin";

    public static HnswBasePlusExactDeltaCheckpointBenchmarkReport Run(
        HnswBasePlusExactDeltaCheckpointOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        string checkpointRootDirectory = Path.GetFullPath(options.CheckpointDirectory);
        Directory.CreateDirectory(checkpointRootDirectory);

        var checkpointRunInfos = new HnswBasePlusExactDeltaCheckpointRunInfo[options.Runs];
        PreparedCheckpointState? finalState = null;
        MeasuredCheckpointRun? finalCheckpoint = null;
        SearchMeasurement? preSearch = null;
        SearchResult[][]? preFirstPassProbeResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            PreparedCheckpointState state = PrepareCheckpointState(options);
            bool finalRun = runIndex == options.Runs - 1;
            if (finalRun)
            {
                WarmupCompositeSearch(options, state.Dataset, state.Composite);
                preSearch = MeasureCompositeSearch(options, state.Dataset, state.Composite, captureResults: true);
                preFirstPassProbeResults = ProbeCompositeFirstPassSearch(options, state.Dataset, state.Composite);
            }

            string runDirectory = CreateCheckpointRunDirectory(checkpointRootDirectory, runIndex + 1);
            MeasuredCheckpointRun measured = MeasureCheckpointRun(state.Composite, runDirectory);
            checkpointRunInfos[runIndex] = CreateCheckpointRunInfo(runIndex + 1, runDirectory, measured);

            if (finalRun)
            {
                finalState = state;
                finalCheckpoint = measured;
            }
        }

        if (finalState is null || finalCheckpoint is null || preSearch is null)
        {
            throw new InvalidOperationException("Checkpoint final-run validation state was not captured.");
        }

        GeneratedDataset dataset = finalState.Dataset;
        HnswBasePlusExactDeltaIndex composite = finalState.Composite;
        long generationBeforeMutations = finalState.GenerationBeforeMutations;
        MutationExecution mutationExecution = finalState.MutationExecution;
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = finalState.PreCounts;
        ulong[] liveIds = finalState.LiveIds;
        TruthSet truth = finalState.Truth;
        string checkpointDirectory = finalCheckpoint.DirectoryPath;
        long generationBeforeCheckpoint = finalCheckpoint.GenerationBeforeCheckpoint;
        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpointDiagnostic = finalCheckpoint.Diagnostic;
        long checkpointElapsedTicks = finalCheckpoint.ElapsedTicks;
        long checkpointAllocatedBytes = finalCheckpoint.ManagedAllocatedBytes;
        HnswBasePlusExactDeltaCheckpointResult checkpointResult = checkpointDiagnostic.Result;
        HnswBasePlusExactDeltaCheckpointCountInfo postCounts = CreateCountInfo(options, composite);
        HnswBasePlusExactDeltaCheckpointRunsInfo checkpointRuns =
            CreateCheckpointRunsInfo(options, checkpointRunInfos);

        HnswBasePlusExactDeltaCheckpointOutputInfo output =
            InspectCheckpointOutput(checkpointDirectory, checkpointResult.LiveVectorCount);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpointDirectory);
        HnswBasePlusExactDeltaCheckpointOpenedValidationInfo openedValidation =
            ValidateOpenedOutput(dataset, options, liveIds, opened, preParity: null);

        HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo noChangesProbe =
            ProbeNoChanges(options, checkpointDirectory, composite);
        bool deletedReservedRejected = ValidateDeletedReservation(options, dataset, composite);

        WarmupCompositeSearch(options, dataset, composite);
        SearchMeasurement postSearch = MeasureCompositeSearch(options, dataset, composite, captureResults: true);
        WarmupOpenedSearch(options, dataset, opened);
        SearchMeasurement openedSearch = MeasureOpenedSearch(options, dataset, opened, captureResults: true);
        HnswBasePlusExactDeltaCheckpointParityInfo parity =
            CompareSearchParity(postSearch.Results, openedSearch.Results, options.Dimension);
        openedValidation = openedValidation with { RebuiltCompositeOpenedSearchParity = parity };

        SearchSectionEvaluation preEvaluation = EvaluateSearchSection(options, dataset, truth, preSearch, liveIds);
        SearchSectionEvaluation postEvaluation = EvaluateSearchSection(options, dataset, truth, postSearch, liveIds);
        SearchSectionEvaluation openedEvaluation = EvaluateSearchSection(options, dataset, truth, openedSearch, liveIds);
        HnswBasePlusExactDeltaRetryDiagnosticsInfo preRetryDiagnostics = CreateRetryDiagnostics(
            options,
            firstPassResults: preFirstPassProbeResults ?? [],
            finalResults: preSearch.Results,
            preEvaluation.Underfill,
            basePhysicalVectorCount: options.BaseVectorCount,
            baseTombstoneCount: options.DeletedBaseCount,
            statusWhenCannotWiden: "notApplicable",
            policyPrefix: "Pre-checkpoint composite");
        HnswBasePlusExactDeltaRetryDiagnosticsInfo postRetryDiagnostics = CreateRetryDiagnostics(
            options,
            firstPassResults: postSearch.Results,
            finalResults: postSearch.Results,
            postEvaluation.Underfill,
            basePhysicalVectorCount: postCounts.BasePhysicalVectorCount,
            baseTombstoneCount: postCounts.BaseTombstoneCount,
            statusWhenCannotWiden: "notApplicable",
            policyPrefix: "Post-checkpoint rebuilt composite");
        HnswBasePlusExactDeltaRetryDiagnosticsInfo openedRetryDiagnostics = CreateNotApplicableRetryDiagnostics(
            options,
            openedEvaluation.Underfill,
            "Opened read-only HnswIndex search has no mutable base-plus-exact-delta adaptive-retry path.");

        bool mutationStatusCountsMatched = MutationStatusCountsMatch(options, mutationExecution);
        bool mutationGenerationMatched =
            mutationExecution.GenerationAfterMutations - generationBeforeMutations ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount;
        bool checkpointCountsMatched = CheckpointCountsMatch(preCounts, checkpointResult);
        bool postCountsMatched = PostCountsMatch(preCounts, postCounts, checkpointResult);
        bool checkpointGenerationAdvanced = checkpointResult.Generation == generationBeforeCheckpoint + 1 &&
            composite.Generation == generationBeforeCheckpoint + 1;
        bool publishedPhasesMeasured = AllMeasured(CreatePhaseSet(checkpointDiagnostic.Diagnostics));
        bool checkpointRepeatedRunEvidencePresent = CheckpointRepeatedRunEvidencePresent(options, checkpointRuns);
        bool allIntegrityPassed =
            preEvaluation.Metrics.ReturnedResultIntegrity.Status == "passed" &&
            postEvaluation.Metrics.ReturnedResultIntegrity.Status == "passed" &&
            openedEvaluation.Metrics.ReturnedResultIntegrity.Status == "passed";
        bool validationPassed =
            mutationStatusCountsMatched &&
            mutationGenerationMatched &&
            checkpointResult.Status == HnswBasePlusExactDeltaCheckpointStatus.Published &&
            checkpointCountsMatched &&
            checkpointGenerationAdvanced &&
            publishedPhasesMeasured &&
            checkpointRepeatedRunEvidencePresent &&
            postCountsMatched &&
            openedValidation.Status == "passed" &&
            parity.AllResultsMatched &&
            allIntegrityPassed &&
            noChangesProbe.Status == "passed" &&
            deletedReservedRejected &&
            output.ValidationOpenStatus == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        HnswBasePlusExactDeltaCheckpointPhaseSetInfo phaseDiagnostics = CreatePhaseSet(checkpointDiagnostic.Diagnostics);
        HnswEvidenceInfo evidence = CreateEvidence();
        HnswEligibilityInfo eligibility = CreateEligibility();

        return new HnswBasePlusExactDeltaCheckpointBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            evidence,
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, commandArguments.ToArray()),
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
                options.PhysicalVectorCount,
                options.QueryCount),
            new TruthInfo(
                "scalar-reference-generated-live-hnsw-base-plus-exact-delta-checkpoint",
                truth.Depth,
                "post-update live base plus live delta minus tombstones, ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, immutable HNSW base build, update application, exact updated truth construction, warmup queries, output-byte scan, validation searches and report writing are excluded from checkpoint timing; each search section times only its own Search calls"),
            new IndexInfo(
                "InternalHnswBasePlusExactDeltaCheckpoint",
                nameof(HnswBasePlusExactDeltaIndex),
                options.Metric.ToString(),
                options.Dimension,
                postCounts.LiveVectorCount,
                "internal HnswBasePlusExactDeltaIndex checkpoint/rebuild smoke report over generated data; no public mutable/update HNSW API, matrix preset, external dataset, memory evidence, concurrency evidence or public claim"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated base vector row order, external base ids 0..baseVectorCount-1; delta ids continue from baseVectorCount",
                $"{options.Metric} only; efSearch is first-pass search width and workspace/adaptive-retry ceiling is reported per search section under retryDiagnostics"),
            new HnswBasePlusExactDeltaCheckpointWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                options.Runs,
                options.WarmupQueries,
                dataset.SeedText,
                "each checkpoint run uses a fresh ignored subdirectory under the supplied checkpoint root; output-byte scan occurs after final-run checkpoint timing",
                "build immutable HNSW base, committed exact-delta inserts, committed base tombstone deletes, configured delta tombstone deletes, duplicate/reserved insert attempts, unknown delete attempts, repeated delete attempts, then checkpoint/rebuild",
                "Base IDs are 0..baseVectorCount-1; committed delta IDs are baseVectorCount..physicalVectorCount-1; deleted IDs remain reserved inside the writable composite instance after checkpoint; unknown IDs start above physicalVectorCount."),
            preCounts,
            CreateMutationInfo(
                options,
                mutationExecution,
                generationBeforeMutations,
                mutationStatusCountsMatched,
                mutationGenerationMatched),
            checkpointRuns,
            new HnswBasePlusExactDeltaCheckpointOperationInfo(
                checkpointResult.Status.ToString(),
                "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                StopwatchTicksToMilliseconds(checkpointElapsedTicks),
                checkpointAllocatedBytes,
                generationBeforeCheckpoint,
                checkpointResult.Generation,
                checkpointGenerationAdvanced,
                phaseDiagnostics,
                "generated data setup, immutable HNSW base build, update application, exact updated truth construction, search timing, no-changes probe, output-byte scan, opened-output validation and report writing"),
            CreateCheckpointResultInfo(checkpointResult),
            postCounts,
            noChangesProbe,
            output,
            openedValidation,
            new HnswBasePlusExactDeltaCheckpointSearchSectionsInfo(
                CreateSearchSection("preCheckpointComposite", "internal pre-checkpoint HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", preSearch, preEvaluation, options, preRetryDiagnostics),
                CreateSearchSection("postCheckpointRebuiltComposite", "internal post-checkpoint rebuilt HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", postSearch, postEvaluation, options, postRetryDiagnostics),
                CreateSearchSection("openedReadOnlyHnsw", "internal opened read-only HnswIndex.Search(query, results, workspace)", openedSearch, openedEvaluation, options, openedRetryDiagnostics)),
            new HnswBasePlusExactDeltaCheckpointMeasurementInfo(
                new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perCheckpointCall",
                    "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                    "generated data setup, HNSW base build, mutation application, exact truth construction, warmup, all search measurements, output-byte scan, no-changes probe, validation and report writing",
                    "single Stopwatch sample around the internal checkpoint call boundary; phase elapsed values come from VEC-133 diagnostics and are not summed or inferred",
                    "Aggregate checkpoint elapsed fields are computed across independently rebuilt equivalent checkpoint attempts; this is not BenchmarkDotNet statistics and not a regression gate.",
                    "Raw checkpoint elapsed milliseconds are emitted in checkpointRuns.runs; checkpoint.elapsedMilliseconds is the final run used for detailed validation."),
                new MeasurementStatusInfo(
                    "measured",
                    checkpointRuns.Aggregate.MeanManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                    "bytesPerCheckpointCall",
                    "Mean across independently rebuilt equivalent checkpoint attempts measured with GC.GetAllocatedBytesForCurrentThread around the runner call to internal CheckpointWithDiagnostics; per-run values and VEC-133 phase allocation values are reported separately and not inferred."),
                phaseDiagnostics,
                new MeasurementStatusInfo(
                    "measured",
                    output.TotalBytes.ToString(CultureInfo.InvariantCulture),
                    "bytes",
                    "Checkpoint output bytes are scanned after the timed checkpoint call has completed."),
                NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-134."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries execute before each measured search section and are excluded from search and checkpoint timing."
                        : "No warmup queries were requested."),
                "Generated data setup, HNSW base build, mutation application, exact truth construction, warmup, output-byte scan, no-changes probe, validation and report writing are excluded from checkpoint timing; checkpoint timing and all search timings are separate."),
            new HnswBasePlusExactDeltaCheckpointValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-hnsw-base-plus-exact-delta-checkpoint-smoke",
                FiniteVectors: true,
                LiveTruthGenerated: true,
                PreCheckpointCompositeComparedToTruth: true,
                CheckpointResultStatusPublished: checkpointResult.Status == HnswBasePlusExactDeltaCheckpointStatus.Published,
                CheckpointResultCountsMatched: checkpointCountsMatched,
                CheckpointGenerationAdvancedExactlyOnce: checkpointGenerationAdvanced,
                PhaseDiagnosticsMeasuredForPublishedCheckpoint: publishedPhasesMeasured,
                CheckpointRepeatedRunEvidencePresent: checkpointRepeatedRunEvidencePresent,
                DetailedValidationRunNumber: checkpointRuns.DetailedValidationRunNumber,
                DetailedValidationUsesFinalRun: checkpointRuns.DetailedValidationRunNumber == options.Runs,
                PostCheckpointCountsMatched: postCountsMatched,
                PostCheckpointRebuiltCompositeComparedToTruth: true,
                OpenedReadOnlyHnswIdVectorValidationPassed: openedValidation.Status == "passed",
                OpenedReadOnlyHnswComparedToTruth: true,
                RebuiltCompositeOpenedHnswSearchParityPassed: parity.AllResultsMatched,
                ReturnedResultIntegrityPassedForAllSearches: allIntegrityPassed,
                NoChangesCheckpointProbePassed: noChangesProbe.Status == "passed",
                DeletedReservedIdsRejectedAfterCheckpoint: deletedReservedRejected,
                OutputBytesScannedOutsideCheckpointDuration: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            eligibility,
            [
                "Private generated HNSW base-plus-exact-delta checkpoint smoke evidence only; not a public benchmark claim.",
                "Generated finite squared-L2, inner-product or cosine data only; no external dataset source, license, version or checksum applies.",
                "This report exercises internal checkpoint/rebuild diagnostics and does not add or imply a public mutable/update HNSW API.",
                "Pre-checkpoint composite retryDiagnostics compares a tight first-pass probe using workspaceEfSearch == efSearch with measured search using the configured workspace/adaptive-retry ceiling.",
                "Checkpoint timing/allocation is measured at the runner call boundary and VEC-133 phase diagnostics are copied from the internal result; phase timings are not inferred or fabricated.",
                "For runs greater than one, checkpoint timing/allocation is measured across independently rebuilt equivalent checkpoint attempts with fresh generated state and fresh checkpoint output subdirectories; detailed validation uses the final run.",
                "Pre-checkpoint composite, post-checkpoint rebuilt composite and opened read-only HNSW searches are timed and allocated separately.",
                "Output bytes are scanned after checkpoint timing has ended.",
                "Process/resident memory, peak memory, concurrency evidence, matrix presets, external datasets, public claims, baseline candidates, comparison artifacts and regression gates are out of scope."
            ]);
    }

    public static void Write(HnswBasePlusExactDeltaCheckpointBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static PreparedCheckpointState PrepareCheckpointState(HnswBasePlusExactDeltaCheckpointOptions options)
    {
        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        BuildMeasurement build = BuildBaseIndex(options, dataset);
        var composite = new HnswBasePlusExactDeltaIndex(build.Index);
        long generationBeforeMutations = composite.Generation;
        MutationExecution mutationExecution = ExecuteMutations(options, dataset, composite);
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = CreateCountInfo(options, composite);
        ulong[] liveIds = BuildLiveIds(options);
        TruthSet truth = GenerateLiveTruth(dataset, options, liveIds);

        return new PreparedCheckpointState(
            dataset,
            composite,
            generationBeforeMutations,
            mutationExecution,
            preCounts,
            liveIds,
            truth);
    }

    private static string CreateCheckpointRunDirectory(string checkpointRootDirectory, int runNumber)
    {
        string directory = Path.Combine(
            checkpointRootDirectory,
            string.Create(CultureInfo.InvariantCulture, $"checkpoint-run-{runNumber:000}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static MeasuredCheckpointRun MeasureCheckpointRun(
        HnswBasePlusExactDeltaIndex composite,
        string checkpointDirectory)
    {
        long generationBeforeCheckpoint = composite.Generation;
        long checkpointAllocationStart = GC.GetAllocatedBytesForCurrentThread();
        long checkpointTimestamp = Stopwatch.GetTimestamp();
        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpointDiagnostic =
            composite.CheckpointWithDiagnostics(checkpointDirectory);
        long checkpointElapsedTicks = Stopwatch.GetTimestamp() - checkpointTimestamp;
        long checkpointAllocatedBytes = GC.GetAllocatedBytesForCurrentThread() - checkpointAllocationStart;

        return new MeasuredCheckpointRun(
            checkpointDirectory,
            generationBeforeCheckpoint,
            checkpointDiagnostic,
            checkpointElapsedTicks,
            checkpointAllocatedBytes);
    }

    private static HnswBasePlusExactDeltaCheckpointRunInfo CreateCheckpointRunInfo(
        int runNumber,
        string checkpointDirectory,
        MeasuredCheckpointRun measured)
    {
        HnswBasePlusExactDeltaCheckpointResult result = measured.Diagnostic.Result;
        bool generationAdvanced = result.Generation == measured.GenerationBeforeCheckpoint + 1;
        return new HnswBasePlusExactDeltaCheckpointRunInfo(
            runNumber,
            checkpointDirectory,
            result.Status.ToString(),
            StopwatchTicksToMilliseconds(measured.ElapsedTicks),
            measured.ManagedAllocatedBytes,
            measured.GenerationBeforeCheckpoint,
            result.Generation,
            generationAdvanced,
            CreatePhaseSet(measured.Diagnostic.Diagnostics));
    }

    private static HnswBasePlusExactDeltaCheckpointRunsInfo CreateCheckpointRunsInfo(
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaCheckpointRunInfo[] runs)
    {
        HnswBasePlusExactDeltaCheckpointRunAggregateInfo aggregate = new(
            runs.Length,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds),
            runs.Average(run => run.ManagedAllocatedBytes),
            runs.Min(run => run.ManagedAllocatedBytes),
            runs.Max(run => run.ManagedAllocatedBytes),
            "Aggregate checkpoint timing/allocation is computed across independently rebuilt equivalent checkpoint attempts. Setup, generated state creation, mutation application, exact truth construction, search measurements, output-byte scan, NoChanges probe, validation and report writing are excluded.");

        return new HnswBasePlusExactDeltaCheckpointRunsInfo(
            runs.Length,
            options.Runs,
            "Detailed validation, output inspection, opened-output validation, NoChanges probe, deleted-ID reservation probe and post/opened search parity use the final checkpoint run.",
            runs,
            aggregate);
    }

    private static HnswBasePlusExactDeltaCheckpointSearchSectionInfo CreateSearchSection(
        string name,
        string timedOperation,
        SearchMeasurement measurement,
        SearchSectionEvaluation evaluation,
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaRetryDiagnosticsInfo retryDiagnostics) =>
        new(
            name,
            timedOperation,
            new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            CreateSearchMeasurement(options, measurement.Runs, timedOperation),
            evaluation.Metrics,
            evaluation.Underfill,
            retryDiagnostics);

    private static MeasurementInfo CreateSearchMeasurement(
        HnswBasePlusExactDeltaCheckpointOptions options,
        SearchRunInfo[] runs,
        string timedOperation) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredSearchCall",
                timedOperation,
                "generated data setup, HNSW base build, mutation application, exact truth construction, checkpoint call, output-byte scan, warmup, final result comparison, validation and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Search aggregate percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                runs.Average(run => run.ManagedAllocatedBytesPerQuery).ToString(CultureInfo.InvariantCulture),
                "bytesPerSearchCall",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each Search(query, results, workspace) call using caller-owned SearchResult[] and workspace; setup, checkpoint, warmup, validation and report writing are excluded."),
            NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-134."),
            new RepeatedRunInfo(
                options.Runs > 1 ? "measured" : "singleRun",
                options.Runs,
                options.Runs > 1,
                options.Runs > 1
                    ? "Multiple measured search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                    : "Only one measured search run executed, so cross-run variance/noise is not measured."),
            CreateRunToRunNoise(runs, timedOperation),
            new WarmupInfo(
                options.WarmupQueries > 0 ? "executed" : "absent",
                options.WarmupQueries,
                options.WarmupQueries > 0
                    ? "Warmup queries executed before this measured search section and excluded from measured timing and allocation totals."
                    : "No warmup queries were requested."));

    private static SearchSectionEvaluation EvaluateSearchSection(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        TruthSet truth,
        SearchMeasurement search,
        ulong[] liveIds)
    {
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            search.Results,
            options.TopK,
            options.Dimension,
            options.Metric);
        HnswBasePlusExactDeltaReturnedResultIntegrityInfo integrity =
            ValidateReturnedResults(dataset, search.Results, options.TopK, liveIds, options.Metric);
        int extraResultCount = CountExtraResults(truth, search.Results, options.TopK);

        return new SearchSectionEvaluation(
            new HnswBasePlusExactDeltaCheckpointMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
                integrity.DistanceMismatchCount,
                comparison.MissingResultCount,
                extraResultCount,
                integrity,
                "set recall@k = returned live ids intersect exact updated top-k live ids divided by min(k, post-update live vector count), summed across measured queries",
                "Every returned result is checked for finite distance, no duplicate ID within its query, generated live ID membership, no tombstoned ID, and selected-metric distance matching recomputation for that returned ID/query within the accepted ResultComparer tolerance. HNSW search is approximate and recall/order are recorded, not required."),
            CreateUnderfill(options, search.Results));
    }

    private static HnswBasePlusExactDeltaReturnedResultIntegrityInfo ValidateReturnedResults(
        GeneratedDataset dataset,
        SearchResult[][] actual,
        int topK,
        IReadOnlyCollection<ulong> liveIds,
        VectorMetric metric)
    {
        var live = new HashSet<ulong>(liveIds);
        int checkedResultCount = 0;
        int queryCountMismatchCount = actual.Length == dataset.QueryCount ? 0 : 1;
        int resultCountViolationCount = 0;
        int nonFiniteDistanceCount = 0;
        int duplicateIdCount = 0;
        int unknownIdCount = 0;
        int tombstonedIdCount = 0;
        int distanceMismatchCount = 0;
        int queryCount = Math.Min(dataset.QueryCount, actual.Length);
        int maxExpectedResults = Math.Min(topK, live.Count);

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

                if (!live.Contains(result.Id))
                {
                    tombstonedIdCount++;
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
            tombstonedIdCount == 0 &&
            distanceMismatchCount == 0;

        return new HnswBasePlusExactDeltaReturnedResultIntegrityInfo(
            passed ? "passed" : "failed",
            checkedResultCount,
            queryCountMismatchCount,
            resultCountViolationCount,
            nonFiniteDistanceCount,
            duplicateIdCount,
            unknownIdCount,
            tombstonedIdCount,
            distanceMismatchCount,
            "For every returned result: distance must be finite; IDs must be unique within a query; ID must be one of the post-update live generated IDs; tombstoned IDs must not be returned; and reported distance must match recomputed selected-metric distance for that query and returned ID within the accepted ResultComparer tolerance.",
            passed
                ? "All returned results are live, not tombstoned, well formed and distance-integrity checked."
                : "One or more returned results failed live-ID, tombstone, well-formedness or distance-integrity checks.");
    }

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswBasePlusExactDeltaCheckpointOptions options) =>
        new(
            options.Metric,
            options.Dimension,
            options.PhysicalVectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            options.Runs,
            options.WarmupQueries);

    private static BuildMeasurement BuildBaseIndex(HnswBasePlusExactDeltaCheckpointOptions options, GeneratedDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var index = new HnswIndex(options.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return new BuildMeasurement(
            index,
            StopwatchTicksToMilliseconds(Stopwatch.GetTimestamp() - start),
            GC.GetAllocatedBytesForCurrentThread() - allocationStart);
    }

    private static MutationExecution ExecuteMutations(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var counts = new MutableMutationStatusCounts();
        int inserted = 0;
        int deletedBase = 0;
        int deletedDelta = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            VectorMutationResult result = composite.TryAdd((ulong)(options.BaseVectorCount + i), dataset.GetVector(options.BaseVectorCount + i));
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                inserted++;
            }
        }

        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            VectorMutationResult result = composite.TryDelete((ulong)i);
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                deletedBase++;
            }
        }

        for (int i = 0; i < options.DeletedDeltaCount; i++)
        {
            VectorMutationResult result = composite.TryDelete((ulong)(options.BaseVectorCount + i));
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                deletedDelta++;
            }
        }

        for (int i = 0; i < options.DuplicateInsertAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(i % options.BaseVectorCount);
            counts.Add(composite.TryAdd(id, dataset.GetVector(options.BaseVectorCount + (i % options.InsertedDeltaCount))).Status);
        }

        ulong firstUnknownId = (ulong)options.PhysicalVectorCount + 1UL;
        for (int i = 0; i < options.UnknownDeleteAttempts; i++)
        {
            counts.Add(composite.TryDelete(firstUnknownId + (ulong)i).Status);
        }

        int committedDeleteCount = options.DeletedBaseCount + options.DeletedDeltaCount;
        for (int i = 0; i < options.RepeatedDeleteAttempts; i++)
        {
            ulong id = i % committedDeleteCount < options.DeletedBaseCount
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(options.BaseVectorCount + ((i - options.DeletedBaseCount) % options.DeletedDeltaCount));
            counts.Add(composite.TryDelete(id).Status);
        }

        return new MutationExecution(inserted, deletedBase, deletedDelta, composite.Generation, counts.ToInfo());
    }

    private static ulong[] BuildLiveIds(HnswBasePlusExactDeltaCheckpointOptions options)
    {
        var ids = new ulong[options.LiveVectorCount];
        int write = 0;
        for (int row = options.DeletedBaseCount; row < options.BaseVectorCount; row++)
        {
            ids[write++] = (ulong)row;
        }

        for (int row = options.DeletedDeltaCount; row < options.InsertedDeltaCount; row++)
        {
            ids[write++] = (ulong)(options.BaseVectorCount + row);
        }

        return ids;
    }

    private static TruthSet GenerateLiveTruth(
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaCheckpointOptions options,
        ulong[] liveIds)
    {
        var results = new TruthItem[dataset.QueryCount][];
        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            var candidates = new TruthItem[liveIds.Length];
            for (int i = 0; i < liveIds.Length; i++)
            {
                ulong id = liveIds[i];
                candidates[i] = new TruthItem(id, ScalarGroundTruth.CalculateDistance(query, dataset.GetVector(checked((int)id)), options.Metric));
            }

            Array.Sort(candidates, CompareTruthItems);
            var top = new TruthItem[options.TopK];
            Array.Copy(candidates, top, options.TopK);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    private static void WarmupCompositeSearch(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(options);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            composite.Search(dataset.GetQuery(i % dataset.QueryCount), results, workspace, options.EfSearch);
        }
    }

    private static SearchResult[][] ProbeCompositeFirstPassSearch(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeFirstPassWorkspace(options);
        var captured = new SearchResult[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = composite.Search(dataset.GetQuery(queryRow), results, workspace, options.EfSearch);
            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            captured[queryRow] = queryResults;
        }

        return captured;
    }

    private static void WarmupOpenedSearch(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswIndex opened)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.LiveVectorCount, options.EffectiveWorkspaceEfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            opened.Search(dataset.GetQuery(i % dataset.QueryCount), results, workspace, options.EfSearch);
        }
    }

    private static SearchMeasurement MeasureCompositeSearch(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool capture = captureResults && runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureCompositeSingleRun(options, dataset, composite, capture);
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (capture)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureCompositeSingleRun(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(options);
        return MeasureQueries(
            options,
            dataset,
            captureResults,
            (query, destination) => composite.Search(query, destination, workspace, options.EfSearch));
    }

    private static SearchMeasurement MeasureOpenedSearch(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswIndex opened,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            var workspace = new HnswSearchWorkspace(options.LiveVectorCount, options.EffectiveWorkspaceEfSearch);
            bool capture = captureResults && runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureQueries(
                options,
                dataset,
                capture,
                (query, destination) => opened.Search(query, destination, workspace, options.EfSearch));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (capture)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureQueries(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        bool captureResults,
        SearchOperation operation)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = operation(query, results);
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

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeWorkspace(HnswBasePlusExactDeltaCheckpointOptions options)
    {
        int maxBaseElements = Math.Max(options.BaseVectorCount, options.LiveVectorCount);
        return new HnswBasePlusExactDeltaSearchWorkspace(
            maxBaseElements,
            options.EfSearch,
            options.EffectiveWorkspaceEfSearch,
            Math.Min(maxBaseElements, options.EffectiveWorkspaceEfSearch),
            options.TopK);
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeFirstPassWorkspace(HnswBasePlusExactDeltaCheckpointOptions options)
    {
        int maxBaseElements = Math.Max(options.BaseVectorCount, options.LiveVectorCount);
        return new HnswBasePlusExactDeltaSearchWorkspace(
            maxBaseElements,
            options.EfSearch,
            Math.Min(maxBaseElements, options.EfSearch),
            options.TopK);
    }

    private static HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo ProbeNoChanges(
        HnswBasePlusExactDeltaCheckpointOptions options,
        string checkpointDirectory,
        HnswBasePlusExactDeltaIndex composite)
    {
        string noChangesDirectory = Path.Combine(
            Path.GetDirectoryName(checkpointDirectory) ?? ".",
            Path.GetFileName(checkpointDirectory) + "-no-changes");
        Directory.CreateDirectory(noChangesDirectory);
        long generationBefore = composite.Generation;
        HnswBasePlusExactDeltaCheckpointDiagnosticResult diagnostic =
            composite.CheckpointWithDiagnostics(noChangesDirectory);
        bool outputEmpty = !Directory.EnumerateFileSystemEntries(noChangesDirectory).Any();
        bool passed = diagnostic.Result.Status == HnswBasePlusExactDeltaCheckpointStatus.NoChanges &&
            composite.Generation == generationBefore &&
            outputEmpty &&
            !AnyExecuted(CreatePhaseSet(diagnostic.Diagnostics));

        return new HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo(
            passed ? "passed" : "failed",
            generationBefore,
            composite.Generation,
            composite.Generation == generationBefore,
            outputEmpty,
            CreatePhaseSet(diagnostic.Diagnostics));
    }

    private static bool ValidateDeletedReservation(
        HnswBasePlusExactDeltaCheckpointOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        if (options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            return false;
        }

        ulong deletedId = options.DeletedBaseCount > 0 ? 0UL : (ulong)options.BaseVectorCount;
        VectorMutationResult result = composite.TryAdd(deletedId, dataset.GetVector(options.BaseVectorCount));
        return result.Status == VectorMutationStatus.DuplicateId;
    }

    private static HnswBasePlusExactDeltaCheckpointOpenedValidationInfo ValidateOpenedOutput(
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaCheckpointOptions options,
        ulong[] expectedLiveIds,
        HnswIndex opened,
        HnswBasePlusExactDeltaCheckpointParityInfo? preParity)
    {
        int idMismatchCount = opened.Count == expectedLiveIds.Length ? 0 : 1;
        int vectorMismatchCount = 0;
        int count = Math.Min(opened.Count, expectedLiveIds.Length);
        ReadOnlySpan<ulong> openedIds = opened.InternalIds;
        ReadOnlySpan<float> openedVectors = opened.InternalVectors;

        for (int row = 0; row < count; row++)
        {
            ulong expectedId = expectedLiveIds[row];
            if (openedIds[row] != expectedId)
            {
                idMismatchCount++;
                continue;
            }

            ReadOnlySpan<float> expectedVector = dataset.GetVector(checked((int)expectedId));
            ReadOnlySpan<float> openedVector = openedVectors.Slice(row * options.Dimension, options.Dimension);
            if (!OpenedVectorPayloadMatches(openedVector, expectedVector, options.Metric))
            {
                vectorMismatchCount++;
            }
        }

        bool passed = idMismatchCount == 0 && vectorMismatchCount == 0;
        return new HnswBasePlusExactDeltaCheckpointOpenedValidationInfo(
            passed ? "passed" : "failed",
            expectedLiveIds.Length,
            opened.Count,
            idMismatchCount,
            vectorMismatchCount,
            preParity ?? new HnswBasePlusExactDeltaCheckpointParityInfo(0, 0, 0, 0, 0, false, "Search parity is populated after post-checkpoint and opened searches are captured."),
            "Opened read-only HNSW checkpoint output must contain live IDs in checkpoint live-view order and vector payloads matching generated live rows under the selected metric storage policy; cosine payloads are unit-normalized by HNSW storage. Search parity is validated separately for the same queries and equivalent workspaces.");
    }

    private static bool OpenedVectorPayloadMatches(
        ReadOnlySpan<float> openedVector,
        ReadOnlySpan<float> expectedVector,
        VectorMetric metric)
    {
        if (metric is VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct)
        {
            return openedVector.SequenceEqual(expectedVector);
        }

        if (metric != VectorMetric.Cosine)
        {
            return false;
        }

        double magnitudeSquared = 0;
        for (int i = 0; i < expectedVector.Length; i++)
        {
            magnitudeSquared += (double)expectedVector[i] * expectedVector[i];
        }

        if (magnitudeSquared <= 0)
        {
            return false;
        }

        double magnitude = Math.Sqrt(magnitudeSquared);
        const float tolerance = 1e-6f;
        for (int i = 0; i < expectedVector.Length; i++)
        {
            float normalized = (float)(expectedVector[i] / magnitude);
            if (MathF.Abs(openedVector[i] - normalized) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    private static HnswBasePlusExactDeltaCheckpointParityInfo CompareSearchParity(
        SearchResult[][] rebuiltComposite,
        SearchResult[][] opened,
        int dimension)
    {
        int writtenCountMismatch = rebuiltComposite.Length == opened.Length ? 0 : 1;
        int idMismatch = 0;
        int orderMismatch = 0;
        int distanceMismatch = 0;
        int queryCount = Math.Min(rebuiltComposite.Length, opened.Length);
        for (int query = 0; query < queryCount; query++)
        {
            SearchResult[] left = rebuiltComposite[query];
            SearchResult[] right = opened[query];
            if (left.Length != right.Length)
            {
                writtenCountMismatch++;
            }

            int count = Math.Min(left.Length, right.Length);
            for (int i = 0; i < count; i++)
            {
                if (left[i].Id != right[i].Id)
                {
                    idMismatch++;
                    orderMismatch++;
                }

                if (!DistanceMatches(left[i].Distance, right[i].Distance, dimension))
                {
                    distanceMismatch++;
                }
            }
        }

        bool passed = writtenCountMismatch == 0 && idMismatch == 0 && orderMismatch == 0 && distanceMismatch == 0;
        return new HnswBasePlusExactDeltaCheckpointParityInfo(
            rebuiltComposite.Length,
            writtenCountMismatch,
            idMismatch,
            orderMismatch,
            distanceMismatch,
            passed,
            "Post-checkpoint rebuilt composite Search and opened read-only HNSW Search are executed for the same queries with fresh caller-owned workspaces and must return the same count, IDs, order and distances within D-026 tolerance.");
    }

    private static HnswBasePlusExactDeltaCheckpointOutputInfo InspectCheckpointOutput(string directory, int vectorCount)
    {
        long manifestBytes = FileLength(directory, ManifestFileName);
        long idsBytes = FileLength(directory, IdsFileName);
        long vectorsBytes = FileLength(directory, VectorsFileName);
        long levelsBytes = FileLength(directory, LevelsFileName);
        long graphBytes = FileLength(directory, GraphFileName);
        long totalBytes = manifestBytes + idsBytes + vectorsBytes + levelsBytes + graphBytes;
        int fileCount = Directory.EnumerateFiles(directory).Count();
        string validationStatus;
        try
        {
            _ = HnswIndex.OpenReadOnly(directory);
            validationStatus = "passed";
        }
        catch
        {
            validationStatus = "failed";
        }

        return new HnswBasePlusExactDeltaCheckpointOutputInfo(
            "recorded",
            directory,
            fileCount,
            totalBytes,
            manifestBytes,
            idsBytes,
            vectorsBytes,
            levelsBytes,
            graphBytes,
            vectorCount,
            vectorCount == 0 ? 0 : (double)totalBytes / vectorCount,
            validationStatus,
            "outsideCheckpointDuration");
    }

    private static long FileLength(string directory, string fileName) =>
        new FileInfo(Path.Combine(directory, fileName)).Length;

    private static HnswBasePlusExactDeltaCheckpointCountInfo CreateCountInfo(
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaIndex composite)
    {
        int physicalCount = checked(composite.BasePhysicalVectorCount + composite.DeltaPhysicalVectorCount);
        return new HnswBasePlusExactDeltaCheckpointCountInfo(
            composite.BasePhysicalVectorCount,
            composite.BaseLiveVectorCount,
            composite.DeltaPhysicalVectorCount,
            composite.DeltaLiveVectorCount,
            composite.BaseTombstoneCount,
            composite.DeltaTombstoneCount,
            composite.TombstoneCount,
            composite.LiveVectorCount,
            composite.DeletedReservedIdCount,
            composite.Generation,
            physicalCount == 0 ? 0 : (double)composite.TombstoneCount / physicalCount,
            options.BaseVectorCount == 0 ? 0 : (double)composite.DeltaPhysicalVectorCount / options.BaseVectorCount,
            "Before checkpoint, base and delta physical rows include tombstoned rows. After a published checkpoint, live rows are folded into a rebuilt immutable HNSW base, delta rows and tombstones are cleared, and deleted/reserved IDs remain retained in the writable composite instance.");
    }

    private static HnswBasePlusExactDeltaCheckpointMutationInfo CreateMutationInfo(
        HnswBasePlusExactDeltaCheckpointOptions options,
        MutationExecution mutationExecution,
        long generationBeforeMutations,
        bool statusCountsMatched,
        bool generationMatched)
    {
        int committed = mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount;
        return new HnswBasePlusExactDeltaCheckpointMutationInfo(
            mutationExecution.InsertedCount,
            mutationExecution.DeletedBaseCount,
            mutationExecution.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            committed,
            generationBeforeMutations,
            mutationExecution.GenerationAfterMutations,
            mutationExecution.GenerationAfterMutations - generationBeforeMutations,
            statusCountsMatched && generationMatched,
            mutationExecution.StatusCounts);
    }

    private static HnswBasePlusExactDeltaCheckpointResultInfo CreateCheckpointResultInfo(
        HnswBasePlusExactDeltaCheckpointResult result) =>
        new(
            result.Status.ToString(),
            result.Generation,
            result.RebuiltBaseVectorCount,
            result.LiveVectorCount,
            result.BasePhysicalVectorCount,
            result.BaseLiveVectorCount,
            result.DeltaPhysicalVectorCount,
            result.DeltaLiveVectorCount,
            result.BaseTombstoneCount,
            result.DeltaTombstoneCount,
            result.TombstoneCount,
            result.DeletedReservedIdCount,
            result.FoldedDeltaVectorCount,
            result.FoldedBaseTombstoneCount,
            result.FoldedDeltaTombstoneCount);

    private static HnswBasePlusExactDeltaCheckpointPhaseSetInfo CreatePhaseSet(
        HnswBasePlusExactDeltaCheckpointDiagnostics diagnostics) =>
        new(
            CreatePhaseInfo(diagnostics.LiveSnapshot),
            CreatePhaseInfo(diagnostics.RebuildBuild),
            CreatePhaseInfo(diagnostics.Save),
            CreatePhaseInfo(diagnostics.OpenValidation),
            CreatePhaseInfo(diagnostics.Publication));

    private static HnswBasePlusExactDeltaCheckpointPhaseInfo CreatePhaseInfo(
        HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics) =>
        new(
            diagnostics.Status.ToString(),
            diagnostics.ElapsedTicks,
            TimeSpan.FromTicks(diagnostics.ElapsedTicks).TotalMilliseconds,
            diagnostics.ManagedAllocatedBytes,
            "VEC-133 internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics phase diagnostic");

    private static bool AllMeasured(HnswBasePlusExactDeltaCheckpointPhaseSetInfo phases) =>
        phases.LiveSnapshot.Status == nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured) &&
        phases.RebuildBuild.Status == nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured) &&
        phases.Save.Status == nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured) &&
        phases.OpenValidation.Status == nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured) &&
        phases.Publication.Status == nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured);

    private static bool AnyExecuted(HnswBasePlusExactDeltaCheckpointPhaseSetInfo phases) =>
        phases.LiveSnapshot.Status != nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted) ||
        phases.RebuildBuild.Status != nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted) ||
        phases.Save.Status != nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted) ||
        phases.OpenValidation.Status != nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted) ||
        phases.Publication.Status != nameof(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted);

    private static bool CheckpointRepeatedRunEvidencePresent(
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaCheckpointRunsInfo checkpointRuns)
    {
        if (checkpointRuns.RunCount != options.Runs ||
            checkpointRuns.Runs.Length != options.Runs ||
            checkpointRuns.Aggregate.RunCount != options.Runs ||
            checkpointRuns.DetailedValidationRunNumber != options.Runs)
        {
            return false;
        }

        var directories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < checkpointRuns.Runs.Length; i++)
        {
            HnswBasePlusExactDeltaCheckpointRunInfo run = checkpointRuns.Runs[i];
            if (run.RunNumber != i + 1 ||
                run.Status != nameof(HnswBasePlusExactDeltaCheckpointStatus.Published) ||
                run.ElapsedMilliseconds < 0 ||
                run.ManagedAllocatedBytes < 0 ||
                !run.GenerationAdvancedExactlyOnce ||
                !AllMeasured(run.Phases) ||
                string.IsNullOrWhiteSpace(run.CheckpointDirectory) ||
                !directories.Add(run.CheckpointDirectory))
            {
                return false;
            }
        }

        return options.Runs == 1 || checkpointRuns.Runs.Length > 1;
    }

    private static bool MutationStatusCountsMatch(HnswBasePlusExactDeltaCheckpointOptions options, MutationExecution mutationExecution) =>
        mutationExecution.StatusCounts.Committed ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount &&
        mutationExecution.StatusCounts.DuplicateId == options.DuplicateInsertAttempts &&
        mutationExecution.StatusCounts.UnknownId == options.UnknownDeleteAttempts &&
        mutationExecution.StatusCounts.AlreadyDeleted == options.RepeatedDeleteAttempts &&
        mutationExecution.StatusCounts.ReadOnly == 0 &&
        mutationExecution.StatusCounts.Unsupported == 0;

    private static bool CheckpointCountsMatch(
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts,
        HnswBasePlusExactDeltaCheckpointResult result) =>
        result.Status == HnswBasePlusExactDeltaCheckpointStatus.Published &&
        result.RebuiltBaseVectorCount == preCounts.LiveVectorCount &&
        result.LiveVectorCount == preCounts.LiveVectorCount &&
        result.BasePhysicalVectorCount == preCounts.LiveVectorCount &&
        result.BaseLiveVectorCount == preCounts.LiveVectorCount &&
        result.DeltaPhysicalVectorCount == 0 &&
        result.DeltaLiveVectorCount == 0 &&
        result.BaseTombstoneCount == 0 &&
        result.DeltaTombstoneCount == 0 &&
        result.TombstoneCount == 0 &&
        result.DeletedReservedIdCount == preCounts.DeletedReservedIdCount &&
        result.FoldedDeltaVectorCount == preCounts.DeltaLiveVectorCount &&
        result.FoldedBaseTombstoneCount == preCounts.BaseTombstoneCount &&
        result.FoldedDeltaTombstoneCount == preCounts.DeltaTombstoneCount;

    private static bool PostCountsMatch(
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts,
        HnswBasePlusExactDeltaCheckpointCountInfo postCounts,
        HnswBasePlusExactDeltaCheckpointResult result) =>
        postCounts.BasePhysicalVectorCount == preCounts.LiveVectorCount &&
        postCounts.BaseLiveVectorCount == preCounts.LiveVectorCount &&
        postCounts.DeltaPhysicalVectorCount == 0 &&
        postCounts.DeltaLiveVectorCount == 0 &&
        postCounts.BaseTombstoneCount == 0 &&
        postCounts.DeltaTombstoneCount == 0 &&
        postCounts.TombstoneCount == 0 &&
        postCounts.LiveVectorCount == preCounts.LiveVectorCount &&
        postCounts.DeletedReservedIdCount == preCounts.DeletedReservedIdCount &&
        postCounts.Generation == result.Generation;

    private static HnswBasePlusExactDeltaUnderfillInfo CreateUnderfill(
        HnswBasePlusExactDeltaCheckpointOptions options,
        SearchResult[][] actual)
    {
        int totalReturned = 0;
        int underfilledQueries = 0;
        int underfilledSlots = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            int returned = actual[i].Length;
            totalReturned += returned;
            if (returned < options.TopK)
            {
                underfilledQueries++;
                underfilledSlots += options.TopK - returned;
            }
        }

        return new HnswBasePlusExactDeltaUnderfillInfo(
            options.QueryCount,
            options.TopK,
            checked(options.QueryCount * options.TopK),
            totalReturned,
            underfilledQueries,
            underfilledSlots,
            "Underfill is recorded when a search section returns fewer than requested top-k live results for a query. This can occur from approximate HNSW traversal and tombstone-filtered overfetch before checkpoint; it is recorded against exact updated truth rather than treated as exact-search failure.");
    }

    private static HnswBasePlusExactDeltaRetryDiagnosticsInfo CreateRetryDiagnostics(
        HnswBasePlusExactDeltaCheckpointOptions options,
        SearchResult[][] firstPassResults,
        SearchResult[][] finalResults,
        HnswBasePlusExactDeltaUnderfillInfo underfill,
        int basePhysicalVectorCount,
        int baseTombstoneCount,
        string statusWhenCannotWiden,
        string policyPrefix)
    {
        int effectiveRetryCeiling = Math.Min(basePhysicalVectorCount, options.EffectiveWorkspaceEfSearch);
        int firstPassReturned = CountReturnedResults(firstPassResults);
        int finalReturned = CountReturnedResults(finalResults);
        int moreResults = 0;
        int differentResults = 0;
        int queryCount = Math.Min(firstPassResults.Length, finalResults.Length);
        for (int query = 0; query < queryCount; query++)
        {
            if (finalResults[query].Length > firstPassResults[query].Length)
            {
                moreResults++;
            }

            if (!SameResults(firstPassResults[query], finalResults[query]))
            {
                differentResults++;
            }
        }

        bool canWiden = options.EffectiveWorkspaceEfSearch > options.EfSearch && effectiveRetryCeiling > options.EfSearch;
        bool baseTombstonesPresent = baseTombstoneCount > 0;
        bool wideningObserved = canWiden && baseTombstonesPresent && differentResults > 0;
        return new HnswBasePlusExactDeltaRetryDiagnosticsInfo(
            canWiden ? "measured" : statusWhenCannotWiden,
            options.EfSearch,
            options.EffectiveWorkspaceEfSearch,
            effectiveRetryCeiling,
            Math.Max(0, effectiveRetryCeiling - options.EfSearch),
            canWiden,
            baseTombstonesPresent,
            firstPassReturned,
            finalReturned,
            moreResults,
            differentResults,
            wideningObserved,
            underfill.UnderfilledQueryCount > 0,
            underfill.UnderfilledQueryCount,
            underfill.UnderfilledSlotCount,
            policyPrefix + " retry diagnostics compare a tight first-pass probe using workspaceEfSearch equal to efSearch with final measured results using the configured workspaceEfSearch ceiling and explicit first-pass efSearch. Result-count or result-set differences show observable adaptive-retry widening effects; internal retry invocation counts are not instrumented by this runner.");
    }

    private static HnswBasePlusExactDeltaRetryDiagnosticsInfo CreateNotApplicableRetryDiagnostics(
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaUnderfillInfo underfill,
        string reason) =>
        new(
            "notApplicable",
            options.EfSearch,
            options.EffectiveWorkspaceEfSearch,
            options.EfSearch,
            0,
            WorkspaceCanWidenBeyondFirstPass: false,
            BaseTombstonesPresent: false,
            FirstPassTotalReturnedResults: underfill.TotalReturnedResults,
            FinalTotalReturnedResults: underfill.TotalReturnedResults,
            QueryCountWithMoreResultsAfterWidening: 0,
            QueryCountWithDifferentResultsAfterWidening: 0,
            RetryWideningObserved: false,
            UnderfillRemainedAfterWidening: underfill.UnderfilledQueryCount > 0,
            underfill.UnderfilledQueryCount,
            underfill.UnderfilledSlotCount,
            reason);

    private static int CountReturnedResults(SearchResult[][] results)
    {
        int count = 0;
        foreach (SearchResult[] queryResults in results)
        {
            count += queryResults.Length;
        }

        return count;
    }

    private static bool SameResults(SearchResult[] left, SearchResult[] right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        for (int i = 0; i < left.Length; i++)
        {
            if (left[i].Id != right[i].Id || left[i].Distance != right[i].Distance)
            {
                return false;
            }
        }

        return true;
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

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs, string operationName)
    {
        bool measured = runs.Length > 1;
        string unavailableReason = "Only one measured search run exists; this field does not establish run-to-run variation.";
        return new RunToRunNoiseInfo(
            measured ? "measured" : "notMeasured",
            runs.Length,
            measured,
            $"Across measured {operationName} runs; warmup, setup, checkpoint, validation and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured search runs executed; simple descriptive run-to-run statistics are recorded for private local checkpoint smoke inspection."
                : "Only one measured search run executed, so run-to-run noise is unavailable and cannot be measured.",
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateSearchMetricNoise(runs, "milliseconds", run => run.ElapsedMilliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "queriesPerSecond", run => run.Qps, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP50Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP95Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "milliseconds", run => run.LatencyP99Milliseconds, measured, unavailableReason),
            CreateSearchMetricNoise(runs, "bytesPerSearchCall", run => run.ManagedAllocatedBytesPerQuery, measured, unavailableReason));
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
            "Computed across measured search runs using the documented private descriptive-statistics formula.");
    }

    private static HnswEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-hnsw-base-plus-exact-delta-checkpoint-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW base-plus-exact-delta checkpoint smoke output is not reviewed public evidence and has no public reporting policy.",
            "No generated mutable/update HNSW checkpoint baseline-candidate policy is accepted.",
            "No generated mutable/update HNSW checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated squared-L2, inner-product or cosine HNSW base-plus-exact-delta checkpoint smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Checkpoint total timing/allocation and VEC-133 phase diagnostics are reported separately from all search timings.",
                "Output bytes are scanned after checkpoint timing has ended.",
                "Managed allocations are smoke fields only; resident/process memory and peak memory are explicitly not measured.",
                "Not eligible for public mutable-HNSW, performance, recall, memory, allocation, baseline, comparison, regression-gate, external-dataset, matrix or concurrency claims."
            ]);

    private static HnswEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW base-plus-exact-delta checkpoint smoke output is not reviewed public evidence and has no public reporting policy.",
            "No generated mutable/update HNSW checkpoint baseline-candidate policy is accepted.",
            "No generated mutable/update HNSW checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static float SquaredEuclideanDistance(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static bool DistanceMatches(float expected, float actual, int dimension)
    {
        if (!float.IsFinite(actual))
        {
            return false;
        }

        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(expected));
        float tolerance = (float)Math.Max(2e-4, relative);
        return MathF.Abs(expected - actual) <= tolerance;
    }

    private static void ValidateOptions(HnswBasePlusExactDeltaCheckpointOptions options)
    {
        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine))
        {
            throw new ArgumentException("generated-hnsw-base-plus-exact-delta-checkpoint supports only SquaredEuclidean, InnerProduct and Cosine.", nameof(options));
        }

        if (options.InsertedDeltaCount <= 0)
        {
            throw new ArgumentException("inserted delta count must be positive.", nameof(options));
        }

        if (options.DeletedBaseCount < 0 || options.DeletedBaseCount > options.BaseVectorCount)
        {
            throw new ArgumentException("deleted base count must be non-negative and no larger than base vector count.", nameof(options));
        }

        if (options.DeletedDeltaCount < 0 || options.DeletedDeltaCount > options.InsertedDeltaCount)
        {
            throw new ArgumentException("deleted delta count must be non-negative and no larger than inserted delta count.", nameof(options));
        }

        if (options.LiveVectorCount <= 0 || options.TopK > options.LiveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-update live vector count.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.RepeatedDeleteAttempts > 0 && options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            throw new ArgumentException("repeated delete attempts require at least one committed delete.", nameof(options));
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

        if (options.EffectiveWorkspaceEfSearch < options.EfSearch || options.EffectiveWorkspaceEfSearch > 4096)
        {
            throw new ArgumentException("workspace-ef-search must be at least ef-search and no more than 4096.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("output path must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.CheckpointDirectory))
        {
            throw new ArgumentException("checkpoint directory must not be empty.", nameof(options));
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

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static string CreateReportId(string? commit, HnswBasePlusExactDeltaCheckpointOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswBasePlusExactDeltaCheckpointOptions.ScenarioName}-{commitPart}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-wefs{options.EffectiveWorkspaceEfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static double StopwatchTicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private delegate int SearchOperation(ReadOnlySpan<float> query, Span<SearchResult> results);

    private sealed record BuildMeasurement(HnswIndex Index, double ElapsedMilliseconds, long ManagedAllocatedBytes);

    private sealed record MutationExecution(
        int InsertedCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        long GenerationAfterMutations,
        GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

    private sealed record PreparedCheckpointState(
        GeneratedDataset Dataset,
        HnswBasePlusExactDeltaIndex Composite,
        long GenerationBeforeMutations,
        MutationExecution MutationExecution,
        HnswBasePlusExactDeltaCheckpointCountInfo PreCounts,
        ulong[] LiveIds,
        TruthSet Truth);

    private sealed record MeasuredCheckpointRun(
        string DirectoryPath,
        long GenerationBeforeCheckpoint,
        HnswBasePlusExactDeltaCheckpointDiagnosticResult Diagnostic,
        long ElapsedTicks,
        long ManagedAllocatedBytes);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);

    private sealed record SearchSectionEvaluation(
        HnswBasePlusExactDeltaCheckpointMetricsInfo Metrics,
        HnswBasePlusExactDeltaUnderfillInfo Underfill);

    private sealed class MutableMutationStatusCounts
    {
        public int Committed { get; private set; }

        public int DuplicateId { get; private set; }

        public int UnknownId { get; private set; }

        public int AlreadyDeleted { get; private set; }

        public int ReadOnly { get; private set; }

        public int Unsupported { get; private set; }

        public void Add(VectorMutationStatus status)
        {
            switch (status)
            {
                case VectorMutationStatus.Committed:
                    Committed++;
                    break;
                case VectorMutationStatus.DuplicateId:
                    DuplicateId++;
                    break;
                case VectorMutationStatus.UnknownId:
                    UnknownId++;
                    break;
                case VectorMutationStatus.AlreadyDeleted:
                    AlreadyDeleted++;
                    break;
                case VectorMutationStatus.ReadOnly:
                    ReadOnly++;
                    break;
                case VectorMutationStatus.Unsupported:
                    Unsupported++;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(status), "Mutation status is not supported.");
            }
        }

        public GeneratedExactUpdateMutationStatusCountInfo ToInfo() =>
            new(Committed, DuplicateId, UnknownId, AlreadyDeleted, ReadOnly, Unsupported);
    }
}
