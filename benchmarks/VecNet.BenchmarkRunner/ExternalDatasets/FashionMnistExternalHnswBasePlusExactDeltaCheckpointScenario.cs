using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario
{
    private const string TaskId = "VEC-138";
    private const string SchemaName = "VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "hnsw.manifest.json";
    private const string IdsFileName = "hnsw.ids.u64";
    private const string VectorsFileName = "hnsw.vectors.f32";
    private const string LevelsFileName = "hnsw.levels.i32";
    private const string GraphFileName = "hnsw.graph.bin";

    public static ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport Run(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadAndValidateDataset(options);

        string checkpointRootDirectory = Path.GetFullPath(options.CheckpointDirectory);
        Directory.CreateDirectory(checkpointRootDirectory);

        var checkpointRunInfos = new HnswBasePlusExactDeltaCheckpointRunInfo[options.Runs];
        PreparedCheckpointState? finalState = null;
        MeasuredCheckpointRun? finalCheckpoint = null;
        SearchMeasurement? preSearch = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            PreparedCheckpointState state = PrepareCheckpointState(options, dataset);
            bool finalRun = runIndex == options.Runs - 1;
            if (finalRun)
            {
                WarmupCompositeSearch(options, dataset, state.Composite);
                preSearch = MeasureCompositeSearch(options, dataset, state.Composite, captureResults: true);
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

        HnswBasePlusExactDeltaIndex composite = finalState.Composite;
        long generationBeforeMutations = finalState.GenerationBeforeMutations;
        MutationExecution mutationExecution = finalState.MutationExecution;
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = finalState.PreCounts;
        ulong[] liveIds = finalState.LiveIds;
        TruthSet truth = finalState.Truth;
        long generationBeforeCheckpoint = finalCheckpoint.GenerationBeforeCheckpoint;
        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpointDiagnostic = finalCheckpoint.Diagnostic;
        HnswBasePlusExactDeltaCheckpointResult checkpointResult = checkpointDiagnostic.Result;
        HnswBasePlusExactDeltaCheckpointCountInfo postCounts = CreateCountInfo(options, composite);
        HnswBasePlusExactDeltaCheckpointRunsInfo checkpointRuns =
            CreateCheckpointRunsInfo(options, checkpointRunInfos);
        HnswBasePlusExactDeltaCheckpointPhaseSetInfo phaseDiagnostics =
            CreatePhaseSet(checkpointDiagnostic.Diagnostics);

        HnswBasePlusExactDeltaCheckpointOutputInfo output =
            InspectCheckpointOutput(finalCheckpoint.DirectoryPath, checkpointResult.LiveVectorCount);
        HnswIndex opened = HnswIndex.OpenReadOnly(finalCheckpoint.DirectoryPath);
        HnswBasePlusExactDeltaCheckpointOpenedValidationInfo openedValidation =
            ValidateOpenedOutput(dataset, options, liveIds, opened, preParity: null);

        HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo noChangesProbe =
            ProbeNoChanges(finalCheckpoint.DirectoryPath, composite);
        bool deletedReservedRejected = ValidateDeletedReservation(options, dataset, composite);

        WarmupCompositeSearch(options, dataset, composite);
        SearchMeasurement postSearch = MeasureCompositeSearch(options, dataset, composite, captureResults: true);
        WarmupOpenedSearch(options, dataset, opened);
        SearchMeasurement openedSearch = MeasureOpenedSearch(options, dataset, opened, captureResults: true);
        HnswBasePlusExactDeltaCheckpointParityInfo parity =
            CompareSearchParity(postSearch.Results, openedSearch.Results, dataset.Dimension);
        openedValidation = openedValidation with { RebuiltCompositeOpenedSearchParity = parity };

        SearchSectionEvaluation preEvaluation = EvaluateSearchSection(options, dataset, truth, preSearch, liveIds);
        SearchSectionEvaluation postEvaluation = EvaluateSearchSection(options, dataset, truth, postSearch, liveIds);
        SearchSectionEvaluation openedEvaluation = EvaluateSearchSection(options, dataset, truth, openedSearch, liveIds);

        bool mutationStatusCountsMatched = MutationStatusCountsMatch(options, mutationExecution);
        bool mutationGenerationMatched =
            mutationExecution.GenerationAfterMutations - generationBeforeMutations ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount;
        bool checkpointCountsMatched = CheckpointCountsMatch(preCounts, checkpointResult);
        bool postCountsMatched = PostCountsMatch(preCounts, postCounts, checkpointResult);
        bool checkpointGenerationAdvanced = checkpointResult.Generation == generationBeforeCheckpoint + 1 &&
            composite.Generation == generationBeforeCheckpoint + 1;
        bool publishedPhasesMeasured = AllMeasured(phaseDiagnostics);
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
        ExternalBenchmarkEvidenceInfo evidence = CreateEvidence();
        ExternalBenchmarkEligibilityInfo eligibility = CreateEligibility();

        return new ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            evidence,
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, commandArguments.ToArray()),
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
            new ExternalHnswBasePlusExactDeltaCheckpointWorkloadInfo(
                dataset.Manifest.DatasetId,
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                options.QueryCount,
                options.TopK,
                dataset.Dimension,
                dataset.Manifest.Shape.SourceDataType,
                dataset.Manifest.Shape.ConvertedDataType,
                "first N query vectors from the admitted Fashion-MNIST query matrix; query rows are never candidate vectors",
                ImmutableBaseStartRow: 0,
                ImmutableBaseEndRowInclusive: options.BaseVectorCount - 1,
                options.BaseVectorCount,
                DeltaStartRow: options.BaseVectorCount,
                DeltaEndRowInclusive: options.PhysicalCandidateVectorCount - 1,
                options.InsertedDeltaCount,
                dataset.BaseCount - options.PhysicalCandidateVectorCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.Runs,
                options.WarmupQueries,
                "contiguous admitted base-matrix rows: immutable base first, exact delta immediately after base, remaining admitted rows unused",
                "candidate external IDs are original Fashion-MNIST base row ordinals; deleted IDs remain reserved inside the writable composite after checkpoint publication",
                "build immutable HNSW base, add exact delta rows, delete base rows from the start, delete delta rows from the start, attempt duplicate/reserved insert, unknown delete and repeated delete, then checkpoint/rebuild",
                string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
                "each checkpoint run uses fresh composite state and a fresh ignored checkpoint-run-NNN subdirectory; output-byte scan occurs after final-run checkpoint timing"),
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
                "Existing admitted Fashion-MNIST exact truth is loaded only as cache/readiness guard; final recall/order use updated live-view truth.",
                dataset.Truth.SourceRawSha256),
            new ExternalHnswBasePlusExactDeltaUpdatedTruthInfo(
                "scalar-reference-external-live-hnsw-base-plus-exact-delta-checkpoint",
                "computed in memory during the scenario from the post-update live view: selected immutable base rows plus committed delta rows minus base and delta tombstones",
                Persisted: false,
                options.QueryCount,
                truth.Depth,
                liveIds.Length,
                FashionMnistExactTruth.TiePolicy(options.Metric),
                FashionMnistExactTruth.DistanceSemantics(options.Metric),
                "existing admitted truth artifact validates cache/truth readiness only and is not final updated truth",
                "live candidate IDs are selected immutable base rows and committed exact delta rows after tombstone suppression"),
            new ScenarioInfo(
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "cache checks, checksum validation, matrix/truth load, immutable HNSW base build, update application, exact updated truth construction, warmup, checkpoint timing, output-byte scan, final-run result capture/comparison and report writing are separated by section"),
            new IndexInfo(
                "InternalExternalHnswBasePlusExactDeltaCheckpoint",
                nameof(HnswBasePlusExactDeltaIndex),
                options.Metric.ToString(),
                dataset.Dimension,
                postCounts.LiveVectorCount,
                "internal HnswBasePlusExactDeltaIndex checkpoint/rebuild smoke report over admitted Fashion-MNIST cache; no public mutable/update HNSW API, matrix, memory evidence, concurrency evidence, package change or public claim"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "admitted Fashion-MNIST base matrix row order, immutable base rows first, original row ordinals as external IDs",
                $"{options.Metric} only"),
            CreateBuildInfo(finalState.Build, options, dataset),
            preCounts,
            CreateMutationInfo(options, mutationExecution, generationBeforeMutations, mutationStatusCountsMatched, mutationGenerationMatched),
            checkpointRuns,
            new HnswBasePlusExactDeltaCheckpointOperationInfo(
                checkpointResult.Status.ToString(),
                "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                StopwatchTicksToMilliseconds(finalCheckpoint.ElapsedTicks),
                finalCheckpoint.ManagedAllocatedBytes,
                generationBeforeCheckpoint,
                checkpointResult.Generation,
                checkpointGenerationAdvanced,
                phaseDiagnostics,
                "cache checks, matrix/truth load, immutable HNSW base build, update application, exact updated truth construction, search timing, no-changes probe, output-byte scan, opened-output validation and report writing"),
            CreateCheckpointResultInfo(checkpointResult),
            postCounts,
            noChangesProbe,
            output,
            openedValidation,
            new ExternalHnswBasePlusExactDeltaCheckpointSearchSectionsInfo(
                CreateSearchSection("preCheckpointSourceComposite", "internal pre-checkpoint external HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", preSearch, preEvaluation, options),
                CreateSearchSection("postCheckpointRebuiltComposite", "internal post-checkpoint rebuilt external HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", postSearch, postEvaluation, options),
                CreateSearchSection("openedReadOnlyHnsw", "internal opened read-only external HnswIndex.Search(query, results, workspace)", openedSearch, openedEvaluation, options)),
            new ExternalHnswBasePlusExactDeltaCheckpointMeasurementInfo(
                new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perCheckpointCall",
                    "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                    "cache checks, matrix/truth load, HNSW base build, mutation application, exact truth construction, warmup, all search measurements, output-byte scan, no-changes probe, validation and report writing",
                    "single Stopwatch sample around the internal checkpoint call boundary; phase elapsed values come from VEC-133 diagnostics and are not summed or inferred",
                    "Aggregate checkpoint elapsed fields are computed across independently rebuilt equivalent external checkpoint attempts; this is not BenchmarkDotNet statistics and not a regression gate.",
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
                NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-138."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries execute before each measured search section and are excluded from search and checkpoint timing."
                        : "No warmup queries were requested."),
                "Cache checks, matrix/truth load, HNSW base build, mutation application, exact truth construction, warmup, output-byte scan, no-changes probe, validation and report writing are excluded from checkpoint timing; checkpoint timing and all search timings are separate."),
            new ExternalHnswBasePlusExactDeltaCheckpointValidationInfo(
                validationPassed ? "passed" : "failed",
                "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-smoke",
                CacheAndTruthReadinessPassed: true,
                ExistingTruthGuardLoaded: true,
                UpdatedTruthGeneratedFromLiveView: true,
                PreCheckpointSourceCompositeComparedToTruth: true,
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
                "Private external Fashion-MNIST HNSW base-plus-exact-delta checkpoint smoke evidence only; not a public benchmark claim.",
                "This command uses only an already admitted Fashion-MNIST cache and existing truth artifact; it does not download, convert, refresh or regenerate dataset/truth artifacts.",
                "Existing admitted truth is a cache/readiness guard only; exact updated truth is computed in memory from the post-update live view.",
                "Source composite versus rebuilt/opened approximate differences are metadata; rebuilt composite versus opened read-only HNSW parity is strict.",
                "Checkpoint timing/allocation is measured at the runner call boundary and VEC-133 phase diagnostics are copied from the internal result; phase timings are not inferred or fabricated.",
                "For runs greater than one, checkpoint timing/allocation is measured across independently rebuilt equivalent checkpoint attempts with fresh external composite state and fresh checkpoint output subdirectories; detailed validation uses the final run.",
                "Pre-checkpoint source composite, post-checkpoint rebuilt composite and opened read-only HNSW searches are timed and allocated separately.",
                "Output bytes are scanned after checkpoint timing has ended.",
                "Process/resident memory, peak memory, concurrency evidence, matrix presets, public claims, baseline candidates, comparison artifacts and regression gates are out of scope."
            ]);
    }

    public static void Write(ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    internal static HnswBasePlusExactDeltaReturnedResultIntegrityInfo ValidateReturnedResults(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        SearchResult[][] actual,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        IReadOnlyCollection<ulong> liveIds)
    {
        var live = new HashSet<ulong>(liveIds);
        int checkedResultCount = 0;
        int queryCountMismatchCount = actual.Length == options.QueryCount ? 0 : 1;
        int resultCountViolationCount = 0;
        int nonFiniteDistanceCount = 0;
        int duplicateIdCount = 0;
        int unknownIdCount = 0;
        int tombstonedIdCount = 0;
        int distanceMismatchCount = 0;
        int queryCount = Math.Min(options.QueryCount, actual.Length);
        int maxExpectedResults = Math.Min(options.TopK, live.Count);

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

                if (result.Id >= (ulong)dataset.BaseCount ||
                    result.Id >= (ulong)options.PhysicalCandidateVectorCount)
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
                    dataset.GetQueryVector(queryRow),
                    dataset.GetBaseVector(checked((int)result.Id)),
                    options.Metric);
                if (!ResultComparer.DistanceMatches(expectedDistance, result.Distance, dataset.Dimension, options.Metric))
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
            "For every returned external checkpoint result: distance must be finite; IDs must be unique within a query; ID must be one of the selected post-update live Fashion-MNIST base-row IDs; tombstoned IDs must not be returned; and reported distance must match recomputed selected-metric distance for that query and returned ID within the accepted ResultComparer tolerance.",
            passed
                ? "All returned external checkpoint results are live, not tombstoned, well formed and distance-integrity checked."
                : "One or more returned external checkpoint results failed live-ID, tombstone, well-formedness or distance-integrity checks.");
    }

    internal static ulong[] BuildLiveIds(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
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

    internal static TruthSet GenerateLiveTruth(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        ulong[] liveIds)
    {
        var results = new TruthItem[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQueryVector(queryRow);
            var candidates = new TruthItem[liveIds.Length];
            for (int i = 0; i < liveIds.Length; i++)
            {
                ulong id = liveIds[i];
                candidates[i] = new TruthItem(id, ScalarGroundTruth.CalculateDistance(query, dataset.GetBaseVector(checked((int)id)), options.Metric));
            }

            Array.Sort(candidates, CompareTruthItems);
            var top = new TruthItem[options.TopK];
            Array.Copy(candidates, top, options.TopK);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    internal static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadAndValidateDataset(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
    {
        var guardOptions = new FashionMnistExternalHnswBenchmarkOptions(
            options.CacheRoot,
            options.OutputPath,
            options.QueryCount,
            options.TopK,
            Runs: 1,
            options.WarmupQueries,
            options.Metric,
            options.M,
            options.EfConstruction,
            options.EfSearch,
            options.HnswSeed);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset =
            FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(guardOptions);

        Require(options.PhysicalCandidateVectorCount <= dataset.BaseCount, "base vectors plus insertions must fit the admitted Fashion-MNIST base matrix.");
        Require(options.TopK <= options.LiveVectorCount, "top-k must not exceed the post-update live vector count.");
        Require(options.WarmupQueries == 0 || options.WarmupQueries <= dataset.QueryMatrixCount, "warmup query count must not exceed admitted query matrix count.");
        Require(dataset.Manifest.DatasetId == FashionMnistDatasetSpecification.GetDatasetId(options.Metric), "External dataset must be the admitted Fashion-MNIST dataset for the selected metric.");
        Require(dataset.Dimension == dataset.Manifest.Shape.Dimension, "Loaded matrix dimension must match admitted manifest dimension.");
        return dataset;
    }

    internal static PreparedCheckpointState PrepareCheckpointState(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset)
    {
        BuildMeasurement build = BuildBaseIndex(options, dataset);
        var composite = new HnswBasePlusExactDeltaIndex(build.Index);
        long generationBeforeMutations = composite.Generation;
        MutationExecution mutationExecution = ExecuteMutations(options, dataset, composite);
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = CreateCountInfo(options, composite);
        ulong[] liveIds = BuildLiveIds(options);
        TruthSet truth = GenerateLiveTruth(dataset, options, liveIds);

        return new PreparedCheckpointState(build, composite, generationBeforeMutations, mutationExecution, preCounts, liveIds, truth);
    }

    internal static BuildMeasurement BuildBaseIndex(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var index = new HnswIndex(dataset.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetBaseVector(row));
        }

        return new BuildMeasurement(
            index,
            StopwatchTicksToMilliseconds(Stopwatch.GetTimestamp() - start),
            GC.GetAllocatedBytesForCurrentThread() - allocationStart);
    }

    internal static HnswBuildInfo CreateBuildInfo(
        BuildMeasurement build,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset) =>
        new(
            "measured",
            build.ElapsedMilliseconds,
            new MeasurementStatusInfo(
                "measured",
                build.ManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                "bytes",
                "Measured with GC.GetAllocatedBytesForCurrentThread around immutable HnswIndex construction and Add calls for selected Fashion-MNIST base rows only; cache checks, matrix/truth loading, composite construction, mutation application and exact updated truth generation are excluded."),
            options.BaseVectorCount,
            dataset.Dimension,
            "internal HnswIndex construction and selected admitted base-vector Add calls",
            "cache checks, checksum validation, matrix load, truth load, composite construction, update application, exact updated truth generation, warmup, measured search, result comparison and report writing");

    internal static MutationExecution ExecuteMutations(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var counts = new MutableMutationStatusCounts();
        int inserted = 0;
        int deletedBase = 0;
        int deletedDelta = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            VectorMutationResult result = composite.TryAdd((ulong)(options.BaseVectorCount + i), dataset.GetBaseVector(options.BaseVectorCount + i));
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
            counts.Add(composite.TryAdd(id, dataset.GetBaseVector(options.BaseVectorCount + (i % options.InsertedDeltaCount))).Status);
        }

        ulong firstUnknownId = (ulong)dataset.BaseCount;
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

    internal static string CreateCheckpointRunDirectory(string checkpointRootDirectory, int runNumber)
    {
        string directory = Path.Combine(
            checkpointRootDirectory,
            string.Create(CultureInfo.InvariantCulture, $"checkpoint-run-{runNumber:000}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    internal static MeasuredCheckpointRun MeasureCheckpointRun(
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

    internal static HnswBasePlusExactDeltaCheckpointRunInfo CreateCheckpointRunInfo(
        int runNumber,
        string checkpointDirectory,
        MeasuredCheckpointRun measured)
    {
        HnswBasePlusExactDeltaCheckpointResult result = measured.Diagnostic.Result;
        return new HnswBasePlusExactDeltaCheckpointRunInfo(
            runNumber,
            checkpointDirectory,
            result.Status.ToString(),
            StopwatchTicksToMilliseconds(measured.ElapsedTicks),
            measured.ManagedAllocatedBytes,
            measured.GenerationBeforeCheckpoint,
            result.Generation,
            result.Generation == measured.GenerationBeforeCheckpoint + 1,
            CreatePhaseSet(measured.Diagnostic.Diagnostics));
    }

    internal static HnswBasePlusExactDeltaCheckpointRunsInfo CreateCheckpointRunsInfo(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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
            "Aggregate checkpoint timing/allocation is computed across independently rebuilt equivalent external Fashion-MNIST checkpoint attempts. Cache checks, matrix/truth loading, state creation, mutation application, exact truth construction, search measurements, output-byte scan, NoChanges probe, validation and report writing are excluded.");

        return new HnswBasePlusExactDeltaCheckpointRunsInfo(
            runs.Length,
            options.Runs,
            "Detailed validation, output inspection, opened-output validation, NoChanges probe, deleted-ID reservation probe and post/opened search parity use the final checkpoint run.",
            runs,
            aggregate);
    }

    internal static void WarmupCompositeSearch(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
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
            composite.Search(dataset.GetQueryVector(i % dataset.QueryMatrixCount), results, workspace);
        }
    }

    internal static void WarmupOpenedSearch(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswIndex opened)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.LiveVectorCount, options.EfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            opened.Search(dataset.GetQueryVector(i % dataset.QueryMatrixCount), results, workspace);
        }
    }

    internal static SearchMeasurement MeasureCompositeSearch(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool capture = captureResults && runIndex == options.Runs - 1;
            HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(options);
            SingleRunMeasurement run = MeasureQueries(
                options,
                dataset,
                capture,
                (query, destination) => composite.Search(query, destination, workspace));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (capture)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    internal static SearchMeasurement MeasureOpenedSearch(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswIndex opened,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            var workspace = new HnswSearchWorkspace(options.LiveVectorCount, options.EfSearch);
            bool capture = captureResults && runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureQueries(
                options,
                dataset,
                capture,
                (query, destination) => opened.Search(query, destination, workspace));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (capture)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureQueries(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
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
            ReadOnlySpan<float> query = dataset.GetQueryVector(queryRow);
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

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeWorkspace(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
    {
        int maxBaseElements = Math.Max(options.BaseVectorCount, options.LiveVectorCount);
        return new HnswBasePlusExactDeltaSearchWorkspace(
            maxBaseElements,
            options.EfSearch,
            Math.Min(maxBaseElements, options.EfSearch),
            options.TopK);
    }

    internal static HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo ProbeNoChanges(
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
        HnswBasePlusExactDeltaCheckpointPhaseSetInfo phases = CreatePhaseSet(diagnostic.Diagnostics);
        bool passed = diagnostic.Result.Status == HnswBasePlusExactDeltaCheckpointStatus.NoChanges &&
            composite.Generation == generationBefore &&
            outputEmpty &&
            !AnyExecuted(phases);

        return new HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo(
            passed ? "passed" : "failed",
            generationBefore,
            composite.Generation,
            composite.Generation == generationBefore,
            outputEmpty,
            phases);
    }

    internal static bool ValidateDeletedReservation(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        if (options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            return false;
        }

        ulong deletedId = options.DeletedBaseCount > 0 ? 0UL : (ulong)options.BaseVectorCount;
        VectorMutationResult result = composite.TryAdd(deletedId, dataset.GetBaseVector(options.BaseVectorCount));
        return result.Status == VectorMutationStatus.DuplicateId;
    }

    internal static HnswBasePlusExactDeltaCheckpointOpenedValidationInfo ValidateOpenedOutput(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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

            ReadOnlySpan<float> expectedVector = dataset.GetBaseVector(checked((int)expectedId));
            ReadOnlySpan<float> openedVector = openedVectors.Slice(row * dataset.Dimension, dataset.Dimension);
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
            "Opened read-only HNSW checkpoint output must contain live IDs in checkpoint live-view order and vector payloads matching admitted Fashion-MNIST live rows under the selected metric storage policy; cosine payloads are unit-normalized by HNSW storage. Search parity is validated separately for the same queries and equivalent workspaces.");
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

    internal static HnswBasePlusExactDeltaCheckpointParityInfo CompareSearchParity(
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
            "Post-checkpoint rebuilt composite Search and opened read-only HNSW Search are executed for the same external queries with fresh caller-owned workspaces and must return the same count, IDs, order and distances within D-026 tolerance.");
    }

    internal static HnswBasePlusExactDeltaCheckpointOutputInfo InspectCheckpointOutput(string directory, int vectorCount)
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

    internal static SearchSectionEvaluation EvaluateSearchSection(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        TruthSet truth,
        SearchMeasurement search,
        ulong[] liveIds)
    {
        ResultComparison comparison = ResultComparer.Compare(
            truth,
            search.Results,
            options.TopK,
            dataset.Dimension,
            options.Metric);
        HnswBasePlusExactDeltaReturnedResultIntegrityInfo integrity =
            ValidateReturnedResults(dataset, search.Results, options, liveIds);
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
                "Every returned result is checked for finite distance, no duplicate ID within its query, Fashion-MNIST live ID membership, no tombstoned ID, and selected-metric distance matching recomputation for that returned ID/query within the accepted ResultComparer tolerance. HNSW search is approximate and recall/order are recorded, not required."),
            CreateUnderfill(options, search.Results));
    }

    internal static HnswBasePlusExactDeltaCheckpointSearchSectionInfo CreateSearchSection(
        string name,
        string timedOperation,
        SearchMeasurement measurement,
        SearchSectionEvaluation evaluation,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options) =>
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
            evaluation.Underfill);

    private static MeasurementInfo CreateSearchMeasurement(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        SearchRunInfo[] runs,
        string timedOperation) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredSearchCall",
                timedOperation,
                "cache checks, matrix/truth load, HNSW base build, mutation application, exact truth construction, checkpoint call, output-byte scan, warmup, final result comparison, validation and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Search aggregate percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                runs.Average(run => run.ManagedAllocatedBytesPerQuery).ToString(CultureInfo.InvariantCulture),
                "bytesPerSearchCall",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each Search(query, results, workspace) call using caller-owned SearchResult[] and workspace; setup, checkpoint, warmup, validation and report writing are excluded."),
            NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-138."),
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

    internal static HnswBasePlusExactDeltaCheckpointCountInfo CreateCountInfo(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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
            "Before checkpoint, base and delta physical rows include tombstoned rows. After a published checkpoint, live Fashion-MNIST rows are folded into a rebuilt immutable HNSW base, delta rows and tombstones are cleared, and deleted/reserved IDs remain retained in the writable composite instance.");
    }

    internal static HnswBasePlusExactDeltaCheckpointMutationInfo CreateMutationInfo(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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

    internal static HnswBasePlusExactDeltaCheckpointResultInfo CreateCheckpointResultInfo(
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

    internal static HnswBasePlusExactDeltaCheckpointPhaseSetInfo CreatePhaseSet(
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

    internal static bool AllMeasured(HnswBasePlusExactDeltaCheckpointPhaseSetInfo phases) =>
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

    internal static bool CheckpointRepeatedRunEvidencePresent(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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

    internal static bool MutationStatusCountsMatch(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        MutationExecution mutationExecution) =>
        mutationExecution.StatusCounts.Committed ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount &&
        mutationExecution.StatusCounts.DuplicateId == options.DuplicateInsertAttempts &&
        mutationExecution.StatusCounts.UnknownId == options.UnknownDeleteAttempts &&
        mutationExecution.StatusCounts.AlreadyDeleted == options.RepeatedDeleteAttempts &&
        mutationExecution.StatusCounts.ReadOnly == 0 &&
        mutationExecution.StatusCounts.Unsupported == 0;

    internal static bool CheckpointCountsMatch(
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

    internal static bool PostCountsMatch(
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
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
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
            $"Across measured {operationName} runs; warmup, cache checks, setup, checkpoint, validation and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured search runs executed; simple descriptive run-to-run statistics are recorded for private local external checkpoint smoke inspection."
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

    private static ExternalBenchmarkEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private external Fashion-MNIST HNSW base-plus-exact-delta checkpoint smoke output is not reviewed public evidence.",
            "No external mutable/update HNSW checkpoint baseline-candidate policy is accepted.",
            "No external mutable/update HNSW checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "External Fashion-MNIST HNSW base-plus-exact-delta checkpoint smoke evidence only; no external matrix or comparison claim applies.",
                "Cache checks, checksum validation, matrix/truth loading, immutable HNSW build, update application, exact updated truth generation, warmup, checkpoint, final-run result capture/comparison and report writing are separated by measured section.",
                "Checkpoint total timing/allocation and VEC-133 phase diagnostics are reported separately from all search timings.",
                "Managed allocations are smoke fields only; resident/process/GC/peak memory is explicitly not measured.",
                "Not eligible for public performance, recall, memory, allocation, mutable-HNSW, baseline, regression-gate, external-dataset matrix, comparison or concurrency claims."
            ]);

    private static ExternalBenchmarkEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "External Fashion-MNIST HNSW base-plus-exact-delta checkpoint reports are private local evidence only until a reviewed public summary policy and public mutable-HNSW admission exist.",
            "No external mutable/update HNSW checkpoint baseline-candidate policy is accepted.",
            "No external mutable/update HNSW checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

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

    internal static void ValidateOptions(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.CacheRoot))
        {
            throw new ArgumentException("Cache root must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.CheckpointDirectory))
        {
            throw new ArgumentException("Checkpoint directory must not be empty.", nameof(options));
        }

        if (options.QueryCount <= 0 || options.TopK <= 0 || options.BaseVectorCount <= 0 || options.InsertedDeltaCount <= 0)
        {
            throw new ArgumentException("query count, top-k, base vector count and inserted delta count must be positive.", nameof(options));
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

        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine))
        {
            throw new ArgumentException("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint supports only SquaredEuclidean, InnerProduct and Cosine.", nameof(options));
        }

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.RepeatedDeleteAttempts > 0 && options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            throw new ArgumentException("repeated delete attempts require at least one committed delete.", nameof(options));
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

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidDataException(message);
        }
    }

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static double StopwatchTicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static string CreateReportId(string? commit, FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName}-{commitPart}-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private delegate int SearchOperation(ReadOnlySpan<float> query, Span<SearchResult> results);

    internal sealed record BuildMeasurement(HnswIndex Index, double ElapsedMilliseconds, long ManagedAllocatedBytes);

    internal sealed record MutationExecution(
        int InsertedCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        long GenerationAfterMutations,
        GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

    internal sealed record PreparedCheckpointState(
        BuildMeasurement Build,
        HnswBasePlusExactDeltaIndex Composite,
        long GenerationBeforeMutations,
        MutationExecution MutationExecution,
        HnswBasePlusExactDeltaCheckpointCountInfo PreCounts,
        ulong[] LiveIds,
        TruthSet Truth);

    internal sealed record MeasuredCheckpointRun(
        string DirectoryPath,
        long GenerationBeforeCheckpoint,
        HnswBasePlusExactDeltaCheckpointDiagnosticResult Diagnostic,
        long ElapsedTicks,
        long ManagedAllocatedBytes);

    internal sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    internal sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);

    internal sealed record SearchSectionEvaluation(
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
