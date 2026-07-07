using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario
{
    private const string TaskId = "VEC-142";
    private const string SchemaName = "VecNet.ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "hnsw.manifest.json";
    private const string IdsFileName = "hnsw.ids.u64";
    private const string VectorsFileName = "hnsw.vectors.f32";
    private const string LevelsFileName = "hnsw.levels.i32";
    private const string GraphFileName = "hnsw.graph.bin";

    public static ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport Run(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions checkpointOptions = options.ToCheckpointOptions();

        MemorySnapshot baseline = CaptureMemorySnapshot();
        HnswMemorySampleInfo baselineSample = CreateSample(
            "baselineProcess",
            "Runtime after runner startup and before Fashion-MNIST cache/truth load where practical.",
            baseline,
            baseline);

        TimedPhase<FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset> loadPhase = SamplePhase(
            "cacheTruthLoad",
            options.SampleIntervalMilliseconds,
            baseline,
            "admitted Fashion-MNIST cache/truth manifest, checksum and matrix load",
            "HNSW build, mutation, exact updated truth generation, checkpoint, search, output-byte scan, validation and report writing",
            () => FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.LoadAndValidateDataset(checkpointOptions));
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = loadPhase.Value;

        string checkpointRootDirectory = Path.GetFullPath(options.CheckpointDirectory);
        Directory.CreateDirectory(checkpointRootDirectory);

        TimedPhase<FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildMeasurement> buildPhase = SamplePhase(
            "immutableHnswBaseBuild",
            options.SampleIntervalMilliseconds,
            baseline,
            "immutable HNSW base construction from selected Fashion-MNIST base rows",
            "cache/truth load, exact delta/tombstone mutation, exact updated truth generation, checkpoint, search, validation and report writing",
            () => FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildBaseIndex(checkpointOptions, dataset));
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildMeasurement build = buildPhase.Value;

        HnswBasePlusExactDeltaIndex composite = null!;
        long generationBeforeMutations = 0;
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MutationExecution mutationExecution = null!;
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = null!;
        TimedPhase<object?> compositeMutationPhase = SamplePhase<object?>(
            "compositeCreationAndExactDeltaTombstoneMutation",
            options.SampleIntervalMilliseconds,
            baseline,
            "HnswBasePlusExactDeltaIndex creation plus exact delta inserts, base tombstones, delta tombstones and mutation probes",
            "cache/truth load, immutable HNSW base build, exact updated truth generation, checkpoint, search, validation and report writing",
            () =>
            {
                composite = new HnswBasePlusExactDeltaIndex(build.Index);
                generationBeforeMutations = composite.Generation;
                mutationExecution = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ExecuteMutations(checkpointOptions, dataset, composite);
                preCounts = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCountInfo(checkpointOptions, composite);
                return null;
            });

        ulong[] liveIds = [];
        TruthSet truth = null!;
        TimedPhase<object?> truthPhase = SamplePhase<object?>(
            "exactUpdatedTruthGeneration",
            options.SampleIntervalMilliseconds,
            baseline,
            "scalar exact updated truth generation from the post-update live view",
            "cache/truth load, immutable HNSW base build, checkpoint, search, validation and report writing",
            () =>
            {
                liveIds = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildLiveIds(checkpointOptions);
                truth = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.GenerateLiveTruth(dataset, checkpointOptions, liveIds);
                return null;
            });

        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchMeasurement preSearch = null!;
        TimedPhase<object?> preSearchPhase = SamplePhase<object?>(
            "preCheckpointSourceCompositeSearch",
            options.SampleIntervalMilliseconds,
            baseline,
            "warmup plus measured pre-checkpoint source composite search",
            "cache/truth load, HNSW base build, mutation, exact truth generation, checkpoint, post-checkpoint searches, validation and report writing",
            () =>
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.WarmupCompositeSearch(checkpointOptions, dataset, composite);
                preSearch = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MeasureCompositeSearch(checkpointOptions, dataset, composite, captureResults: true);
                return null;
            });

        string runDirectory = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCheckpointRunDirectory(checkpointRootDirectory, 1);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MeasuredCheckpointRun measuredCheckpoint = null!;
        TimedPhase<object?> checkpointPhase = SamplePhase<object?>(
            "checkpointPublication",
            options.SampleIntervalMilliseconds,
            baseline,
            "single internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics publication run",
            "cache/truth load, base build, mutation, exact truth generation, search measurements, output-byte scan, NoChanges probe, validation and report writing",
            () =>
            {
                measuredCheckpoint = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MeasureCheckpointRun(composite, runDirectory);
                return null;
            });

        HnswBasePlusExactDeltaCheckpointResult checkpointResult = measuredCheckpoint.Diagnostic.Result;
        HnswBasePlusExactDeltaCheckpointPhaseSetInfo phaseDiagnostics =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreatePhaseSet(measuredCheckpoint.Diagnostic.Diagnostics);
        HnswBasePlusExactDeltaCheckpointRunInfo checkpointRun =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCheckpointRunInfo(1, runDirectory, measuredCheckpoint);
        HnswBasePlusExactDeltaCheckpointRunsInfo checkpointRuns =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCheckpointRunsInfo(checkpointOptions, [checkpointRun]);
        HnswBasePlusExactDeltaCheckpointCountInfo postCounts =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCountInfo(checkpointOptions, composite);

        HnswIndex opened = null!;
        TimedPhase<object?> openPhase = SamplePhase<object?>(
            "openedReadOnlyHnswOpen",
            options.SampleIntervalMilliseconds,
            baseline,
            "explicit HnswIndex.OpenReadOnly over the published checkpoint output",
            "cache/truth load, base build, mutation, exact truth generation, checkpoint publication, search, output-byte scan, validation and report writing",
            () =>
            {
                opened = HnswIndex.OpenReadOnly(runDirectory);
                return null;
            });

        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchMeasurement postSearch = null!;
        TimedPhase<object?> postSearchPhase = SamplePhase<object?>(
            "postCheckpointRebuiltCompositeSearch",
            options.SampleIntervalMilliseconds,
            baseline,
            "warmup plus measured post-checkpoint rebuilt composite search",
            "cache/truth load, base build, mutation, exact truth generation, checkpoint, opened HNSW search, validation and report writing",
            () =>
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.WarmupCompositeSearch(checkpointOptions, dataset, composite);
                postSearch = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MeasureCompositeSearch(checkpointOptions, dataset, composite, captureResults: true);
                return null;
            });

        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchMeasurement openedSearch = null!;
        TimedPhase<object?> openedSearchPhase = SamplePhase<object?>(
            "openedReadOnlyHnswSearch",
            options.SampleIntervalMilliseconds,
            baseline,
            "warmup plus measured opened read-only HNSW search",
            "cache/truth load, base build, mutation, exact truth generation, checkpoint, rebuilt composite search, validation and report writing",
            () =>
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.WarmupOpenedSearch(checkpointOptions, dataset, opened);
                openedSearch = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MeasureOpenedSearch(checkpointOptions, dataset, opened, captureResults: true);
                return null;
            });

        HnswBasePlusExactDeltaCheckpointOutputInfo output = null!;
        HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo noChangesProbe = null!;
        bool deletedReservedRejected = false;
        HnswBasePlusExactDeltaCheckpointOpenedValidationInfo openedValidation = null!;
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchSectionEvaluation preEvaluation = null!;
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchSectionEvaluation postEvaluation = null!;
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.SearchSectionEvaluation openedEvaluation = null!;
        HnswBasePlusExactDeltaCheckpointParityInfo parity = null!;
        TimedPhase<object?> finalValidationPhase = SamplePhase<object?>(
            "finalValidation",
            options.SampleIntervalMilliseconds,
            baseline,
            "checkpoint output-byte scan, opened payload validation, rebuilt/opened parity, search/truth evaluation, NoChanges probe and deleted-ID reservation validation",
            "cache/truth load, base build, mutation, exact truth generation, checkpoint publication and measured search calls",
            () =>
            {
                output = InspectCheckpointOutput(runDirectory, checkpointResult.LiveVectorCount, validationOpenStatus: "passed");
                parity = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CompareSearchParity(postSearch.Results, openedSearch.Results, dataset.Dimension);
                openedValidation = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ValidateOpenedOutput(dataset, checkpointOptions, liveIds, opened, parity);
                noChangesProbe = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ProbeNoChanges(runDirectory, composite);
                deletedReservedRejected = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ValidateDeletedReservation(checkpointOptions, dataset, composite);
                preEvaluation = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.EvaluateSearchSection(checkpointOptions, dataset, truth, preSearch, liveIds);
                postEvaluation = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.EvaluateSearchSection(checkpointOptions, dataset, truth, postSearch, liveIds);
                openedEvaluation = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.EvaluateSearchSection(checkpointOptions, dataset, truth, openedSearch, liveIds);
                return null;
            });

        bool mutationStatusCountsMatched = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.MutationStatusCountsMatch(checkpointOptions, mutationExecution);
        bool mutationGenerationMatched =
            mutationExecution.GenerationAfterMutations - generationBeforeMutations ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount;
        bool checkpointCountsMatched = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CheckpointCountsMatch(preCounts, checkpointResult);
        bool postCountsMatched = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.PostCountsMatch(preCounts, postCounts, checkpointResult);
        bool checkpointGenerationAdvanced = checkpointResult.Generation == measuredCheckpoint.GenerationBeforeCheckpoint + 1 &&
            composite.Generation == measuredCheckpoint.GenerationBeforeCheckpoint + 1;
        bool publishedPhasesMeasured = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.AllMeasured(phaseDiagnostics);
        bool checkpointRunCountIsOne = checkpointRuns.RunCount == 1 &&
            checkpointRuns.Runs.Length == 1 &&
            checkpointRuns.Aggregate.RunCount == 1 &&
            checkpointRuns.DetailedValidationRunNumber == 1;
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
            checkpointRunCountIsOne &&
            postCountsMatched &&
            openedValidation.Status == "passed" &&
            parity.AllResultsMatched &&
            allIntegrityPassed &&
            noChangesProbe.Status == "passed" &&
            deletedReservedRejected &&
            output.ValidationOpenStatus == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        ExternalHnswCheckpointMemorySmokeActualMemoryInfo actualMemory = CreateActualMemory(
            baselineSample,
            loadPhase.EndSample,
            buildPhase.EndSample,
            compositeMutationPhase.EndSample,
            truthPhase.EndSample,
            preSearchPhase.EndSample,
            checkpointPhase.EndSample,
            openPhase.EndSample,
            postSearchPhase.EndSample,
            openedSearchPhase.EndSample,
            finalValidationPhase.EndSample);
        ExternalHnswCheckpointMemorySmokePeakMemoryInfo peakMemory = CreatePeakMemory(
            options,
            loadPhase.Peak,
            buildPhase.Peak,
            compositeMutationPhase.Peak,
            truthPhase.Peak,
            preSearchPhase.Peak,
            checkpointPhase.Peak,
            openPhase.Peak,
            postSearchPhase.Peak,
            openedSearchPhase.Peak,
            finalValidationPhase.Peak);
        ExternalHnswCheckpointMemorySmokeLayoutLowerBoundsInfo lowerBounds =
            CreateLowerBounds(options, build.Index, opened, preCounts, postCounts);
        ExternalHnswCheckpointMemorySmokeStorageOutputInfo storageOutput = CreateStorageOutput(output);

        return new ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, commandArguments.ToArray()),
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
            CreateWorkload(options, dataset),
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
                "scalar-reference-external-live-hnsw-base-plus-exact-delta-checkpoint-memory-smoke",
                "computed in memory during the scenario from the post-update live view: selected immutable base rows plus committed delta rows minus base and delta tombstones",
                Persisted: false,
                options.QueryCount,
                truth.Depth,
                liveIds.Length,
                "ascending scalar-reference squared-L2 distance, then ascending external ID for exact equal distances",
                "VecNet canonical squared-L2 over admitted converted Fashion-MNIST float32 vectors",
                "existing admitted truth artifact validates cache/truth readiness only and is not final updated truth",
                "live candidate IDs are selected immutable base rows and committed exact delta rows after tombstone suppression"),
            new ScenarioInfo(
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "This mode executes the accepted VEC-138 Fashion-MNIST checkpoint shape once and records actual whole-process boundary samples, observed sampled peaks, payload-only lower bounds and checkpoint output file facts in separate sections."),
            new IndexInfo(
                "InternalExternalHnswBasePlusExactDeltaCheckpointMemorySmoke",
                nameof(HnswBasePlusExactDeltaIndex),
                VectorMetric.SquaredEuclidean.ToString(),
                dataset.Dimension,
                postCounts.LiveVectorCount,
                "internal HnswBasePlusExactDeltaIndex checkpoint/rebuild memory smoke over admitted Fashion-MNIST cache; no public mutable/update HNSW API, memory/capacity claim, package change or regression gate"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "admitted Fashion-MNIST base matrix row order, immutable base rows first, original row ordinals as external IDs",
                "SquaredEuclidean only"),
            new ExternalHnswCheckpointMemorySmokeMeasuredPhasesInfo(
                loadPhase.Phase,
                buildPhase.Phase,
                compositeMutationPhase.Phase,
                truthPhase.Phase,
                preSearchPhase.Phase,
                checkpointPhase.Phase,
                phaseDiagnostics,
                openPhase.Phase,
                postSearchPhase.Phase,
                openedSearchPhase.Phase,
                finalValidationPhase.Phase),
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateBuildInfo(build, checkpointOptions, dataset),
            preCounts,
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateMutationInfo(checkpointOptions, mutationExecution, generationBeforeMutations, mutationStatusCountsMatched, mutationGenerationMatched),
            checkpointRuns,
            new HnswBasePlusExactDeltaCheckpointOperationInfo(
                checkpointResult.Status.ToString(),
                "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                StopwatchTicksToMilliseconds(measuredCheckpoint.ElapsedTicks),
                measuredCheckpoint.ManagedAllocatedBytes,
                measuredCheckpoint.GenerationBeforeCheckpoint,
                checkpointResult.Generation,
                checkpointGenerationAdvanced,
                phaseDiagnostics,
                "cache checks, matrix/truth load, immutable HNSW base build, update application, exact updated truth construction, search timing, no-changes probe, output-byte scan, opened-output validation and report writing"),
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateCheckpointResultInfo(checkpointResult),
            postCounts,
            noChangesProbe,
            output,
            openedValidation,
            new ExternalHnswBasePlusExactDeltaCheckpointSearchSectionsInfo(
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateSearchSection("preCheckpointSourceComposite", "internal pre-checkpoint external HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", preSearch, preEvaluation, checkpointOptions),
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateSearchSection("postCheckpointRebuiltComposite", "internal post-checkpoint rebuilt external HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", postSearch, postEvaluation, checkpointOptions),
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.CreateSearchSection("openedReadOnlyHnsw", "internal opened read-only external HnswIndex.Search(query, results, workspace)", openedSearch, openedEvaluation, checkpointOptions)),
            actualMemory,
            peakMemory,
            lowerBounds,
            storageOutput,
            new ExternalHnswCheckpointMemorySmokeMeasurementInfo(
                new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perSingleCheckpointPublicationCall",
                    "internal HnswBasePlusExactDeltaIndex.CheckpointWithDiagnostics(directoryPath)",
                    "cache checks, matrix/truth load, HNSW base build, mutation application, exact truth construction, warmup, all search measurements, output-byte scan, no-changes probe, validation and report writing",
                    "single Stopwatch sample around the internal checkpoint call boundary; VEC-133 phase elapsed values are reported separately",
                    "This memory smoke intentionally executes one checkpoint publication run by default and does not analyze repeated-run variance.",
                    "Raw checkpoint elapsed milliseconds are emitted in checkpointRuns.runs[0] and checkpoint.elapsedMilliseconds."),
                new MeasurementStatusInfo(
                    "measured",
                    measuredCheckpoint.ManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                    "bytesPerCheckpointCall",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around the runner call to internal CheckpointWithDiagnostics; VEC-133 phase allocation values are reported separately and not inferred."),
                phaseDiagnostics,
                new MeasurementStatusInfo(
                    "measured",
                    "actualMemory",
                    "wholeProcessBoundarySamples",
                    "Actual whole-process/process and GC-observed boundary samples are reported under actualMemory."),
                new MeasurementStatusInfo(
                    "sampled",
                    "peakMemory",
                    "observedSampledWholeProcessPeaks",
                    string.Create(CultureInfo.InvariantCulture, $"Observed sampled peaks are reported under peakMemory with a {options.SampleIntervalMilliseconds} ms sampling interval and missed-short-peak caveats.")),
                new MeasurementStatusInfo(
                    "estimatedLowerBound",
                    "layoutLowerBounds",
                    "payloadOnlyBytes",
                    "Payload-only lower-bound estimates are reported under layoutLowerBounds and are not actual retained memory."),
                new MeasurementStatusInfo(
                    "fileFacts",
                    output.TotalBytes.ToString(CultureInfo.InvariantCulture),
                    "bytes",
                    "Checkpoint output bytes are final file facts scanned after checkpoint publication and outside checkpoint timing; they are not memory."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries execute before measured search sections and are excluded from search and checkpoint timing."
                        : "No warmup queries were requested."),
                "Memory boundary samples include runner/runtime state and coexisting retained objects. Checkpoint timing, search timing, actual samples, sampled peaks, layout lower bounds and checkpoint output bytes are reported as separate evidence categories."),
            new ExternalHnswCheckpointMemorySmokeValidationInfo(
                validationPassed ? "passed" : "failed",
                "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke",
                CacheAndTruthReadinessPassed: true,
                ExistingTruthGuardLoaded: true,
                UpdatedTruthGeneratedFromLiveView: true,
                PreCheckpointSourceCompositeComparedToTruth: true,
                CheckpointResultStatusPublished: checkpointResult.Status == HnswBasePlusExactDeltaCheckpointStatus.Published,
                CheckpointResultCountsMatched: checkpointCountsMatched,
                CheckpointGenerationAdvancedExactlyOnce: checkpointGenerationAdvanced,
                PhaseDiagnosticsMeasuredForPublishedCheckpoint: publishedPhasesMeasured,
                CheckpointRunCountIsOne: checkpointRunCountIsOne,
                PostCheckpointCountsMatched: postCountsMatched,
                PostCheckpointRebuiltCompositeComparedToTruth: true,
                OpenedReadOnlyHnswOpened: opened.Count == liveIds.Length,
                OpenedReadOnlyHnswIdVectorValidationPassed: openedValidation.Status == "passed",
                OpenedReadOnlyHnswComparedToTruth: true,
                RebuiltCompositeOpenedHnswSearchParityPassed: parity.AllResultsMatched,
                ReturnedResultIntegrityPassedForAllSearches: allIntegrityPassed,
                NoChangesCheckpointProbePassed: noChangesProbe.Status == "passed",
                DeletedReservedIdsRejectedAfterCheckpoint: deletedReservedRejected,
                ActualPeakLowerBoundAndStorageSectionsSeparated: true,
                OutputBytesAreSeparateFileFacts: true,
                UnsupportedFieldsExplicitlyMarked: true,
                WorkingSetContextOnly: true,
                SampledPeakLabelsPresent: true,
                OutputBytesScannedOutsideCheckpointDuration: output.ScanTimingScope == "outsideCheckpointDuration",
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private external Fashion-MNIST HNSW base-plus-exact-delta checkpoint memory smoke evidence only; not a public memory, capacity, storage-size, recall, latency, allocation, package, platform, preview-readiness, baseline, comparison or regression claim.",
                "The workload is the accepted VEC-138 Fashion-MNIST checkpoint smoke shape with exactly one checkpoint publication run for VEC-142.",
                "Actual memory samples are local whole-process boundary samples and include Fashion-MNIST input arrays, runner objects, runtime state, source composite state and opened HNSW state where retained.",
                "Observed sampled peaks can miss short-lived peaks between samples and are not true maxima.",
                "Working set and process peak working set are OS/cache-sensitive context only.",
                "layoutLowerBounds contains payload-only lower-bound estimates and excludes runtime/object/header/slack/fragmentation/input-array costs.",
                "storageOutput contains checkpoint output file facts only and is separate from actualMemory, peakMemory and layoutLowerBounds.",
                "Public claim, preview-readiness, baseline-candidate, comparison-artifact and regression-gate eligibility are false."
            ]);
    }

    public static void Write(ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static ExternalHnswCheckpointMemorySmokeWorkloadInfo CreateWorkload(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset) =>
        new(
            dataset.Manifest.DatasetId,
            dataset.BaseCount,
            dataset.QueryMatrixCount,
            options.QueryCount,
            options.TopK,
            dataset.Dimension,
            VectorMetric.SquaredEuclidean.ToString(),
            ImmutableBaseStartRow: 0,
            options.BaseVectorCount - 1,
            options.BaseVectorCount,
            DeltaStartRow: options.BaseVectorCount,
            options.PhysicalCandidateVectorCount - 1,
            options.InsertedDeltaCount,
            UnusedStartRow: options.PhysicalCandidateVectorCount,
            dataset.BaseCount - 1,
            dataset.BaseCount - options.PhysicalCandidateVectorCount,
            options.DeletedBaseCount,
            options.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            options.LiveVectorCount,
            options.DeletedReservedIdCount,
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.CheckpointRunCount,
            options.WarmupQueries,
            options.SampleIntervalMilliseconds,
            "contiguous admitted base-matrix rows: immutable base rows 0..57999, exact delta rows 58000..58999, remaining admitted rows 59000..59999 unused for the default accepted smoke",
            "candidate external IDs are original Fashion-MNIST base row ordinals; deleted IDs remain reserved inside the writable composite after checkpoint publication",
            "build immutable HNSW base, add exact delta rows, delete base rows from the start, delete delta rows from the start, attempt duplicate/reserved insert, unknown delete and repeated delete, then checkpoint/rebuild",
            FormatHex(options.Seed),
            FormatHex(options.HnswSeed),
            "Actual samples, observed sampled peaks, payload-only layout lower bounds and checkpoint output bytes are separate sections.");

    private static HnswBasePlusExactDeltaCheckpointOutputInfo InspectCheckpointOutput(
        string directory,
        int vectorCount,
        string validationOpenStatus)
    {
        long manifestBytes = FileLength(directory, ManifestFileName);
        long idsBytes = FileLength(directory, IdsFileName);
        long vectorsBytes = FileLength(directory, VectorsFileName);
        long levelsBytes = FileLength(directory, LevelsFileName);
        long graphBytes = FileLength(directory, GraphFileName);
        long totalBytes = manifestBytes + idsBytes + vectorsBytes + levelsBytes + graphBytes;
        return new HnswBasePlusExactDeltaCheckpointOutputInfo(
            "recorded",
            directory,
            Directory.EnumerateFiles(directory).Count(),
            totalBytes,
            manifestBytes,
            idsBytes,
            vectorsBytes,
            levelsBytes,
            graphBytes,
            vectorCount,
            vectorCount == 0 ? 0 : (double)totalBytes / vectorCount,
            validationOpenStatus,
            "outsideCheckpointDuration");
    }

    private static ExternalHnswCheckpointMemorySmokeStorageOutputInfo CreateStorageOutput(
        HnswBasePlusExactDeltaCheckpointOutputInfo output) =>
        new(
            "fileFacts",
            "Final checkpoint output file lengths scanned after successful checkpoint publication and outside checkpoint timing.",
            "private ignored benchmark-runner artifact path",
            output.DirectoryPath,
            output.FileCount,
            output.TotalBytes,
            output.ManifestBytes,
            output.IdsBytes,
            output.VectorsBytes,
            output.LevelsBytes,
            output.GraphBytes,
            output.OutputVectorCount,
            output.BytesPerLiveVector,
            output.ScanTimingScope,
            NotMeasured("bytes", "Active output-directory byte sampling during checkpoint publication is not implemented; final output bytes are scanned after publication only."),
            NotMeasured("bytes", "Peak temporary disk bytes are not measured and are not inferred from final checkpoint output bytes."),
            "Checkpoint output bytes are persisted file facts, not memory; they are not actual memory, sampled peak memory or layout lower-bound memory.");

    private static ExternalHnswCheckpointMemorySmokeActualMemoryInfo CreateActualMemory(
        HnswMemorySampleInfo baseline,
        HnswMemorySampleInfo postLoad,
        HnswMemorySampleInfo postBuild,
        HnswMemorySampleInfo postMutation,
        HnswMemorySampleInfo postTruth,
        HnswMemorySampleInfo postPreSearch,
        HnswMemorySampleInfo postCheckpoint,
        HnswMemorySampleInfo postOpen,
        HnswMemorySampleInfo postPostSearch,
        HnswMemorySampleInfo postOpenedSearch,
        HnswMemorySampleInfo postValidation) =>
        new(
            "measured",
            "wholeProcessBoundarySamples",
            "Samples use GC.GetGCMemoryInfo and Process.GetCurrentProcess after retained-state boundaries. No forced full-GC stabilization is applied.",
            "Actual whole-process/process and GC-observed boundary samples are separate from observed sampled peaks, payload lower-bound estimates and checkpoint output file facts.",
            baseline,
            postLoad,
            postBuild,
            postMutation,
            postTruth,
            postPreSearch,
            postCheckpoint,
            postOpen,
            postPostSearch,
            postOpenedSearch,
            postValidation,
            new ExternalHnswCheckpointMemorySmokeUnsupportedInfo(
                NotAvailable("bytes", "Object-accurate Dictionary<ulong,int> retained bytes, buckets, entries, object headers and slack capacity are not exposed by HNSW/composite internals."),
                NotAvailable("bytes", "Object-accurate HnswGraphLayer object/header/alignment retained bytes are not exposed by HnswIndex."),
                NotAvailable("bytes", "Object-accurate tombstone HashSet retained memory is not exposed by HnswBasePlusExactDeltaIndex."),
                NotAvailable("bytes", "Object-accurate deleted/reserved-ID HashSet retained memory is not exposed by HnswBasePlusExactDeltaIndex."),
                NotAvailable("bytes", "Managed object headers, array headers, alignment and backing-array slack cannot be attributed by VecNet structure in this runner."),
                NotAvailable("bytes", "NeighborCandidate array element/object layout is not reported as an object-accurate retained-memory value."),
                NotMeasured("bytes", "Index-only private bytes are not measured because Fashion-MNIST inputs, exact truth arrays, result arrays, runner objects and runtime state coexist in the same process."),
                NotMeasured("bytes", "Source-composite-only retained memory is not measured because source composite, opened HNSW, input arrays and validation state coexist in this runner process."),
                NotMeasured("bytes", "Opened-only retained memory is not measured because the source composite, opened index and Fashion-MNIST inputs coexist in this runner process."),
                NotMeasured("bytes", "True process peak memory is not measured; sampled peak fields can miss short-lived peaks between samples."),
                NotMeasured("bytes", "Peak temporary disk usage is not measured because active directory/temp-file sampling is not implemented.")),
            [
                "Whole-process samples cannot isolate VecNet index-only retained private bytes from Fashion-MNIST input arrays, exact truth arrays, result buffers, validation arrays, runner objects or runtime state.",
                "Working set and process peak working set are OS/cache-sensitive context only.",
                "GC committed and fragmented values are runtime counters, not object-accurate VecNet retained memory attribution.",
                "Source composite and opened HNSW intentionally coexist after OpenReadOnly in this single-process smoke."
            ]);

    private static ExternalHnswCheckpointMemorySmokePeakMemoryInfo CreatePeakMemory(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options,
        HnswMemoryPeakOperationInfo load,
        HnswMemoryPeakOperationInfo build,
        HnswMemoryPeakOperationInfo mutation,
        HnswMemoryPeakOperationInfo truth,
        HnswMemoryPeakOperationInfo preSearch,
        HnswMemoryPeakOperationInfo checkpoint,
        HnswMemoryPeakOperationInfo open,
        HnswMemoryPeakOperationInfo postSearch,
        HnswMemoryPeakOperationInfo openedSearch,
        HnswMemoryPeakOperationInfo validation) =>
        new(
            "sampled",
            "observedSampledWholeProcessPeaks",
            "Peak memory is actively sampled whole-process process/GC memory for runner phase boundaries. It is not index-only attribution and not a true maximum.",
            load,
            build,
            mutation,
            truth,
            preSearch,
            checkpoint,
            open,
            postSearch,
            openedSearch,
            validation,
            NotMeasured("bytes", "Peak temporary disk is not measured; final checkpoint output bytes are reported under storageOutput only."),
            NotMeasured("bytes", "True process peak memory is not measured; observed sampled peak fields can miss short-lived peaks between samples."),
            [
                string.Create(CultureInfo.InvariantCulture, $"Runner phases are sampled every {options.SampleIntervalMilliseconds} ms plus explicit start/end samples."),
                "Observed sampled peaks can miss short-lived allocations between samples.",
                "Working-set peaks are context-only and OS/cache-sensitive.",
                "Peak values are whole-process values and include runner/runtime state."
            ]);

    private static ExternalHnswCheckpointMemorySmokeLayoutLowerBoundsInfo CreateLowerBounds(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options,
        HnswIndex sourceBaseIndex,
        HnswIndex opened,
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts,
        HnswBasePlusExactDeltaCheckpointCountInfo postCounts)
    {
        HnswLayerLowerBounds sourceBase = CreateGraphLowerBounds(options.BaseVectorCount, sourceBaseIndex.MaxLayer, options.M);
        HnswLayerLowerBounds rebuiltOpened = CreateGraphLowerBounds(opened.Count, opened.MaxLayer, options.M);
        int dimension = sourceBaseIndex.Dimension;
        long sourceBaseVectorBytes = checked((long)options.BaseVectorCount * dimension * sizeof(float));
        long sourceBaseIdBytes = checked((long)options.BaseVectorCount * sizeof(ulong));
        long sourceBaseLevelBytes = checked((long)options.BaseVectorCount * sizeof(int));
        long sourceBaseIdMapBytes = checked((long)options.BaseVectorCount * (sizeof(ulong) + sizeof(int)));
        long deltaVectorBytes = checked((long)preCounts.DeltaPhysicalVectorCount * dimension * sizeof(float));
        long deltaIdBytes = checked((long)preCounts.DeltaPhysicalVectorCount * sizeof(ulong));
        long baseTombstoneBytes = checked((long)preCounts.BaseTombstoneCount * sizeof(ulong));
        long deltaTombstoneBytes = checked((long)preCounts.DeltaTombstoneCount * sizeof(ulong));
        long deletedReservedBytes = checked((long)preCounts.DeletedReservedIdCount * sizeof(ulong));
        long compositeWorkspaceBytes = EstimateCompositeWorkspaceBytes(Math.Max(options.BaseVectorCount, options.LiveVectorCount), options.EfSearch, options.TopK);
        long openedWorkspaceBytes = EstimateHnswWorkspaceBytes(options.LiveVectorCount, options.EfSearch);
        long rebuiltVectorBytes = checked((long)opened.Count * dimension * sizeof(float));
        long rebuiltIdBytes = checked((long)opened.Count * sizeof(ulong));
        long rebuiltLevelBytes = checked((long)opened.Count * sizeof(int));
        long rebuiltIdMapBytes = checked((long)opened.Count * (sizeof(ulong) + sizeof(int)));
        long sourceCompositeLowerBound = checked(
            sourceBaseVectorBytes +
            sourceBaseIdBytes +
            sourceBaseLevelBytes +
            sourceBase.GraphPayloadLowerBoundBytes +
            sourceBaseIdMapBytes +
            deltaVectorBytes +
            deltaIdBytes +
            baseTombstoneBytes +
            deltaTombstoneBytes +
            deletedReservedBytes);
        long rebuiltOpenedLowerBound = checked(
            rebuiltVectorBytes +
            rebuiltIdBytes +
            rebuiltLevelBytes +
            rebuiltOpened.GraphPayloadLowerBoundBytes +
            rebuiltIdMapBytes);

        return new ExternalHnswCheckpointMemorySmokeLayoutLowerBoundsInfo(
            "estimatedLowerBound",
            "payload-only; not actual retained memory",
            dimension,
            preCounts.BasePhysicalVectorCount,
            preCounts.DeltaPhysicalVectorCount,
            preCounts.LiveVectorCount,
            postCounts.BasePhysicalVectorCount,
            sourceBaseVectorBytes,
            sourceBaseIdBytes,
            sourceBaseLevelBytes,
            sourceBase.GraphCountPayloadLowerBoundBytes,
            sourceBase.GraphNeighborPayloadLowerBoundBytes,
            sourceBase.GraphPayloadLowerBoundBytes,
            sourceBaseIdMapBytes,
            deltaVectorBytes,
            deltaIdBytes,
            baseTombstoneBytes,
            deltaTombstoneBytes,
            deletedReservedBytes,
            new MeasurementStatusInfo(
                "estimatedLowerBound",
                compositeWorkspaceBytes.ToString(CultureInfo.InvariantCulture),
                "bytes",
                "Caller-owned composite search workspace payload floor for max(basePhysical, live) and efSearch/topK; excludes object/array headers, alignment and slack."),
            new MeasurementStatusInfo(
                "estimatedLowerBound",
                openedWorkspaceBytes.ToString(CultureInfo.InvariantCulture),
                "bytes",
                "Caller-owned HnswSearchWorkspace payload floor for live vector count and efSearch; excludes object/array headers, alignment and slack."),
            rebuiltVectorBytes,
            rebuiltIdBytes,
            rebuiltLevelBytes,
            rebuiltOpened.GraphCountPayloadLowerBoundBytes,
            rebuiltOpened.GraphNeighborPayloadLowerBoundBytes,
            rebuiltOpened.GraphPayloadLowerBoundBytes,
            rebuiltIdMapBytes,
            sourceCompositeLowerBound,
            rebuiltOpenedLowerBound,
            sourceBase.Layers,
            rebuiltOpened.Layers,
            "sourceCompositeLowerBound = base vector/id/level/graph/id-map payload floors + exact delta vector/id payload floors + tombstone/deleted-reserved ID payload floors; rebuiltOpenedLowerBound = live rebuilt HNSW vector/id/level/graph/id-map payload floors.",
            "Excludes managed object headers, array headers, Dictionary/HashSet buckets/entries/capacity overhead, graph layer object overhead, backing-array slack from growth, NeighborCandidate layout, Fashion-MNIST input arrays, exact truth arrays, captured validation results, JSON serialization objects, runtime/JIT state, process fragmentation and temporary checkpoint/open copies except where observed by sampled peak fields.");
    }

    private static HnswLayerLowerBounds CreateGraphLowerBounds(int vectorCount, int maxLayer, int m)
    {
        int layerCount = Math.Max(0, maxLayer + 1);
        var layers = new HnswMemoryLayerLowerBoundInfo[layerCount];
        long graphCountBytes = 0;
        long graphNeighborBytes = 0;
        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = layer == 0 ? checked(m * 2) : m;
            long layerCountBytes = checked((long)vectorCount * sizeof(int));
            long layerNeighborBytes = checked((long)vectorCount * stride * sizeof(int));
            layers[layer] = new HnswMemoryLayerLowerBoundInfo(layer, stride, layerCountBytes, layerNeighborBytes);
            graphCountBytes = checked(graphCountBytes + layerCountBytes);
            graphNeighborBytes = checked(graphNeighborBytes + layerNeighborBytes);
        }

        return new HnswLayerLowerBounds(graphCountBytes, graphNeighborBytes, checked(graphCountBytes + graphNeighborBytes), layers);
    }

    private static TimedPhase<T> SamplePhase<T>(
        string name,
        int sampleIntervalMilliseconds,
        MemorySnapshot baseline,
        string timedScope,
        string excludedOperations,
        Func<T> operation)
    {
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long timestamp = Stopwatch.GetTimestamp();
        using var sampler = new ProcessMemorySampler(name, sampleIntervalMilliseconds);
        sampler.Start();
        T value = operation();
        ProcessMemorySamplerResult result = sampler.Stop();
        long elapsedTicks = Stopwatch.GetTimestamp() - timestamp;
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;
        HnswMemoryPeakOperationInfo peak = CreatePeak(name, sampleIntervalMilliseconds, result, baseline, timedScope, excludedOperations);
        return new TimedPhase<T>(
            value,
            new ExternalHnswCheckpointMemorySmokePhaseInfo(
                name,
                "measured",
                StopwatchTicksToMilliseconds(elapsedTicks),
                allocatedBytes,
                string.Create(CultureInfo.InvariantCulture, $"Observed sampled whole-process peak with {sampleIntervalMilliseconds} ms interval; actual end boundary sample recorded under actualMemory."),
                timedScope,
                excludedOperations),
            peak.EndSample,
            peak);
    }

    private static HnswMemoryPeakOperationInfo CreatePeak(
        string name,
        int sampleIntervalMilliseconds,
        ProcessMemorySamplerResult result,
        MemorySnapshot baseline,
        string timedScope,
        string excludedOperations) =>
        new(
            name,
            "sampled",
            sampleIntervalMilliseconds,
            result.SampleCount,
            CreateSample(name + "Start", "Whole-process sample immediately before " + timedScope + ".", result.Start, baseline),
            CreateSample(name + "End", "Whole-process sample immediately after " + timedScope + ".", result.End, baseline),
            SampledPeak(result.Peak.ManagedHeapSizeBytes, baseline.ManagedHeapSizeBytes, contextOnly: false, "Highest sampled GC heap size during " + name + "; observed sampled peak, not a true maximum."),
            SampledPeak(result.Peak.GcCommittedBytes, baseline.GcCommittedBytes, contextOnly: false, "Highest sampled GC committed bytes during " + name + "; observed sampled peak, not a true maximum."),
            SampledPeak(result.Peak.ProcessPrivateBytes, baseline.ProcessPrivateBytes, contextOnly: false, "Highest sampled Process.PrivateMemorySize64 during " + name + "; whole-process observed sampled peak, not index-only attribution."),
            SampledPeak(result.Peak.ProcessWorkingSetBytes, baseline.ProcessWorkingSetBytes, contextOnly: true, "Highest sampled Process.WorkingSet64 during " + name + "; OS/cache-sensitive context-only observed sampled peak."),
            "Sampling can miss short-lived peaks between samples; this field is an observed sampled peak, not a mathematical maximum.",
            "Samples are whole-process values and cannot attribute bytes only to VecNet HNSW/composite structures.",
            timedScope,
            excludedOperations);

    private static ExternalHnswCheckpointMemorySmokeEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private external Fashion-MNIST checkpoint memory smoke output is not reviewed public evidence and has no public reporting policy.",
            "One private local memory smoke does not establish public HNSW API/package preview readiness.",
            "No external mutable/update HNSW memory baseline-candidate policy is accepted.",
            "External mutable/update HNSW memory smoke reports are not accepted comparison artifacts.",
            "No external mutable/update HNSW memory regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "External Fashion-MNIST checkpoint memory smoke evidence only; no generated fallback, matrix, competitor comparison, package or platform claim applies.",
                "Actual samples are local whole-process boundary samples and are separated from sampled peaks, lower-bound layout estimates and checkpoint output file facts.",
                "Observed sampled peaks are not true maxima and can miss short-lived peaks.",
                "Working set is OS/cache-sensitive context only.",
                "Not a public claim, preview-readiness result, baseline candidate, comparison artifact, regression gate, Linux validation or BenchmarkDotNet-grade evidence."
            ]);

    private static ExternalHnswCheckpointMemorySmokeEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private external Fashion-MNIST checkpoint memory smoke output is not reviewed public evidence.",
            "This private local smoke report does not establish public HNSW API/package preview readiness.",
            "No external HNSW base-plus-exact-delta memory baseline-candidate policy is accepted.",
            "No external HNSW base-plus-exact-delta memory comparison-artifact policy is accepted.",
            "No external HNSW base-plus-exact-delta memory regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static void ValidateOptions(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options)
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

        if (options.SampleIntervalMilliseconds is < 1 or > 1000)
        {
            throw new ArgumentException("sample interval must be in the range 1..1000 milliseconds.", nameof(options));
        }

        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ValidateOptions(options.ToCheckpointOptions());
    }

    private static HnswMemorySampleInfo CreateSample(string name, string boundary, MemorySnapshot current, MemorySnapshot baseline) =>
        new(
            name,
            boundary,
            Measured(current.ManagedHeapSizeBytes, baseline.ManagedHeapSizeBytes, contextOnly: false, "GC.GetGCMemoryInfo().HeapSizeBytes at sample boundary."),
            Measured(current.GcCommittedBytes, baseline.GcCommittedBytes, contextOnly: false, "GC.GetGCMemoryInfo().TotalCommittedBytes at sample boundary where exposed by the runtime."),
            Measured(current.GcFragmentedBytes, baseline.GcFragmentedBytes, contextOnly: false, "GC.GetGCMemoryInfo().FragmentedBytes at sample boundary where exposed by the runtime."),
            Measured(current.ProcessPrivateBytes, baseline.ProcessPrivateBytes, contextOnly: false, "Process.PrivateMemorySize64 at sample boundary; whole-process local value, not index-only attribution."),
            Measured(current.ProcessWorkingSetBytes, baseline.ProcessWorkingSetBytes, contextOnly: true, "Process.WorkingSet64 at sample boundary; OS/cache-sensitive context only, not a retained-memory claim."),
            Measured(current.ProcessPeakWorkingSetBytes, baseline.ProcessPeakWorkingSetBytes, contextOnly: true, "Process.PeakWorkingSet64 at sample boundary; process-lifetime OS/cache-sensitive context only."));

    private static MemorySnapshot CaptureMemorySnapshot()
    {
        GCMemoryInfo gc = GC.GetGCMemoryInfo();
        using Process process = Process.GetCurrentProcess();
        process.Refresh();
        return new MemorySnapshot(
            gc.HeapSizeBytes,
            gc.TotalCommittedBytes,
            gc.FragmentedBytes,
            process.PrivateMemorySize64,
            process.WorkingSet64,
            process.PeakWorkingSet64);
    }

    private static HnswMemoryMetricInfo Measured(long value, long baseline, bool contextOnly, string reason) =>
        new("measured", value, value - baseline, "bytes", contextOnly, reason);

    private static HnswMemoryMetricInfo SampledPeak(long value, long baseline, bool contextOnly, string reason) =>
        new("sampled", value, value - baseline, "bytes", contextOnly, reason);

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static MeasurementStatusInfo NotAvailable(string unit, string reason) =>
        new("notAvailable", "absent", unit, reason);

    private static long EstimateHnswWorkspaceBytes(int maxElements, int maxEf) =>
        checked(
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)));

    private static long EstimateCompositeWorkspaceBytes(int maxBaseElements, int efSearch, int topK) =>
        checked(EstimateHnswWorkspaceBytes(maxBaseElements, efSearch) + ((long)topK * sizeof(ulong)) + ((long)topK * sizeof(float)));

    private static long FileLength(string directory, string fileName) =>
        new FileInfo(Path.Combine(directory, fileName)).Length;

    private static double StopwatchTicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static string CreateReportId(string? commit, FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName}-{commitPart}-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-1r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private sealed record TimedPhase<T>(
        T Value,
        ExternalHnswCheckpointMemorySmokePhaseInfo Phase,
        HnswMemorySampleInfo EndSample,
        HnswMemoryPeakOperationInfo Peak);

    private sealed record HnswLayerLowerBounds(
        long GraphCountPayloadLowerBoundBytes,
        long GraphNeighborPayloadLowerBoundBytes,
        long GraphPayloadLowerBoundBytes,
        HnswMemoryLayerLowerBoundInfo[] Layers);

    private sealed record MemorySnapshot(
        long ManagedHeapSizeBytes,
        long GcCommittedBytes,
        long GcFragmentedBytes,
        long ProcessPrivateBytes,
        long ProcessWorkingSetBytes,
        long ProcessPeakWorkingSetBytes);

    private sealed record ProcessMemorySamplerResult(
        MemorySnapshot Start,
        MemorySnapshot End,
        MemorySnapshot Peak,
        int SampleCount);

    private sealed class ProcessMemorySampler : IDisposable
    {
        private readonly string _name;
        private readonly int _intervalMilliseconds;
        private readonly List<MemorySnapshot> _samples = [];
        private readonly object _gate = new();
        private volatile bool _stopRequested;
        private Thread? _thread;

        internal ProcessMemorySampler(string name, int intervalMilliseconds)
        {
            _name = name;
            _intervalMilliseconds = intervalMilliseconds;
        }

        internal void Start()
        {
            lock (_gate)
            {
                _samples.Add(CaptureMemorySnapshot());
            }

            _thread = new Thread(Run)
            {
                IsBackground = true,
                Name = "VecNet.ExternalCheckpointMemorySmoke." + _name
            };
            _thread.Start();
        }

        internal ProcessMemorySamplerResult Stop()
        {
            _stopRequested = true;
            _thread?.Join();
            lock (_gate)
            {
                _samples.Add(CaptureMemorySnapshot());
                MemorySnapshot start = _samples[0];
                MemorySnapshot end = _samples[^1];
                MemorySnapshot peak = new(
                    _samples.Max(sample => sample.ManagedHeapSizeBytes),
                    _samples.Max(sample => sample.GcCommittedBytes),
                    _samples.Max(sample => sample.GcFragmentedBytes),
                    _samples.Max(sample => sample.ProcessPrivateBytes),
                    _samples.Max(sample => sample.ProcessWorkingSetBytes),
                    _samples.Max(sample => sample.ProcessPeakWorkingSetBytes));
                return new ProcessMemorySamplerResult(start, end, peak, _samples.Count);
            }
        }

        public void Dispose()
        {
            if (_thread is not null && _thread.IsAlive)
            {
                _stopRequested = true;
                _thread.Join();
            }
        }

        private void Run()
        {
            while (!_stopRequested)
            {
                Thread.Sleep(_intervalMilliseconds);
                MemorySnapshot sample = CaptureMemorySnapshot();
                lock (_gate)
                {
                    _samples.Add(sample);
                }
            }
        }
    }
}
