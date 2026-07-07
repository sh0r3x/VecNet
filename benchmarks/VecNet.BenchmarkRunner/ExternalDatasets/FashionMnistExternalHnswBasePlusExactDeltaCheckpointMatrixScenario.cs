using System.Globalization;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario
{
    private const string TaskId = "VEC-140";
    private const string SchemaName = "VecNet.ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest";
    private const string SchemaVersion = "0.1";
    private const string DatasetId = "fashion-mnist-784-euclidean";
    private const int Dimension = 784;
    private const int AdmittedBaseMatrixRowCount = 60_000;
    private const int QueryCount = 50;
    private const int WarmupQueries = 3;
    private const int CheckpointRuns = 2;
    private const uint MatrixSeed = 0x5EED2139;
    private const ulong HnswMatrixSeedBase = 0x484E535700013900UL;
    private const int M = 16;
    private const int EfConstruction = 128;
    private const int EfSearch = 192;
    private const int DuplicateInsertAttempts = 1;
    private const int UnknownDeleteAttempts = 1;
    private const int RepeatedDeleteAttempts = 1;
    private const string HnswProfileName = "wide-m16-ef192";

    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly ExternalCheckpointUpdateProfile[] UpdateProfiles =
    [
        new(
            "low-churn",
            BaseRowCount: 59_000,
            DeltaRowCount: 500,
            DeletedBaseCount: 100,
            DeletedDeltaCount: 0,
            ExpectedPhysicalCandidateCount: 59_500,
            ExpectedLiveCount: 59_400,
            ExpectedDeletedReservedIdCount: 100,
            "light overlay churn with exact delta present and low tombstone pressure"),
        new(
            "tombstone-heavy",
            BaseRowCount: 56_000,
            DeltaRowCount: 2_000,
            DeletedBaseCount: 5_000,
            DeletedDeltaCount: 500,
            ExpectedPhysicalCandidateCount: 58_000,
            ExpectedLiveCount: 52_500,
            ExpectedDeletedReservedIdCount: 5_500,
            "heavy base tombstone suppression plus delta tombstones")
    ];

    public static ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest Run(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(options.PresetName);
        MatrixCase[] cases = ExpandCases(options);
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo cacheTruth = CreateCacheTruthInfo(options, out string? cacheBlockReason);
        var caseManifests = new ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;
        int blocked = 0;

        if (cacheBlockReason is not null)
        {
            for (int i = 0; i < cases.Length; i++)
            {
                blocked++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    cases[i],
                    CreateCaseArguments(cases[i].Options),
                    report: null,
                    "blocked",
                    "blocked",
                    "cacheTruthReadiness",
                    cacheBlockReason,
                    includeLinkedPaths: false);
            }
        }
        else
        {
            for (int i = 0; i < cases.Length; i++)
            {
                MatrixCase matrixCase = cases[i];
                string[] caseArguments = CreateCaseArguments(matrixCase.Options);
                try
                {
                    ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report =
                        FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(matrixCase.Options, caseArguments);
                    FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Write(report, matrixCase.Options.OutputPath);

                    ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary linkedValidation =
                        ValidateLinkedReport(matrixCase, report);
                    bool casePassed =
                        string.Equals(report.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) &&
                        string.Equals(linkedValidation.Status, "passed", StringComparison.OrdinalIgnoreCase);
                    if (casePassed)
                    {
                        passed++;
                    }
                    else
                    {
                        failed++;
                    }

                    caseManifests[i] = CreateCaseManifest(
                        i + 1,
                        matrixCase,
                        caseArguments,
                        report,
                        casePassed ? "passed" : "failed",
                        casePassed ? "passed" : "failed",
                        errorCategory: casePassed ? null : "linkedReportValidation",
                        errorMessage: casePassed ? null : "Linked VEC-138 report did not satisfy matrix validation requirements.",
                        includeLinkedPaths: true,
                        linkedValidation);
                }
                catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
                {
                    blocked++;
                    caseManifests[i] = CreateCaseManifest(
                        i + 1,
                        matrixCase,
                        caseArguments,
                        report: null,
                        "blocked",
                        "blocked",
                        "caseRuntimeBlock",
                        ex.Message,
                        includeLinkedPaths: false);
                }
                catch (Exception ex) when (ex is ArgumentException or UnauthorizedAccessException or InvalidOperationException)
                {
                    failed++;
                    caseManifests[i] = CreateCaseManifest(
                        i + 1,
                        matrixCase,
                        caseArguments,
                        report: null,
                        "failed",
                        "failed",
                        "caseFailure",
                        ex.Message,
                        includeLinkedPaths: false);
                }
            }
        }

        RepositoryInfo repository = RepositoryInfo.Create();
        ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate aggregate =
            CreateAggregate(caseManifests, passed, failed, blocked, cacheTruth, cacheBlockReason);

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
            CreateReportId(repository.Commit, presetName),
            presetName,
            DateTimeOffset.UtcNow,
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixPostureInfo(
                "local-evidence",
                "private-raw",
                "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix",
                "Private/local external Fashion-MNIST checkpoint matrix evidence only; no public mutable/update HNSW claim."),
            cacheTruth,
            CreateDesign(presetName, cases),
            cases.Length,
            failed == 0 && blocked == 0 ? "passed" : "failed",
            caseManifests,
            aggregate,
            CreateEligibility(presetName),
            [
                "Private external Fashion-MNIST HNSW base-plus-exact-delta checkpoint matrix evidence only; not a public benchmark, baseline candidate, comparison artifact, regression gate or public mutable/update HNSW claim.",
                "Each successful case reuses the accepted VEC-138 external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint report schema and writes one linked private per-case report.",
                "The matrix uses only the already admitted local Fashion-MNIST cache and existing exact truth artifact; it does not download, convert, admit, refresh or regenerate Fashion-MNIST.",
                "Existing Fashion-MNIST exact truth is a cache/readiness guard only; linked VEC-138 reports compute exact updated truth from each case's post-update live view.",
                "Linked report paths are manifest-relative and checkpoint outputs are case-scoped under checkpoint-output/checkpoint-run-NNN subdirectories.",
                "Process/resident/peak memory, concurrency evidence, HNSW filtering, hnswlib/FAISS comparison, public docs, package metadata, package publication and public claims are out of scope."
            ]);
    }

    public static MatrixCase[] ExpandCases(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options)
    {
        string presetName = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(options.PresetName);
        CaseDefinition[] definitions = GetCaseDefinitions(presetName);
        string manifestDirectory = GetManifestDirectory(options.ManifestPath);
        var cases = new MatrixCase[definitions.Length];

        for (int i = 0; i < definitions.Length; i++)
        {
            CaseDefinition definition = definitions[i];
            ExternalCheckpointUpdateProfile updateProfile = definition.UpdateProfile;
            int caseNumber = i + 1;
            uint workloadSeed = unchecked(MatrixSeed + (uint)i);
            ulong hnswSeed = HnswMatrixSeedBase + (ulong)caseNumber;
            string caseId = CreateCaseId(caseNumber, definition.TopK, updateProfile.Name);
            string caseDirectory = Path.Combine(options.OutputDirectory, caseId);
            string reportPath = Path.Combine(caseDirectory, "checkpoint-report.json");
            string checkpointDirectory = Path.Combine(caseDirectory, "checkpoint-output");
            var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
                options.CacheRoot,
                reportPath,
                checkpointDirectory,
                QueryCount,
                definition.TopK,
                updateProfile.BaseRowCount,
                updateProfile.DeltaRowCount,
                updateProfile.DeletedBaseCount,
                updateProfile.DeletedDeltaCount,
                DuplicateInsertAttempts,
                UnknownDeleteAttempts,
                RepeatedDeleteAttempts,
                CheckpointRuns,
                WarmupQueries,
                VectorMetric.SquaredEuclidean,
                workloadSeed,
                M,
                EfConstruction,
                EfSearch,
                hnswSeed);

            cases[i] = new MatrixCase(
                caseId,
                updateProfile.Name,
                HnswProfileName,
                CreateRelativePath(manifestDirectory, reportPath),
                CreateRelativePath(manifestDirectory, checkpointDirectory),
                updateProfile,
                caseOptions);
        }

        return cases;
    }

    public static int GetMaxTopK(string presetName) => GetCaseDefinitions(presetName).Max(item => item.TopK);

    public static int GetMaxPhysicalCandidateVectorCount(string presetName) =>
        GetCaseDefinitions(presetName).Max(item => item.UpdateProfile.ExpectedPhysicalCandidateCount);

    public static void WriteManifest(ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest manifest, string manifestPath) =>
        ReportWriter.WriteJson(manifest, manifestPath);

    public static string[] CreateCaseArguments(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options) =>
    [
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
        "--cache-root", options.CacheRoot,
        "--output", options.OutputPath,
        "--checkpoint-directory", options.CheckpointDirectory,
        "--query-count", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--base-vectors", options.BaseVectorCount.ToString(CultureInfo.InvariantCulture),
        "--insertions", options.InsertedDeltaCount.ToString(CultureInfo.InvariantCulture),
        "--deletes", options.DeletedBaseCount.ToString(CultureInfo.InvariantCulture),
        "--delta-deletes", options.DeletedDeltaCount.ToString(CultureInfo.InvariantCulture),
        "--duplicate-inserts", options.DuplicateInsertAttempts.ToString(CultureInfo.InvariantCulture),
        "--unknown-deletes", options.UnknownDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--repeated-deletes", options.RepeatedDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--metric", "squared-euclidean",
        "--seed", FormatHex(options.Seed),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--hnsw-seed", FormatHex(options.HnswSeed)
    ];

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo CreateCacheTruthInfo(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options,
        out string? blockReason)
    {
        try
        {
            FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset =
                FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(
                    new FashionMnistExternalHnswBenchmarkOptions(
                        options.CacheRoot,
                        Path.Combine(options.OutputDirectory, "cache-truth-validation.json"),
                        QueryCount,
                        GetMaxTopK(options.PresetName),
                        Runs: 1,
                        WarmupQueries,
                        VectorMetric.SquaredEuclidean,
                        M,
                        EfConstruction,
                        EfSearch,
                        HnswMatrixSeedBase));

            if (dataset.BaseCount < GetMaxPhysicalCandidateVectorCount(options.PresetName))
            {
                throw new InvalidDataException("Admitted Fashion-MNIST base matrix does not contain enough rows for the largest checkpoint matrix case.");
            }

            blockReason = null;
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo(
                "available",
                options.CacheRoot,
                dataset.Manifest.DatasetId,
                Dimension,
                VectorMetric.SquaredEuclidean.ToString(),
                "Loaded existing admitted Fashion-MNIST cache only; no download, conversion, admission, refresh or truth regeneration path is used by VEC-140.",
                "Loaded existing exact truth artifact only as cache/readiness guard; linked VEC-138 reports generate exact updated truth from live post-update views.",
                dataset.Paths.RelativeManifestPath,
                dataset.ManifestSha256,
                dataset.Manifest.Truth.RelativePath,
                dataset.TruthSha256,
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                dataset.Truth.QuerySubsetCount,
                dataset.Truth.TruthDepth,
                QueryCount,
                GetMaxTopK(options.PresetName),
                GetMaxPhysicalCandidateVectorCount(options.PresetName),
                ErrorMessage: null);
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException or ArgumentException or UnauthorizedAccessException)
        {
            blockReason = ex.Message;
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo(
                "unavailable",
                options.CacheRoot,
                DatasetId,
                Dimension,
                VectorMetric.SquaredEuclidean.ToString(),
                "Admitted local Fashion-MNIST cache is required; VEC-140 must not download, convert, admit, refresh or regenerate data.",
                "Existing exact truth artifact with sufficient query subset and truth depth is required; VEC-140 must not refresh truth.",
                AdmissionManifestPath: null,
                AdmissionManifestSha256: null,
                TruthRelativePath: null,
                TruthSha256: null,
                BaseVectorCount: null,
                QueryMatrixCount: null,
                TruthQuerySubsetCount: null,
                TruthDepth: null,
                QueryCount,
                GetMaxTopK(options.PresetName),
                GetMaxPhysicalCandidateVectorCount(options.PresetName),
                ex.Message);
        }
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        MatrixCase matrixCase,
        string[] commandArguments,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report,
        string status,
        string validationStatus,
        string? errorCategory,
        string? errorMessage,
        bool includeLinkedPaths,
        ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary? linkedValidation = null)
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options = matrixCase.Options;
        ExternalCheckpointUpdateProfile profile = matrixCase.UpdateProfile;

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.UpdateProfileName,
            matrixCase.HnswProfileName,
            DatasetId,
            options.Metric.ToString(),
            Dimension,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            FormatHex(options.Seed),
            FormatHex(options.HnswSeed),
            options.M,
            options.EfConstruction,
            options.EfSearch,
            ImmutableBaseStartRow: 0,
            options.BaseVectorCount - 1,
            options.BaseVectorCount,
            DeltaStartRow: options.BaseVectorCount,
            options.PhysicalCandidateVectorCount - 1,
            options.InsertedDeltaCount,
            UnusedStartRow: options.PhysicalCandidateVectorCount,
            AdmittedBaseMatrixRowCount - 1,
            AdmittedBaseMatrixRowCount - options.PhysicalCandidateVectorCount,
            options.DeletedBaseCount,
            options.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            options.PhysicalCandidateVectorCount,
            options.LiveVectorCount,
            profile.ExpectedDeletedReservedIdCount,
            includeLinkedPaths ? matrixCase.RelativeReportPath : null,
            includeLinkedPaths ? matrixCase.RelativeCheckpointDirectoryPath : null,
            commandArguments,
            report?.ReportId,
            status,
            validationStatus,
            linkedValidation ?? CreateUnavailableLinkedValidation(),
            CreateRepeatedRunSummary(options, report),
            CreatePhaseDiagnosticsSummary(report?.CheckpointRuns.Runs.Select(run => run.Phases).ToArray()),
            CreateOutputSummary(report),
            CreateSearchSummary(report?.Searches.PreCheckpointSourceComposite),
            CreateSearchSummary(report?.Searches.PostCheckpointRebuiltComposite),
            CreateSearchSummary(report?.Searches.OpenedReadOnlyHnsw),
            CreateOpenedValidationSummary(report),
            CreateParitySummary(report),
            CreateDeletedReservationSummary(profile, report),
            CreateNoChangesSummary(report),
            CreateCountSummary(options, profile, report),
            CreateMemorySummary(),
            CreateEligibilitySummary(report),
            errorCategory,
            errorMessage);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary ValidateLinkedReport(
        MatrixCase matrixCase,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report)
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options = matrixCase.Options;
        bool schemaMatched =
            report.SchemaName == "VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport" &&
            report.SchemaVersion == "0.1";
        bool scenarioMatched =
            report.TaskId == "VEC-138" &&
            report.ScenarioName == FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName;
        bool caseParametersMatched =
            report.Workload.TopK == options.TopK &&
            report.Workload.MeasuredQueryCount == QueryCount &&
            report.Workload.WarmupQueryCount == WarmupQueries &&
            report.Workload.CheckpointRunCount == CheckpointRuns &&
            report.Workload.ImmutableBaseRowCount == options.BaseVectorCount &&
            report.Workload.DeltaRowCount == options.InsertedDeltaCount &&
            report.Workload.DeletedBaseVectorCount == options.DeletedBaseCount &&
            report.Workload.DeletedDeltaVectorCount == options.DeletedDeltaCount &&
            report.Hnsw.M == M &&
            report.Hnsw.EfConstruction == EfConstruction &&
            report.Hnsw.EfSearch == EfSearch &&
            report.Hnsw.RandomSeed == FormatHex(options.HnswSeed);
        bool checkpointSections =
            report.CheckpointRuns.RunCount == CheckpointRuns &&
            report.Checkpoint.Status == nameof(HnswBasePlusExactDeltaCheckpointStatus.Published) &&
            report.Output.Status == "recorded" &&
            report.NoChangesProbe.Status == "passed";
        bool phaseDiagnostics = AllMeasured(report.Checkpoint.Phases);
        bool openedValidation = report.OpenedValidation.Status == "passed";
        bool parity = report.OpenedValidation.RebuiltCompositeOpenedSearchParity.AllResultsMatched;
        bool deletedReservation = report.Validation.DeletedReservedIdsRejectedAfterCheckpoint;
        bool eligibilityFalse = CreateEligibilitySummary(report).AllEligibilityFlagsFalse;
        bool passed = schemaMatched &&
            scenarioMatched &&
            caseParametersMatched &&
            checkpointSections &&
            phaseDiagnostics &&
            openedValidation &&
            parity &&
            deletedReservation &&
            eligibilityFalse;

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary(
            passed ? "passed" : "failed",
            LinkedReportInspected: true,
            schemaMatched,
            scenarioMatched,
            caseParametersMatched,
            checkpointSections,
            phaseDiagnostics,
            openedValidation,
            parity,
            deletedReservation,
            eligibilityFalse);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary CreateUnavailableLinkedValidation() =>
        new(
            "notAvailable",
            LinkedReportInspected: false,
            SchemaMatched: null,
            ScenarioMatched: null,
            CaseParametersMatched: null,
            RequiredCheckpointSectionsPresent: null,
            PhaseDiagnosticsPresent: null,
            OpenedValidationPresent: null,
            RebuiltOpenedParityPassed: null,
            DeletedReservationValidated: null,
            EligibilityFalse: null);

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary CreateRepeatedRunSummary(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
                "notAvailable",
                options.Runs,
                CompletedRunCount: null,
                PublishedRunCount: null,
                NoChangesRunCount: null,
                FailedRunCount: null,
                DetailedValidationRunNumber: null,
                DetailedValidationUsesFinalRun: null,
                MeanElapsedMilliseconds: null,
                MinElapsedMilliseconds: null,
                MaxElapsedMilliseconds: null,
                MeanManagedAllocatedBytes: null,
                MinManagedAllocatedBytes: null,
                MaxManagedAllocatedBytes: null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
            "recorded",
            options.Runs,
            report.CheckpointRuns.Runs.Length,
            report.CheckpointRuns.Runs.Count(run => run.Status == nameof(HnswBasePlusExactDeltaCheckpointStatus.Published)),
            report.CheckpointRuns.Runs.Count(run => run.Status == nameof(HnswBasePlusExactDeltaCheckpointStatus.NoChanges)),
            report.CheckpointRuns.Runs.Count(run =>
                run.Status != nameof(HnswBasePlusExactDeltaCheckpointStatus.Published) &&
                run.Status != nameof(HnswBasePlusExactDeltaCheckpointStatus.NoChanges)),
            report.CheckpointRuns.DetailedValidationRunNumber,
            report.CheckpointRuns.DetailedValidationRunNumber == options.Runs,
            report.CheckpointRuns.Aggregate.MeanElapsedMilliseconds,
            report.CheckpointRuns.Aggregate.MinElapsedMilliseconds,
            report.CheckpointRuns.Aggregate.MaxElapsedMilliseconds,
            report.CheckpointRuns.Aggregate.MeanManagedAllocatedBytes,
            report.CheckpointRuns.Aggregate.MinManagedAllocatedBytes,
            report.CheckpointRuns.Aggregate.MaxManagedAllocatedBytes);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary CreatePhaseDiagnosticsSummary(
        HnswBasePlusExactDeltaCheckpointPhaseSetInfo[]? phases)
    {
        if (phases is null || phases.Length == 0)
        {
            return UnavailablePhaseDiagnostics();
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary(
            "recorded",
            SummarizePhase(phases.Select(item => item.LiveSnapshot).ToArray()),
            SummarizePhase(phases.Select(item => item.RebuildBuild).ToArray()),
            SummarizePhase(phases.Select(item => item.Save).ToArray()),
            SummarizePhase(phases.Select(item => item.OpenValidation).ToArray()),
            SummarizePhase(phases.Select(item => item.Publication).ToArray()));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary UnavailablePhaseDiagnostics() =>
        new(
            "notAvailable",
            UnavailablePhase(),
            UnavailablePhase(),
            UnavailablePhase(),
            UnavailablePhase(),
            UnavailablePhase());

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary SummarizePhase(
        HnswBasePlusExactDeltaCheckpointPhaseInfo[] phases)
    {
        int measured = phases.Count(item => item.Status == "Measured");
        int notExecuted = phases.Count(item => item.Status == "NotExecuted");
        int missing = phases.Length - measured - notExecuted;
        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary(
            missing == 0 && measured > 0 ? "recorded" : "partial",
            measured,
            notExecuted,
            missing,
            phases.Sum(item => item.ElapsedMilliseconds),
            phases.Sum(item => item.ManagedAllocatedBytes));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary UnavailablePhase() =>
        new("notAvailable", MeasuredCount: 0, NotExecutedCount: 0, MissingCount: 1, TotalElapsedMilliseconds: null, TotalManagedAllocatedBytes: null);

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary CreateOutputSummary(
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary("notAvailable", null, null, null, null, null, null, null, null, null, null, null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary(
            report.Output.Status,
            report.Output.FileCount,
            report.Output.TotalBytes,
            report.Output.ManifestBytes,
            report.Output.IdsBytes,
            report.Output.VectorsBytes,
            report.Output.LevelsBytes,
            report.Output.GraphBytes,
            report.Output.OutputVectorCount,
            report.Output.BytesPerLiveVector,
            report.Output.ValidationOpenStatus,
            report.Output.ScanTimingScope);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary CreateSearchSummary(
        HnswBasePlusExactDeltaCheckpointSearchSectionInfo? section)
    {
        if (section is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary("notAvailable", null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary(
            "recorded",
            section.Metrics.RecallAtK,
            section.Metrics.OrderedAgreement,
            section.Metrics.DistanceToleranceStatus,
            section.Metrics.DistanceMismatchCount,
            section.Metrics.MissingResultCount,
            section.Metrics.ExtraResultCount,
            section.Metrics.ReturnedResultIntegrity.Status,
            section.Metrics.ReturnedResultIntegrity.CheckedResultCount,
            section.Metrics.ReturnedResultIntegrity.UnknownIdCount,
            section.Metrics.ReturnedResultIntegrity.TombstonedIdCount,
            section.Metrics.ReturnedResultIntegrity.DistanceMismatchCount,
            section.Underfill.UnderfilledQueryCount,
            section.Underfill.UnderfilledSlotCount,
            section.Search.Aggregate.MeanQps,
            section.Search.Aggregate.MeanLatencyP95Milliseconds,
            section.Search.Aggregate.MeanManagedAllocatedBytesPerQuery);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary CreateOpenedValidationSummary(
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary("notAvailable", null, null, null, null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary(
            report.OpenedValidation.Status,
            report.OpenedValidation.ExpectedVectorCount,
            report.OpenedValidation.OpenedVectorCount,
            report.OpenedValidation.IdMismatchCount,
            report.OpenedValidation.VectorMismatchCount);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary CreateParitySummary(
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        HnswBasePlusExactDeltaCheckpointParityInfo? parity = report?.OpenedValidation.RebuiltCompositeOpenedSearchParity;
        if (parity is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary("notAvailable", null, null, null, null, null, null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary(
            parity.AllResultsMatched ? "passed" : "failed",
            parity.QueryCount,
            parity.WrittenCountMismatchCount,
            parity.IdMismatchCount,
            parity.OrderMismatchCount,
            parity.DistanceMismatchCount,
            parity.AllResultsMatched);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary CreateDeletedReservationSummary(
        ExternalCheckpointUpdateProfile profile,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary(
                "notAvailable",
                DeletedReservedIdsRejectedAfterCheckpoint: null,
                profile.ExpectedDeletedReservedIdCount,
                ActualDeletedReservedIdCount: null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary(
            report.Validation.DeletedReservedIdsRejectedAfterCheckpoint ? "passed" : "failed",
            report.Validation.DeletedReservedIdsRejectedAfterCheckpoint,
            profile.ExpectedDeletedReservedIdCount,
            report.PostCheckpointCounts.DeletedReservedIdCount);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary CreateNoChangesSummary(
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary(
                "notAvailable",
                GenerationUnchanged: null,
                OutputDirectoryRemainedEmpty: null,
                UnavailablePhaseDiagnostics());
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary(
            report.NoChangesProbe.Status,
            report.NoChangesProbe.GenerationUnchanged,
            report.NoChangesProbe.OutputDirectoryRemainedEmpty,
            CreatePhaseDiagnosticsSummary([report.NoChangesProbe.Phases]));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary CreateCountSummary(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options,
        ExternalCheckpointUpdateProfile profile,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary(
                "notAvailable",
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.PhysicalCandidateVectorCount,
                options.LiveVectorCount,
                profile.ExpectedDeletedReservedIdCount,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary(
            "recorded",
            options.BaseVectorCount,
            options.InsertedDeltaCount,
            options.PhysicalCandidateVectorCount,
            options.LiveVectorCount,
            profile.ExpectedDeletedReservedIdCount,
            report.PreCheckpointCounts.LiveVectorCount,
            report.PreCheckpointCounts.TombstoneCount,
            report.PreCheckpointCounts.DeletedReservedIdCount,
            report.PostCheckpointCounts.BasePhysicalVectorCount,
            report.PostCheckpointCounts.LiveVectorCount,
            report.PostCheckpointCounts.TombstoneCount,
            report.PostCheckpointCounts.DeletedReservedIdCount,
            report.PreCheckpointCounts.TombstoneRatio,
            report.PreCheckpointCounts.DeltaInsertRatio);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixMemorySummary CreateMemorySummary() =>
        new("notMeasured", "bytes", "Actual/process/resident/peak memory is not measured by VEC-140.");

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary CreateEligibilitySummary(
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
                "notAvailable",
                LinkedReportInspected: false,
                NonFalseEligibilityFlagCount: 0,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                ComparisonPublicationEligible: false,
                RegressionGateEligible: false,
                AllEligibilityFlagsFalse: false);
        }

        bool publicClaimEligible = report.Evidence.PublicClaimEligible || report.Validation.PublicClaimEligible || report.Eligibility.PublicClaimEligible;
        bool baselineCandidateEligible = report.Evidence.BaselineCandidateEligible || report.Validation.BaselineCandidateEligible || report.Eligibility.BaselineCandidateEligible;
        bool comparisonArtifactEligible = report.Validation.ComparisonArtifactEligible;
        bool regressionGateEligible = report.Evidence.RegressionGateEligible || report.Validation.RegressionGateEligible || report.Eligibility.RegressionGateEligible;
        int nonFalse = CountTrue(publicClaimEligible, baselineCandidateEligible, comparisonArtifactEligible, regressionGateEligible);

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
            "recorded",
            LinkedReportInspected: true,
            nonFalse,
            publicClaimEligible,
            baselineCandidateEligible,
            comparisonArtifactEligible,
            ComparisonPublicationEligible: false,
            regressionGateEligible,
            AllEligibilityFlagsFalse: nonFalse == 0);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate CreateAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked,
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo cacheTruth,
        string? cacheBlockReason)
    {
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded =
            cases.Where(item => item.LinkedReportId is not null).ToArray();

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate(
            passed,
            failed,
            SkippedCaseCount: 0,
            blocked,
            recorded.Length,
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthAggregate(
                cacheTruth.Status,
                string.Equals(cacheTruth.Status, "available", StringComparison.OrdinalIgnoreCase),
                cacheBlockReason is not null && blocked == cases.Length,
                blocked,
                cacheBlockReason),
            CreateRepeatedRunAggregate(cases, recorded),
            CreateAggregatePhaseDiagnostics(recorded.Select(item => item.PhaseDiagnostics).ToArray()),
            CreateOutputAggregate(recorded),
            CreateSearchAggregate(recorded.Select(item => item.PreCheckpointSourceCompositeSearch).ToArray()),
            CreateSearchAggregate(recorded.Select(item => item.PostCheckpointRebuiltCompositeSearch).ToArray()),
            CreateSearchAggregate(recorded.Select(item => item.OpenedReadOnlyHnswSearch).ToArray()),
            CreateOpenedAggregate(recorded),
            CreateParityAggregate(recorded),
            CreateDeletedReservationAggregate(recorded),
            CreateNoChangesAggregate(recorded),
            CreateMemorySummary(),
            CreateRecursiveEligibilityAggregate(cases, recorded));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunAggregate CreateRepeatedRunAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunAggregate(
                "notAvailable",
                CheckpointRuns,
                RecordedCaseCount: 0,
                RequestedRunCountTotal: cases.Length * CheckpointRuns,
                CompletedRunCount: 0,
                PublishedRunCount: 0,
                NoChangesRunCount: 0,
                FailedRunCount: 0,
                FinalRunDetailedValidationCaseCount: 0,
                MeanElapsedMilliseconds: null,
                MaxElapsedMilliseconds: null,
                MeanManagedAllocatedBytes: null,
                MaxManagedAllocatedBytes: null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunAggregate(
            "recorded",
            CheckpointRuns,
            recorded.Length,
            cases.Length * CheckpointRuns,
            recorded.Sum(item => item.RepeatedCheckpointRuns.CompletedRunCount ?? 0),
            recorded.Sum(item => item.RepeatedCheckpointRuns.PublishedRunCount ?? 0),
            recorded.Sum(item => item.RepeatedCheckpointRuns.NoChangesRunCount ?? 0),
            recorded.Sum(item => item.RepeatedCheckpointRuns.FailedRunCount ?? 0),
            recorded.Count(item => item.RepeatedCheckpointRuns.DetailedValidationUsesFinalRun == true),
            recorded.Average(item => item.RepeatedCheckpointRuns.MeanElapsedMilliseconds ?? 0),
            recorded.Max(item => item.RepeatedCheckpointRuns.MaxElapsedMilliseconds ?? 0),
            recorded.Average(item => item.RepeatedCheckpointRuns.MeanManagedAllocatedBytes ?? 0),
            recorded.Max(item => item.RepeatedCheckpointRuns.MaxManagedAllocatedBytes ?? 0));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary CreateAggregatePhaseDiagnostics(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary[] summaries)
    {
        ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary[] recorded =
            summaries.Where(item => item.Status == "recorded").ToArray();
        if (recorded.Length == 0)
        {
            return UnavailablePhaseDiagnostics();
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary(
            "recorded",
            SumPhases(recorded.Select(item => item.LiveSnapshot)),
            SumPhases(recorded.Select(item => item.RebuildBuild)),
            SumPhases(recorded.Select(item => item.Save)),
            SumPhases(recorded.Select(item => item.OpenValidation)),
            SumPhases(recorded.Select(item => item.Publication)));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary SumPhases(
        IEnumerable<ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary> phases)
    {
        ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary[] materialized = phases.ToArray();
        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary(
            "recorded",
            materialized.Sum(item => item.MeasuredCount),
            materialized.Sum(item => item.NotExecutedCount),
            materialized.Sum(item => item.MissingCount),
            materialized.Sum(item => item.TotalElapsedMilliseconds ?? 0),
            materialized.Sum(item => item.TotalManagedAllocatedBytes ?? 0));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputAggregate CreateOutputAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputAggregate(
                "notAvailable",
                RecordedCaseCount: 0,
                TotalFileCount: 0,
                TotalBytes: 0,
                ManifestBytes: 0,
                IdsBytes: 0,
                VectorsBytes: 0,
                LevelsBytes: 0,
                GraphBytes: 0,
                MinBytesPerLiveVector: null,
                MaxBytesPerLiveVector: null,
                ScanTimingScope: "outsideCheckpointDuration");
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputAggregate(
            "recorded",
            recorded.Length,
            recorded.Sum(item => item.OutputSummary.FileCount ?? 0),
            recorded.Sum(item => item.OutputSummary.TotalBytes ?? 0),
            recorded.Sum(item => item.OutputSummary.ManifestBytes ?? 0),
            recorded.Sum(item => item.OutputSummary.IdsBytes ?? 0),
            recorded.Sum(item => item.OutputSummary.VectorsBytes ?? 0),
            recorded.Sum(item => item.OutputSummary.LevelsBytes ?? 0),
            recorded.Sum(item => item.OutputSummary.GraphBytes ?? 0),
            recorded.Min(item => item.OutputSummary.BytesPerLiveVector),
            recorded.Max(item => item.OutputSummary.BytesPerLiveVector),
            "outsideCheckpointDuration");
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary CreateSearchAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary[] summaries)
    {
        ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary[] recorded =
            summaries.Where(item => item.Status == "recorded").ToArray();
        if (recorded.Length == 0)
        {
            return new ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary("notAvailable", 0, null, null, null, null, 0, 0, 0, 0, null);
        }

        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary(
            "recorded",
            recorded.Length,
            recorded.Min(item => item.RecallAtK),
            recorded.Max(item => item.RecallAtK),
            recorded.Min(item => item.OrderedAgreement),
            recorded.Max(item => item.OrderedAgreement),
            recorded.Count(item => item.ReturnedResultIntegrityStatus != "passed"),
            recorded.Count(item => item.DistanceToleranceStatus != "passed"),
            recorded.Sum(item => item.UnderfilledQueryCount ?? 0),
            recorded.Sum(item => item.UnderfilledSlotCount ?? 0),
            recorded.Max(item => item.MeanManagedAllocatedBytesPerQuery));
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedAggregate CreateOpenedAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded) =>
        recorded.Length == 0
            ? new ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedAggregate("notAvailable", 0, 0, 0, 0)
            : new ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedAggregate(
                "recorded",
                recorded.Length,
                recorded.Count(item => item.OpenedValidation.Status == "passed"),
                recorded.Sum(item => item.OpenedValidation.IdMismatchCount ?? 0),
                recorded.Sum(item => item.OpenedValidation.VectorMismatchCount ?? 0));

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixParityAggregate CreateParityAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded) =>
        recorded.Length == 0
            ? new ExternalHnswBasePlusExactDeltaCheckpointMatrixParityAggregate("notAvailable", 0, 0, 0, 0, 0)
            : new ExternalHnswBasePlusExactDeltaCheckpointMatrixParityAggregate(
                "recorded",
                recorded.Length,
                recorded.Count(item => item.RebuiltOpenedParity.AllResultsMatched == true),
                recorded.Sum(item => item.RebuiltOpenedParity.IdMismatchCount ?? 0),
                recorded.Sum(item => item.RebuiltOpenedParity.OrderMismatchCount ?? 0),
                recorded.Sum(item => item.RebuiltOpenedParity.DistanceMismatchCount ?? 0));

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationAggregate CreateDeletedReservationAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded) =>
        recorded.Length == 0
            ? new ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationAggregate("notAvailable", 0, 0, 0, 0)
            : new ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationAggregate(
                "recorded",
                recorded.Length,
                recorded.Count(item => item.DeletedReservation.DeletedReservedIdsRejectedAfterCheckpoint == true),
                recorded.Sum(item => item.DeletedReservation.ExpectedDeletedReservedIdCount),
                recorded.Sum(item => item.DeletedReservation.ActualDeletedReservedIdCount ?? 0));

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesAggregate CreateNoChangesAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded) =>
        recorded.Length == 0
            ? new ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesAggregate("notAvailable", 0, 0, 0, 0)
            : new ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesAggregate(
                "recorded",
                recorded.Length,
                recorded.Count(item => item.NoChanges.Status == "passed"),
                recorded.Count(item => item.NoChanges.GenerationUnchanged == false),
                recorded.Count(item => item.NoChanges.OutputDirectoryRemainedEmpty == false));

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary CreateRecursiveEligibilityAggregate(
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded)
    {
        int nonFalse = recorded.Sum(item => item.RecursiveEligibility.NonFalseEligibilityFlagCount);
        bool allReportsInspected = recorded.Length == cases.Length;
        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
            allReportsInspected ? "recorded" : "partial",
            LinkedReportInspected: allReportsInspected,
            nonFalse,
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            AllEligibilityFlagsFalse: allReportsInspected && nonFalse == 0);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixDesignInfo CreateDesign(
        string presetName,
        MatrixCase[] cases)
    {
        CaseDefinition[] definitions = GetCaseDefinitions(presetName);
        return new ExternalHnswBasePlusExactDeltaCheckpointMatrixDesignInfo(
            DatasetId,
            Dimension,
            VectorMetric.SquaredEuclidean.ToString(),
            QueryCount,
            WarmupQueries,
            CheckpointRuns,
            definitions.Select(item => item.TopK).Distinct().Order().ToArray(),
            definitions
                .Select(item => item.UpdateProfile)
                .Distinct()
                .Select(profile => new ExternalHnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo(
                    profile.Name,
                    ImmutableBaseStartRow: 0,
                    profile.BaseRowCount - 1,
                    profile.BaseRowCount,
                    DeltaStartRow: profile.BaseRowCount,
                    profile.BaseRowCount + profile.DeltaRowCount - 1,
                    profile.DeltaRowCount,
                    UnusedStartRow: profile.ExpectedPhysicalCandidateCount,
                    AdmittedBaseMatrixRowCount - 1,
                    AdmittedBaseMatrixRowCount - profile.ExpectedPhysicalCandidateCount,
                    profile.DeletedBaseCount,
                    profile.DeletedDeltaCount,
                    profile.ExpectedPhysicalCandidateCount,
                    profile.ExpectedLiveCount,
                    profile.ExpectedDeletedReservedIdCount,
                    profile.Description))
                .ToArray(),
            new ExternalHnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo(HnswProfileName, M, EfConstruction, EfSearch),
            FormatHex(MatrixSeed),
            FormatHex(HnswMatrixSeedBase),
            "per-case workload seed = matrix seed + zero-based case index",
            "per-case HNSW seed = HNSW matrix seed base + one-based case number",
            cases.Select(item => new ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseSeedInfo(
                Array.IndexOf(cases, item) + 1,
                item.CaseId,
                FormatHex(item.Options.Seed),
                FormatHex(item.Options.HnswSeed))).ToArray(),
            "Immutable base rows start at row 0, exact delta rows immediately follow base rows, unused rows remain outside the live candidate set and Fashion-MNIST row ordinals are external IDs.",
            "Linked VEC-138 reports compute exact updated truth in memory from the post-update live view and use it for pre-checkpoint, rebuilt and opened search validation.",
            "Each case writes checkpoint-output/checkpoint-run-001 and checkpoint-run-002 beneath its case directory.",
            "Checkpoint timing/allocation and search timings are copied from linked VEC-138 reports; output-byte scanning is outside checkpoint duration.",
            "Private/local external checkpoint matrix evidence only; no public API, package, public claim, memory, concurrency, filtering or competitor comparison evidence.");
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibility CreateEligibility(string presetName) =>
        new(
            "local-evidence",
            "private-raw",
            presetName,
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "Private external Fashion-MNIST HNSW base-plus-exact-delta checkpoint matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "No external mutable/update HNSW checkpoint matrix baseline-candidate policy is accepted.",
            "No external mutable/update HNSW checkpoint matrix comparison artifact is accepted.",
            "No accepted public comparison-summary policy exists for this private external checkpoint matrix.",
            "No external mutable/update HNSW checkpoint matrix regression-gate policy, threshold or hard gate is accepted.");

    private static CaseDefinition[] GetCaseDefinitions(string presetName)
    {
        string normalizedPresetName = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(presetName);
        CaseDefinition[] standard =
        [
            new(TopK: 10, UpdateProfiles[0]),
            new(TopK: 10, UpdateProfiles[1]),
            new(TopK: 100, UpdateProfiles[0]),
            new(TopK: 100, UpdateProfiles[1])
        ];

        return normalizedPresetName switch
        {
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.SmokePresetName => [standard[0]],
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.StandardPresetName => standard,
            _ => throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta checkpoint matrix preset '{presetName}'.")
        };
    }

    private static string CreateCaseId(int caseNumber, int topK, string updateProfileName) =>
        string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}-{topK}k-{updateProfileName}-{HnswProfileName}");

    private static bool AllMeasured(HnswBasePlusExactDeltaCheckpointPhaseSetInfo phases) =>
        phases.LiveSnapshot.Status == "Measured" &&
        phases.RebuildBuild.Status == "Measured" &&
        phases.Save.Status == "Measured" &&
        phases.OpenValidation.Status == "Measured" &&
        phases.Publication.Status == "Measured";

    private static int CountTrue(params bool[] values) => values.Count(static value => value);

    private static string GetManifestDirectory(string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(directory) ? "." : directory);
    }

    private static string CreateRelativePath(string manifestDirectory, string path) =>
        Path.GetRelativePath(manifestDirectory, Path.GetFullPath(path)).Replace('\\', '/');

    private static string CreateReportId(string? commit, string presetName)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName}-{commitPart}-{presetName}-{QueryCount}q-{CheckpointRuns}r-{MatrixSeed:X8}");
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    public sealed record MatrixCase(
        string CaseId,
        string UpdateProfileName,
        string HnswProfileName,
        string RelativeReportPath,
        string RelativeCheckpointDirectoryPath,
        ExternalCheckpointUpdateProfile UpdateProfile,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions Options);

    public sealed record ExternalCheckpointUpdateProfile(
        string Name,
        int BaseRowCount,
        int DeltaRowCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        int ExpectedPhysicalCandidateCount,
        int ExpectedLiveCount,
        int ExpectedDeletedReservedIdCount,
        string Description);

    private sealed record CaseDefinition(int TopK, ExternalCheckpointUpdateProfile UpdateProfile);
}
