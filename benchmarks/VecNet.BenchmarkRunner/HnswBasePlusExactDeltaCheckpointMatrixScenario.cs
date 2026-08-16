using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class HnswBasePlusExactDeltaCheckpointMatrixScenario
{
    private const string TaskId = "VEC-136";
    private const string SchemaName = "VecNet.HnswBasePlusExactDeltaCheckpointMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly CheckpointMatrixPresetCase[] SmokeCases =
    [
        new(Dimension: 32, TopK: 1, UpdateProfileName: "low-churn", InsertedDeltaCount: 4, DeletedBaseCount: 2, DeletedDeltaCount: 0, M: 4, EfConstruction: 16, EfSearch: 16),
        new(Dimension: 32, TopK: 10, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 8, DeletedBaseCount: 8, DeletedDeltaCount: 2, M: 4, EfConstruction: 32, EfSearch: 32)
    ];

    private static readonly CheckpointMatrixPresetCase[] StandardCases =
    [
        new(Dimension: 32, TopK: 1, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 32, TopK: 10, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 32, TopK: 10, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 32, TopK: 100, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 128, TopK: 1, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 128, TopK: 10, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 128, TopK: 10, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 128, TopK: 100, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 386, TopK: 1, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 386, TopK: 10, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 386, TopK: 10, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 386, TopK: 100, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 768, TopK: 1, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 768, TopK: 10, UpdateProfileName: "low-churn", InsertedDeltaCount: 32, DeletedBaseCount: 16, DeletedDeltaCount: 0, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 768, TopK: 10, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192),
        new(Dimension: 768, TopK: 100, UpdateProfileName: "tombstone-heavy", InsertedDeltaCount: 64, DeletedBaseCount: 96, DeletedDeltaCount: 32, M: 16, EfConstruction: 128, EfSearch: 192)
    ];

    public static HnswBasePlusExactDeltaCheckpointMatrixManifest Run(
        HnswBasePlusExactDeltaCheckpointMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = HnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(options.PresetName);
        MatrixCase[] cases = ExpandCases(options);
        var caseManifests = new HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;
        int blocked = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            MatrixCase matrixCase = cases[i];
            string[] caseArguments = CreateCaseArguments(matrixCase.Options);

            try
            {
                HnswBasePlusExactDeltaCheckpointBenchmarkReport report =
                    HnswBasePlusExactDeltaCheckpointScenario.Run(matrixCase.Options, caseArguments);
                HnswBasePlusExactDeltaCheckpointScenario.Write(report, matrixCase.Options.OutputPath);

                bool casePassed = string.Equals(report.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase);
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
                    report.Validation.Status,
                    errorMessage: null);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                blocked++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    matrixCase,
                    caseArguments,
                    report: null,
                    status: "blocked",
                    validationStatus: "blocked",
                    ex.Message);
            }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
            {
                failed++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    matrixCase,
                    caseArguments,
                    report: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        HnswBasePlusExactDeltaCheckpointMatrixAggregate aggregate =
            CreateAggregate(caseManifests, passed, failed, blocked);

        return new HnswBasePlusExactDeltaCheckpointMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            CreateDesign(presetName, options.Metric),
            caseManifests.Length,
            failed == 0 && blocked == 0 ? "passed" : "failed",
            caseManifests,
            aggregate,
            CreateEligibility(presetName),
            [
                "Private generated HNSW base-plus-exact-delta checkpoint matrix evidence only; not a public benchmark, baseline candidate, comparison artifact, regression gate or public mutable/update HNSW claim.",
                "Each case reuses the accepted VEC-134 generated-hnsw-base-plus-exact-delta-checkpoint report schema and writes one linked private per-case report.",
                "Checkpoint timing, allocation and phase diagnostics are owned by the linked VEC-134 reports; the matrix only orchestrates cases and summarizes linked evidence.",
                "Linked report paths are stored relative to the manifest directory. Checkpoint output paths are case-scoped and preserve VEC-134 checkpoint-run subdirectories.",
                "Generated finite squared-L2, inner-product or cosine data only; no external dataset, memory evidence, concurrency evidence, public documentation, package metadata, baseline candidate, comparison artifact or regression gate is introduced."
            ]);
    }

    public static MatrixCase[] ExpandCases(HnswBasePlusExactDeltaCheckpointMatrixOptions options)
    {
        CheckpointMatrixPresetCase[] presetCases = GetPresetCases(options.PresetName);
        string manifestDirectory = GetManifestDirectory(options.ManifestPath);
        var cases = new MatrixCase[presetCases.Length];

        for (int i = 0; i < presetCases.Length; i++)
        {
            CheckpointMatrixPresetCase presetCase = presetCases[i];
            uint dataSeed = unchecked(options.Seed + (uint)i);
            ulong hnswSeed = CreateHnswSeed(options.Seed, i);
            int caseRuns = IsStandard(options.PresetName) ? 2 : options.Runs;
            string caseId = CreateCaseId(i + 1, presetCase);
            string caseDirectory = Path.Combine(options.OutputDirectory, caseId);
            string reportPath = Path.Combine(caseDirectory, "checkpoint-report.json");
            string checkpointDirectory = Path.Combine(caseDirectory, "checkpoint-output");
            string relativeReportPath = CreateRelativePath(manifestDirectory, reportPath);
            string relativeCheckpointDirectory = CreateRelativePath(manifestDirectory, checkpointDirectory);
            var caseOptions = new HnswBasePlusExactDeltaCheckpointOptions(
                options.Metric,
                presetCase.Dimension,
                options.BaseVectorCount,
                options.QueryCount,
                presetCase.TopK,
                dataSeed,
                presetCase.InsertedDeltaCount,
                presetCase.DeletedBaseCount,
                presetCase.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                reportPath,
                checkpointDirectory,
                caseRuns,
                options.WarmupQueries,
                presetCase.M,
                presetCase.EfConstruction,
                presetCase.EfSearch,
                hnswSeed);

            cases[i] = new MatrixCase(
                caseId,
                "fixed-hnsw",
                presetCase.UpdateProfileName,
                relativeReportPath,
                relativeCheckpointDirectory,
                caseOptions);
        }

        return cases;
    }

    public static int GetMinimumBaseVectorCount(string presetName)
    {
        CheckpointMatrixPresetCase[] presetCases = GetPresetCases(presetName);
        int minimum = 1;
        foreach (CheckpointMatrixPresetCase presetCase in presetCases)
        {
            int minimumForLiveTopK = checked(presetCase.TopK - presetCase.InsertedDeltaCount + presetCase.DeletedBaseCount + presetCase.DeletedDeltaCount);
            minimum = Math.Max(minimum, Math.Max(presetCase.DeletedBaseCount, minimumForLiveTopK));
        }

        return minimum;
    }

    public static void WriteManifest(HnswBasePlusExactDeltaCheckpointMatrixManifest manifest, string manifestPath) =>
        ReportWriter.WriteJson(manifest, manifestPath);

    public static string[] CreateCaseArguments(HnswBasePlusExactDeltaCheckpointOptions options) =>
    [
        HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.BaseVectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--insertions", options.InsertedDeltaCount.ToString(CultureInfo.InvariantCulture),
        "--deletes", options.DeletedBaseCount.ToString(CultureInfo.InvariantCulture),
        "--delta-deletes", options.DeletedDeltaCount.ToString(CultureInfo.InvariantCulture),
        "--duplicate-inserts", options.DuplicateInsertAttempts.ToString(CultureInfo.InvariantCulture),
        "--unknown-deletes", options.UnknownDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--repeated-deletes", options.RepeatedDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--hnsw-seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
        "--output", options.OutputPath,
        "--checkpoint-directory", options.CheckpointDirectory
    ];

    private static HnswBasePlusExactDeltaCheckpointMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        MatrixCase matrixCase,
        string[] commandArguments,
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        HnswBasePlusExactDeltaCheckpointOptions options = matrixCase.Options;
        int expectedLiveVectorCount = options.LiveVectorCount;

        return new HnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.HnswProfileName,
            matrixCase.UpdateProfileName,
            options.Metric.ToString(),
            options.Dimension,
            options.BaseVectorCount,
            options.PhysicalVectorCount,
            expectedLiveVectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
            options.M,
            options.EfConstruction,
            options.EfSearch,
            options.InsertedDeltaCount,
            options.DeletedBaseCount,
            options.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            matrixCase.RelativeReportPath,
            matrixCase.RelativeCheckpointDirectoryPath,
            commandArguments,
            report?.ReportId,
            status,
            validationStatus,
            CreateValidationSummary(report),
            CreateRepeatedRunSummary(report),
            CreateCheckpointSummary(report),
            CreateSearchSummary(report?.Searches.PreCheckpointComposite),
            CreateSearchSummary(report?.Searches.PostCheckpointRebuiltComposite),
            CreateSearchSummary(report?.Searches.OpenedReadOnlyHnsw),
            CreateCountSummary(options, report),
            CreateEligibilitySummary(report),
            errorMessage);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixValidationSummary CreateValidationSummary(
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaCheckpointMatrixValidationSummary("notAvailable", null, null, null, null, null, null, null, null, null, null, null, null, null)
            : new HnswBasePlusExactDeltaCheckpointMatrixValidationSummary(
                report.Validation.Status,
                report.Validation.CheckpointResultStatusPublished,
                report.Validation.CheckpointResultCountsMatched,
                report.Validation.CheckpointGenerationAdvancedExactlyOnce,
                report.Validation.PhaseDiagnosticsMeasuredForPublishedCheckpoint,
                report.Validation.CheckpointRepeatedRunEvidencePresent,
                report.Validation.DetailedValidationRunNumber,
                report.Validation.DetailedValidationUsesFinalRun,
                report.Validation.OpenedReadOnlyHnswIdVectorValidationPassed,
                report.Validation.RebuiltCompositeOpenedHnswSearchParityPassed,
                report.Validation.ReturnedResultIntegrityPassedForAllSearches,
                report.Validation.NoChangesCheckpointProbePassed,
                report.Validation.DeletedReservedIdsRejectedAfterCheckpoint,
                report.Validation.OutputBytesScannedOutsideCheckpointDuration);

    private static HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary CreateRepeatedRunSummary(
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary("notAvailable", null, null, null, null, null, null, null, null)
            : new HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
                "recorded",
                report.CheckpointRuns.RunCount,
                report.CheckpointRuns.DetailedValidationRunNumber,
                report.CheckpointRuns.Aggregate.MeanElapsedMilliseconds,
                report.CheckpointRuns.Aggregate.MinElapsedMilliseconds,
                report.CheckpointRuns.Aggregate.MaxElapsedMilliseconds,
                report.CheckpointRuns.Aggregate.MeanManagedAllocatedBytes,
                report.CheckpointRuns.Aggregate.MinManagedAllocatedBytes,
                report.CheckpointRuns.Aggregate.MaxManagedAllocatedBytes);

    private static HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary CreateCheckpointSummary(
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary("notAvailable", null, null, null, null, null, null, null, null)
            : new HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary(
                report.Checkpoint.Status,
                report.Checkpoint.ElapsedMilliseconds,
                report.Checkpoint.ManagedAllocatedBytes,
                report.Checkpoint.GenerationBeforeCheckpoint,
                report.Checkpoint.GenerationAfterCheckpoint,
                report.Checkpoint.GenerationAdvancedExactlyOnce,
                report.Output.FileCount,
                report.Output.TotalBytes,
                report.Output.ScanTimingScope);

    private static HnswBasePlusExactDeltaCheckpointMatrixSearchSummary CreateSearchSummary(
        HnswBasePlusExactDeltaCheckpointSearchSectionInfo? section) =>
        section is null
            ? new HnswBasePlusExactDeltaCheckpointMatrixSearchSummary("notAvailable", null, null, null, null, null, null, null, null)
            : new HnswBasePlusExactDeltaCheckpointMatrixSearchSummary(
                "recorded",
                section.Metrics.RecallAtK,
                section.Metrics.OrderedAgreement,
                section.Metrics.ReturnedResultIntegrity.Status,
                section.Underfill.UnderfilledQueryCount,
                section.Underfill.UnderfilledSlotCount,
                section.Search.Aggregate.MeanQps,
                section.Search.Aggregate.MeanLatencyP95Milliseconds,
                section.Search.Aggregate.MeanManagedAllocatedBytesPerQuery);

    private static HnswBasePlusExactDeltaCheckpointMatrixCountSummary CreateCountSummary(
        HnswBasePlusExactDeltaCheckpointOptions options,
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswBasePlusExactDeltaCheckpointMatrixCountSummary(
                "notAvailable",
                options.BaseVectorCount,
                options.PhysicalVectorCount,
                options.LiveVectorCount,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null);
        }

        return new HnswBasePlusExactDeltaCheckpointMatrixCountSummary(
            "recorded",
            options.BaseVectorCount,
            options.PhysicalVectorCount,
            options.LiveVectorCount,
            report.PreCheckpointCounts.LiveVectorCount,
            report.PreCheckpointCounts.TombstoneCount,
            report.PostCheckpointCounts.BasePhysicalVectorCount,
            report.PostCheckpointCounts.LiveVectorCount,
            report.PostCheckpointCounts.TombstoneCount,
            report.PostCheckpointCounts.DeletedReservedIdCount,
            report.PreCheckpointCounts.TombstoneRatio,
            report.PreCheckpointCounts.DeltaInsertRatio);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary CreateEligibilitySummary(
        HnswBasePlusExactDeltaCheckpointBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
                "notAvailable",
                LinkedReportInspected: false,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                AllEligibilityFlagsFalse: false);
        }

        bool publicClaimEligible =
            report.Evidence.PublicClaimEligible ||
            report.Validation.PublicClaimEligible ||
            report.Eligibility.PublicClaimEligible;
        bool baselineCandidateEligible =
            report.Evidence.BaselineCandidateEligible ||
            report.Validation.BaselineCandidateEligible ||
            report.Eligibility.BaselineCandidateEligible;
        bool comparisonArtifactEligible = report.Validation.ComparisonArtifactEligible;
        bool regressionGateEligible =
            report.Evidence.RegressionGateEligible ||
            report.Validation.RegressionGateEligible ||
            report.Eligibility.RegressionGateEligible;

        return new HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
            "recorded",
            LinkedReportInspected: true,
            publicClaimEligible,
            baselineCandidateEligible,
            comparisonArtifactEligible,
            regressionGateEligible,
            AllEligibilityFlagsFalse: !publicClaimEligible && !baselineCandidateEligible && !comparisonArtifactEligible && !regressionGateEligible);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixAggregate CreateAggregate(
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded =
            cases.Where(item => item.LinkedReportId is not null).ToArray();
        HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary recursiveEligibility =
            CreateAggregateEligibility(recorded, cases.Length);

        return new HnswBasePlusExactDeltaCheckpointMatrixAggregate(
            passed,
            failed,
            SkippedCaseCount: 0,
            blocked,
            recorded.Length,
            recorded.Sum(item => item.RepeatedCheckpointRuns.RunCount ?? 0),
            recorded.Count(item => string.Equals(item.ValidationSummary.Status, "passed", StringComparison.OrdinalIgnoreCase)),
            recorded.Count(item => item.ValidationSummary.CheckpointRepeatedRunEvidencePresent == true),
            CreateAggregateSearch(recorded.Select(item => item.PreCheckpointSearch).ToArray()),
            CreateAggregateSearch(recorded.Select(item => item.PostCheckpointSearch).ToArray()),
            CreateAggregateSearch(recorded.Select(item => item.OpenedReadOnlySearch).ToArray()),
            CreateAggregateCheckpoint(recorded),
            recursiveEligibility);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary CreateAggregateSearch(
        HnswBasePlusExactDeltaCheckpointMatrixSearchSummary[] summaries)
    {
        HnswBasePlusExactDeltaCheckpointMatrixSearchSummary[] recorded =
            summaries.Where(item => string.Equals(item.Status, "recorded", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (recorded.Length == 0)
        {
            return new HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary("notAvailable", 0, null, null, null, null, null, null, null);
        }

        return new HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary(
            "recorded",
            recorded.Length,
            recorded.Min(item => item.RecallAtK),
            recorded.Max(item => item.RecallAtK),
            recorded.Min(item => item.OrderedAgreement),
            recorded.Max(item => item.OrderedAgreement),
            recorded.Sum(item => item.UnderfilledQueryCount ?? 0),
            recorded.Sum(item => item.UnderfilledSlotCount ?? 0),
            recorded.Max(item => item.MeanManagedAllocatedBytesPerQuery));
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixAggregateCheckpointSummary CreateAggregateCheckpoint(
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new HnswBasePlusExactDeltaCheckpointMatrixAggregateCheckpointSummary("notAvailable", 0, 0, null, null, null, null, null);
        }

        return new HnswBasePlusExactDeltaCheckpointMatrixAggregateCheckpointSummary(
            "recorded",
            recorded.Length,
            recorded.Count(item => string.Equals(item.CheckpointSummary.Status, "Published", StringComparison.Ordinal)),
            recorded.Average(item => item.CheckpointSummary.FinalRunElapsedMilliseconds ?? 0),
            recorded.Max(item => item.CheckpointSummary.FinalRunElapsedMilliseconds ?? 0),
            recorded.Average(item => item.CheckpointSummary.FinalRunManagedAllocatedBytes ?? 0),
            recorded.Max(item => item.CheckpointSummary.FinalRunManagedAllocatedBytes ?? 0),
            recorded.Sum(item => item.CheckpointSummary.OutputTotalBytes ?? 0));
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary CreateAggregateEligibility(
        HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] recorded,
        int caseCount)
    {
        bool publicClaimEligible = recorded.Any(item => item.RecursiveEligibility.PublicClaimEligible);
        bool baselineCandidateEligible = recorded.Any(item => item.RecursiveEligibility.BaselineCandidateEligible);
        bool comparisonArtifactEligible = recorded.Any(item => item.RecursiveEligibility.ComparisonArtifactEligible);
        bool regressionGateEligible = recorded.Any(item => item.RecursiveEligibility.RegressionGateEligible);
        bool allReportsInspected = recorded.Length == caseCount;

        return new HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
            allReportsInspected ? "recorded" : "partial",
            LinkedReportInspected: allReportsInspected,
            publicClaimEligible,
            baselineCandidateEligible,
            comparisonArtifactEligible,
            regressionGateEligible,
            AllEligibilityFlagsFalse: allReportsInspected && !publicClaimEligible && !baselineCandidateEligible && !comparisonArtifactEligible && !regressionGateEligible);
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixDesignInfo CreateDesign(string presetName, VectorMetric metric)
    {
        CheckpointMatrixPresetCase[] presetCases = GetPresetCases(presetName);
        return new HnswBasePlusExactDeltaCheckpointMatrixDesignInfo(
            metric.ToString(),
            presetCases.Select(item => item.Dimension).Distinct().Order().ToArray(),
            presetCases.Select(item => item.TopK).Distinct().Order().ToArray(),
            presetCases
                .Select(item => new HnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo("fixed-hnsw", item.M, item.EfConstruction, item.EfSearch))
                .Distinct()
                .ToArray(),
            presetCases
                .GroupBy(item => item.UpdateProfileName)
                .Select(group =>
                {
                    CheckpointMatrixPresetCase item = group.First();
                    return new HnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo(
                        item.UpdateProfileName,
                        item.InsertedDeltaCount,
                        item.DeletedBaseCount,
                        item.DeletedDeltaCount,
                        DescribeUpdateProfile(item.UpdateProfileName));
                })
                .OrderBy(item => item.Name, StringComparer.Ordinal)
                .ToArray(),
            "Generated finite squared-L2, inner-product or cosine base vectors, delta vectors and queries only; exact updated truth and checkpoint measurements are produced by linked VEC-134 reports.",
            IsStandard(presetName)
                ? "Standard is exactly 16 bounded generated cases covering dimensions 32, 128, 386 and 768; top-k 1, 10 and 100; low-churn and tombstone-heavy profiles; fixed M=16, efConstruction=128, efSearch=192; and two checkpoint runs per case."
                : "Smoke is exactly two bounded generated cases for quick local validation.",
            "Internal checkpoint matrix evidence only; no public mutable/update HNSW API, external dataset, actual/peak memory evidence, concurrency evidence, baseline candidate, comparison artifact, regression gate, package metadata or public claim.");
    }

    private static HnswBasePlusExactDeltaCheckpointMatrixEligibility CreateEligibility(string presetName) =>
        new(
            "local-evidence",
            "private-raw",
            IsStandard(presetName) ? "standard" : "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW base-plus-exact-delta checkpoint matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "No generated mutable/update HNSW checkpoint matrix baseline-candidate policy is accepted.",
            "No generated mutable/update HNSW checkpoint matrix comparison artifact is accepted.",
            "No generated mutable/update HNSW checkpoint matrix regression-gate policy, threshold or hard gate is accepted.");

    private static CheckpointMatrixPresetCase[] GetPresetCases(string presetName)
    {
        string normalizedPresetName = HnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            HnswBasePlusExactDeltaCheckpointMatrixOptions.SmokePresetName => SmokeCases,
            HnswBasePlusExactDeltaCheckpointMatrixOptions.StandardPresetName => StandardCases,
            _ => throw new ArgumentException($"Unsupported HNSW base-plus-exact-delta checkpoint matrix preset '{presetName}'.")
        };
    }

    private static bool IsStandard(string presetName) =>
        string.Equals(
            HnswBasePlusExactDeltaCheckpointMatrixOptions.NormalizePresetName(presetName),
            HnswBasePlusExactDeltaCheckpointMatrixOptions.StandardPresetName,
            StringComparison.Ordinal);

    private static string CreateCaseId(int caseNumber, CheckpointMatrixPresetCase presetCase) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"case-{caseNumber:D3}-{presetCase.UpdateProfileName}-{presetCase.Dimension}d-{presetCase.TopK}k");

    private static ulong CreateHnswSeed(uint baseSeed, int caseIndex) =>
        0x484E5357_00013600UL ^ ((ulong)baseSeed << 16) ^ (uint)(caseIndex + 1);

    private static string GetManifestDirectory(string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(directory) ? "." : directory);
    }

    private static string CreateRelativePath(string manifestDirectory, string path) =>
        Path.GetRelativePath(manifestDirectory, Path.GetFullPath(path)).Replace('\\', '/');

    private static string DescribeUpdateProfile(string updateProfileName) =>
        string.Equals(updateProfileName, "low-churn", StringComparison.Ordinal)
            ? "moderate exact delta with light base tombstones and no delta tombstones"
            : "heavier base tombstones plus delta tombstones";

    public sealed record MatrixCase(
        string CaseId,
        string HnswProfileName,
        string UpdateProfileName,
        string RelativeReportPath,
        string RelativeCheckpointDirectoryPath,
        HnswBasePlusExactDeltaCheckpointOptions Options);

    private sealed record CheckpointMatrixPresetCase(
        int Dimension,
        int TopK,
        string UpdateProfileName,
        int InsertedDeltaCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        int M,
        int EfConstruction,
        int EfSearch);
}
