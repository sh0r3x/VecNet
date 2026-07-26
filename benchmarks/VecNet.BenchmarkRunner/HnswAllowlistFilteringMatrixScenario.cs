using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class HnswAllowlistFilteringMatrixScenario
{
    private const string TaskId = "VEC-150";
    private const string SchemaName = "VecNet.HnswAllowlistFilteringMatrixManifest";
    private const string SchemaVersion = "0.1";
    private const ulong HnswMatrixSeedBase = 0x484E535700014800UL;

    private static readonly AllowlistMatrixPresetCase[] SmokeCases =
    [
        new("smoke-empty-k10", "empty", "low-churn", "fallback underfill", Dimension: 32, BaseVectorCount: 512, InsertedDeltaCount: 64, DeletedBaseCount: 8, DeletedDeltaCount: 2, TopK: 10, M: 8, EfConstruction: 64, EfSearch: 64),
        new("smoke-fallback-boundary-k10", "fallback-boundary", "low-churn", "exact efSearch boundary", Dimension: 32, BaseVectorCount: 512, InsertedDeltaCount: 64, DeletedBaseCount: 8, DeletedDeltaCount: 2, TopK: 10, M: 8, EfConstruction: 64, EfSearch: 64),
        new("smoke-broad-k10", "broad", "low-churn", "broad emission integrity", Dimension: 32, BaseVectorCount: 512, InsertedDeltaCount: 64, DeletedBaseCount: 8, DeletedDeltaCount: 2, TopK: 10, M: 8, EfConstruction: 64, EfSearch: 64),
        new("smoke-broad-tombstone-heavy-k10", "broad", "tombstone-heavy", "broad emission with heavier tombstones", Dimension: 96, BaseVectorCount: 1024, InsertedDeltaCount: 128, DeletedBaseCount: 128, DeletedDeltaCount: 48, TopK: 10, M: 8, EfConstruction: 64, EfSearch: 64)
    ];

    private static readonly AllowlistMatrixPresetCase[] StandardCases = CreateStandardCases();

    public static HnswAllowlistFilteringMatrixManifest Run(
        HnswAllowlistFilteringMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = HnswAllowlistFilteringMatrixOptions.NormalizePresetName(options.PresetName);
        MatrixCase[] cases = ExpandCases(options);
        var caseManifests = new HnswAllowlistFilteringMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;
        int blocked = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            MatrixCase matrixCase = cases[i];
            string[] caseArguments = CreateCaseArguments(matrixCase.Options);

            try
            {
                HnswAllowlistFilteringBenchmarkReport report =
                    HnswAllowlistFilteringScenario.Run(matrixCase.Options, caseArguments);
                HnswAllowlistFilteringScenario.Write(report, matrixCase.Options.OutputPath);

                HnswAllowlistFilteringMatrixEligibilitySummary recursiveEligibility = CreateEligibilitySummary(report);
                bool casePassed =
                    string.Equals(report.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) &&
                    recursiveEligibility.AllEligibilityFlagsFalse;
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
                    casePassed ? report.Validation.Status : "failed",
                    recursiveEligibility,
                    errorMessage: recursiveEligibility.AllEligibilityFlagsFalse ? null : "linked report contains public/baseline/comparison/regression eligibility set to true");
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
                    CreateEligibilitySummary(report: null),
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
                    CreateEligibilitySummary(report: null),
                    ex.Message);
            }
        }

        HnswAllowlistFilteringMatrixAggregate aggregate =
            CreateAggregate(caseManifests, passed, failed, blocked);
        string validationStatus =
            failed == 0 && blocked == 0 && aggregate.RecursiveEligibility.AllEligibilityFlagsFalse ? "passed" : "failed";

        return new HnswAllowlistFilteringMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            HnswAllowlistFilteringMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswAllowlistFilteringMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            CreateDesign(presetName, options.Seed, options.Metric),
            caseManifests.Length,
            validationStatus,
            caseManifests,
            aggregate,
            CreateEligibility(presetName),
            [
                "Private generated HNSW allowlist filtering matrix evidence only; not a public benchmark, public claim, baseline candidate, comparison artifact or regression gate.",
                "Each successful case reuses the accepted VEC-149 VecNet.HnswAllowlistFilteringBenchmarkReport 0.1 report schema without changing measured search-call semantics.",
                "Linked report, opened-index and checkpoint paths are stored relative to the manifest directory.",
                "The matrix summarizes branch coverage, exact fallback parity, broad emission recall/order, underfill, allowlist metadata, tombstones, mutation/counts, returned-result integrity and allocation maxima from linked reports.",
                "Generated finite squared-L2 data only; no Fashion-MNIST, external dataset, public docs, package metadata, production dependency or public filtering claim is introduced."
            ]);
    }

    public static MatrixCase[] ExpandCases(HnswAllowlistFilteringMatrixOptions options)
    {
        AllowlistMatrixPresetCase[] presetCases = GetPresetCases(options.PresetName);
        string manifestDirectory = GetManifestDirectory(options.ManifestPath);
        var cases = new MatrixCase[presetCases.Length];

        for (int i = 0; i < presetCases.Length; i++)
        {
            AllowlistMatrixPresetCase presetCase = presetCases[i];
            uint dataSeed = unchecked(options.Seed + (uint)i);
            ulong hnswSeed = unchecked(HnswMatrixSeedBase + (ulong)i + 1);
            string caseDirectory = Path.Combine(options.OutputDirectory, presetCase.CaseId);
            string reportPath = Path.Combine(caseDirectory, "allowlist-filtered-report.json");
            string openedIndexDirectory = Path.Combine(caseDirectory, "opened-index");
            string checkpointDirectory = Path.Combine(caseDirectory, "checkpoint-output");
            var caseOptions = new HnswAllowlistFilteringOptions(
                options.Metric,
                presetCase.Dimension,
                presetCase.BaseVectorCount,
                options.QueryCount,
                presetCase.TopK,
                dataSeed,
                presetCase.InsertedDeltaCount,
                presetCase.DeletedBaseCount,
                presetCase.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                presetCase.FilterProfile,
                reportPath,
                openedIndexDirectory,
                checkpointDirectory,
                options.Runs,
                options.WarmupQueries,
                presetCase.M,
                presetCase.EfConstruction,
                presetCase.EfSearch,
                hnswSeed);

            cases[i] = new MatrixCase(
                presetCase.CaseId,
                presetCase.FilterProfile,
                presetCase.UpdateProfileName,
                presetCase.BranchFocus,
                CreateRelativePath(manifestDirectory, reportPath),
                CreateRelativePath(manifestDirectory, openedIndexDirectory),
                CreateRelativePath(manifestDirectory, checkpointDirectory),
                caseOptions);
        }

        return cases;
    }

    public static void WriteManifest(HnswAllowlistFilteringMatrixManifest manifest, string manifestPath) =>
        ReportWriter.WriteJson(manifest, manifestPath);

    public static string[] CreateCaseArguments(HnswAllowlistFilteringOptions options) =>
    [
        HnswAllowlistFilteringOptions.ScenarioName,
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
        "--filter", options.FilterProfile,
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--hnsw-seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
        "--output", options.OutputPath,
        "--opened-index-directory", options.OpenedIndexDirectory,
        "--checkpoint-directory", options.CheckpointDirectory
    ];

    private static HnswAllowlistFilteringMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        MatrixCase matrixCase,
        string[] commandArguments,
        HnswAllowlistFilteringBenchmarkReport? report,
        string status,
        string validationStatus,
        HnswAllowlistFilteringMatrixEligibilitySummary recursiveEligibility,
        string? errorMessage)
    {
        HnswAllowlistFilteringOptions options = matrixCase.Options;

        return new HnswAllowlistFilteringMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.FilterProfile,
            matrixCase.UpdateProfileName,
            matrixCase.BranchFocus,
            options.Metric.ToString(),
            options.Dimension,
            options.BaseVectorCount,
            options.PhysicalVectorCount,
            options.LiveVectorCount,
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
            matrixCase.RelativeOpenedIndexDirectoryPath,
            matrixCase.RelativeCheckpointDirectoryPath,
            commandArguments,
            report?.ReportId,
            status,
            validationStatus,
            CreateBranchSummary(options, report),
            CreateFallbackSummary(report),
            CreateBroadSummary(report),
            CreateUnderfillSummary(options, report),
            CreateAllowlistSummary(report),
            CreateTombstoneSummary(report),
            CreateDeltaScanSummary(report),
            CreateMutationSummary(options, report),
            CreateCountSummary(options, report),
            CreateIntegritySummary(report),
            CreateAllocationSummary(report),
            recursiveEligibility,
            errorMessage);
    }

    private static HnswAllowlistFilteringMatrixBranchSummary CreateBranchSummary(
        HnswAllowlistFilteringOptions options,
        HnswAllowlistFilteringBenchmarkReport? report) =>
        report is null
            ? new HnswAllowlistFilteringMatrixBranchSummary("notAvailable", 0, 0, options.EfSearch, null, null, null)
            : new HnswAllowlistFilteringMatrixBranchSummary(
                "recorded",
                report.Branches.ExactFallbackQueryCount,
                report.Branches.BroadEmissionQueryCount,
                report.Branches.BranchThresholdEfSearch,
                report.Branches.ExpectedBranch,
                report.Branches.BranchConsistencyStatus,
                report.Branches.BranchMismatchCount);

    private static HnswAllowlistFilteringMatrixFallbackSummary CreateFallbackSummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixFallbackSummary("notAvailable", 0, null, null, null, null);
        }

        HnswAllowlistSearchSectionInfo[] sections = GetSections(report);
        bool applicable = report.Branches.ExactFallbackQueryCount > 0;
        return new HnswAllowlistFilteringMatrixFallbackSummary(
            applicable ? "recorded" : "notApplicable",
            report.Branches.ExactFallbackQueryCount,
            applicable ? sections.All(section => string.Equals(section.ExactFallbackValidation.Status, "passed", StringComparison.OrdinalIgnoreCase)) : null,
            applicable ? sections.Sum(section => section.ExactFallbackValidation.CountMismatchCount) : null,
            applicable ? sections.Sum(section => section.ExactFallbackValidation.IdOrOrderMismatchCount) : null,
            applicable ? sections.Sum(section => section.ExactFallbackValidation.DistanceMismatchCount) : null);
    }

    private static HnswAllowlistFilteringMatrixBroadEmissionSummary CreateBroadSummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixBroadEmissionSummary("notAvailable", 0, null, null, null, null, null, null, null);
        }

        HnswAllowlistBroadEmissionValidationInfo[] broad =
            GetSections(report).Select(section => section.BroadEmissionValidation).ToArray();
        HnswAllowlistBroadEmissionValidationInfo[] recorded =
            broad.Where(item => string.Equals(item.Status, "passed", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (recorded.Length == 0)
        {
            return new HnswAllowlistFilteringMatrixBroadEmissionSummary(
                "notApplicable",
                report.Branches.BroadEmissionQueryCount,
                null,
                null,
                null,
                null,
                null,
                null,
                null);
        }

        return new HnswAllowlistFilteringMatrixBroadEmissionSummary(
            "recorded",
            report.Branches.BroadEmissionQueryCount,
            recorded.Min(item => item.RecallAtK),
            recorded.Max(item => item.RecallAtK),
            recorded.Min(item => item.OrderedAgreement),
            recorded.Max(item => item.OrderedAgreement),
            recorded.Sum(item => item.MissingResultCount),
            recorded.Sum(item => item.ExtraResultCount),
            recorded.Sum(item => item.DistanceMismatchCount));
    }

    private static HnswAllowlistFilteringMatrixUnderfillSummary CreateUnderfillSummary(
        HnswAllowlistFilteringOptions options,
        HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixUnderfillSummary(
                "notAvailable",
                options.QueryCount,
                options.TopK,
                checked(options.QueryCount * options.TopK),
                null,
                null,
                null,
                null);
        }

        HnswAllowlistUnderfillInfo[] underfills = GetSections(report).Select(section => section.Underfill).ToArray();
        return new HnswAllowlistFilteringMatrixUnderfillSummary(
            "recorded",
            report.Workload.QueryCount,
            report.Workload.TopK,
            underfills.Sum(item => item.TotalRequestedResultSlots),
            underfills.Sum(item => item.TotalReturnedResults),
            underfills.Sum(item => item.TotalExactTruthAvailableResults),
            underfills.Sum(item => item.UnderfilledQueryCount),
            underfills.Sum(item => item.UnderfilledSlotCount));
    }

    private static HnswAllowlistFilteringMatrixAllowlistSummary CreateAllowlistSummary(HnswAllowlistFilteringBenchmarkReport? report) =>
        report is null
            ? new HnswAllowlistFilteringMatrixAllowlistSummary("notAvailable", null, null, null, null, null, null, null, null, null, null, null, null, null)
            : new HnswAllowlistFilteringMatrixAllowlistSummary(
                "recorded",
                report.Allowlist.Profile,
                report.Allowlist.InputIdCountPerQuery,
                report.Allowlist.DistinctInputIdCountPerQuery,
                report.Allowlist.KnownIdCountPerQuery,
                report.Allowlist.UnknownIdCountPerQuery,
                report.Allowlist.DuplicateInputIdCountPerQuery,
                report.Allowlist.TombstonedInputIdCountPerQuery,
                report.Allowlist.KnownLiveAllowedCountPerQuery,
                report.Allowlist.LiveBaseAllowedCountPerQuery,
                report.Allowlist.LiveDeltaAllowedCountPerQuery,
                report.Allowlist.KnownLiveAllowedMin,
                report.Allowlist.KnownLiveAllowedMean,
                report.Allowlist.KnownLiveAllowedMax);

    private static HnswAllowlistFilteringMatrixTombstoneSummary CreateTombstoneSummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixTombstoneSummary("notAvailable", null, null, null, null, null, null, null);
        }

        HnswAllowlistSearchSectionInfo[] sections = GetSections(report);
        return new HnswAllowlistFilteringMatrixTombstoneSummary(
            "recorded",
            report.PreCheckpointCounts.BaseTombstoneCount,
            report.PreCheckpointCounts.DeltaTombstoneCount,
            report.PreCheckpointCounts.TombstoneCount,
            report.Allowlist.TombstonedInputIdCountPerQuery,
            sections.Sum(section => section.TombstoneSuppression.ReturnedBaseTombstoneCount),
            sections.Sum(section => section.TombstoneSuppression.ReturnedDeltaTombstoneCount),
            sections.All(section => string.Equals(section.TombstoneSuppression.Status, "passed", StringComparison.OrdinalIgnoreCase)));
    }

    private static HnswAllowlistFilteringMatrixDeltaScanSummary CreateDeltaScanSummary(HnswAllowlistFilteringBenchmarkReport? report) =>
        report is null
            ? new HnswAllowlistFilteringMatrixDeltaScanSummary("notAvailable", null, null, null, null, null, null)
            : new HnswAllowlistFilteringMatrixDeltaScanSummary(
                "recorded",
                report.Searches.SourceComposite.ExactFilteredDeltaScan.LiveDeltaScannedCountPerQuery,
                report.Searches.SourceComposite.ExactFilteredDeltaScan.AllowedLiveDeltaCountPerQuery,
                report.Searches.SourceComposite.ExactFilteredDeltaScan.TotalEmittedDeltaResultCount,
                report.Searches.RebuiltComposite.ExactFilteredDeltaScan.LiveDeltaScannedCountPerQuery,
                report.Searches.RebuiltComposite.ExactFilteredDeltaScan.AllowedLiveDeltaCountPerQuery,
                report.Searches.RebuiltComposite.ExactFilteredDeltaScan.TotalEmittedDeltaResultCount);

    private static HnswAllowlistFilteringMatrixMutationSummary CreateMutationSummary(
        HnswAllowlistFilteringOptions options,
        HnswAllowlistFilteringBenchmarkReport? report) =>
        report is null
            ? new HnswAllowlistFilteringMatrixMutationSummary(
                "notAvailable",
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                null,
                null,
                null)
            : new HnswAllowlistFilteringMatrixMutationSummary(
                "recorded",
                report.Mutations.InsertedCount,
                report.Mutations.DeletedBaseCount,
                report.Mutations.DeletedDeltaCount,
                report.Mutations.DuplicateInsertAttempts,
                report.Mutations.UnknownDeleteAttempts,
                report.Mutations.RepeatedDeleteAttempts,
                report.Mutations.CommittedMutationCount,
                report.Mutations.GenerationDeltaMatchesCommittedMutations,
                report.Mutations.GenerationAfterMutations);

    private static HnswAllowlistFilteringMatrixCountSummary CreateCountSummary(
        HnswAllowlistFilteringOptions options,
        HnswAllowlistFilteringBenchmarkReport? report) =>
        report is null
            ? new HnswAllowlistFilteringMatrixCountSummary(
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
                null)
            : new HnswAllowlistFilteringMatrixCountSummary(
                "recorded",
                report.PreCheckpointCounts.BasePhysicalVectorCount,
                checked(report.PreCheckpointCounts.BasePhysicalVectorCount + report.PreCheckpointCounts.DeltaPhysicalVectorCount),
                options.LiveVectorCount,
                report.PreCheckpointCounts.LiveVectorCount,
                report.PreCheckpointCounts.TombstoneCount,
                report.PostCheckpointCounts.LiveVectorCount,
                report.PostCheckpointCounts.TombstoneCount,
                report.PostCheckpointCounts.DeletedReservedIdCount,
                report.PreCheckpointCounts.TombstoneRatio,
                report.PreCheckpointCounts.DeltaInsertRatio);

    private static HnswAllowlistFilteringMatrixIntegritySummary CreateIntegritySummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixIntegritySummary("notAvailable", null, null, null, null, null, null, null, null);
        }

        HnswAllowlistReturnedResultIntegrityInfo[] integrity =
            GetSections(report).Select(section => section.ReturnedResultIntegrity).ToArray();
        return new HnswAllowlistFilteringMatrixIntegritySummary(
            "recorded",
            integrity.All(item => string.Equals(item.Status, "passed", StringComparison.OrdinalIgnoreCase)),
            integrity.Sum(item => item.CheckedResultCount),
            integrity.Sum(item => item.UnknownIdCount),
            integrity.Sum(item => item.TombstonedIdCount),
            integrity.Sum(item => item.NotAllowedIdCount),
            integrity.Sum(item => item.DuplicateIdCount),
            integrity.Sum(item => item.NonFiniteDistanceCount),
            integrity.Sum(item => item.DistanceMismatchCount));
    }

    private static HnswAllowlistFilteringMatrixAllocationSummary CreateAllocationSummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixAllocationSummary("notAvailable", null, null, null);
        }

        AggregateTimingInfo[] aggregates = GetSections(report).Select(section => section.Search.Aggregate).ToArray();
        return new HnswAllowlistFilteringMatrixAllocationSummary(
            "recorded",
            aggregates.Max(item => item.MeanManagedAllocatedBytesPerQuery),
            aggregates.Max(item => item.MaxManagedAllocatedBytesPerQuery),
            aggregates.Max(item => item.MaxManagedAllocatedBytes));
    }

    private static HnswAllowlistFilteringMatrixEligibilitySummary CreateEligibilitySummary(HnswAllowlistFilteringBenchmarkReport? report)
    {
        if (report is null)
        {
            return new HnswAllowlistFilteringMatrixEligibilitySummary(
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
        bool comparisonArtifactEligible =
            report.Validation.ComparisonArtifactEligible ||
            report.Eligibility.ComparisonArtifactEligible;
        bool regressionGateEligible =
            report.Evidence.RegressionGateEligible ||
            report.Validation.RegressionGateEligible ||
            report.Eligibility.RegressionGateEligible;

        return new HnswAllowlistFilteringMatrixEligibilitySummary(
            "recorded",
            LinkedReportInspected: true,
            publicClaimEligible,
            baselineCandidateEligible,
            comparisonArtifactEligible,
            regressionGateEligible,
            AllEligibilityFlagsFalse: !publicClaimEligible && !baselineCandidateEligible && !comparisonArtifactEligible && !regressionGateEligible);
    }

    private static HnswAllowlistFilteringMatrixAggregate CreateAggregate(
        HnswAllowlistFilteringMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        HnswAllowlistFilteringMatrixCaseManifest[] recorded =
            cases.Where(item => item.LinkedReportId is not null).ToArray();
        return new HnswAllowlistFilteringMatrixAggregate(
            passed,
            failed,
            SkippedCaseCount: 0,
            blocked,
            recorded.Length,
            CreateAggregateBranchCoverage(recorded),
            CreateAggregateFallback(recorded),
            CreateAggregateBroad(recorded),
            CreateAggregateUnderfill(recorded),
            CreateAggregateAllowlist(recorded),
            CreateAggregateMutationCounts(recorded),
            CreateAggregateIntegrity(recorded),
            CreateAggregateAllocations(recorded),
            CreateAggregateEligibility(recorded, cases.Length));
    }

    private static HnswAllowlistFilteringMatrixAggregateBranchCoverage CreateAggregateBranchCoverage(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new HnswAllowlistFilteringMatrixAggregateBranchCoverage("notAvailable", 0, 0, 0, 0, []);
        }

        return new HnswAllowlistFilteringMatrixAggregateBranchCoverage(
            "recorded",
            recorded.Count(item => item.BranchSummary.ExactFallbackQueryCount > 0 && item.BranchSummary.BroadEmissionQueryCount == 0),
            recorded.Count(item => item.BranchSummary.BroadEmissionQueryCount > 0 && item.BranchSummary.ExactFallbackQueryCount == 0),
            recorded.Count(item => item.BranchSummary.ExactFallbackQueryCount > 0 && item.BranchSummary.BroadEmissionQueryCount > 0),
            recorded.Count(item => item.BranchSummary.BranchMismatchCount > 0),
            recorded.Select(item => item.FilterProfile).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray());
    }

    private static HnswAllowlistFilteringMatrixAggregateFallbackSummary CreateAggregateFallback(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        HnswAllowlistFilteringMatrixCaseManifest[] fallback =
            recorded.Where(item => string.Equals(item.ExactFallbackParity.Status, "recorded", StringComparison.OrdinalIgnoreCase)).ToArray();
        return new HnswAllowlistFilteringMatrixAggregateFallbackSummary(
            fallback.Length == 0 ? "notApplicable" : "recorded",
            fallback.Length,
            fallback.Count(item => item.ExactFallbackParity.AllSearchesPassed == true),
            fallback.Sum(item => item.ExactFallbackParity.CountMismatchCount ?? 0),
            fallback.Sum(item => item.ExactFallbackParity.IdOrOrderMismatchCount ?? 0),
            fallback.Sum(item => item.ExactFallbackParity.DistanceMismatchCount ?? 0));
    }

    private static HnswAllowlistFilteringMatrixAggregateBroadEmissionSummary CreateAggregateBroad(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        HnswAllowlistFilteringMatrixBroadEmissionSummary[] broad =
            recorded.Select(item => item.BroadEmission)
                .Where(item => string.Equals(item.Status, "recorded", StringComparison.OrdinalIgnoreCase))
                .ToArray();
        return new HnswAllowlistFilteringMatrixAggregateBroadEmissionSummary(
            broad.Length == 0 ? "notApplicable" : "recorded",
            broad.Length,
            broad.Length == 0 ? null : broad.Min(item => item.MinRecallAtK),
            broad.Length == 0 ? null : broad.Max(item => item.MaxRecallAtK),
            broad.Length == 0 ? null : broad.Min(item => item.MinOrderedAgreement),
            broad.Length == 0 ? null : broad.Max(item => item.MaxOrderedAgreement),
            broad.Sum(item => item.MissingResultCount ?? 0),
            broad.Sum(item => item.ExtraResultCount ?? 0),
            broad.Sum(item => item.DistanceMismatchCount ?? 0));
    }

    private static HnswAllowlistFilteringMatrixAggregateUnderfillSummary CreateAggregateUnderfill(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded) =>
        new(
            recorded.Length == 0 ? "notAvailable" : "recorded",
            recorded.Length,
            recorded.Sum(item => item.Underfill.TotalRequestedResultSlots),
            recorded.Sum(item => item.Underfill.TotalReturnedResults ?? 0),
            recorded.Sum(item => item.Underfill.TotalExactTruthAvailableResults ?? 0),
            recorded.Sum(item => item.Underfill.UnderfilledQueryCount ?? 0),
            recorded.Sum(item => item.Underfill.UnderfilledSlotCount ?? 0));

    private static HnswAllowlistFilteringMatrixAggregateAllowlistSummary CreateAggregateAllowlist(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new HnswAllowlistFilteringMatrixAggregateAllowlistSummary("notAvailable", 0, 0, 0, 0, 0, 0, 0);
        }

        return new HnswAllowlistFilteringMatrixAggregateAllowlistSummary(
            "recorded",
            recorded.Length,
            recorded.Min(item => item.Allowlist.KnownLiveAllowedCountPerQuery ?? 0),
            recorded.Average(item => item.Allowlist.KnownLiveAllowedCountPerQuery ?? 0),
            recorded.Max(item => item.Allowlist.KnownLiveAllowedCountPerQuery ?? 0),
            recorded.Sum(item => (item.Allowlist.UnknownIdCountPerQuery ?? 0) * item.QueryCount),
            recorded.Sum(item => (item.Allowlist.DuplicateInputIdCountPerQuery ?? 0) * item.QueryCount),
            recorded.Sum(item => (item.Allowlist.TombstonedInputIdCountPerQuery ?? 0) * item.QueryCount));
    }

    private static HnswAllowlistFilteringMatrixAggregateMutationCountSummary CreateAggregateMutationCounts(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new HnswAllowlistFilteringMatrixAggregateMutationCountSummary("notAvailable", 0, 0, 0, 0, 0, 0, 0);
        }

        return new HnswAllowlistFilteringMatrixAggregateMutationCountSummary(
            "recorded",
            recorded.Length,
            recorded.Sum(item => item.Mutations.InsertedDeltaVectorCount),
            recorded.Sum(item => item.Mutations.DeletedBaseVectorCount),
            recorded.Sum(item => item.Mutations.DeletedDeltaVectorCount),
            recorded.Sum(item => item.Counts.PreCheckpointTombstoneCount ?? 0),
            recorded.Min(item => item.Counts.PreCheckpointLiveVectorCount ?? 0),
            recorded.Max(item => item.Counts.PreCheckpointLiveVectorCount ?? 0));
    }

    private static HnswAllowlistFilteringMatrixAggregateIntegritySummary CreateAggregateIntegrity(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded) =>
        new(
            recorded.Length == 0 ? "notAvailable" : "recorded",
            recorded.Length,
            recorded.Count(item => item.ReturnedResultIntegrity.PassedForAllSearches == true),
            recorded.Sum(item => item.ReturnedResultIntegrity.CheckedResultCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.UnknownIdCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.TombstonedIdCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.NotAllowedIdCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.DuplicateIdCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.NonFiniteDistanceCount ?? 0),
            recorded.Sum(item => item.ReturnedResultIntegrity.DistanceMismatchCount ?? 0));

    private static HnswAllowlistFilteringMatrixAggregateAllocationSummary CreateAggregateAllocations(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded)
    {
        if (recorded.Length == 0)
        {
            return new HnswAllowlistFilteringMatrixAggregateAllocationSummary("notAvailable", 0, null, null, null);
        }

        return new HnswAllowlistFilteringMatrixAggregateAllocationSummary(
            "recorded",
            recorded.Length,
            recorded.Max(item => item.Allocations.MaxMeanManagedAllocatedBytesPerSearchCall),
            recorded.Max(item => item.Allocations.MaxManagedAllocatedBytesPerSearchCall),
            recorded.Max(item => item.Allocations.MaxManagedAllocatedBytesPerRun));
    }

    private static HnswAllowlistFilteringMatrixEligibilitySummary CreateAggregateEligibility(
        HnswAllowlistFilteringMatrixCaseManifest[] recorded,
        int caseCount)
    {
        bool publicClaimEligible = recorded.Any(item => item.RecursiveEligibility.PublicClaimEligible);
        bool baselineCandidateEligible = recorded.Any(item => item.RecursiveEligibility.BaselineCandidateEligible);
        bool comparisonArtifactEligible = recorded.Any(item => item.RecursiveEligibility.ComparisonArtifactEligible);
        bool regressionGateEligible = recorded.Any(item => item.RecursiveEligibility.RegressionGateEligible);
        bool allReportsInspected = recorded.Length == caseCount;

        return new HnswAllowlistFilteringMatrixEligibilitySummary(
            allReportsInspected ? "recorded" : "partial",
            LinkedReportInspected: allReportsInspected,
            publicClaimEligible,
            baselineCandidateEligible,
            comparisonArtifactEligible,
            regressionGateEligible,
            AllEligibilityFlagsFalse: allReportsInspected && !publicClaimEligible && !baselineCandidateEligible && !comparisonArtifactEligible && !regressionGateEligible);
    }

    private static HnswAllowlistFilteringMatrixDesignInfo CreateDesign(string presetName, uint seed, VectorMetric metric)
    {
        AllowlistMatrixPresetCase[] presetCases = GetPresetCases(presetName);
        AllowlistMatrixPresetCase first = presetCases[0];
        return new HnswAllowlistFilteringMatrixDesignInfo(
            metric.ToString(),
            presetCases.Select(item => item.Dimension).Distinct().Order().ToArray(),
            presetCases.Select(item => item.TopK).Distinct().Order().ToArray(),
            presetCases.Select(item => item.FilterProfile).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
            presetCases
                .GroupBy(item => item.UpdateProfileName)
                .Select(group =>
                {
                    AllowlistMatrixPresetCase item = group.First();
                    return new HnswAllowlistFilteringMatrixUpdateProfileInfo(
                        item.UpdateProfileName,
                        item.InsertedDeltaCount,
                        item.DeletedBaseCount,
                        item.DeletedDeltaCount,
                        DescribeUpdateProfile(item.UpdateProfileName));
                })
                .OrderBy(item => item.Name, StringComparer.Ordinal)
                .ToArray(),
            new HnswAllowlistFilteringMatrixHnswProfileInfo("fixed-hnsw", first.M, first.EfConstruction, first.EfSearch),
            string.Create(CultureInfo.InvariantCulture, $"0x{seed:X8}"),
            string.Create(CultureInfo.InvariantCulture, $"0x{HnswMatrixSeedBase:X16}"),
            "caseDatasetSeed = matrixSeed + (caseNumber - 1); caseHnswSeed = hnswMatrixSeedBase + caseNumber; case numbers are one-based within the selected preset.",
            "Generated finite squared-L2 data only; allowlist generation, exact live-view truth, build/open/checkpoint and validation remain inside linked VEC-149 reports and outside measured search-call samples.",
            string.Equals(presetName, HnswAllowlistFilteringMatrixOptions.StandardPresetName, StringComparison.Ordinal)
                ? "Standard is 18 bounded generated cases over dimensions 32, 96 and 384; filter profiles empty, very-selective, fallback-boundary, broad and all; top-k 10 and 100; low-churn and tombstone-heavy update profiles; fixed M=16, efConstruction=128 and efSearch=192."
                : "Smoke is four bounded generated cases covering empty, fallback-boundary and broad profiles plus a tombstone-heavy broad case.",
            "Private generated HNSW filtering matrix evidence only; no Fashion-MNIST, external datasets, public docs, package changes, baseline candidates, comparison artifacts, regression gates or public claims.");
    }

    private static HnswAllowlistFilteringMatrixEligibility CreateEligibility(string presetName) =>
        new(
            "local-evidence",
            "private-raw",
            string.Equals(presetName, HnswAllowlistFilteringMatrixOptions.StandardPresetName, StringComparison.Ordinal) ? "standard" : "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW allowlist filtering matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "No generated HNSW allowlist filtering matrix baseline-candidate policy is accepted.",
            "No generated HNSW allowlist filtering matrix comparison artifact is accepted.",
            "No generated HNSW allowlist filtering matrix regression-gate policy, threshold or hard gate is accepted.");

    private static AllowlistMatrixPresetCase[] GetPresetCases(string presetName)
    {
        string normalized = HnswAllowlistFilteringMatrixOptions.NormalizePresetName(presetName);
        return normalized switch
        {
            HnswAllowlistFilteringMatrixOptions.SmokePresetName => SmokeCases,
            HnswAllowlistFilteringMatrixOptions.StandardPresetName => StandardCases,
            _ => throw new ArgumentException($"Unsupported generated HNSW allowlist filtering matrix preset '{presetName}'.")
        };
    }

    private static AllowlistMatrixPresetCase[] CreateStandardCases()
    {
        int[] dimensions = [32, 96, 384];
        var cases = new List<AllowlistMatrixPresetCase>(18);
        foreach (int dimension in dimensions)
        {
            string suffix = $"{dimension}d";
            cases.Add(new($"standard-empty-{suffix}-k10", "empty", "low-churn", "fallback underfill", dimension, 2048, 256, 64, 16, 10, 16, 128, 192));
            cases.Add(new($"standard-very-selective-{suffix}-k10", "very-selective", "low-churn", "fallback underfill/parity", dimension, 2048, 256, 64, 16, 10, 16, 128, 192));
            cases.Add(new($"standard-fallback-boundary-{suffix}-k10", "fallback-boundary", "low-churn", "exact efSearch boundary", dimension, 2048, 256, 64, 16, 10, 16, 128, 192));
            cases.Add(new($"standard-broad-{suffix}-k10", "broad", "low-churn", "emission integrity/recall", dimension, 2048, 256, 64, 16, 10, 16, 128, 192));
            cases.Add(new($"standard-broad-tombstone-heavy-{suffix}-k100", "broad", "tombstone-heavy", "emission underfill/tombstones", dimension, 2048, 512, 512, 256, 100, 16, 128, 192));
            cases.Add(new($"standard-all-tombstone-heavy-{suffix}-k100", "all", "tombstone-heavy", "broad all-live visibility", dimension, 2048, 512, 512, 256, 100, 16, 128, 192));
        }

        return cases.ToArray();
    }

    private static HnswAllowlistSearchSectionInfo[] GetSections(HnswAllowlistFilteringBenchmarkReport report) =>
    [
        report.Searches.ImmutableHnsw,
        report.Searches.OpenedHnsw,
        report.Searches.SourceComposite,
        report.Searches.RebuiltComposite,
        report.Searches.CheckpointOpenedHnsw
    ];

    private static string GetManifestDirectory(string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(directory) ? "." : directory);
    }

    private static string CreateRelativePath(string manifestDirectory, string path) =>
        Path.GetRelativePath(manifestDirectory, Path.GetFullPath(path)).Replace('\\', '/');

    private static string DescribeUpdateProfile(string updateProfileName) =>
        string.Equals(updateProfileName, "low-churn", StringComparison.Ordinal)
            ? "moderate exact delta with light base and delta tombstones"
            : "heavier base tombstones plus delta tombstones";

    public sealed record MatrixCase(
        string CaseId,
        string FilterProfile,
        string UpdateProfileName,
        string BranchFocus,
        string RelativeReportPath,
        string RelativeOpenedIndexDirectoryPath,
        string RelativeCheckpointDirectoryPath,
        HnswAllowlistFilteringOptions Options);

    private sealed record AllowlistMatrixPresetCase(
        string CaseId,
        string FilterProfile,
        string UpdateProfileName,
        string BranchFocus,
        int Dimension,
        int BaseVectorCount,
        int InsertedDeltaCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        int TopK,
        int M,
        int EfConstruction,
        int EfSearch);
}
