using System.Globalization;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario
{
    private const string TaskId = "VEC-129";
    private const string SchemaName = "VecNet.ExternalHnswBasePlusExactDeltaMatrixManifest";
    private const string SchemaVersion = "0.1";
    private const int Dimension = 784;
    private const ulong HnswSeedBase = 0x484E535700012800UL;

    private static readonly int[] SmokeTopKValues = [10];
    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly ExternalDeltaHnswProfile[] HnswProfiles =
    [
        new("wide-m16-ef192", M: 16, EfConstruction: 128, EfSearch: 192),
        new("wide-m16-ef384", M: 16, EfConstruction: 128, EfSearch: 384)
    ];
    private static readonly ExternalDeltaUpdateProfile[] UpdateProfiles =
    [
        new("low-churn", BaseRowCount: 59_000, DeltaRowCount: 500, DeletedBaseCount: 100, DeletedDeltaCount: 0, ExpectedLiveCount: 59_400, "light overlay churn with exact delta present and low tombstone pressure"),
        new("tombstone-heavy", BaseRowCount: 56_000, DeltaRowCount: 2_000, DeletedBaseCount: 5_000, DeletedDeltaCount: 500, ExpectedLiveCount: 52_500, "heavy base tombstone suppression plus delta tombstones")
    ];

    public static ExternalHnswBasePlusExactDeltaMatrixManifest Run(
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(options.PresetName);
        MatrixCase[] cases = ExpandCases(options);
        ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo cacheTruth = CreateCacheTruthInfo(options, out string? cacheBlockReason);
        var caseManifests = new ExternalHnswBasePlusExactDeltaMatrixCaseManifest[cases.Length];
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
                    status: "blocked",
                    validationStatus: "blocked",
                    cacheBlockReason);
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
                    ExternalHnswBasePlusExactDeltaBenchmarkReport report =
                        FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(matrixCase.Options, caseArguments);
                    FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(report, matrixCase.Options.OutputPath);

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
                catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
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
                catch (Exception ex) when (ex is ArgumentException or UnauthorizedAccessException or InvalidOperationException)
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
        }

        return new ExternalHnswBasePlusExactDeltaMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            cacheTruth,
            CreateDesign(options, presetName),
            caseManifests.Length,
            caseManifests,
            CreateAggregate(caseManifests, passed, failed, blocked),
            CreateEligibility(),
            [
                "Private external Fashion-MNIST HNSW base-plus-exact-delta matrix evidence only; not a public benchmark, baseline candidate, comparison artifact, regression gate or public mutable/update HNSW claim.",
                "Each successful case reuses the accepted VEC-127 external-fashion-mnist-hnsw-base-plus-exact-delta report schema and writes a linked private per-case report.",
                "The matrix uses only the already admitted local Fashion-MNIST cache and existing exact truth artifact; it does not download, convert, admit, refresh or regenerate Fashion-MNIST.",
                "Existing Fashion-MNIST exact truth is a cache/readiness guard only; linked reports compute exact updated truth from each case's post-update live view.",
                "Per-case timing and managed allocation remain the accepted VEC-127 internal composite Search(query, results, workspace) boundary.",
                "Durable mutable overlay persistence, checkpoint/rebuild, filtering, hnswlib/FAISS comparison, memory evidence, concurrency evidence and public claims are out of scope."
            ]);
    }

    public static MatrixCase[] ExpandCases(FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options)
    {
        string presetName = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(options.PresetName);
        int[] topKValues = GetTopKValues(presetName);
        ExternalDeltaUpdateProfile[] updateProfiles = GetUpdateProfiles(presetName);
        ExternalDeltaHnswProfile[] hnswProfiles = GetHnswProfiles(presetName);
        var cases = new List<MatrixCase>(topKValues.Length * updateProfiles.Length * hnswProfiles.Length);
        int caseIndex = 0;

        foreach (int topK in topKValues)
        {
            foreach (ExternalDeltaUpdateProfile updateProfile in updateProfiles)
            {
                foreach (ExternalDeltaHnswProfile hnswProfile in hnswProfiles)
                {
                    if (hnswProfile.EfSearch < topK)
                    {
                        throw new InvalidOperationException("External Fashion-MNIST HNSW base-plus-exact-delta matrix efSearch must be at least top-k for every case.");
                    }

                    int caseNumber = caseIndex + 1;
                    uint workloadSeed = unchecked(options.Seed + (uint)caseIndex);
                    ulong hnswSeed = HnswSeedBase + (ulong)caseNumber;
                    string caseId = CreateCaseId(caseNumber, topK, updateProfile.Name, hnswProfile.Name);
                    string relativeReportPath = $"{caseId}.json";
                    string outputPath = Path.Combine(options.OutputDirectory, relativeReportPath);
                    var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
                        options.CacheRoot,
                        outputPath,
                        options.QueryCount,
                        topK,
                        updateProfile.BaseRowCount,
                        updateProfile.DeltaRowCount,
                        updateProfile.DeletedBaseCount,
                        updateProfile.DeletedDeltaCount,
                        options.DuplicateInsertAttempts,
                        options.UnknownDeleteAttempts,
                        options.RepeatedDeleteAttempts,
                        options.Runs,
                        options.WarmupQueries,
                        options.Metric,
                        workloadSeed,
                        hnswProfile.M,
                        hnswProfile.EfConstruction,
                        hnswProfile.EfSearch,
                        hnswSeed);

                    cases.Add(new MatrixCase(caseId, updateProfile.Name, hnswProfile.Name, relativeReportPath, updateProfile, hnswProfile, caseOptions));
                    caseIndex++;
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetTopKValues(presetName).Max();

    public static int GetMaxPhysicalCandidateVectorCount(string presetName)
    {
        _ = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(presetName);
        return GetUpdateProfiles(presetName).Max(profile => profile.BaseRowCount + profile.DeltaRowCount);
    }

    public static void WriteManifest(ExternalHnswBasePlusExactDeltaMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(FashionMnistExternalHnswBasePlusExactDeltaOptions options) =>
    [
        FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName,
        "--cache-root", options.CacheRoot,
        "--output", options.OutputPath,
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
        "--metric", ToCommandLineMetric(options.Metric),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--hnsw-seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}")
    ];

    private static ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo CreateCacheTruthInfo(
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options,
        out string? blockReason)
    {
        try
        {
            FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset =
                FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(
                    new FashionMnistExternalHnswBenchmarkOptions(
                        options.CacheRoot,
                        Path.Combine(options.OutputDirectory, "cache-truth-validation.json"),
                        options.QueryCount,
                        GetMaxTopK(options.PresetName),
                        options.Runs,
                        options.WarmupQueries,
                        options.Metric,
                        M: 16,
                        EfConstruction: 128,
                        EfSearch: Math.Max(192, GetMaxTopK(options.PresetName)),
                        HnswSeedBase));

            if (dataset.BaseCount < GetMaxPhysicalCandidateVectorCount(options.PresetName))
            {
                throw new InvalidDataException("Admitted Fashion-MNIST base matrix does not contain enough rows for the largest matrix base-plus-delta case.");
            }

            blockReason = null;
            return new ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo(
                "available",
                options.CacheRoot,
                dataset.Manifest.DatasetId,
                Dimension,
                options.Metric.ToString(),
                "Loaded existing admitted Fashion-MNIST cache only; no download, conversion, admission, refresh or truth regeneration path is used by VEC-129.",
                "Loaded existing exact truth artifact only as cache/readiness guard; linked VEC-127 reports generate exact updated truth from live post-update views.",
                dataset.Paths.RelativeManifestPath,
                dataset.ManifestSha256,
                dataset.Manifest.Truth.RelativePath,
                dataset.TruthSha256,
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                dataset.Truth.QuerySubsetCount,
                dataset.Truth.TruthDepth,
                ErrorMessage: null);
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException or ArgumentException or UnauthorizedAccessException)
        {
            blockReason = ex.Message;
            return new ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo(
                "unavailable",
                options.CacheRoot,
                FashionMnistDatasetSpecification.GetDatasetId(options.Metric),
                Dimension,
                options.Metric.ToString(),
                "Admitted local Fashion-MNIST cache is required; VEC-129 must not download, convert, admit, refresh or regenerate data.",
                "Existing exact truth artifact with sufficient query subset and truth depth is required; VEC-129 must not refresh truth.",
                AdmissionManifestPath: null,
                AdmissionManifestSha256: null,
                TruthRelativePath: null,
                TruthSha256: null,
                BaseVectorCount: null,
                QueryMatrixCount: null,
                TruthQuerySubsetCount: null,
                TruthDepth: null,
                ex.Message);
        }
    }

    private static ExternalHnswBasePlusExactDeltaMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        MatrixCase matrixCase,
        string[] commandArguments,
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        FashionMnistExternalHnswBasePlusExactDeltaOptions options = matrixCase.Options;
        return new ExternalHnswBasePlusExactDeltaMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.UpdateProfileName,
            matrixCase.HnswProfileName,
            FashionMnistDatasetSpecification.GetDatasetId(options.Metric),
            options.Metric.ToString(),
            Dimension,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
            options.M,
            options.EfConstruction,
            options.EfSearch,
            ImmutableBaseStartRow: 0,
            options.BaseVectorCount - 1,
            options.BaseVectorCount,
            DeltaStartRow: options.BaseVectorCount,
            options.PhysicalCandidateVectorCount - 1,
            options.InsertedDeltaCount,
            UnusedCandidateRowCount: 60_000 - options.PhysicalCandidateVectorCount,
            options.DeletedBaseCount,
            options.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            options.PhysicalCandidateVectorCount,
            options.LiveVectorCount,
            matrixCase.RelativeReportPath,
            commandArguments,
            report?.ReportId,
            status,
            validationStatus,
            CreateRecallOrderSummary(report),
            CreateIntegritySummary(report),
            CreateUnderfillSummary(options, report),
            CreateAllocationSummary(report),
            CreateMutationSummary(options, report),
            CreateCountSummary(options, report),
            CreateEligibilitySummary(report),
            errorMessage);
    }

    private static ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary CreateRecallOrderSummary(
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary("notAvailable", null, null, null, null, null, null)
            : new ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary(
                "recorded",
                report.Metrics.RecallAtK,
                report.Metrics.OrderedAgreement,
                report.Metrics.DistanceToleranceStatus,
                report.Metrics.DistanceMismatchCount,
                report.Metrics.MissingResultCount,
                report.Metrics.ExtraResultCount);

    private static ExternalHnswBasePlusExactDeltaMatrixIntegritySummary CreateIntegritySummary(
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixIntegritySummary("notAvailable", null, null, null, null, null, null, null, null)
            : new ExternalHnswBasePlusExactDeltaMatrixIntegritySummary(
                report.Metrics.ReturnedResultIntegrity.Status,
                report.Metrics.ReturnedResultIntegrity.CheckedResultCount,
                report.Metrics.ReturnedResultIntegrity.QueryCountMismatchCount,
                report.Metrics.ReturnedResultIntegrity.ResultCountViolationCount,
                report.Metrics.ReturnedResultIntegrity.NonFiniteDistanceCount,
                report.Metrics.ReturnedResultIntegrity.DuplicateIdCount,
                report.Metrics.ReturnedResultIntegrity.UnknownIdCount,
                report.Metrics.ReturnedResultIntegrity.TombstonedIdCount,
                report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);

    private static ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary CreateUnderfillSummary(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary(
                "notAvailable",
                options.QueryCount,
                options.TopK,
                checked(options.QueryCount * options.TopK),
                null,
                null,
                null)
            : new ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary(
                "recorded",
                report.Underfill.QueryCount,
                report.Underfill.RequestedResultCountPerQuery,
                report.Underfill.TotalRequestedResultSlots,
                report.Underfill.TotalReturnedResults,
                report.Underfill.UnderfilledQueryCount,
                report.Underfill.UnderfilledSlotCount);

    private static ExternalHnswBasePlusExactDeltaMatrixAllocationSummary CreateAllocationSummary(
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixAllocationSummary("notAvailable", null, null, null, null, null, null, null, null)
            : new ExternalHnswBasePlusExactDeltaMatrixAllocationSummary(
                "recorded",
                report.Search.Aggregate.MeanElapsedMilliseconds,
                report.Search.Aggregate.MeanLatencyP50Milliseconds,
                report.Search.Aggregate.MeanLatencyP95Milliseconds,
                report.Search.Aggregate.MeanLatencyP99Milliseconds,
                report.Search.Aggregate.MeanQps,
                report.Search.Aggregate.MeanManagedAllocatedBytesPerQuery,
                report.Measurement.ManagedAllocations.Status,
                report.Measurement.Memory.Status);

    private static ExternalHnswBasePlusExactDeltaMatrixMutationSummary CreateMutationSummary(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixMutationSummary(
                "notAvailable",
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null)
            : new ExternalHnswBasePlusExactDeltaMatrixMutationSummary(
                "recorded",
                report.Mutations.InsertedCount,
                report.Mutations.DeletedBaseCount,
                report.Mutations.DeletedDeltaCount,
                report.Mutations.DuplicateInsertAttempts,
                report.Mutations.UnknownDeleteAttempts,
                report.Mutations.RepeatedDeleteAttempts,
                report.Mutations.CommittedMutationCount,
                report.Mutations.StatusCounts.Committed,
                report.Mutations.StatusCounts.DuplicateId,
                report.Mutations.StatusCounts.UnknownId,
                report.Mutations.StatusCounts.AlreadyDeleted,
                report.Mutations.GenerationDeltaMatchesCommittedMutations,
                report.Mutations.GenerationDelta,
                report.Mutations.GenerationAfterMutations);

    private static ExternalHnswBasePlusExactDeltaMatrixCountSummary CreateCountSummary(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixCountSummary(
                "notAvailable",
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.PhysicalCandidateVectorCount,
                options.LiveVectorCount,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null)
            : new ExternalHnswBasePlusExactDeltaMatrixCountSummary(
                "recorded",
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.PhysicalCandidateVectorCount,
                options.LiveVectorCount,
                report.Counts.BasePhysicalVectorCount,
                report.Counts.BaseLiveVectorCount,
                report.Counts.DeltaPhysicalVectorCount,
                report.Counts.DeltaLiveVectorCount,
                report.Counts.BaseTombstoneCount,
                report.Counts.DeltaTombstoneCount,
                report.Counts.TombstoneCount,
                report.Counts.LiveVectorCount,
                report.Counts.DeletedReservedIdCount,
                report.Counts.Generation,
                report.Counts.TombstoneRatio,
                report.Counts.DeltaInsertRatio);

    private static ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary CreateEligibilitySummary(
        ExternalHnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary(false, false, false, false, false, false)
            : new ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary(
                report.Eligibility.PublicClaimEligible,
                report.Eligibility.BaselineCandidateEligible,
                report.Eligibility.RegressionGateEligible,
                report.Validation.PublicClaimEligible,
                report.Validation.BaselineCandidateEligible,
                report.Validation.RegressionGateEligible);

    private static ExternalHnswBasePlusExactDeltaMatrixDesignInfo CreateDesign(
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options,
        string presetName) =>
        new(
            FashionMnistDatasetSpecification.GetDatasetId(options.Metric),
            Dimension,
            options.Metric.ToString(),
            options.QueryCount,
            GetTopKValues(presetName),
            GetUpdateProfiles(presetName)
                .Select(profile => new ExternalHnswBasePlusExactDeltaMatrixUpdateProfileInfo(
                    profile.Name,
                    BaseStartRow: 0,
                    profile.BaseRowCount - 1,
                    profile.BaseRowCount,
                    DeltaStartRow: profile.BaseRowCount,
                    profile.BaseRowCount + profile.DeltaRowCount - 1,
                    profile.DeltaRowCount,
                    60_000 - profile.BaseRowCount - profile.DeltaRowCount,
                    profile.DeletedBaseCount,
                    profile.DeletedDeltaCount,
                    profile.ExpectedLiveCount,
                    profile.Description))
                .ToArray(),
            GetHnswProfiles(presetName)
                .Select(profile => new ExternalHnswBasePlusExactDeltaMatrixHnswProfileInfo(profile.Name, profile.M, profile.EfConstruction, profile.EfSearch))
                .ToArray(),
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            "per-case workload seed = matrix seed + zero-based case index; contiguous Fashion-MNIST row policy remains deterministic",
            "per-case HNSW seed = 0x484E535700012800 + one-based case number",
            "Immutable base rows start at row 0, exact delta rows immediately follow the base rows, base tombstones delete from the start of selected base rows and delta tombstones delete from the start of selected delta rows.",
            "Linked VEC-127 reports compute exact updated truth in memory from the post-update live view using scalar-reference selected-metric distance and ascending external ID ties.",
            "Measured search samples include only internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace) with caller-owned buffers/workspace; cache checks, loading, build, mutations, exact truth, warmup, comparison and manifest writing are excluded.",
            "Private/local external matrix evidence only; no public mutable/update HNSW API, durable mutable overlay persistence, checkpoint/rebuild, filtering, hnswlib/FAISS comparison, memory/concurrency evidence or public claim.");

    private static ExternalHnswBasePlusExactDeltaMatrixAggregate CreateAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked) =>
        new(
            passed,
            failed,
            SkippedCaseCount: 0,
            blocked,
            cases.Count(item => item.IntegritySummary.Status != "notAvailable" &&
                !string.Equals(item.IntegritySummary.Status, "passed", StringComparison.OrdinalIgnoreCase)),
            cases.Count(item => item.RecallOrderSummary.DistanceToleranceStatus is not null &&
                !string.Equals(item.RecallOrderSummary.DistanceToleranceStatus, "passed", StringComparison.OrdinalIgnoreCase)),
            CreateRecallAggregate(cases),
            CreateOrderAggregate(cases),
            CreateUnderfillAggregate(cases),
            CreateAllocationAggregate(cases),
            CreateMutationAggregate(cases),
            CreateCountAggregate(cases),
            CreateEligibilityAggregate(cases));

    private static ExternalHnswBasePlusExactDeltaMatrixRecallAggregate CreateRecallAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases)
    {
        double[] values = cases.Select(item => item.RecallOrderSummary.RecallAtK).OfType<double>().ToArray();
        return new ExternalHnswBasePlusExactDeltaMatrixRecallAggregate(
            MinOrNull(values),
            MaxOrNull(values),
            cases
                .GroupBy(item => item.TopK)
                .OrderBy(group => group.Key)
                .Select(group => CreateGroupedDoubleSummary(group.Key.ToString(CultureInfo.InvariantCulture), group.Select(item => item.RecallOrderSummary.RecallAtK).OfType<double>()))
                .ToArray());
    }

    private static ExternalHnswBasePlusExactDeltaMatrixOrderAggregate CreateOrderAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases)
    {
        double[] values = cases.Select(item => item.RecallOrderSummary.OrderedAgreement).OfType<double>().ToArray();
        return new ExternalHnswBasePlusExactDeltaMatrixOrderAggregate(
            MinOrNull(values),
            MaxOrNull(values),
            cases
                .GroupBy(item => item.TopK)
                .OrderBy(group => group.Key)
                .Select(group => CreateGroupedDoubleSummary(group.Key.ToString(CultureInfo.InvariantCulture), group.Select(item => item.RecallOrderSummary.OrderedAgreement).OfType<double>()))
                .ToArray());
    }

    private static ExternalHnswBasePlusExactDeltaMatrixUnderfillAggregate CreateUnderfillAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases) =>
        new(
            cases.Count(item => item.UnderfillSummary.UnderfilledSlotCount.GetValueOrDefault() > 0),
            cases.Sum(item => item.UnderfillSummary.UnderfilledQueryCount.GetValueOrDefault()),
            cases.Sum(item => item.UnderfillSummary.UnderfilledSlotCount.GetValueOrDefault()),
            cases
                .GroupBy(item => string.Create(CultureInfo.InvariantCulture, $"{item.TopK}k-{item.UpdateProfileName}"))
                .OrderBy(group => group.Key, StringComparer.Ordinal)
                .Select(group => new ExternalHnswBasePlusExactDeltaMatrixWorstUnderfillSummary(
                    group.Key,
                    group.Max(item => item.UnderfillSummary.UnderfilledSlotCount.GetValueOrDefault())))
                .ToArray());

    private static ExternalHnswBasePlusExactDeltaMatrixAllocationAggregate CreateAllocationAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases)
    {
        double[] values = cases.Select(item => item.AllocationSummary.MeanManagedAllocatedBytesPerSearchCall).OfType<double>().ToArray();
        return new ExternalHnswBasePlusExactDeltaMatrixAllocationAggregate(
            MaxOrNull(values),
            values.Count(value => value > 0));
    }

    private static ExternalHnswBasePlusExactDeltaMatrixMutationAggregate CreateMutationAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases) =>
        new(
            cases.Count(item => item.MutationSummary.GenerationDeltaMatchesCommittedMutations == false ||
                item.MutationSummary.StatusCommitted != item.MutationSummary.CommittedMutationCount),
            cases.Sum(item => item.MutationSummary.CommittedMutationCount.GetValueOrDefault()),
            cases.Sum(item => item.MutationSummary.StatusDuplicateId.GetValueOrDefault()),
            cases.Sum(item => item.MutationSummary.StatusUnknownId.GetValueOrDefault()),
            cases.Sum(item => item.MutationSummary.StatusAlreadyDeleted.GetValueOrDefault()));

    private static ExternalHnswBasePlusExactDeltaMatrixCountAggregate CreateCountAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases)
    {
        int[] liveCounts = cases.Select(item => item.CountSummary.LiveVectorCount).OfType<int>().ToArray();
        int[] tombstones = cases.Select(item => item.CountSummary.TombstoneCount).OfType<int>().ToArray();
        double[] tombstoneRatios = cases.Select(item => item.CountSummary.TombstoneRatio).OfType<double>().ToArray();
        double[] deltaRatios = cases.Select(item => item.CountSummary.DeltaInsertRatio).OfType<double>().ToArray();
        return new ExternalHnswBasePlusExactDeltaMatrixCountAggregate(
            liveCounts.Length == 0 ? null : liveCounts.Min(),
            liveCounts.Length == 0 ? null : liveCounts.Max(),
            tombstones.Length == 0 ? null : tombstones.Max(),
            MaxOrNull(tombstoneRatios),
            MaxOrNull(deltaRatios));
    }

    private static ExternalHnswBasePlusExactDeltaMatrixEligibilityAggregate CreateEligibilityAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases)
    {
        int linkedReportNonFalseCount = cases.Count(item =>
            item.EligibilitySummary.PublicClaimEligible ||
            item.EligibilitySummary.BaselineCandidateEligible ||
            item.EligibilitySummary.RegressionGateEligible ||
            item.EligibilitySummary.ValidationPublicClaimEligible ||
            item.EligibilitySummary.ValidationBaselineCandidateEligible ||
            item.EligibilitySummary.ValidationRegressionGateEligible);
        return new ExternalHnswBasePlusExactDeltaMatrixEligibilityAggregate(
            linkedReportNonFalseCount,
            ManifestPublicClaimEligible: false,
            ManifestBaselineCandidateEligible: false,
            ManifestRegressionGateEligible: false,
            ComparisonPublicationEligible: false);
    }

    private static ExternalHnswBasePlusExactDeltaMatrixGroupedDoubleSummary CreateGroupedDoubleSummary(
        string group,
        IEnumerable<double> values)
    {
        double[] materialized = values.ToArray();
        return new ExternalHnswBasePlusExactDeltaMatrixGroupedDoubleSummary(group, MinOrNull(materialized), MaxOrNull(materialized));
    }

    private static ExternalHnswBasePlusExactDeltaMatrixEligibility CreateEligibility() =>
        new(
            "local-evidence",
            "private-raw",
            "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            ComparisonPublicationEligible: false,
            "Private external Fashion-MNIST HNSW base-plus-exact-delta matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "No external mutable/update HNSW matrix baseline-candidate policy is accepted.",
            "No external mutable/update HNSW matrix regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            "No accepted public comparison-summary policy exists for this private external mutable/update HNSW matrix.");

    private static int[] GetTopKValues(string presetName)
    {
        string normalizedPresetName = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.SmokePresetName => SmokeTopKValues,
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.StandardPresetName => StandardTopKValues,
            _ => throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta matrix preset '{presetName}'.")
        };
    }

    private static ExternalDeltaUpdateProfile[] GetUpdateProfiles(string presetName)
    {
        string normalizedPresetName = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.SmokePresetName => [UpdateProfiles[0]],
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.StandardPresetName => UpdateProfiles,
            _ => throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta matrix preset '{presetName}'.")
        };
    }

    private static ExternalDeltaHnswProfile[] GetHnswProfiles(string presetName)
    {
        string normalizedPresetName = FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.SmokePresetName => [HnswProfiles[0]],
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.StandardPresetName => HnswProfiles,
            _ => throw new ArgumentException($"Unsupported external Fashion-MNIST HNSW base-plus-exact-delta matrix preset '{presetName}'.")
        };
    }

    private static string CreateCaseId(int caseNumber, int topK, string updateProfileName, string hnswProfileName) =>
        string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}-{topK}k-{updateProfileName}-{hnswProfileName}");

    private static string ToCommandLineMetric(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? "cosine" : "squared-euclidean";

    private static double? MinOrNull(double[] values) => values.Length == 0 ? null : values.Min();

    private static double? MaxOrNull(double[] values) => values.Length == 0 ? null : values.Max();

    public sealed record MatrixCase(
        string CaseId,
        string UpdateProfileName,
        string HnswProfileName,
        string RelativeReportPath,
        ExternalDeltaUpdateProfile UpdateProfile,
        ExternalDeltaHnswProfile HnswProfile,
        FashionMnistExternalHnswBasePlusExactDeltaOptions Options);

    public sealed record ExternalDeltaHnswProfile(string Name, int M, int EfConstruction, int EfSearch);

    public sealed record ExternalDeltaUpdateProfile(
        string Name,
        int BaseRowCount,
        int DeltaRowCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        int ExpectedLiveCount,
        string Description);
}
