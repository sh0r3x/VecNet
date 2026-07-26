using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class HnswBasePlusExactDeltaMatrixScenario
{
    private const string TaskId = "VEC-125";
    private const string SchemaName = "VecNet.HnswBasePlusExactDeltaMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly int[] SmokeDimensions = [16];
    private static readonly int[] SmokeTopKValues = [1, 10];
    private static readonly VectorMetric[] SupportedMetrics = [VectorMetric.SquaredEuclidean, VectorMetric.Cosine];
    private static readonly HnswMatrixProfile[] SmokeHnswProfiles =
    [
        new("balanced-m4", M: 4, EfConstruction: 16, EfSearch: 16)
    ];
    private static readonly UpdateMatrixProfile[] SmokeUpdateProfiles =
    [
        new("low-churn", InsertedDeltaCount: 4, DeletedBaseCount: 2, DeletedDeltaCount: 0, "small exact delta with light base tombstones")
    ];

    private static readonly int[] StandardDimensions = [32, 128];
    private static readonly int[] StandardTopKValues = [10, 50];
    private static readonly HnswMatrixProfile[] StandardHnswProfiles =
    [
        new("balanced-m8", M: 8, EfConstruction: 64, EfSearch: 64)
    ];
    private static readonly UpdateMatrixProfile[] StandardUpdateProfiles =
    [
        new("low-churn", InsertedDeltaCount: 16, DeletedBaseCount: 8, DeletedDeltaCount: 0, "moderate exact delta with light base tombstones"),
        new("tombstone-heavy", InsertedDeltaCount: 16, DeletedBaseCount: 32, DeletedDeltaCount: 8, "heavier base tombstones plus delta tombstones")
    ];

    public static HnswBasePlusExactDeltaMatrixManifest Run(
        HnswBasePlusExactDeltaMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = HnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(options.PresetName);
        MatrixCase[] cases = ExpandCases(options);
        var caseManifests = new HnswBasePlusExactDeltaMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;
        int blocked = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            MatrixCase matrixCase = cases[i];
            string[] caseArguments = CreateCaseArguments(matrixCase.Options);

            try
            {
                HnswBasePlusExactDeltaBenchmarkReport report =
                    HnswBasePlusExactDeltaGeneratedScenario.Run(matrixCase.Options, caseArguments);
                HnswBasePlusExactDeltaGeneratedScenario.Write(report, matrixCase.Options.OutputPath);

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

        return new HnswBasePlusExactDeltaMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            HnswBasePlusExactDeltaMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswBasePlusExactDeltaMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            CreateDesign(presetName),
            caseManifests.Length,
            caseManifests,
            new HnswBasePlusExactDeltaMatrixAggregate(passed, failed, SkippedCaseCount: 0, blocked),
            CreateEligibility(),
            [
                "Private generated HNSW base-plus-exact-delta matrix evidence only; not a public benchmark, baseline candidate, regression gate or public mutable/update HNSW claim.",
                "Each case reuses the accepted VEC-124 generated-hnsw-base-plus-exact-delta report schema and writes a linked private per-case report when execution succeeds.",
                "Generated finite SquaredEuclidean and Cosine data only; no external dataset, hnswlib, FAISS, checkpoint/rebuild, durable mutable overlay persistence, filtering or direct HNSW graph mutation is introduced.",
                "The standard preset covers two dimensions, two top-k values and two update/tombstone profiles with efSearch greater than or equal to top-k.",
                "Per-case summaries repeat recall, ordered-agreement, underfill, mutation and count metadata from linked VEC-124 reports for matrix-level inspection."
            ]);
    }

    public static MatrixCase[] ExpandCases(HnswBasePlusExactDeltaMatrixOptions options)
    {
        MatrixPreset preset = GetPreset(options.PresetName);
        var cases = new List<MatrixCase>(
            SupportedMetrics.Length * preset.Dimensions.Length * preset.TopKValues.Length * preset.HnswProfiles.Length * preset.UpdateProfiles.Length);
        int caseIndex = 0;

        foreach (VectorMetric metric in SupportedMetrics)
        {
            foreach (int dimension in preset.Dimensions)
            {
                foreach (int topK in preset.TopKValues)
                {
                    foreach (HnswMatrixProfile hnswProfile in preset.HnswProfiles)
                    {
                        if (hnswProfile.EfSearch < topK)
                        {
                            throw new InvalidOperationException("Matrix HNSW profile efSearch must be at least top-k for every case.");
                        }

                        foreach (UpdateMatrixProfile updateProfile in preset.UpdateProfiles)
                        {
                            uint dataSeed = unchecked(options.Seed + (uint)caseIndex);
                            ulong hnswSeed = CreateHnswSeed(options.Seed, caseIndex);
                            string caseId = CreateCaseId(caseIndex + 1, metric, hnswProfile.Name, updateProfile.Name, dimension, topK);
                            string relativeReportPath = $"{caseId}.json";
                            string outputPath = Path.Combine(options.OutputDirectory, relativeReportPath);
                            var caseOptions = new HnswBasePlusExactDeltaGeneratedOptions(
                                metric,
                                dimension,
                                options.BaseVectorCount,
                                options.QueryCount,
                                topK,
                                dataSeed,
                                updateProfile.InsertedDeltaCount,
                                updateProfile.DeletedBaseCount,
                                updateProfile.DeletedDeltaCount,
                                options.DuplicateInsertAttempts,
                                options.UnknownDeleteAttempts,
                                options.RepeatedDeleteAttempts,
                                outputPath,
                                options.Runs,
                                options.WarmupQueries,
                                hnswProfile.M,
                                hnswProfile.EfConstruction,
                                hnswProfile.EfSearch,
                                hnswSeed);
                            cases.Add(new MatrixCase(caseId, hnswProfile.Name, updateProfile.Name, relativeReportPath, caseOptions));
                            caseIndex++;
                        }
                    }
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMinimumBaseVectorCount(string presetName)
    {
        MatrixPreset preset = GetPreset(presetName);
        int maxTopK = preset.TopKValues.Max();
        int maxDeletePressure = preset.UpdateProfiles.Max(profile => profile.DeletedBaseCount + profile.DeletedDeltaCount - profile.InsertedDeltaCount);
        return Math.Max(maxTopK, maxTopK + Math.Max(0, maxDeletePressure));
    }

    public static void WriteManifest(HnswBasePlusExactDeltaMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(HnswBasePlusExactDeltaGeneratedOptions options) =>
    [
        HnswBasePlusExactDeltaGeneratedOptions.ScenarioName,
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
        "--output", options.OutputPath
    ];

    private static HnswBasePlusExactDeltaMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        MatrixCase matrixCase,
        string[] commandArguments,
        HnswBasePlusExactDeltaBenchmarkReport? report,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        HnswBasePlusExactDeltaGeneratedOptions options = matrixCase.Options;
        int expectedLiveVectorCount = checked(options.BaseVectorCount + options.InsertedDeltaCount - options.DeletedBaseCount - options.DeletedDeltaCount);

        return new HnswBasePlusExactDeltaMatrixCaseManifest(
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
            commandArguments,
            report?.ReportId,
            status,
            validationStatus,
            CreateRecallSummary(report),
            CreateUnderfillSummary(options, report),
            CreateMutationSummary(options, report),
            CreateCountSummary(options, report),
            errorMessage);
    }

    private static HnswBasePlusExactDeltaMatrixRecallSummary CreateRecallSummary(
        HnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaMatrixRecallSummary("notAvailable", null, null, null, null, null, null)
            : new HnswBasePlusExactDeltaMatrixRecallSummary(
                "recorded",
                report.Metrics.RecallAtK,
                report.Metrics.OrderedAgreement,
                report.Metrics.DistanceToleranceStatus,
                report.Metrics.MissingResultCount,
                report.Metrics.ExtraResultCount,
                report.Metrics.ReturnedResultIntegrity.Status);

    private static HnswBasePlusExactDeltaMatrixUnderfillSummary CreateUnderfillSummary(
        HnswBasePlusExactDeltaGeneratedOptions options,
        HnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaMatrixUnderfillSummary(
                "notAvailable",
                options.QueryCount,
                options.TopK,
                checked(options.QueryCount * options.TopK),
                null,
                null,
                null)
            : new HnswBasePlusExactDeltaMatrixUnderfillSummary(
                "recorded",
                report.Underfill.QueryCount,
                report.Underfill.RequestedResultCountPerQuery,
                report.Underfill.TotalRequestedResultSlots,
                report.Underfill.TotalReturnedResults,
                report.Underfill.UnderfilledQueryCount,
                report.Underfill.UnderfilledSlotCount);

    private static HnswBasePlusExactDeltaMatrixMutationSummary CreateMutationSummary(
        HnswBasePlusExactDeltaGeneratedOptions options,
        HnswBasePlusExactDeltaBenchmarkReport? report) =>
        report is null
            ? new HnswBasePlusExactDeltaMatrixMutationSummary(
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
                null)
            : new HnswBasePlusExactDeltaMatrixMutationSummary(
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
                report.Mutations.GenerationAfterMutations);

    private static HnswBasePlusExactDeltaMatrixCountSummary CreateCountSummary(
        HnswBasePlusExactDeltaGeneratedOptions options,
        HnswBasePlusExactDeltaBenchmarkReport? report)
    {
        int expectedLiveVectorCount = checked(options.BaseVectorCount + options.InsertedDeltaCount - options.DeletedBaseCount - options.DeletedDeltaCount);
        if (report is null)
        {
            return new HnswBasePlusExactDeltaMatrixCountSummary(
                "notAvailable",
                options.BaseVectorCount,
                options.PhysicalVectorCount,
                expectedLiveVectorCount,
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
                null);
        }

        return new HnswBasePlusExactDeltaMatrixCountSummary(
            "recorded",
            report.Counts.BasePhysicalVectorCount,
            checked(report.Counts.BasePhysicalVectorCount + report.Counts.DeltaPhysicalVectorCount),
            expectedLiveVectorCount,
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
    }

    private static HnswBasePlusExactDeltaMatrixDesignInfo CreateDesign(string presetName)
    {
        MatrixPreset preset = GetPreset(presetName);
        return new HnswBasePlusExactDeltaMatrixDesignInfo(
            string.Join(",", SupportedMetrics.Select(metric => metric.ToString())),
            preset.Dimensions,
            preset.TopKValues,
            preset.HnswProfiles
                .Select(profile => new HnswBasePlusExactDeltaMatrixHnswProfileInfo(profile.Name, profile.M, profile.EfConstruction, profile.EfSearch))
                .ToArray(),
            preset.UpdateProfiles
                .Select(profile => new HnswBasePlusExactDeltaMatrixUpdateProfileInfo(profile.Name, profile.InsertedDeltaCount, profile.DeletedBaseCount, profile.DeletedDeltaCount, profile.Description))
                .ToArray(),
            "Generated finite SquaredEuclidean or Cosine base vectors, delta vectors and queries only; exact updated truth is computed by the linked VEC-124 report from the post-update live view.",
            "Smoke is a small local validation preset; standard covers at least two dimensions, two top-k values and two update/tombstone profiles.",
            "Internal composite matrix evidence only; no public mutable/update HNSW API, durable mutable overlay persistence, checkpoint/rebuild, external dataset, baseline, regression gate or public claim.");
    }

    private static HnswBasePlusExactDeltaMatrixEligibility CreateEligibility() =>
        new(
            "local-evidence",
            "private-raw",
            "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW base-plus-exact-delta matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "No generated mutable/update HNSW matrix baseline-candidate policy is accepted.",
            "No generated mutable/update HNSW matrix regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static MatrixPreset GetPreset(string presetName)
    {
        string normalizedPresetName = HnswBasePlusExactDeltaMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            HnswBasePlusExactDeltaMatrixOptions.SmokePresetName => new MatrixPreset(SmokeDimensions, SmokeTopKValues, SmokeHnswProfiles, SmokeUpdateProfiles),
            HnswBasePlusExactDeltaMatrixOptions.StandardPresetName => new MatrixPreset(StandardDimensions, StandardTopKValues, StandardHnswProfiles, StandardUpdateProfiles),
            _ => throw new ArgumentException($"Unsupported HNSW base-plus-exact-delta matrix preset '{presetName}'.")
        };
    }

    private static string CreateCaseId(int caseNumber, VectorMetric metric, string hnswProfileName, string updateProfileName, int dimension, int topK) =>
        string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}-{metric}-{hnswProfileName}-{updateProfileName}-{dimension}d-{topK}k");

    private static ulong CreateHnswSeed(uint baseSeed, int caseIndex) =>
        0x484E5357_00012500UL ^ ((ulong)baseSeed << 16) ^ (uint)(caseIndex + 1);

    public sealed record MatrixCase(
        string CaseId,
        string HnswProfileName,
        string UpdateProfileName,
        string RelativeReportPath,
        HnswBasePlusExactDeltaGeneratedOptions Options);

    private sealed record MatrixPreset(
        int[] Dimensions,
        int[] TopKValues,
        HnswMatrixProfile[] HnswProfiles,
        UpdateMatrixProfile[] UpdateProfiles);

    private sealed record HnswMatrixProfile(string Name, int M, int EfConstruction, int EfSearch);

    private sealed record UpdateMatrixProfile(
        string Name,
        int InsertedDeltaCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        string Description);
}
