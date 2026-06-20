using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactCheckpointMatrixScenario
{
    private const string TaskId = "VEC-069";
    private const string SchemaName = "VecNet.ExactCheckpointBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly CheckpointMatrixPresetCase[] SmokeCases =
    [
        new(VectorMetric.SquaredEuclidean, Dimension: 32, BaseVectorCount: 32, InsertedDeltaCount: 4, DeletedBaseCount: 2, TopK: 5, QueryCount: 2, AllowlistKind: "broad", CandidateSetKind: "selective"),
        new(VectorMetric.InnerProduct, Dimension: 128, BaseVectorCount: 48, InsertedDeltaCount: 8, DeletedBaseCount: 6, TopK: 10, QueryCount: 3, AllowlistKind: "selective", CandidateSetKind: "broad"),
        new(VectorMetric.Cosine, Dimension: 32, BaseVectorCount: 40, InsertedDeltaCount: 8, DeletedBaseCount: 5, TopK: 10, QueryCount: 2, AllowlistKind: "very-selective", CandidateSetKind: "selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 386, BaseVectorCount: 64, InsertedDeltaCount: 16, DeletedBaseCount: 20, TopK: 25, QueryCount: 2, AllowlistKind: "empty", CandidateSetKind: "all")
    ];

    private static readonly CheckpointMatrixPresetCase[] StandardCases =
    [
        new(VectorMetric.SquaredEuclidean, Dimension: 32, BaseVectorCount: 64, InsertedDeltaCount: 8, DeletedBaseCount: 4, TopK: 10, QueryCount: 3, AllowlistKind: "all", CandidateSetKind: "broad"),
        new(VectorMetric.SquaredEuclidean, Dimension: 128, BaseVectorCount: 96, InsertedDeltaCount: 24, DeletedBaseCount: 12, TopK: 10, QueryCount: 4, AllowlistKind: "broad", CandidateSetKind: "selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 386, BaseVectorCount: 128, InsertedDeltaCount: 32, DeletedBaseCount: 32, TopK: 25, QueryCount: 5, AllowlistKind: "selective", CandidateSetKind: "very-selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 768, BaseVectorCount: 160, InsertedDeltaCount: 40, DeletedBaseCount: 60, TopK: 50, QueryCount: 3, AllowlistKind: "empty", CandidateSetKind: "all"),
        new(VectorMetric.InnerProduct, Dimension: 32, BaseVectorCount: 64, InsertedDeltaCount: 16, DeletedBaseCount: 8, TopK: 10, QueryCount: 4, AllowlistKind: "selective", CandidateSetKind: "broad"),
        new(VectorMetric.InnerProduct, Dimension: 128, BaseVectorCount: 128, InsertedDeltaCount: 32, DeletedBaseCount: 16, TopK: 25, QueryCount: 5, AllowlistKind: "all", CandidateSetKind: "empty"),
        new(VectorMetric.InnerProduct, Dimension: 386, BaseVectorCount: 160, InsertedDeltaCount: 40, DeletedBaseCount: 48, TopK: 50, QueryCount: 3, AllowlistKind: "very-selective", CandidateSetKind: "selective"),
        new(VectorMetric.InnerProduct, Dimension: 768, BaseVectorCount: 240, InsertedDeltaCount: 60, DeletedBaseCount: 80, TopK: 100, QueryCount: 4, AllowlistKind: "broad", CandidateSetKind: "all"),
        new(VectorMetric.Cosine, Dimension: 32, BaseVectorCount: 72, InsertedDeltaCount: 12, DeletedBaseCount: 24, TopK: 10, QueryCount: 3, AllowlistKind: "empty", CandidateSetKind: "selective"),
        new(VectorMetric.Cosine, Dimension: 128, BaseVectorCount: 128, InsertedDeltaCount: 16, DeletedBaseCount: 64, TopK: 25, QueryCount: 4, AllowlistKind: "selective", CandidateSetKind: "very-selective"),
        new(VectorMetric.Cosine, Dimension: 386, BaseVectorCount: 192, InsertedDeltaCount: 64, DeletedBaseCount: 32, TopK: 50, QueryCount: 5, AllowlistKind: "broad", CandidateSetKind: "empty"),
        new(VectorMetric.Cosine, Dimension: 768, BaseVectorCount: 256, InsertedDeltaCount: 80, DeletedBaseCount: 120, TopK: 100, QueryCount: 3, AllowlistKind: "all", CandidateSetKind: "broad")
    ];

    public static GeneratedExactCheckpointMatrixManifest Run(
        GeneratedExactCheckpointMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = GeneratedExactCheckpointMatrixOptions.NormalizePresetName(options.PresetName);
        GeneratedExactCheckpointMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new GeneratedExactCheckpointMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            GeneratedExactCheckpointMatrixCase matrixCase = cases[i];
            string[] caseArguments = CreateCaseArguments(matrixCase.Options);
            try
            {
                GeneratedExactCheckpointBenchmarkReport report = GeneratedExactCheckpointScenario.Run(matrixCase.Options, caseArguments);
                GeneratedExactCheckpointScenario.Write(report, matrixCase.Options.OutputPath);

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
                    report.ReportId,
                    casePassed ? "passed" : "failed",
                    report.Validation.Status,
                    errorMessage: null);
            }
            catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException or InvalidOperationException)
            {
                failed++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    matrixCase,
                    caseArguments,
                    reportId: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        return new GeneratedExactCheckpointMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            GeneratedExactCheckpointMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactCheckpointMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            failed == 0 ? "passed" : "failed",
            caseManifests,
            new GeneratedExactCheckpointMatrixAggregate(passed, failed),
            new GeneratedExactCheckpointMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                PreviewReadinessEligible: false,
                "Generated exact checkpoint matrix output is private local evidence only; no reviewed public checkpoint-performance or persisted-size summary policy exists.",
                "No exact-checkpoint baseline-candidate policy is accepted for VEC-069.",
                "No exact-checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted for VEC-069.",
                "Checkpoint matrix output does not establish resource, durability, crash, concurrency, Linux or release-package readiness."),
            [
                "Generated exact checkpoint matrix evidence only; no external datasets, real workload labels or public storage/performance evidence are used.",
                "Each case reuses the accepted generated-exact-checkpoint scenario and VecNet.ExactCheckpointBenchmarkReport schema 0.1 measurement semantics.",
                "Checkpoint duration in linked reports measures only public ExactFlatIndex.Checkpoint(directoryPath) as defined by VEC-067.",
                "Setup, data generation, base build, mutation execution, truth construction, filter/candidate input generation, candidate-set construction, validation, reopened-output checks, output-byte scans, cleanup and report writing stay outside checkpoint timing.",
                "Live-view Save, post-checkpoint search timing, checkpoint allocations, resident/process memory and retained HashSet capacity remain not measured or unavailable in linked VEC-067 reports.",
                "The standard preset broadens generated exact checkpoint metric, dimension, base count, insert/delete pressure, top-k, query-count and selectivity coverage but remains private evidence, not a baseline candidate, regression gate, preview-readiness result or public benchmark claim.",
                "No HNSW checkpoint/durability, active durable-location replacement, VectorData, SQL/database comparison, compression, SSD/DiskANN, public documentation or production dependency is introduced by this matrix manifest."
            ]);
    }

    public static GeneratedExactCheckpointMatrixCase[] ExpandCases(GeneratedExactCheckpointMatrixOptions options)
    {
        CheckpointMatrixPresetCase[] presetCases = GetPresetCases(options.PresetName);
        var cases = new GeneratedExactCheckpointMatrixCase[presetCases.Length];
        for (int i = 0; i < presetCases.Length; i++)
        {
            CheckpointMatrixPresetCase presetCase = presetCases[i];
            uint seed = unchecked(options.Seed + (uint)i);
            var caseOptions = new GeneratedExactCheckpointOptions(
                presetCase.Metric,
                presetCase.Dimension,
                presetCase.BaseVectorCount,
                presetCase.QueryCount,
                presetCase.TopK,
                seed,
                presetCase.InsertedDeltaCount,
                presetCase.DeletedBaseCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                presetCase.AllowlistKind,
                presetCase.CandidateSetKind,
                options.DuplicateIdsPerQuery,
                options.UnknownIdsPerQuery,
                CreateReportPath(options.OutputDirectory, i + 1, presetCase),
                options.Runs,
                options.WarmupQueries);
            cases[i] = new GeneratedExactCheckpointMatrixCase(CreateCaseId(i + 1, presetCase), caseOptions);
        }

        return cases;
    }

    public static void WriteManifest(GeneratedExactCheckpointMatrixManifest manifest, string manifestPath) =>
        ReportWriter.WriteJson(manifest, manifestPath);

    public static string[] CreateCaseArguments(GeneratedExactCheckpointOptions options) =>
    [
        GeneratedExactCheckpointOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.BaseVectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--insertions", options.InsertedDeltaCount.ToString(CultureInfo.InvariantCulture),
        "--deletes", options.DeletedBaseCount.ToString(CultureInfo.InvariantCulture),
        "--duplicate-inserts", options.DuplicateInsertAttempts.ToString(CultureInfo.InvariantCulture),
        "--unknown-deletes", options.UnknownDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--repeated-deletes", options.RepeatedDeleteAttempts.ToString(CultureInfo.InvariantCulture),
        "--allowlist", options.AllowlistKind,
        "--candidate-set", options.CandidateSetKind,
        "--duplicate-ids", options.DuplicateIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--unknown-ids", options.UnknownIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--output", options.OutputPath
    ];

    private static GeneratedExactCheckpointMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        GeneratedExactCheckpointMatrixCase matrixCase,
        string[] commandArguments,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        GeneratedExactCheckpointOptions options = matrixCase.Options;
        double expectedTombstoneRatio = options.PhysicalVectorCount == 0
            ? 0
            : (double)options.DeletedBaseCount / options.PhysicalVectorCount;

        return new GeneratedExactCheckpointMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            options.Metric.ToString(),
            options.Dimension,
            options.BaseVectorCount,
            options.InsertedDeltaCount,
            options.DeletedBaseCount,
            options.PhysicalVectorCount,
            options.LiveVectorCount,
            expectedTombstoneRatio,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            options.AllowlistKind,
            options.CandidateSetKind,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            "new-or-empty-directory",
            "per-run fresh ignored artifact directory under the per-case report output directory",
            options.OutputPath,
            commandArguments,
            reportId,
            status,
            validationStatus,
            errorMessage);
    }

    private static string CreateReportPath(
        string outputDirectory,
        int caseNumber,
        CheckpointMatrixPresetCase presetCase) =>
        Path.Combine(outputDirectory, CreateCaseId(caseNumber, presetCase) + ".json");

    private static string CreateCaseId(int caseNumber, CheckpointMatrixPresetCase presetCase) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"case-{caseNumber:D3}-{presetCase.Metric.ToString().ToLowerInvariant()}-{presetCase.Dimension}d-{presetCase.BaseVectorCount}b-{presetCase.InsertedDeltaCount}i-{presetCase.DeletedBaseCount}d-{presetCase.TopK}k-{presetCase.AllowlistKind}-allowlist-{presetCase.CandidateSetKind}-candidate-set");

    private static CheckpointMatrixPresetCase[] GetPresetCases(string presetName)
    {
        string normalizedPresetName = GeneratedExactCheckpointMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            GeneratedExactCheckpointMatrixOptions.SmokePresetName => SmokeCases,
            GeneratedExactCheckpointMatrixOptions.StandardPresetName => StandardCases,
            _ => throw new ArgumentException($"Unsupported generated exact checkpoint matrix preset '{presetName}'.")
        };
    }

    public sealed record GeneratedExactCheckpointMatrixCase(string CaseId, GeneratedExactCheckpointOptions Options);

    private sealed record CheckpointMatrixPresetCase(
        VectorMetric Metric,
        int Dimension,
        int BaseVectorCount,
        int InsertedDeltaCount,
        int DeletedBaseCount,
        int TopK,
        int QueryCount,
        string AllowlistKind,
        string CandidateSetKind);
}
