using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactUpdateMatrixScenario
{
    private const string TaskId = "VEC-062";
    private const string SchemaName = "VecNet.ExactUpdateBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly UpdateMatrixPresetCase[] SmokeCases =
    [
        new(VectorMetric.SquaredEuclidean, Dimension: 32, BaseVectorCount: 32, QueryCount: 2, TopK: 5, InsertedDeltaCount: 4, DeletedBaseCount: 2, AllowlistKind: "broad", CandidateSetKind: "selective"),
        new(VectorMetric.InnerProduct, Dimension: 128, BaseVectorCount: 48, QueryCount: 3, TopK: 10, InsertedDeltaCount: 8, DeletedBaseCount: 6, AllowlistKind: "selective", CandidateSetKind: "broad"),
        new(VectorMetric.Cosine, Dimension: 32, BaseVectorCount: 40, QueryCount: 2, TopK: 10, InsertedDeltaCount: 8, DeletedBaseCount: 5, AllowlistKind: "very-selective", CandidateSetKind: "selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 386, BaseVectorCount: 64, QueryCount: 2, TopK: 25, InsertedDeltaCount: 16, DeletedBaseCount: 20, AllowlistKind: "empty", CandidateSetKind: "all")
    ];

    private static readonly UpdateMatrixPresetCase[] StandardCases =
    [
        new(VectorMetric.SquaredEuclidean, Dimension: 32, BaseVectorCount: 64, QueryCount: 3, TopK: 10, InsertedDeltaCount: 8, DeletedBaseCount: 4, AllowlistKind: "all", CandidateSetKind: "broad"),
        new(VectorMetric.SquaredEuclidean, Dimension: 128, BaseVectorCount: 96, QueryCount: 4, TopK: 10, InsertedDeltaCount: 24, DeletedBaseCount: 12, AllowlistKind: "broad", CandidateSetKind: "selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 386, BaseVectorCount: 128, QueryCount: 5, TopK: 25, InsertedDeltaCount: 32, DeletedBaseCount: 32, AllowlistKind: "selective", CandidateSetKind: "very-selective"),
        new(VectorMetric.SquaredEuclidean, Dimension: 768, BaseVectorCount: 160, QueryCount: 3, TopK: 50, InsertedDeltaCount: 40, DeletedBaseCount: 60, AllowlistKind: "empty", CandidateSetKind: "all"),
        new(VectorMetric.InnerProduct, Dimension: 32, BaseVectorCount: 64, QueryCount: 4, TopK: 10, InsertedDeltaCount: 16, DeletedBaseCount: 8, AllowlistKind: "selective", CandidateSetKind: "broad"),
        new(VectorMetric.InnerProduct, Dimension: 128, BaseVectorCount: 128, QueryCount: 5, TopK: 25, InsertedDeltaCount: 32, DeletedBaseCount: 16, AllowlistKind: "all", CandidateSetKind: "empty"),
        new(VectorMetric.InnerProduct, Dimension: 386, BaseVectorCount: 160, QueryCount: 3, TopK: 50, InsertedDeltaCount: 40, DeletedBaseCount: 48, AllowlistKind: "very-selective", CandidateSetKind: "selective"),
        new(VectorMetric.InnerProduct, Dimension: 768, BaseVectorCount: 240, QueryCount: 4, TopK: 100, InsertedDeltaCount: 60, DeletedBaseCount: 80, AllowlistKind: "broad", CandidateSetKind: "all"),
        new(VectorMetric.Cosine, Dimension: 32, BaseVectorCount: 72, QueryCount: 3, TopK: 10, InsertedDeltaCount: 12, DeletedBaseCount: 24, AllowlistKind: "empty", CandidateSetKind: "selective"),
        new(VectorMetric.Cosine, Dimension: 128, BaseVectorCount: 128, QueryCount: 4, TopK: 25, InsertedDeltaCount: 16, DeletedBaseCount: 64, AllowlistKind: "selective", CandidateSetKind: "very-selective"),
        new(VectorMetric.Cosine, Dimension: 386, BaseVectorCount: 192, QueryCount: 5, TopK: 50, InsertedDeltaCount: 64, DeletedBaseCount: 32, AllowlistKind: "broad", CandidateSetKind: "empty"),
        new(VectorMetric.Cosine, Dimension: 768, BaseVectorCount: 256, QueryCount: 3, TopK: 100, InsertedDeltaCount: 80, DeletedBaseCount: 120, AllowlistKind: "all", CandidateSetKind: "broad")
    ];

    public static GeneratedExactUpdateMatrixManifest Run(
        GeneratedExactUpdateMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = GeneratedExactUpdateMatrixOptions.NormalizePresetName(options.PresetName);
        GeneratedExactUpdateMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new GeneratedExactUpdateMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            GeneratedExactUpdateMatrixCase matrixCase = cases[i];
            try
            {
                string[] caseArguments = CreateCaseArguments(matrixCase.Options);
                GeneratedExactUpdateBenchmarkReport report = GeneratedExactUpdateScenario.Run(matrixCase.Options, caseArguments);
                GeneratedExactUpdateScenario.Write(report, matrixCase.Options.OutputPath);

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
                    CreateCaseArguments(matrixCase.Options),
                    reportId: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        return new GeneratedExactUpdateMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            GeneratedExactUpdateMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactUpdateMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            caseManifests,
            new GeneratedExactUpdateMatrixAggregate(passed, failed),
            new GeneratedExactUpdateMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated exact update matrix output is private local evidence only; no reviewed public update-performance summary policy exists.",
                "No exact-update baseline-candidate policy is accepted for VEC-062.",
                "No exact-update regression-gate policy, threshold, comparison artifact or hard gate is accepted for VEC-062."),
            [
                "Generated exact update matrix evidence only; no external datasets, real workload labels or public update-performance evidence are used.",
                "Each case reuses the accepted generated-exact-update scenario and VecNet.ExactUpdateBenchmarkReport schema 0.1 measurement semantics.",
                "Per-case generated data setup, base index build, mutation execution, live truth construction, allowlist/candidate input generation, post-mutation candidate-set construction, warmup, final-run result capture/comparison and report writing remain excluded from measured search latency and allocation.",
                "Mutation latency/allocation, live-view Save cost, resident/process memory, checkpoint/rebuild timing and retained tombstone HashSet memory remain not measured or unavailable in per-case VEC-061 reports.",
                "The standard preset broadens generated exact update metric, dimension, base count, insert/delete pressure, top-k, query-count and selectivity coverage but remains private evidence, not a baseline candidate, regression gate or public benchmark claim.",
                "No checkpoint/rebuild, HNSW update/durability, durable delta segment, stored label, VectorData, SQL/database, compression, SSD/DiskANN, comparison artifact, public documentation or production dependency is introduced by this matrix manifest."
            ]);
    }

    public static GeneratedExactUpdateMatrixCase[] ExpandCases(GeneratedExactUpdateMatrixOptions options)
    {
        UpdateMatrixPresetCase[] presetCases = GetPresetCases(options.PresetName);
        var cases = new GeneratedExactUpdateMatrixCase[presetCases.Length];
        for (int i = 0; i < presetCases.Length; i++)
        {
            UpdateMatrixPresetCase presetCase = presetCases[i];
            uint seed = unchecked(options.Seed + (uint)i);
            var caseOptions = new GeneratedExactUpdateOptions(
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
            cases[i] = new GeneratedExactUpdateMatrixCase(caseOptions);
        }

        return cases;
    }

    public static void WriteManifest(GeneratedExactUpdateMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(GeneratedExactUpdateOptions options) =>
    [
        GeneratedExactUpdateOptions.ScenarioName,
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

    private static GeneratedExactUpdateMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        GeneratedExactUpdateMatrixCase matrixCase,
        string[] commandArguments,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        GeneratedExactUpdateOptions options = matrixCase.Options;
        int expectedLiveVectorCount = checked(options.BaseVectorCount + options.InsertedDeltaCount - options.DeletedBaseCount);
        double expectedTombstoneRatio = options.PhysicalVectorCount == 0
            ? 0
            : (double)options.DeletedBaseCount / options.PhysicalVectorCount;

        return new GeneratedExactUpdateMatrixCaseManifest(
            caseNumber,
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
            options.InsertedDeltaCount,
            options.DeletedBaseCount,
            expectedTombstoneRatio,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            options.AllowlistKind,
            options.CandidateSetKind,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
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
        UpdateMatrixPresetCase presetCase) =>
        Path.Combine(
            outputDirectory,
            string.Create(
                CultureInfo.InvariantCulture,
                $"case-{caseNumber:D3}-{presetCase.Metric.ToString().ToLowerInvariant()}-{presetCase.Dimension}d-{presetCase.BaseVectorCount}b-{presetCase.InsertedDeltaCount}i-{presetCase.DeletedBaseCount}d-{presetCase.TopK}k-{presetCase.AllowlistKind}-allowlist-{presetCase.CandidateSetKind}-candidate-set.json"));

    private static UpdateMatrixPresetCase[] GetPresetCases(string presetName)
    {
        string normalizedPresetName = GeneratedExactUpdateMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            GeneratedExactUpdateMatrixOptions.SmokePresetName => SmokeCases,
            GeneratedExactUpdateMatrixOptions.StandardPresetName => StandardCases,
            _ => throw new ArgumentException($"Unsupported generated exact update matrix preset '{presetName}'.")
        };
    }

    public sealed record GeneratedExactUpdateMatrixCase(GeneratedExactUpdateOptions Options);

    private sealed record UpdateMatrixPresetCase(
        VectorMetric Metric,
        int Dimension,
        int BaseVectorCount,
        int QueryCount,
        int TopK,
        int InsertedDeltaCount,
        int DeletedBaseCount,
        string AllowlistKind,
        string CandidateSetKind);
}
