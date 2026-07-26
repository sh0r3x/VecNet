using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class DurableHnswGeneratedMatrixScenario
{
    private const string TaskId = "VEC-076";
    private const string SchemaName = "VecNet.DurableHnswBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";
    private const string LinkedReportSchemaName = "VecNet.DurableHnswBenchmarkReport";
    private const string LinkedReportSchemaVersion = "0.1";
    private const string LinkedReportTaskId = "VEC-074";
    private const string LinkedReportScenarioName = DurableHnswGeneratedOptions.ScenarioName;

    private static readonly DurableHnswMatrixPresetCase[] SmokeCases =
    [
        new("low-ef-m4", VectorMetric.SquaredEuclidean, Dimension: 16, VectorCount: 64, QueryCount: 3, TopK: 5, M: 4, EfConstruction: 16, EfSearch: 8, Runs: 1, WarmupQueries: 0),
        new("balanced-m8", VectorMetric.SquaredEuclidean, Dimension: 32, VectorCount: 128, QueryCount: 4, TopK: 10, M: 8, EfConstruction: 32, EfSearch: 24, Runs: 1, WarmupQueries: 1),
        new("wide-m12", VectorMetric.SquaredEuclidean, Dimension: 128, VectorCount: 192, QueryCount: 5, TopK: 25, M: 12, EfConstruction: 64, EfSearch: 64, Runs: 2, WarmupQueries: 1),
        new("tail-balanced-m8", VectorMetric.SquaredEuclidean, Dimension: 386, VectorCount: 96, QueryCount: 3, TopK: 25, M: 8, EfConstruction: 64, EfSearch: 64, Runs: 1, WarmupQueries: 1),
        new("low-ef-m4", VectorMetric.Cosine, Dimension: 16, VectorCount: 64, QueryCount: 3, TopK: 5, M: 4, EfConstruction: 16, EfSearch: 8, Runs: 1, WarmupQueries: 0),
        new("balanced-m8", VectorMetric.Cosine, Dimension: 32, VectorCount: 128, QueryCount: 4, TopK: 10, M: 8, EfConstruction: 32, EfSearch: 24, Runs: 1, WarmupQueries: 1),
        new("wide-m12", VectorMetric.Cosine, Dimension: 128, VectorCount: 192, QueryCount: 5, TopK: 25, M: 12, EfConstruction: 64, EfSearch: 64, Runs: 2, WarmupQueries: 1),
        new("tail-balanced-m8", VectorMetric.Cosine, Dimension: 386, VectorCount: 96, QueryCount: 3, TopK: 25, M: 8, EfConstruction: 64, EfSearch: 64, Runs: 1, WarmupQueries: 1)
    ];

    public static DurableHnswGeneratedMatrixManifest Run(
        DurableHnswGeneratedMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = DurableHnswGeneratedMatrixOptions.NormalizePresetName(options.PresetName);
        DurableHnswGeneratedMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new DurableHnswGeneratedMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            DurableHnswGeneratedMatrixCase matrixCase = cases[i];
            string[] caseArguments = CreateCaseArguments(matrixCase.Options);
            try
            {
                DurableHnswGeneratedOptions parsedCaseOptions = CommandLine.ParseDurableHnswGenerated(caseArguments);
                DurableHnswBenchmarkReport report = DurableHnswGeneratedScenario.Run(parsedCaseOptions, caseArguments);
                DurableHnswGeneratedScenario.Write(report, parsedCaseOptions.OutputPath);

                bool casePassed = IsLinkedReportCompatible(report);
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
                    errorType: null,
                    errorMessage: null);
            }
            catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException or InvalidOperationException)
            {
                failed++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    matrixCase,
                    caseArguments,
                    report: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.GetType().Name,
                    ex.Message);
            }
        }

        bool allLinkedReportsValidationPassed = caseManifests.All(
            item =>
                string.Equals(item.Status, "passed", StringComparison.OrdinalIgnoreCase) &&
                string.Equals(item.ValidationStatus, "passed", StringComparison.OrdinalIgnoreCase) &&
                string.Equals(item.LinkedReportSchemaName, LinkedReportSchemaName, StringComparison.Ordinal) &&
                string.Equals(item.LinkedReportSchemaVersion, LinkedReportSchemaVersion, StringComparison.Ordinal) &&
                string.Equals(item.LinkedReportTaskId, LinkedReportTaskId, StringComparison.Ordinal) &&
                string.Equals(item.LinkedReportScenarioName, LinkedReportScenarioName, StringComparison.Ordinal));
        bool allLinkedReportsPrivateRaw = allLinkedReportsValidationPassed;
        bool allLinkedReportsEligibilityFalse = allLinkedReportsValidationPassed;

        return new DurableHnswGeneratedMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            DurableHnswGeneratedMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(DurableHnswGeneratedMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            caseManifests,
            new DurableHnswGeneratedMatrixAggregate(passed, failed),
            new DurableHnswGeneratedMatrixValidation(
                failed == 0 && allLinkedReportsValidationPassed ? "passed" : "failed",
                passed,
                failed,
                LinkedReportSchemaName,
                LinkedReportSchemaVersion,
                LinkedReportScenarioName,
                allLinkedReportsValidationPassed,
                allLinkedReportsPrivateRaw,
                allLinkedReportsEligibilityFalse),
            new DurableHnswGeneratedMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                "Private generated durable-HNSW matrix output is not reviewed public evidence and has no public reporting policy.",
                "A bounded generated local smoke matrix does not establish hostile durable-open fuzzing, external data behavior, actual memory, crash/interruption behavior, concurrency, Linux validation, packageability or public API readiness.",
                "No durable-HNSW baseline-candidate policy is accepted for VEC-076.",
                "No durable-HNSW comparison schema or compatibility policy is accepted for VEC-076.",
                "No durable-HNSW threshold, hard gate or regression decision policy is accepted for VEC-076."),
            [
                "Generated SquaredEuclidean and Cosine durable-HNSW matrix smoke evidence only; no external datasets, public claims, preview readiness or baseline policy are introduced.",
                "Each successful case delegates to hnsw-generated-durable and writes a VecNet.DurableHnswBenchmarkReport schema 0.1 linked report.",
                "The linked VEC-074 report remains the source of truth for build, save, open, opened-search, recall, parity, returned-result integrity, read-only mutation posture and storage-byte file facts.",
                "The matrix manifest records case identity, command replay, linked report paths, snapshot roots, pass/fail accounting and false eligibility without averaging timings, comparing baselines or defining thresholds.",
                "No src/VecNet change, public HNSW profile admission, filtering, direct HNSW mutation, compression, VectorData, SSD/DiskANN or production dependency is introduced."
            ]);
    }

    public static DurableHnswGeneratedMatrixCase[] ExpandCases(DurableHnswGeneratedMatrixOptions options)
    {
        DurableHnswMatrixPresetCase[] presetCases = GetPresetCases(options.PresetName);
        var cases = new DurableHnswGeneratedMatrixCase[presetCases.Length];
        for (int i = 0; i < presetCases.Length; i++)
        {
            DurableHnswMatrixPresetCase presetCase = presetCases[i];
            uint dataSeed = unchecked(options.Seed + (uint)(i + 1));
            ulong hnswSeed = CreateHnswSeed(options.Seed, i + 1);
            string caseId = CreateCaseId(i + 1, presetCase);
            var caseOptions = new DurableHnswGeneratedOptions(
                presetCase.Metric,
                presetCase.Dimension,
                presetCase.VectorCount,
                presetCase.QueryCount,
                presetCase.TopK,
                dataSeed,
                Path.Combine(options.OutputDirectory, caseId + ".json"),
                Path.Combine(options.OutputDirectory, caseId + "-snapshot"),
                presetCase.Runs,
                presetCase.WarmupQueries,
                presetCase.M,
                presetCase.EfConstruction,
                presetCase.EfSearch,
                hnswSeed);
            cases[i] = new DurableHnswGeneratedMatrixCase(caseId, presetCase.ProfileName, caseOptions);
        }

        return cases;
    }

    public static void WriteManifest(DurableHnswGeneratedMatrixManifest manifest, string manifestPath) =>
        ReportWriter.WriteJson(manifest, manifestPath);

    public static string[] CreateCaseArguments(DurableHnswGeneratedOptions options) =>
    [
        DurableHnswGeneratedOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.VectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--hnsw-seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
        "--output", options.OutputPath,
        "--snapshot-directory", options.SnapshotDirectory
    ];

    private static DurableHnswGeneratedMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        DurableHnswGeneratedMatrixCase matrixCase,
        string[] commandArguments,
        DurableHnswBenchmarkReport? report,
        string status,
        string validationStatus,
        string? errorType,
        string? errorMessage)
    {
        DurableHnswGeneratedOptions options = matrixCase.Options;
        return new DurableHnswGeneratedMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.ProfileName,
            options.Metric.ToString(),
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            string.Create(CultureInfo.InvariantCulture, $"0x{options.HnswSeed:X16}"),
            options.M,
            options.EfConstruction,
            options.EfSearch,
            options.OutputPath,
            options.SnapshotDirectory,
            commandArguments,
            report?.ReportId,
            report?.SchemaName,
            report?.SchemaVersion,
            report?.TaskId,
            report?.ScenarioName,
            status,
            validationStatus,
            errorType,
            errorMessage);
    }

    private static bool IsLinkedReportCompatible(DurableHnswBenchmarkReport report) =>
        string.Equals(report.SchemaName, LinkedReportSchemaName, StringComparison.Ordinal) &&
        string.Equals(report.SchemaVersion, LinkedReportSchemaVersion, StringComparison.Ordinal) &&
        string.Equals(report.TaskId, LinkedReportTaskId, StringComparison.Ordinal) &&
        string.Equals(report.ScenarioName, LinkedReportScenarioName, StringComparison.Ordinal) &&
        string.Equals(report.ClaimClass, "local-evidence", StringComparison.Ordinal) &&
        string.Equals(report.PrivacyClass, "private-raw", StringComparison.Ordinal) &&
        string.Equals(report.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) &&
        !report.Evidence.PublicClaimEligible &&
        !report.Evidence.PreviewReadinessEligible &&
        !report.Evidence.BaselineCandidateEligible &&
        !report.Evidence.ComparisonArtifactEligible &&
        !report.Evidence.RegressionGateEligible &&
        !report.Validation.PublicClaimEligible &&
        !report.Validation.PreviewReadinessEligible &&
        !report.Validation.BaselineCandidateEligible &&
        !report.Validation.ComparisonArtifactEligible &&
        !report.Validation.RegressionGateEligible &&
        !report.Eligibility.PublicClaimEligible &&
        !report.Eligibility.PreviewReadinessEligible &&
        !report.Eligibility.BaselineCandidateEligible &&
        !report.Eligibility.ComparisonArtifactEligible &&
        !report.Eligibility.RegressionGateEligible &&
        report.Validation.ReportIsPrivateRaw;

    private static string CreateCaseId(int caseNumber, DurableHnswMatrixPresetCase presetCase) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"case-{caseNumber:D3}-{presetCase.Metric}-{presetCase.ProfileName}-{presetCase.Dimension}d-{presetCase.VectorCount}v-{presetCase.QueryCount}q-{presetCase.TopK}k");

    private static DurableHnswMatrixPresetCase[] GetPresetCases(string presetName)
    {
        string normalizedPresetName = DurableHnswGeneratedMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            DurableHnswGeneratedMatrixOptions.SmokePresetName => SmokeCases,
            _ => throw new ArgumentException($"Unsupported durable HNSW matrix preset '{presetName}'.")
        };
    }

    private static ulong CreateHnswSeed(uint baseSeed, int caseNumber) =>
        0x484E5357_44550000UL ^ ((ulong)baseSeed << 16) ^ (uint)caseNumber;

    public sealed record DurableHnswGeneratedMatrixCase(
        string CaseId,
        string ProfileName,
        DurableHnswGeneratedOptions Options);

    private sealed record DurableHnswMatrixPresetCase(
        string ProfileName,
        VectorMetric Metric,
        int Dimension,
        int VectorCount,
        int QueryCount,
        int TopK,
        int M,
        int EfConstruction,
        int EfSearch,
        int Runs,
        int WarmupQueries);
}
