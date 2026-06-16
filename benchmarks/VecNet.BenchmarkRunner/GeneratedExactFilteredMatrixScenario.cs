using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactFilteredMatrixScenario
{
    private const string TaskId = "VEC-047";
    private const string SchemaName = "VecNet.ExactFilteredBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly VectorMetric[] SmokeMetrics =
    [
        VectorMetric.SquaredEuclidean,
        VectorMetric.InnerProduct
    ];

    private static readonly int[] SmokeDimensions = [32, 128];
    private static readonly int[] SmokeTopKValues = [10];
    private static readonly string[] SmokeFilterKinds = ["broad", "selective"];

    private static readonly VectorMetric[] StandardMetrics =
    [
        VectorMetric.SquaredEuclidean,
        VectorMetric.InnerProduct,
        VectorMetric.Cosine
    ];

    private static readonly int[] StandardDimensions = [32, 128, 386, 768];
    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly string[] StandardFilterKinds = ["all", "broad", "selective", "very-selective", "empty"];

    public static GeneratedExactFilteredMatrixManifest Run(
        GeneratedExactFilteredMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = GeneratedExactFilteredMatrixOptions.NormalizePresetName(options.PresetName);
        GeneratedExactFilteredMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new GeneratedExactFilteredMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            GeneratedExactFilteredMatrixCase matrixCase = cases[i];
            try
            {
                GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(
                    matrixCase.Options,
                    CreateCaseArguments(matrixCase.Options));
                GeneratedExactFilteredScenario.Write(report, matrixCase.Options.OutputPath);

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
                    reportId: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        return new GeneratedExactFilteredMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            GeneratedExactFilteredMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            caseManifests,
            new GeneratedExactFilteredMatrixAggregate(passed, failed),
            new GeneratedExactFilteredMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated exact-filter matrix output is private local smoke evidence only; no reviewed public summary policy exists.",
                "No generated exact-filter baseline-candidate policy is accepted in VEC-047.",
                "No generated exact-filter regression-gate policy is accepted in VEC-047."),
            [
                "Generated exact-filter matrix smoke evidence only; no external datasets, real workload labels or public selectivity evidence are used.",
                "Each case reuses the accepted exact-generated-filtered scenario and VecNet.ExactFilteredBenchmarkReport schema 0.1 measurement semantics.",
                "Per-case generated data setup, exact-flat index build, synthetic allowlist generation, workspace construction, warmup, scalar filtered truth, final-run result capture/comparison and report writing remain excluded from measured search latency and allocation.",
                "The standard preset broadens generated exact-filter metric, dimension, top-k and synthetic selectivity coverage but remains private smoke evidence, not a baseline candidate, regression gate or public benchmark claim.",
                "No filtered comparison artifact, retained ID-to-ordinal map, precompiled filter, stored label, external dataset filter, HNSW/ANN filtering, optimization decision, public documentation or production dependency is introduced by this matrix manifest."
            ]);
    }

    public static GeneratedExactFilteredMatrixCase[] ExpandCases(GeneratedExactFilteredMatrixOptions options)
    {
        FilteredMatrixPreset preset = GetPreset(options.PresetName);
        var cases = new List<GeneratedExactFilteredMatrixCase>(
            preset.Metrics.Length * preset.Dimensions.Length * preset.TopKValues.Length * preset.FilterKinds.Length);
        int caseIndex = 0;

        foreach (VectorMetric metric in preset.Metrics)
        {
            foreach (int dimension in preset.Dimensions)
            {
                foreach (int topK in preset.TopKValues)
                {
                    foreach (string filterKind in preset.FilterKinds)
                    {
                        uint seed = unchecked(options.Seed + (uint)caseIndex);
                        var caseOptions = new GeneratedExactFilteredOptions(
                            metric,
                            dimension,
                            options.VectorCount,
                            options.QueryCount,
                            topK,
                            seed,
                            filterKind,
                            options.DuplicateIdsPerQuery,
                            options.UnknownIdsPerQuery,
                            CreateReportPath(options.OutputDirectory, caseIndex + 1, metric, dimension, topK, filterKind),
                            options.Runs,
                            options.WarmupQueries);
                        cases.Add(new GeneratedExactFilteredMatrixCase(caseOptions));
                        caseIndex++;
                    }
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetPreset(presetName).TopKValues.Max();

    public static void WriteManifest(GeneratedExactFilteredMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(GeneratedExactFilteredOptions options) =>
    [
        GeneratedExactFilteredOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.VectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--filter", options.FilterKind,
        "--duplicate-ids", options.DuplicateIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--unknown-ids", options.UnknownIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--output", options.OutputPath
    ];

    private static GeneratedExactFilteredMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        GeneratedExactFilteredMatrixCase matrixCase,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        GeneratedExactFilteredOptions options = matrixCase.Options;
        return new GeneratedExactFilteredMatrixCaseManifest(
            caseNumber,
            options.Metric.ToString(),
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            options.FilterKind,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            options.OutputPath,
            reportId,
            status,
            validationStatus,
            errorMessage);
    }

    private static string CreateReportPath(
        string outputDirectory,
        int caseNumber,
        VectorMetric metric,
        int dimension,
        int topK,
        string filterKind) =>
        Path.Combine(
            outputDirectory,
            string.Create(
                CultureInfo.InvariantCulture,
                $"case-{caseNumber:D3}-{metric.ToString().ToLowerInvariant()}-{dimension}d-{topK}k-{filterKind}.json"));

    private static FilteredMatrixPreset GetPreset(string presetName)
    {
        string normalizedPresetName = GeneratedExactFilteredMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            GeneratedExactFilteredMatrixOptions.SmokePresetName => new FilteredMatrixPreset(SmokeMetrics, SmokeDimensions, SmokeTopKValues, SmokeFilterKinds),
            GeneratedExactFilteredMatrixOptions.StandardPresetName => new FilteredMatrixPreset(StandardMetrics, StandardDimensions, StandardTopKValues, StandardFilterKinds),
            _ => throw new ArgumentException($"Unsupported generated exact-filter matrix preset '{presetName}'.")
        };
    }

    public sealed record GeneratedExactFilteredMatrixCase(GeneratedExactFilteredOptions Options);

    private sealed record FilteredMatrixPreset(VectorMetric[] Metrics, int[] Dimensions, int[] TopKValues, string[] FilterKinds);
}
