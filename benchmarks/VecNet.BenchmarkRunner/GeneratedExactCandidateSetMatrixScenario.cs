using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactCandidateSetMatrixScenario
{
    private const string TaskId = "VEC-054";
    private const string SchemaName = "VecNet.ExactCandidateSetBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly VectorMetric[] SmokeMetrics =
    [
        VectorMetric.SquaredEuclidean,
        VectorMetric.InnerProduct
    ];

    private static readonly int[] SmokeDimensions = [32, 128];
    private static readonly int[] SmokeTopKValues = [10];
    private static readonly string[] SmokeCandidateSetKinds = ["broad", "selective"];

    private static readonly VectorMetric[] StandardMetrics =
    [
        VectorMetric.SquaredEuclidean,
        VectorMetric.InnerProduct,
        VectorMetric.Cosine
    ];

    private static readonly int[] StandardDimensions = [32, 128, 386, 768];
    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly string[] StandardCandidateSetKinds = ["all", "broad", "selective", "very-selective", "empty"];

    public static GeneratedExactCandidateSetMatrixManifest Run(
        GeneratedExactCandidateSetMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = GeneratedExactCandidateSetMatrixOptions.NormalizePresetName(options.PresetName);
        GeneratedExactCandidateSetMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new GeneratedExactCandidateSetMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            GeneratedExactCandidateSetMatrixCase matrixCase = cases[i];
            try
            {
                GeneratedExactCandidateSetBenchmarkReport report = GeneratedExactCandidateSetScenario.Run(
                    matrixCase.Options,
                    CreateCaseArguments(matrixCase.Options));
                GeneratedExactCandidateSetScenario.Write(report, matrixCase.Options.OutputPath);

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

        return new GeneratedExactCandidateSetMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            GeneratedExactCandidateSetMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            caseManifests,
            new GeneratedExactCandidateSetMatrixAggregate(passed, failed),
            new GeneratedExactCandidateSetMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated exact candidate-set matrix output is private local smoke evidence only; no reviewed public summary policy exists.",
                "No generated exact candidate-set baseline-candidate policy is accepted in VEC-054.",
                "No generated exact candidate-set regression-gate policy is accepted in VEC-054."),
            [
                "Generated exact candidate-set matrix smoke evidence only; no external datasets, real workload labels or public selectivity evidence are used.",
                "Each case reuses the accepted generated-exact-candidate-set scenario and VecNet.ExactCandidateSetBenchmarkReport schema 0.1 measurement semantics.",
                "Per-case generated data setup, exact-flat index build, candidate ID generation, candidate-set construction, warmup, scalar filtered truth, final-run result capture/comparison and report writing remain excluded from measured search latency and allocation.",
                "The standard preset broadens generated exact candidate-set metric, dimension, top-k and synthetic selectivity coverage but remains private smoke evidence, not a baseline candidate, regression gate or public benchmark claim.",
                "No raw allowlist-versus-candidate-set conclusion, comparison artifact, stored label, external dataset filter, HNSW/ANN filtering, optimization decision, public documentation or production dependency is introduced by this matrix manifest."
            ]);
    }

    public static GeneratedExactCandidateSetMatrixCase[] ExpandCases(GeneratedExactCandidateSetMatrixOptions options)
    {
        CandidateSetMatrixPreset preset = GetPreset(options.PresetName);
        var cases = new List<GeneratedExactCandidateSetMatrixCase>(
            preset.Metrics.Length * preset.Dimensions.Length * preset.TopKValues.Length * preset.CandidateSetKinds.Length);
        int caseIndex = 0;

        foreach (VectorMetric metric in preset.Metrics)
        {
            foreach (int dimension in preset.Dimensions)
            {
                foreach (int topK in preset.TopKValues)
                {
                    foreach (string candidateSetKind in preset.CandidateSetKinds)
                    {
                        uint seed = unchecked(options.Seed + (uint)caseIndex);
                        var caseOptions = new GeneratedExactCandidateSetOptions(
                            metric,
                            dimension,
                            options.VectorCount,
                            options.QueryCount,
                            topK,
                            seed,
                            candidateSetKind,
                            options.DuplicateIdsPerQuery,
                            options.UnknownIdsPerQuery,
                            CreateReportPath(options.OutputDirectory, caseIndex + 1, metric, dimension, topK, candidateSetKind),
                            options.Runs,
                            options.WarmupQueries);
                        cases.Add(new GeneratedExactCandidateSetMatrixCase(caseOptions));
                        caseIndex++;
                    }
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetPreset(presetName).TopKValues.Max();

    public static void WriteManifest(GeneratedExactCandidateSetMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(GeneratedExactCandidateSetOptions options) =>
    [
        GeneratedExactCandidateSetOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.VectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--candidate-set", options.CandidateSetKind,
        "--duplicate-ids", options.DuplicateIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--unknown-ids", options.UnknownIdsPerQuery.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--output", options.OutputPath
    ];

    private static GeneratedExactCandidateSetMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        GeneratedExactCandidateSetMatrixCase matrixCase,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        GeneratedExactCandidateSetOptions options = matrixCase.Options;
        return new GeneratedExactCandidateSetMatrixCaseManifest(
            caseNumber,
            options.Metric.ToString(),
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            options.CandidateSetKind,
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
        string candidateSetKind) =>
        Path.Combine(
            outputDirectory,
            string.Create(
                CultureInfo.InvariantCulture,
                $"case-{caseNumber:D3}-{metric.ToString().ToLowerInvariant()}-{dimension}d-{topK}k-{candidateSetKind}.json"));

    private static CandidateSetMatrixPreset GetPreset(string presetName)
    {
        string normalizedPresetName = GeneratedExactCandidateSetMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            GeneratedExactCandidateSetMatrixOptions.SmokePresetName => new CandidateSetMatrixPreset(SmokeMetrics, SmokeDimensions, SmokeTopKValues, SmokeCandidateSetKinds),
            GeneratedExactCandidateSetMatrixOptions.StandardPresetName => new CandidateSetMatrixPreset(StandardMetrics, StandardDimensions, StandardTopKValues, StandardCandidateSetKinds),
            _ => throw new ArgumentException($"Unsupported generated exact candidate-set matrix preset '{presetName}'.")
        };
    }

    public sealed record GeneratedExactCandidateSetMatrixCase(GeneratedExactCandidateSetOptions Options);

    private sealed record CandidateSetMatrixPreset(VectorMetric[] Metrics, int[] Dimensions, int[] TopKValues, string[] CandidateSetKinds);
}
