using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactMatrixScenario
{
    private const string TaskId = "VEC-015";

    private static readonly VectorMetric[] Metrics =
    [
        VectorMetric.SquaredEuclidean,
        VectorMetric.InnerProduct,
        VectorMetric.Cosine
    ];

    private static readonly int[] SmokeDimensions = [32, 128, 386];
    private static readonly int[] SmokeTopKValues = [1, 10];
    private static readonly int[] StandardDimensions = [32, 128, 386, 768];
    private static readonly int[] StandardTopKValues = [1, 10, 100];

    public static GeneratedExactMatrixManifest Run(GeneratedExactMatrixOptions options, IReadOnlyList<string> commandArguments)
    {
        string presetName = GeneratedExactMatrixOptions.NormalizePresetName(options.PresetName);
        GeneratedExactSearchOptions[] cases = ExpandCases(options);
        var caseManifests = new GeneratedExactMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            GeneratedExactSearchOptions caseOptions = cases[i];
            try
            {
                BenchmarkReport report = GeneratedExactSearchScenario.Run(caseOptions, CreateCaseArguments(caseOptions));
                ReportWriter.Write(report, caseOptions.OutputPath);

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
                    caseOptions,
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
                    caseOptions,
                    reportId: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        var manifest = new GeneratedExactMatrixManifest(
            SchemaName: "VecNet.BenchmarkMatrixManifest",
            SchemaVersion: "0.1",
            TaskId: TaskId,
            ScenarioName: GeneratedExactMatrixOptions.ScenarioName,
            PresetName: presetName,
            GeneratedAtUtc: DateTimeOffset.UtcNow,
            Repository: RepositoryInfo.Create(),
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            OutputDirectory: options.OutputDirectory,
            CaseCount: caseManifests.Length,
            Cases: caseManifests,
            Aggregate: new GeneratedExactMatrixAggregate(passed, failed),
            Eligibility: new GeneratedExactMatrixEligibility(
                ClaimClass: "local-evidence",
                PrivacyClass: "private-raw",
                EvidenceStatus: "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                Reason: "Generated exact matrix output is private smoke evidence only; it does not implement baseline comparison math, regression thresholds or public claim review."),
            Notes:
            [
                "Generated finite data only; no external datasets are used.",
                "Each case uses scalar-reference truth and public ExactFlatIndex search through the existing exact-generated scenario.",
                "Repeated-run, warmup and managed allocation measurement semantics are inherited from the per-case schema 0.1 reports.",
                "Single-query concurrency only; ANN, persistence, filtering, updates and concurrency are out of scope for VEC-015."
            ]);

        return BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);
    }

    public static GeneratedExactSearchOptions[] ExpandCases(GeneratedExactMatrixOptions options)
    {
        MatrixPreset preset = GetPreset(options.PresetName);
        var cases = new List<GeneratedExactSearchOptions>(Metrics.Length * preset.Dimensions.Length * preset.TopKValues.Length);
        int caseIndex = 0;
        foreach (VectorMetric metric in Metrics)
        {
            foreach (int dimension in preset.Dimensions)
            {
                foreach (int topK in preset.TopKValues)
                {
                    if (topK > options.VectorCount)
                    {
                        throw new ArgumentException("top-k must be less than or equal to the vector count for every matrix case.");
                    }

                    uint seed = unchecked(options.Seed + (uint)caseIndex);
                    cases.Add(new GeneratedExactSearchOptions(
                        metric,
                        dimension,
                        options.VectorCount,
                        options.QueryCount,
                        topK,
                        seed,
                        CreateReportPath(options.OutputDirectory, caseIndex + 1, metric, dimension, topK),
                        null,
                        options.Runs,
                        options.WarmupQueries));
                    caseIndex++;
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetPreset(presetName).TopKValues.Max();

    public static void WriteManifest(GeneratedExactMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    private static GeneratedExactMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        GeneratedExactSearchOptions options,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage) =>
        new(
            caseNumber,
            options.Metric.ToString(),
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
            options.OutputPath,
            reportId,
            status,
            validationStatus,
            errorMessage);

    private static string[] CreateCaseArguments(GeneratedExactSearchOptions options) =>
    [
        GeneratedExactSearchOptions.ScenarioName,
        "--metric", options.Metric.ToString(),
        "--dimension", options.Dimension.ToString(CultureInfo.InvariantCulture),
        "--vectors", options.VectorCount.ToString(CultureInfo.InvariantCulture),
        "--queries", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}"),
        "--output", options.OutputPath
    ];

    private static string CreateReportPath(
        string outputDirectory,
        int caseNumber,
        VectorMetric metric,
        int dimension,
        int topK) =>
        Path.Combine(
            outputDirectory,
            string.Create(
                CultureInfo.InvariantCulture,
                $"case-{caseNumber:D2}-{metric.ToString().ToLowerInvariant()}-{dimension}d-{topK}k.json"));

    private static MatrixPreset GetPreset(string presetName)
    {
        string normalizedPresetName = GeneratedExactMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            GeneratedExactMatrixOptions.SmokePresetName => new MatrixPreset(SmokeDimensions, SmokeTopKValues),
            GeneratedExactMatrixOptions.StandardPresetName => new MatrixPreset(StandardDimensions, StandardTopKValues),
            _ => throw new ArgumentException($"Unsupported matrix preset '{presetName}'.")
        };
    }

    private sealed record MatrixPreset(int[] Dimensions, int[] TopKValues);
}
