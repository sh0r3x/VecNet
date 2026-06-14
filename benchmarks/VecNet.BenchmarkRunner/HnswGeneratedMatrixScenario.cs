using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class HnswGeneratedMatrixScenario
{
    private const string TaskId = "VEC-037";
    private const string SchemaName = "VecNet.HnswBenchmarkMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly int[] SmokeDimensions = [16, 32];
    private static readonly int[] SmokeTopKValues = [1, 10];
    private static readonly HnswMatrixProfile[] SmokeProfiles =
    [
        new("low-ef-m4", M: 4, EfConstruction: 16, EfSearch: 10),
        new("balanced-m8", M: 8, EfConstruction: 32, EfSearch: 24)
    ];

    private static readonly int[] StandardDimensions = [32, 128, 386];
    private static readonly int[] StandardTopKValues = [1, 10, 50];
    private static readonly HnswMatrixProfile[] StandardProfiles =
    [
        new("low-ef-m4", M: 4, EfConstruction: 32, EfSearch: 50),
        new("balanced-m8", M: 8, EfConstruction: 64, EfSearch: 100),
        new("wide-m16", M: 16, EfConstruction: 128, EfSearch: 150)
    ];

    public static HnswGeneratedMatrixManifest Run(HnswGeneratedMatrixOptions options, IReadOnlyList<string> commandArguments)
    {
        string presetName = HnswGeneratedMatrixOptions.NormalizePresetName(options.PresetName);
        HnswMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new HnswGeneratedMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            HnswMatrixCase matrixCase = cases[i];
            try
            {
                HnswBenchmarkReport report = HnswGeneratedScenario.Run(
                    matrixCase.Options,
                    CreateCaseArguments(matrixCase.Options));
                HnswGeneratedScenario.Write(report, matrixCase.Options.OutputPath);

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

        return new HnswGeneratedMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            HnswGeneratedMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            options.OutputDirectory,
            caseManifests.Length,
            caseManifests,
            new HnswGeneratedMatrixAggregate(passed, failed),
            new HnswGeneratedMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Generated HNSW matrix output is private local smoke evidence only; no reviewed public summary policy exists.",
                "Generated HNSW baseline-candidate policy has not been accepted.",
                "Generated HNSW regression-gate policy has not been accepted."),
            [
                "Generated squared-L2 HNSW matrix smoke evidence only; no external datasets are used.",
                "Each case reuses the existing hnsw-generated scenario and VecNet.HnswBenchmarkReport schema 0.1 measurement semantics.",
                "Per-case HNSW build, exact truth generation, warmup, final-run result capture/comparison and report writing remain excluded from measured search latency and QPS.",
                "The standard preset broadens generated HNSW parameter coverage but remains private smoke evidence, not a baseline candidate, regression gate or public benchmark claim.",
                "No public HNSW API, HNSW algorithm behavior, comparison artifact, persistence, filtering, updates, optimization, external dataset HNSW mode or production dependency is introduced by this matrix manifest."
            ]);
    }

    public static HnswMatrixCase[] ExpandCases(HnswGeneratedMatrixOptions options)
    {
        HnswMatrixPreset preset = GetPreset(options.PresetName);
        var cases = new List<HnswMatrixCase>(preset.Dimensions.Length * preset.TopKValues.Length * preset.Profiles.Length);
        int caseIndex = 0;

        foreach (int dimension in preset.Dimensions)
        {
            foreach (int topK in preset.TopKValues)
            {
                foreach (HnswMatrixProfile profile in preset.Profiles)
                {
                    uint dataSeed = unchecked(options.Seed + (uint)caseIndex);
                    ulong hnswSeed = CreateHnswSeed(options.Seed, caseIndex);
                    var caseOptions = new HnswGeneratedOptions(
                        VectorMetric.SquaredEuclidean,
                        dimension,
                        options.VectorCount,
                        options.QueryCount,
                        topK,
                        dataSeed,
                        CreateReportPath(options.OutputDirectory, caseIndex + 1, profile.Name, dimension, topK),
                        options.Runs,
                        options.WarmupQueries,
                        profile.M,
                        profile.EfConstruction,
                        profile.EfSearch,
                        hnswSeed);
                    cases.Add(new HnswMatrixCase(profile.Name, caseOptions));
                    caseIndex++;
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetPreset(presetName).TopKValues.Max();

    public static void WriteManifest(HnswGeneratedMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(HnswGeneratedOptions options) =>
    [
        HnswGeneratedOptions.ScenarioName,
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
        "--output", options.OutputPath
    ];

    private static HnswGeneratedMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        HnswMatrixCase matrixCase,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        HnswGeneratedOptions options = matrixCase.Options;
        return new HnswGeneratedMatrixCaseManifest(
            caseNumber,
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
            reportId,
            status,
            validationStatus,
            errorMessage);
    }

    private static string CreateReportPath(
        string outputDirectory,
        int caseNumber,
        string profileName,
        int dimension,
        int topK) =>
        Path.Combine(
            outputDirectory,
            string.Create(
                CultureInfo.InvariantCulture,
                $"case-{caseNumber:D2}-{profileName}-{dimension}d-{topK}k.json"));

    private static HnswMatrixPreset GetPreset(string presetName)
    {
        string normalizedPresetName = HnswGeneratedMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            HnswGeneratedMatrixOptions.SmokePresetName => new HnswMatrixPreset(SmokeDimensions, SmokeTopKValues, SmokeProfiles),
            HnswGeneratedMatrixOptions.StandardPresetName => new HnswMatrixPreset(StandardDimensions, StandardTopKValues, StandardProfiles),
            _ => throw new ArgumentException($"Unsupported HNSW matrix preset '{presetName}'.")
        };
    }

    private static ulong CreateHnswSeed(uint baseSeed, int caseIndex) =>
        0x484E5357_00000000UL ^ ((ulong)baseSeed << 16) ^ (uint)(caseIndex + 1);

    public sealed record HnswMatrixCase(string ProfileName, HnswGeneratedOptions Options);

    private sealed record HnswMatrixPreset(int[] Dimensions, int[] TopKValues, HnswMatrixProfile[] Profiles);

    private sealed record HnswMatrixProfile(string Name, int M, int EfConstruction, int EfSearch);
}
