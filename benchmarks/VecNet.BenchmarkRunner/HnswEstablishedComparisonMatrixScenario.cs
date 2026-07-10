using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class HnswEstablishedComparisonMatrixScenario
{
    private const string TaskId = "VEC-119";
    private const string SchemaName = "VecNet.HnswEstablishedComparisonMatrixManifest";
    private const string SchemaVersion = "0.1";

    private static readonly int[] SmokeTopKValues = [10];
    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly HnswComparisonMatrixProfile[] MatrixProfiles =
    [
        new("balanced-m8", M: 8, EfConstruction: 64, EfSearch: 128),
        new("wide-m16", M: 16, EfConstruction: 128, EfSearch: 192),
        new("default-m16", M: 16, EfConstruction: 200, EfSearch: 200)
    ];

    public static HnswEstablishedComparisonMatrixManifest Run(
        HnswEstablishedComparisonMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = HnswEstablishedComparisonMatrixOptions.NormalizePresetName(options.PresetName);
        HnswComparisonMatrixCase[] cases = ExpandCases(options);
        var caseManifests = new HnswEstablishedComparisonMatrixCaseManifest[cases.Length];
        int passed = 0;
        int failed = 0;
        int blocked = 0;

        for (int i = 0; i < cases.Length; i++)
        {
            HnswComparisonMatrixCase matrixCase = cases[i];
            try
            {
                HnswEstablishedComparisonReport report = HnswEstablishedComparisonScenario.Run(
                    matrixCase.Options,
                    CreateCaseArguments(matrixCase.Options));
                HnswEstablishedComparisonScenario.Write(report, matrixCase.Options.OutputPath);

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
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
            {
                blocked++;
                caseManifests[i] = CreateCaseManifest(
                    i + 1,
                    matrixCase,
                    reportId: null,
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
                    reportId: null,
                    status: "failed",
                    validationStatus: "failed",
                    ex.Message);
            }
        }

        return new HnswEstablishedComparisonMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            HnswEstablishedComparisonMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswEstablishedComparisonMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            CreateSourcePinning(),
            CreateDesign(presetName),
            caseManifests.Length,
            caseManifests,
            new HnswEstablishedComparisonMatrixAggregate(passed, failed, SkippedCaseCount: 0, blocked),
            CreateEligibility(),
            [
                "Private generated hnswlib comparison matrix only; not a public performance, recall, memory, capacity, storage-size, baseline, comparison-publication or regression-gate claim.",
                "Each case reuses the accepted VEC-118 hnswlib-generated-comparison report schema and writes a linked private per-case report only when the external tool actually runs.",
                "Generated squared-L2 data only; no Fashion-MNIST or other external dataset is used by this matrix.",
                "Representative dimensions 128, 384 and 768 are present in both smoke and standard presets; optional adversarial dimension 386 is not part of the matrix presets and must not replace 384.",
                "The accepted balanced-m8, wide-m16 and default-m16 profiles are present in both presets."
            ]);
    }

    public static HnswComparisonMatrixCase[] ExpandCases(HnswEstablishedComparisonMatrixOptions options)
    {
        string presetName = HnswEstablishedComparisonMatrixOptions.NormalizePresetName(options.PresetName);
        int[] topKValues = GetTopKValues(presetName);
        var cases = new List<HnswComparisonMatrixCase>(
            HnswEstablishedComparisonOptions.RepresentativeDimensions.Length * topKValues.Length * MatrixProfiles.Length);
        int caseIndex = 0;

        foreach (int dimension in HnswEstablishedComparisonOptions.RepresentativeDimensions)
        {
            foreach (int topK in topKValues)
            {
                foreach (HnswComparisonMatrixProfile profile in MatrixProfiles)
                {
                    if (profile.EfSearch < topK)
                    {
                        throw new InvalidOperationException("Matrix profile efSearch must be at least top-k for every case.");
                    }

                    uint dataSeed = unchecked(options.Seed + (uint)caseIndex);
                    ulong hnswSeed = CreateHnswSeed(options.Seed, caseIndex);
                    string caseId = CreateCaseId(caseIndex + 1, profile.Name, dimension, topK);
                    string caseDirectory = Path.Combine(options.OutputDirectory, caseId);
                    var caseOptions = new HnswEstablishedComparisonOptions(
                        VectorMetric.SquaredEuclidean,
                        dimension,
                        options.VectorCount,
                        options.QueryCount,
                        topK,
                        dataSeed,
                        Path.Combine(options.OutputDirectory, $"{caseId}.json"),
                        Path.Combine(caseDirectory, "work"),
                        Path.Combine(caseDirectory, "vecnet-snapshot"),
                        Path.Combine(caseDirectory, "hnswlib-index.bin"),
                        options.HnswlibPythonPath,
                        options.Runs,
                        options.WarmupQueries,
                        profile.M,
                        profile.EfConstruction,
                        profile.EfSearch,
                        hnswSeed);
                    cases.Add(new HnswComparisonMatrixCase(caseId, profile.Name, caseOptions));
                    caseIndex++;
                }
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetTopKValues(presetName).Max();

    public static void WriteManifest(HnswEstablishedComparisonMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(HnswEstablishedComparisonOptions options) =>
    [
        HnswEstablishedComparisonOptions.ScenarioName,
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
        "--hnswlib-python", options.HnswlibPythonPath,
        "--output", options.OutputPath,
        "--work-directory", options.WorkDirectory,
        "--vecnet-snapshot-directory", options.VecNetSnapshotDirectory,
        "--hnswlib-index", options.HnswlibIndexPath
    ];

    private static HnswEstablishedComparisonMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        HnswComparisonMatrixCase matrixCase,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        HnswEstablishedComparisonOptions options = matrixCase.Options;
        return new HnswEstablishedComparisonMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.ProfileName,
            options.Metric.ToString(),
            options.Dimension,
            GetDimensionRole(options.Dimension),
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

    private static HnswEstablishedComparisonSourcePinningInfo CreateSourcePinning() =>
        new(
            "hnswlib",
            HnswEstablishedComparisonOptions.HnswlibPackageName,
            HnswEstablishedComparisonOptions.HnswlibPackageSource,
            HnswEstablishedComparisonOptions.HnswlibVersion,
            HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256,
            HnswEstablishedComparisonOptions.HnswlibLicense,
            "Apache-2.0 dependency is used only by private, non-shipping comparison tooling and is not distributed with VecNet.",
            "hnswlib executes as Python/native external tooling through a private ignored virtual environment; VecNet remains managed .NET in-process.",
            "No hnswlib, Python or native asset is referenced by src/VecNet or included in the VecNet package.");

    private static HnswEstablishedComparisonMatrixDesignInfo CreateDesign(string presetName) =>
        new(
            HnswEstablishedComparisonOptions.RepresentativeDimensions,
            HnswEstablishedComparisonOptions.OptionalAdversarialDimensions,
            MatrixProfiles
                .Select(profile => new HnswEstablishedComparisonMatrixProfileInfo(profile.Name, profile.M, profile.EfConstruction, profile.EfSearch))
                .ToArray(),
            GetTopKValues(presetName),
            VectorMetric.SquaredEuclidean.ToString(),
            "Accepted matrix design preserves representative generated dimensions 128, 384 and 768 for every preset.",
            "Dimension 386 is optional adversarial/tail coverage only and must not replace representative dimension 384.",
            "Smoke and standard presets both include balanced-m8, wide-m16 and default-m16 profiles; standard adds top-k 100 coverage.");

    private static HnswEstablishedComparisonMatrixEligibility CreateEligibility() =>
        new(
            "local-evidence",
            "private-raw",
            "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "Private hnswlib comparison matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "Established-implementation comparison matrix output is not a VecNet baseline candidate.",
            "No accepted public comparison-summary policy exists for VecNet versus hnswlib output.",
            "No hnswlib comparison regression-gate policy or threshold exists.");

    private static int[] GetTopKValues(string presetName)
    {
        string normalizedPresetName = HnswEstablishedComparisonMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            HnswEstablishedComparisonMatrixOptions.SmokePresetName => SmokeTopKValues,
            HnswEstablishedComparisonMatrixOptions.StandardPresetName => StandardTopKValues,
            _ => throw new ArgumentException($"Unsupported hnswlib comparison matrix preset '{presetName}'.")
        };
    }

    private static string GetDimensionRole(int dimension) =>
        HnswEstablishedComparisonOptions.RepresentativeDimensions.Contains(dimension)
            ? "representative"
            : HnswEstablishedComparisonOptions.OptionalAdversarialDimensions.Contains(dimension)
                ? "optional-adversarial-tail"
                : "custom";

    private static string CreateCaseId(int caseNumber, string profileName, int dimension, int topK) =>
        string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D2}-{profileName}-{dimension}d-{topK}k");

    private static ulong CreateHnswSeed(uint baseSeed, int caseIndex) =>
        0x484E5357_00011900UL ^ ((ulong)baseSeed << 16) ^ (uint)(caseIndex + 1);

    public sealed record HnswComparisonMatrixCase(
        string CaseId,
        string ProfileName,
        HnswEstablishedComparisonOptions Options);

    private sealed record HnswComparisonMatrixProfile(
        string Name,
        int M,
        int EfConstruction,
        int EfSearch);
}
