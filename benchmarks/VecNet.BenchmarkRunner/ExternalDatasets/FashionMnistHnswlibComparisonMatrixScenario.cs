using System.Globalization;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistHnswlibComparisonMatrixScenario
{
    private const string TaskId = "VEC-121";
    private const string SchemaName = "VecNet.FashionMnistHnswlibComparisonMatrixManifest";
    private const string SchemaVersion = "0.1";
    private const string DatasetId = "fashion-mnist-784-euclidean";
    private const int Dimension = 784;

    private static readonly int[] SmokeTopKValues = [10];
    private static readonly int[] StandardTopKValues = [10, 100];
    private static readonly FashionMnistComparisonMatrixProfile[] MatrixProfiles =
    [
        new("balanced-m8", M: 8, EfConstruction: 64, EfSearch: 128),
        new("wide-m16", M: 16, EfConstruction: 128, EfSearch: 192),
        new("default-m16", M: 16, EfConstruction: 200, EfSearch: 200)
    ];

    public static FashionMnistHnswlibComparisonMatrixManifest Run(
        FashionMnistHnswlibComparisonMatrixOptions options,
        IReadOnlyList<string> commandArguments)
    {
        string presetName = FashionMnistHnswlibComparisonMatrixOptions.NormalizePresetName(options.PresetName);
        FashionMnistComparisonMatrixCase[] cases = ExpandCases(options);
        FashionMnistHnswlibComparisonMatrixCacheTruthInfo cacheTruth = CreateCacheTruthInfo(options, out string? cacheBlockReason);
        var caseManifests = new FashionMnistHnswlibComparisonMatrixCaseManifest[cases.Length];
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
                    reportId: null,
                    status: "blocked",
                    validationStatus: "blocked",
                    cacheBlockReason);
            }
        }
        else
        {
            for (int i = 0; i < cases.Length; i++)
            {
                FashionMnistComparisonMatrixCase matrixCase = cases[i];
                try
                {
                    FashionMnistHnswlibComparisonReport report = FashionMnistHnswlibComparisonScenario.Run(
                        matrixCase.Options,
                        CreateCaseArguments(matrixCase.Options));
                    FashionMnistHnswlibComparisonScenario.Write(report, matrixCase.Options.OutputPath);

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
        }

        return new FashionMnistHnswlibComparisonMatrixManifest(
            SchemaName,
            SchemaVersion,
            TaskId,
            FashionMnistHnswlibComparisonMatrixOptions.ScenarioName,
            presetName,
            DateTimeOffset.UtcNow,
            RepositoryInfo.Create(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistHnswlibComparisonMatrixOptions.ScenarioName, commandArguments.ToArray()),
            options.OutputDirectory,
            cacheTruth,
            CreateSourcePinning(),
            CreateDesign(presetName),
            caseManifests.Length,
            caseManifests,
            new FashionMnistHnswlibComparisonMatrixAggregate(passed, failed, SkippedCaseCount: 0, blocked),
            CreateEligibility(),
            [
                "Private Fashion-MNIST hnswlib comparison matrix only; not a public performance, recall, memory, allocation, capacity, storage-size, baseline, comparison-publication or regression-gate claim.",
                "Each successful case reuses the accepted VEC-120 external-fashion-mnist-hnswlib-comparison report schema and writes a linked private per-case report.",
                "The matrix uses only the already admitted local Fashion-MNIST cache and existing exact truth artifact; it does not download, convert, admit or refresh Fashion-MNIST.",
                "Standard preset includes balanced-m8, wide-m16 and default-m16 profiles at top-k 10 and 100; every profile enforces efSearch >= top-k.",
                "Blocked cases do not receive linked report IDs and do not write fake successful per-case reports."
            ]);
    }

    public static FashionMnistComparisonMatrixCase[] ExpandCases(FashionMnistHnswlibComparisonMatrixOptions options)
    {
        string presetName = FashionMnistHnswlibComparisonMatrixOptions.NormalizePresetName(options.PresetName);
        int[] topKValues = GetTopKValues(presetName);
        var cases = new List<FashionMnistComparisonMatrixCase>(topKValues.Length * MatrixProfiles.Length);
        int caseIndex = 0;

        foreach (int topK in topKValues)
        {
            foreach (FashionMnistComparisonMatrixProfile profile in MatrixProfiles)
            {
                if (profile.EfSearch < topK)
                {
                    throw new InvalidOperationException("Fashion-MNIST matrix profile efSearch must be at least top-k for every case.");
                }

                ulong seed = unchecked(options.Seed + (ulong)caseIndex);
                string caseId = CreateCaseId(caseIndex + 1, profile.Name, topK);
                string caseDirectory = Path.Combine(options.OutputDirectory, caseId);
                var caseOptions = new FashionMnistHnswlibComparisonOptions(
                    options.CacheRoot,
                    Path.Combine(options.OutputDirectory, $"{caseId}.json"),
                    Path.Combine(caseDirectory, "work"),
                    Path.Combine(caseDirectory, "vecnet-snapshot"),
                    Path.Combine(caseDirectory, "hnswlib-index.bin"),
                    options.HnswlibPythonPath,
                    options.QueryCount,
                    topK,
                    options.Runs,
                    options.WarmupQueries,
                    profile.M,
                    profile.EfConstruction,
                    profile.EfSearch,
                    seed);

                cases.Add(new FashionMnistComparisonMatrixCase(caseId, profile.Name, caseOptions));
                caseIndex++;
            }
        }

        return cases.ToArray();
    }

    public static int GetMaxTopK(string presetName) => GetTopKValues(presetName).Max();

    public static void WriteManifest(FashionMnistHnswlibComparisonMatrixManifest manifest, string manifestPath)
    {
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(manifestPath, ReportWriter.Serialize(manifest));
    }

    public static string[] CreateCaseArguments(FashionMnistHnswlibComparisonOptions options) =>
    [
        FashionMnistHnswlibComparisonOptions.ScenarioName,
        "--cache-root", options.CacheRoot,
        "--output", options.OutputPath,
        "--work-directory", options.WorkDirectory,
        "--vecnet-snapshot-directory", options.VecNetSnapshotDirectory,
        "--hnswlib-index", options.HnswlibIndexPath,
        "--hnswlib-python", options.HnswlibPythonPath,
        "--query-count", options.QueryCount.ToString(CultureInfo.InvariantCulture),
        "--top-k", options.TopK.ToString(CultureInfo.InvariantCulture),
        "--runs", options.Runs.ToString(CultureInfo.InvariantCulture),
        "--warmup-queries", options.WarmupQueries.ToString(CultureInfo.InvariantCulture),
        "--m", options.M.ToString(CultureInfo.InvariantCulture),
        "--ef-construction", options.EfConstruction.ToString(CultureInfo.InvariantCulture),
        "--ef-search", options.EfSearch.ToString(CultureInfo.InvariantCulture),
        "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X16}")
    ];

    private static FashionMnistHnswlibComparisonMatrixCacheTruthInfo CreateCacheTruthInfo(
        FashionMnistHnswlibComparisonMatrixOptions options,
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
                        VectorMetric.SquaredEuclidean,
                        M: 8,
                        EfConstruction: 64,
                        EfSearch: Math.Max(128, GetMaxTopK(options.PresetName)),
                        options.Seed));

            blockReason = null;
            return new FashionMnistHnswlibComparisonMatrixCacheTruthInfo(
                "available",
                options.CacheRoot,
                dataset.Manifest.DatasetId,
                Dimension,
                VectorMetric.SquaredEuclidean.ToString(),
                "Loaded existing admitted Fashion-MNIST cache only; no download, conversion, admission or refresh path is used by VEC-121.",
                "Loaded existing exact truth artifact from the admitted cache path.",
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
            return new FashionMnistHnswlibComparisonMatrixCacheTruthInfo(
                "unavailable",
                options.CacheRoot,
                DatasetId,
                Dimension,
                VectorMetric.SquaredEuclidean.ToString(),
                "Admitted local Fashion-MNIST cache is required; VEC-121 must not download, convert, admit or refresh data.",
                "Existing exact truth artifact is required; VEC-121 must not refresh truth.",
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

    private static FashionMnistHnswlibComparisonMatrixCaseManifest CreateCaseManifest(
        int caseNumber,
        FashionMnistComparisonMatrixCase matrixCase,
        string? reportId,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        FashionMnistHnswlibComparisonOptions options = matrixCase.Options;
        return new FashionMnistHnswlibComparisonMatrixCaseManifest(
            caseNumber,
            matrixCase.CaseId,
            matrixCase.ProfileName,
            DatasetId,
            VectorMetric.SquaredEuclidean.ToString(),
            Dimension,
            options.QueryCount,
            options.TopK,
            options.Runs,
            options.WarmupQueries,
            string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X16}"),
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
            "hnswlib executes as Python/native external tooling through a private ignored environment; VecNet remains managed .NET in-process.",
            "No hnswlib, Python or native asset is referenced by src/VecNet or included in the VecNet package.");

    private static FashionMnistHnswlibComparisonMatrixDesignInfo CreateDesign(string presetName) =>
        new(
            MatrixProfiles
                .Select(profile => new FashionMnistHnswlibComparisonMatrixProfileInfo(profile.Name, profile.M, profile.EfConstruction, profile.EfSearch))
                .ToArray(),
            GetTopKValues(presetName),
            DatasetId,
            Dimension,
            VectorMetric.SquaredEuclidean.ToString(),
            "Fashion-MNIST external comparison matrix uses admitted base matrix row order, external ids 0..baseCount-1 and the first configured query vectors with existing exact truth.",
            "Smoke and standard presets both include balanced-m8, wide-m16 and default-m16 profiles; standard adds top-k 100 coverage.");

    private static FashionMnistHnswlibComparisonMatrixEligibility CreateEligibility() =>
        new(
            "local-evidence",
            "private-raw",
            "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonPublicationEligible: false,
            RegressionGateEligible: false,
            "Private Fashion-MNIST hnswlib comparison matrix output has not been reviewed for public reporting and is not a public VecNet claim.",
            "Established external-comparison matrix output is not a VecNet baseline candidate.",
            "No accepted public comparison-summary policy exists for VecNet versus hnswlib output.",
            "No hnswlib comparison regression-gate policy or threshold exists.");

    private static int[] GetTopKValues(string presetName)
    {
        string normalizedPresetName = FashionMnistHnswlibComparisonMatrixOptions.NormalizePresetName(presetName);
        return normalizedPresetName switch
        {
            FashionMnistHnswlibComparisonMatrixOptions.SmokePresetName => SmokeTopKValues,
            FashionMnistHnswlibComparisonMatrixOptions.StandardPresetName => StandardTopKValues,
            _ => throw new ArgumentException($"Unsupported Fashion-MNIST hnswlib comparison matrix preset '{presetName}'.")
        };
    }

    private static string CreateCaseId(int caseNumber, string profileName, int topK) =>
        string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D2}-{profileName}-{topK}k");

    public sealed record FashionMnistComparisonMatrixCase(
        string CaseId,
        string ProfileName,
        FashionMnistHnswlibComparisonOptions Options);

    private sealed record FashionMnistComparisonMatrixProfile(
        string Name,
        int M,
        int EfConstruction,
        int EfSearch);
}
