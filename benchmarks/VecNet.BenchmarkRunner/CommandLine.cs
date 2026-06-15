using System.Globalization;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner;

public static class CommandLine
{
    public static GeneratedExactSearchOptions Parse(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactSearchOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactSearchOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2009);
        string outputPath = values.TryGetValue("output", out string? output)
            ? output
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        string? baselineReportId = GetOptionalNonWhiteSpace(values, "baseline-report-id");

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        return new GeneratedExactSearchOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            baselineReportId,
            runs,
            warmupQueries);
    }

    public static GeneratedExactMatrixOptions ParseMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedMatrixOption);
        string presetName = GeneratedExactMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 8);
        int runs = GetPositiveInt(values, "runs", 1);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2014);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "matrix-manifest.json");

        int maxTopK = GeneratedExactMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new GeneratedExactMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            outputDirectory,
            manifestPath);
    }

    public static BenchmarkComparisonOptions ParseComparison(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? BenchmarkComparisonOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, BenchmarkComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedComparisonOption);
        string baselinePath = GetRequiredNonWhiteSpace(values, "baseline");
        string currentPath = GetRequiredNonWhiteSpace(values, "current");
        string outputPath = values.TryGetValue("output", out string? output)
            ? output
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                "comparisons",
                $"benchmark-comparison-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");

        return new BenchmarkComparisonOptions(baselinePath, currentPath, outputPath);
    }

    public static HnswGeneratedMatrixOptions ParseHnswGeneratedMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswGeneratedMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswGeneratedMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswGeneratedMatrixOption);
        string presetName = HnswGeneratedMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? HnswGeneratedMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 4);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2037);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "hnsw-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        int maxTopK = HnswGeneratedMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum HNSW matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new HnswGeneratedMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            outputDirectory,
            manifestPath);
    }

    public static HnswGeneratedOptions ParseHnswGenerated(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswGeneratedOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswGeneratedOption);
        HnswGeneratedOptions defaults = HnswGeneratedOptions.Default;
        VectorMetric metric = GetEnum(values, "metric", defaults.Metric);
        int dimension = GetPositiveInt(values, "dimension", defaults.Dimension);
        int vectorCount = GetPositiveInt(values, "vectors", defaults.VectorCount);
        int queryCount = GetPositiveInt(values, "queries", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        uint seed = GetSeed(values, "seed", defaults.Seed);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("hnsw-generated supports only SquaredEuclidean.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (efSearch < topK)
        {
            throw new ArgumentException("ef-search must be greater than or equal to top-k.");
        }

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be in the range 1..4096.");
        }

        return new HnswGeneratedOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            runs,
            warmupQueries,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    public static FashionMnistExternalDatasetOptions ParseExternalFashionMnist(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalDatasetOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalDatasetOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedExternalFashionMnistOption);
        FashionMnistExternalDatasetOptions defaults = FashionMnistExternalDatasetOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int truthDepth = GetPositiveInt(values, "truth-depth", defaults.TruthDepth);
        bool download = GetBoolean(values, "download", defaults.DownloadRawFiles);

        return new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, download);
    }

    public static FashionMnistExternalExactBenchmarkOptions ParseExternalFashionMnistExact(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalExactBenchmarkOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalExactBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedExternalFashionMnistExactOption);
        FashionMnistExternalExactBenchmarkOptions defaults = FashionMnistExternalExactBenchmarkOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : defaults.OutputPath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        VectorMetric metric = GetExternalFashionMnistMetric(values, "metric", defaults.Metric);

        return new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, queryCount, topK, runs, warmupQueries, metric);
    }

    public static FashionMnistExternalHnswBenchmarkOptions ParseExternalFashionMnistHnsw(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalHnswBenchmarkOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalHnswBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedExternalFashionMnistHnswOption);
        FashionMnistExternalHnswBenchmarkOptions defaults = FashionMnistExternalHnswBenchmarkOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : defaults.OutputPath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        VectorMetric metric = GetExternalFashionMnistMetric(values, "metric", defaults.Metric);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK)
        {
            throw new ArgumentException("Option --ef-search must be greater than or equal to --top-k.");
        }

        if (efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be in the range 1..4096.");
        }

        return new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            queryCount,
            topK,
            runs,
            warmupQueries,
            metric,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    private static Dictionary<string, string> ParseOptionValues(
        IReadOnlyList<string> args,
        int startIndex,
        Func<string, bool> isSupportedOption)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = startIndex; i < args.Count; i += 2)
        {
            string name = args[i];
            if (!name.StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Count)
            {
                throw new ArgumentException($"Expected an option/value pair at '{name}'.");
            }

            string optionName = name[2..];
            if (!isSupportedOption(optionName))
            {
                throw new ArgumentException($"Unsupported option '{name}'.");
            }

            string value = args[i + 1];
            if (value.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Option '{name}' requires a value.");
            }

            values[optionName] = value;
        }

        return values;
    }

    private static TEnum GetEnum<TEnum>(Dictionary<string, string> values, string name, TEnum defaultValue)
        where TEnum : struct, Enum
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!Enum.TryParse(value, ignoreCase: true, out TEnum parsed) || !Enum.IsDefined(parsed))
        {
            throw new ArgumentException($"Option --{name} has unsupported value '{value}'.");
        }

        return parsed;
    }

    private static int GetPositiveInt(Dictionary<string, string> values, string name, int defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new ArgumentException($"Option --{name} must be a positive integer.");
        }

        return parsed;
    }

    private static int GetNonNegativeInt(Dictionary<string, string> values, string name, int defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed < 0)
        {
            throw new ArgumentException($"Option --{name} must be a non-negative integer.");
        }

        return parsed;
    }

    private static bool IsSupportedGeneratedOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "baseline-report-id", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedComparisonOption(string name) =>
        string.Equals(name, "baseline", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "current", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "truth-depth", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "download", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistExactOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistHnswOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswGeneratedOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswGeneratedMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static VectorMetric GetExternalFashionMnistMetric(Dictionary<string, string> values, string name, VectorMetric defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (string.Equals(value, "squared-euclidean", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, nameof(VectorMetric.SquaredEuclidean), StringComparison.OrdinalIgnoreCase))
        {
            return VectorMetric.SquaredEuclidean;
        }

        throw new ArgumentException($"Option --{name} has unsupported value '{value}'.");
    }

    private static uint GetSeed(Dictionary<string, string> values, string name, uint defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        NumberStyles styles = value.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? NumberStyles.AllowHexSpecifier
            : NumberStyles.None;
        string normalized = styles == NumberStyles.AllowHexSpecifier ? value[2..] : value;

        if (!uint.TryParse(normalized, styles, CultureInfo.InvariantCulture, out uint parsed))
        {
            throw new ArgumentException($"Option --{name} must be an unsigned integer or hexadecimal value.");
        }

        return parsed;
    }

    private static string? GetOptionalNonWhiteSpace(Dictionary<string, string> values, string name)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return null;
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"Option --{name} must not be empty.");
        }

        return value;
    }

    private static string GetRequiredNonWhiteSpace(Dictionary<string, string> values, string name)
    {
        string? value = GetOptionalNonWhiteSpace(values, name);
        if (value is null)
        {
            throw new ArgumentException($"Option --{name} is required.");
        }

        return value;
    }

    private static bool GetBoolean(Dictionary<string, string> values, string name, bool defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!bool.TryParse(value, out bool parsed))
        {
            throw new ArgumentException($"Option --{name} must be true or false.");
        }

        return parsed;
    }

    private static ulong GetUInt64Seed(Dictionary<string, string> values, string name, ulong defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        NumberStyles styles = value.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? NumberStyles.AllowHexSpecifier
            : NumberStyles.None;
        string normalized = styles == NumberStyles.AllowHexSpecifier ? value[2..] : value;

        if (!ulong.TryParse(normalized, styles, CultureInfo.InvariantCulture, out ulong parsed))
        {
            throw new ArgumentException($"Option --{name} must be an unsigned integer or hexadecimal value.");
        }

        return parsed;
    }
}
