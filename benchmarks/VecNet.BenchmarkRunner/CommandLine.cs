using System.Globalization;

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
}
