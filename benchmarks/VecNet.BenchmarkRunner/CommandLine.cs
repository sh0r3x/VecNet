using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class CommandLine
{
    private const string ScenarioName = "exact-generated";

    public static GeneratedExactSearchOptions Parse(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? ScenarioName : args[0];
        if (!string.Equals(scenario, ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = args.Count == 0 ? 0 : 1; i < args.Count; i += 2)
        {
            string name = args[i];
            if (!name.StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Count)
            {
                throw new ArgumentException($"Expected an option/value pair at '{name}'.");
            }

            values[name[2..]] = args[i + 1];
        }

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
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
            baselineReportId);
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
