using System.Text.Json;
using System.Text.Json.Serialization;

namespace VecNet.BenchmarkRunner;

public static class ReportWriter
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    public static string Serialize<T>(T value) =>
        JsonSerializer.Serialize(value, Options);

    public static T? Deserialize<T>(string json) =>
        JsonSerializer.Deserialize<T>(json, Options);

    public static void Write(BenchmarkReport report, string outputPath)
    {
        WriteJson(report, outputPath);
    }

    public static void WriteComparison(BenchmarkComparisonArtifact comparison, string outputPath)
    {
        WriteJson(comparison, outputPath);
    }

    private static void WriteJson<T>(T value, string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(outputPath, Serialize(value));
    }
}
