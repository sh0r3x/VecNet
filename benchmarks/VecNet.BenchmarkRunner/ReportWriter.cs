using System.Text.Json;

namespace VecNet.BenchmarkRunner;

public static class ReportWriter
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static string Serialize(BenchmarkReport report) =>
        JsonSerializer.Serialize(report, Options);

    public static void Write(BenchmarkReport report, string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(outputPath, Serialize(report));
    }
}
