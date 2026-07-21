namespace VecNet.BenchmarkRunner;

public static class PublicBenchmarkSummaryWriter
{
    public static string Serialize(PublicBenchmarkSummary summary)
    {
        EnsureValid(summary);
        return ReportWriter.Serialize(summary);
    }

    public static void Write(PublicBenchmarkSummary summary, string outputPath)
    {
        EnsureValid(summary);
        ReportWriter.WriteJson(summary, outputPath);
    }

    public static PublicBenchmarkSummaryValidationResult Validate(PublicBenchmarkSummary? summary) =>
        PublicBenchmarkSummaryGate.Validate(summary);

    private static void EnsureValid(PublicBenchmarkSummary summary)
    {
        PublicBenchmarkSummaryValidationResult validation = Validate(summary);
        if (!validation.IsValid)
        {
            throw new InvalidOperationException(
                "Public benchmark summary failed validation: " +
                string.Join("; ", validation.Errors));
        }
    }
}
