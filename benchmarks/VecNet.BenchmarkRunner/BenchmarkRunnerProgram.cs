using System.Globalization;

namespace VecNet.BenchmarkRunner;

public static class BenchmarkRunnerProgram
{
    public static int Run(string[] args)
    {
        try
        {
            if (args.Length > 0 && IsHelp(args[0]))
            {
                WriteUsage(Console.Out);
                return 0;
            }

            GeneratedExactSearchOptions options = CommandLine.Parse(args);
            BenchmarkReport report = GeneratedExactSearchScenario.Run(options, args);
            ReportWriter.Write(report, options.OutputPath);

            Console.WriteLine(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"Wrote private benchmark report to {options.OutputPath}"));
            return 0;
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(ex.Message);
            WriteUsage(Console.Error);
            return 1;
        }
    }

    private static bool IsHelp(string value) =>
        string.Equals(value, "-h", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(value, "--help", StringComparison.OrdinalIgnoreCase);

    private static void WriteUsage(TextWriter writer)
    {
        writer.WriteLine("Usage:");
        writer.WriteLine("  exact-generated --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --seed 0x5EED2009 --output VecNet.BenchmarkRunner.Artifacts/report.json [--baseline-report-id report-id]");
    }
}
