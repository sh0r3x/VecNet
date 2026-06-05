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

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactMatrixOptions matrixOptions = CommandLine.ParseMatrix(args);
                GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(matrixOptions, args);
                GeneratedExactMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
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
        writer.WriteLine("  exact-generated --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --runs 1 --warmup-queries 0 --seed 0x5EED2009 --output VecNet.BenchmarkRunner.Artifacts/report.json [--baseline-report-id report-id]");
        writer.WriteLine("  exact-generated-matrix --preset smoke --vectors 128 --queries 8 --runs 1 --warmup-queries 0 --seed 0x5EED2014 --output-dir VecNet.BenchmarkRunner.Artifacts/matrix --manifest VecNet.BenchmarkRunner.Artifacts/matrix/matrix-manifest.json");
    }
}
