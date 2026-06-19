using System.Globalization;
using VecNet.BenchmarkRunner.ExternalDatasets;

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

            if (args.Length > 0 && string.Equals(args[0], BenchmarkComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                BenchmarkComparisonOptions comparisonOptions = CommandLine.ParseComparison(args);
                BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(comparisonOptions, args);
                BenchmarkComparisonScenario.Write(comparison, comparisonOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private warning-only comparison artifact to {comparisonOptions.OutputPath} with status {comparison.Compatibility.Status}."));
                return 0;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactFilteredOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactFilteredOptions filteredOptions = CommandLine.ParseGeneratedExactFiltered(args);
                GeneratedExactFilteredBenchmarkReport filteredReport = GeneratedExactFilteredScenario.Run(filteredOptions, args);
                GeneratedExactFilteredScenario.Write(filteredReport, filteredOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact-filter benchmark report to {filteredOptions.OutputPath} with validation status {filteredReport.Validation.Status}."));
                return string.Equals(filteredReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactFilteredMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactFilteredMatrixOptions matrixOptions = CommandLine.ParseGeneratedExactFilteredMatrix(args);
                GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(matrixOptions, args);
                GeneratedExactFilteredMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact-filter matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactCandidateSetOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactCandidateSetOptions candidateSetOptions = CommandLine.ParseGeneratedExactCandidateSet(args);
                GeneratedExactCandidateSetBenchmarkReport candidateSetReport = GeneratedExactCandidateSetScenario.Run(candidateSetOptions, args);
                GeneratedExactCandidateSetScenario.Write(candidateSetReport, candidateSetOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact candidate-set benchmark report to {candidateSetOptions.OutputPath} with validation status {candidateSetReport.Validation.Status}."));
                return string.Equals(candidateSetReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactCandidateSetMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactCandidateSetMatrixOptions matrixOptions = CommandLine.ParseGeneratedExactCandidateSetMatrix(args);
                GeneratedExactCandidateSetMatrixManifest manifest = GeneratedExactCandidateSetMatrixScenario.Run(matrixOptions, args);
                GeneratedExactCandidateSetMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact candidate-set matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactUpdateOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactUpdateOptions updateOptions = CommandLine.ParseGeneratedExactUpdate(args);
                GeneratedExactUpdateBenchmarkReport updateReport = GeneratedExactUpdateScenario.Run(updateOptions, args);
                GeneratedExactUpdateScenario.Write(updateReport, updateOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact update benchmark report to {updateOptions.OutputPath} with validation status {updateReport.Validation.Status}."));
                return string.Equals(updateReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswGeneratedMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswGeneratedMatrixOptions matrixOptions = CommandLine.ParseHnswGeneratedMatrix(args);
                HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(matrixOptions, args);
                HnswGeneratedMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private HNSW matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswGeneratedOptions hnswOptions = CommandLine.ParseHnswGenerated(args);
                HnswBenchmarkReport hnswReport = HnswGeneratedScenario.Run(hnswOptions, args);
                HnswGeneratedScenario.Write(hnswReport, hnswOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW benchmark report to {hnswOptions.OutputPath} with validation status {hnswReport.Validation.Status}."));
                return string.Equals(hnswReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalDatasetOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalDatasetOptions externalOptions = CommandLine.ParseExternalFashionMnist(args);
                FashionMnistAdmissionResult result = FashionMnistExternalDatasetScenario.Run(externalOptions, args);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external dataset manifest to {result.ManifestPath} and exact validation evidence to {result.EvidencePath} with status {result.Evidence.Validation.Status}."));
                return string.Equals(result.Evidence.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalExactBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalExactBenchmarkOptions externalOptions = CommandLine.ParseExternalFashionMnistExact(args);
                ExternalBenchmarkReport externalReport = FashionMnistExternalExactBenchmarkScenario.Run(externalOptions, args);
                FashionMnistExternalExactBenchmarkScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external exact benchmark report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBenchmarkOptions externalOptions = CommandLine.ParseExternalFashionMnistHnsw(args);
                ExternalHnswBenchmarkReport externalReport = FashionMnistExternalHnswBenchmarkScenario.Run(externalOptions, args);
                FashionMnistExternalHnswBenchmarkScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW benchmark report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
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
        catch (Exception ex) when (ex is ArgumentException or InvalidDataException or IOException or UnauthorizedAccessException or HttpRequestException)
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
        writer.WriteLine("  exact-generated-matrix --preset smoke|standard --vectors 128 --queries 8 --runs 1 --warmup-queries 0 --seed 0x5EED2014 --output-dir VecNet.BenchmarkRunner.Artifacts/matrix --manifest VecNet.BenchmarkRunner.Artifacts/matrix/matrix-manifest.json");
        writer.WriteLine("  exact-generated-filtered --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --filter all|broad|selective|very-selective|empty --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2046 --output VecNet.BenchmarkRunner.Artifacts/exact-generated-filtered.json");
        writer.WriteLine("  exact-generated-filtered-matrix --preset smoke|standard --vectors 128 --queries 4 --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2047 --output-dir VecNet.BenchmarkRunner.Artifacts/exact-filtered-matrix --manifest VecNet.BenchmarkRunner.Artifacts/exact-filtered-matrix/exact-filtered-matrix-manifest.json");
        writer.WriteLine("  generated-exact-candidate-set --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --candidate-set all|broad|selective|very-selective|empty --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2053 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-candidate-set.json");
        writer.WriteLine("  generated-exact-candidate-set-matrix --preset smoke|standard --vectors 128 --queries 4 --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2054 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-exact-candidate-set-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-exact-candidate-set-matrix/exact-candidate-set-matrix-manifest.json");
        writer.WriteLine("  generated-exact-update --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --insertions 1000 --deletes 1000 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --allowlist broad --candidate-set selective --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2061 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-update.json");
        writer.WriteLine("  compare-generated-exact --baseline VecNet.BenchmarkRunner.Artifacts/baseline.json --current VecNet.BenchmarkRunner.Artifacts/current.json --output VecNet.BenchmarkRunner.Artifacts/comparisons/comparison.json");
        writer.WriteLine("  hnsw-generated --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --runs 1 --warmup-queries 0 --seed 0x5EED2036 --m 16 --ef-construction 200 --ef-search 50 --hnsw-seed 0x0000000564543034 --output VecNet.BenchmarkRunner.Artifacts/hnsw-generated.json");
        writer.WriteLine("  hnsw-generated-matrix --preset smoke|standard --vectors 128 --queries 4 --runs 1 --warmup-queries 0 --seed 0x5EED2037 --output-dir VecNet.BenchmarkRunner.Artifacts/hnsw-matrix --manifest VecNet.BenchmarkRunner.Artifacts/hnsw-matrix/hnsw-matrix-manifest.json");
        writer.WriteLine("  external-fashion-mnist --cache-root VecNet.DatasetCache --query-count 100 --truth-depth 10 --download false");
        writer.WriteLine("  external-fashion-mnist-exact --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-exact.json --query-count 3 --top-k 10 --runs 3 --warmup-queries 3 --metric squared-euclidean");
        writer.WriteLine("  external-fashion-mnist-hnsw --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw.json --query-count 3 --top-k 10 --runs 3 --warmup-queries 3 --metric squared-euclidean --m 8 --ef-construction 64 --ef-search 100 --hnsw-seed 0x484E535700000039");
    }
}
