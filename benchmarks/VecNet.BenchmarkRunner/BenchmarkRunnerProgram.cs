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

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactCheckpointOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactCheckpointOptions checkpointOptions = CommandLine.ParseGeneratedExactCheckpoint(args);
                GeneratedExactCheckpointBenchmarkReport checkpointReport = GeneratedExactCheckpointScenario.Run(checkpointOptions, args);
                GeneratedExactCheckpointScenario.Write(checkpointReport, checkpointOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact checkpoint benchmark report to {checkpointOptions.OutputPath} with validation status {checkpointReport.Validation.Status}."));
                return string.Equals(checkpointReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactOpenedSearchOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactOpenedSearchOptions openedSearchOptions = CommandLine.ParseGeneratedExactOpenedSearch(args);
                GeneratedExactOpenedSearchBenchmarkReport openedSearchReport = GeneratedExactOpenedSearchScenario.Run(openedSearchOptions, args);
                GeneratedExactOpenedSearchScenario.Write(openedSearchReport, openedSearchOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact opened read-only search allocation report to {openedSearchOptions.OutputPath} with validation status {openedSearchReport.Validation.Status}."));
                return string.Equals(openedSearchReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactMemorySmokeOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactMemorySmokeOptions memoryOptions = CommandLine.ParseGeneratedExactMemorySmoke(args);
                GeneratedExactMemorySmokeReport memoryReport = GeneratedExactMemorySmokeScenario.Run(memoryOptions, args);
                GeneratedExactMemorySmokeScenario.Write(memoryReport, memoryOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact memory smoke report to {memoryOptions.OutputPath} with validation status {memoryReport.Validation.Status}."));
                return string.Equals(memoryReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactPracticalUpdateOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactPracticalUpdateOptions practicalUpdateOptions = CommandLine.ParseGeneratedExactPracticalUpdate(args);
                GeneratedExactPracticalUpdateBenchmarkReport practicalUpdateReport = GeneratedExactPracticalUpdateScenario.Run(practicalUpdateOptions, args);
                GeneratedExactPracticalUpdateScenario.Write(practicalUpdateReport, practicalUpdateOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact practical-update report to {practicalUpdateOptions.OutputPath} with validation status {practicalUpdateReport.Validation.Status}."));
                return string.Equals(practicalUpdateReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactCheckpointMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactCheckpointMatrixOptions matrixOptions = CommandLine.ParseGeneratedExactCheckpointMatrix(args);
                GeneratedExactCheckpointMatrixManifest manifest = GeneratedExactCheckpointMatrixScenario.Run(matrixOptions, args);
                GeneratedExactCheckpointMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact checkpoint matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], GeneratedExactUpdateMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                GeneratedExactUpdateMatrixOptions matrixOptions = CommandLine.ParseGeneratedExactUpdateMatrix(args);
                GeneratedExactUpdateMatrixManifest manifest = GeneratedExactUpdateMatrixScenario.Run(matrixOptions, args);
                GeneratedExactUpdateMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated exact update matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], DurableHnswGeneratedMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                DurableHnswGeneratedMatrixOptions matrixOptions = CommandLine.ParseDurableHnswGeneratedMatrix(args);
                DurableHnswGeneratedMatrixManifest manifest = DurableHnswGeneratedMatrixScenario.Run(matrixOptions, args);
                DurableHnswGeneratedMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated durable HNSW matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed case(s) and {manifest.Aggregate.FailedCaseCount} failed case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], DurableHnswGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                DurableHnswGeneratedOptions durableHnswOptions = CommandLine.ParseDurableHnswGenerated(args);
                DurableHnswBenchmarkReport durableHnswReport = DurableHnswGeneratedScenario.Run(durableHnswOptions, args);
                DurableHnswGeneratedScenario.Write(durableHnswReport, durableHnswOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated durable HNSW benchmark report to {durableHnswOptions.OutputPath} with validation status {durableHnswReport.Validation.Status}."));
                return string.Equals(durableHnswReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswAllowlistFilteringOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswAllowlistFilteringOptions filteringOptions = CommandLine.ParseHnswAllowlistFiltering(args);
                HnswAllowlistFilteringBenchmarkReport filteringReport = HnswAllowlistFilteringScenario.Run(filteringOptions, args);
                HnswAllowlistFilteringScenario.Write(filteringReport, filteringOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW allowlist-filtered report to {filteringOptions.OutputPath} with validation status {filteringReport.Validation.Status}."));
                return string.Equals(filteringReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswAllowlistFilteringMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswAllowlistFilteringMatrixOptions matrixOptions = CommandLine.ParseHnswAllowlistFilteringMatrix(args);
                HnswAllowlistFilteringMatrixManifest manifest = HnswAllowlistFilteringMatrixScenario.Run(matrixOptions, args);
                HnswAllowlistFilteringMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW allowlist-filtered matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswMemorySmokeOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswMemorySmokeOptions memoryOptions = CommandLine.ParseHnswMemorySmoke(args);
                HnswMemorySmokeReport memoryReport = HnswMemorySmokeScenario.Run(memoryOptions, args);
                HnswMemorySmokeScenario.Write(memoryReport, memoryOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW memory smoke report to {memoryOptions.OutputPath} with validation status {memoryReport.Validation.Status}."));
                return string.Equals(memoryReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswEstablishedComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswEstablishedComparisonOptions comparisonOptions = CommandLine.ParseHnswEstablishedComparison(args);
                HnswEstablishedComparisonReport comparisonReport = HnswEstablishedComparisonScenario.Run(comparisonOptions, args);
                HnswEstablishedComparisonScenario.Write(comparisonReport, comparisonOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private hnswlib generated comparison report to {comparisonOptions.OutputPath} with validation status {comparisonReport.Validation.Status}."));
                return string.Equals(comparisonReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswEstablishedComparisonMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswEstablishedComparisonMatrixOptions matrixOptions = CommandLine.ParseHnswEstablishedComparisonMatrix(args);
                HnswEstablishedComparisonMatrixManifest manifest = HnswEstablishedComparisonMatrixScenario.Run(matrixOptions, args);
                HnswEstablishedComparisonMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private hnswlib generated comparison matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistHnswlibComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistHnswlibComparisonOptions comparisonOptions = CommandLine.ParseFashionMnistHnswlibComparison(args);
                FashionMnistHnswlibComparisonReport comparisonReport = FashionMnistHnswlibComparisonScenario.Run(comparisonOptions, args);
                FashionMnistHnswlibComparisonScenario.Write(comparisonReport, comparisonOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST hnswlib comparison report to {comparisonOptions.OutputPath} with validation status {comparisonReport.Validation.Status}."));
                return string.Equals(comparisonReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistHnswlibComparisonMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistHnswlibComparisonMatrixOptions matrixOptions = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args);
                FashionMnistHnswlibComparisonMatrixManifest manifest = FashionMnistHnswlibComparisonMatrixScenario.Run(matrixOptions, args);
                FashionMnistHnswlibComparisonMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST hnswlib comparison matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
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

            if (args.Length > 0 && string.Equals(args[0], HnswBasePlusExactDeltaMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswBasePlusExactDeltaMatrixOptions matrixOptions = CommandLine.ParseHnswBasePlusExactDeltaMatrix(args);
                HnswBasePlusExactDeltaMatrixManifest manifest = HnswBasePlusExactDeltaMatrixScenario.Run(matrixOptions, args);
                HnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW base-plus-exact-delta matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswBasePlusExactDeltaCheckpointMatrixOptions matrixOptions = CommandLine.ParseHnswBasePlusExactDeltaCheckpointMatrix(args);
                HnswBasePlusExactDeltaCheckpointMatrixManifest manifest = HnswBasePlusExactDeltaCheckpointMatrixScenario.Run(matrixOptions, args);
                HnswBasePlusExactDeltaCheckpointMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW base-plus-exact-delta checkpoint matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswBasePlusExactDeltaCheckpointOptions checkpointOptions = CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(args);
                HnswBasePlusExactDeltaCheckpointBenchmarkReport checkpointReport = HnswBasePlusExactDeltaCheckpointScenario.Run(checkpointOptions, args);
                HnswBasePlusExactDeltaCheckpointScenario.Write(checkpointReport, checkpointOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW base-plus-exact-delta checkpoint report to {checkpointOptions.OutputPath} with validation status {checkpointReport.Validation.Status}."));
                return string.Equals(checkpointReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], HnswBasePlusExactDeltaGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                HnswBasePlusExactDeltaGeneratedOptions deltaOptions = CommandLine.ParseHnswBasePlusExactDeltaGenerated(args);
                HnswBasePlusExactDeltaBenchmarkReport deltaReport = HnswBasePlusExactDeltaGeneratedScenario.Run(deltaOptions, args);
                HnswBasePlusExactDeltaGeneratedScenario.Write(deltaReport, deltaOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private generated HNSW base-plus-exact-delta report to {deltaOptions.OutputPath} with validation status {deltaReport.Validation.Status}."));
                return string.Equals(deltaReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
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

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalDurableHnswBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalDurableHnswBenchmarkOptions externalOptions = CommandLine.ParseExternalFashionMnistDurableHnsw(args);
                ExternalDurableHnswBenchmarkReport externalReport = FashionMnistExternalDurableHnswBenchmarkScenario.Run(externalOptions, args);
                FashionMnistExternalDurableHnswBenchmarkScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external durable HNSW benchmark report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBasePlusExactDeltaOptions externalOptions = CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(args);
                ExternalHnswBasePlusExactDeltaBenchmarkReport externalReport = FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(externalOptions, args);
                FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW base-plus-exact-delta report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions externalOptions = CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(args);
                ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport externalReport = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(externalOptions, args);
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW base-plus-exact-delta checkpoint report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswAllowlistFilteringOptions externalOptions = CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(args);
                ExternalHnswAllowlistFilteringBenchmarkReport externalReport = FashionMnistExternalHnswAllowlistFilteringScenario.Run(externalOptions, args);
                FashionMnistExternalHnswAllowlistFilteringScenario.Write(externalReport, externalOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW allowlist filtering report to {externalOptions.OutputPath} with validation status {externalReport.Validation.Status}."));
                return string.Equals(externalReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions memoryOptions = CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(args);
                ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport memoryReport = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Run(memoryOptions, args);
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Write(memoryReport, memoryOptions.OutputPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW base-plus-exact-delta checkpoint memory smoke report to {memoryOptions.OutputPath} with validation status {memoryReport.Validation.Status}."));
                return string.Equals(memoryReport.Validation.Status, "passed", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions matrixOptions = CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(args);
                ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest manifest = FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.Run(matrixOptions, args);
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW base-plus-exact-delta checkpoint matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
            }

            if (args.Length > 0 && string.Equals(args[0], FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
            {
                FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions matrixOptions = CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(args);
                ExternalHnswBasePlusExactDeltaMatrixManifest manifest = FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.Run(matrixOptions, args);
                FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, matrixOptions.ManifestPath);

                Console.WriteLine(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"Wrote private Fashion-MNIST external HNSW base-plus-exact-delta matrix manifest to {matrixOptions.ManifestPath} with {manifest.Aggregate.PassedCaseCount} passed, {manifest.Aggregate.FailedCaseCount} failed and {manifest.Aggregate.BlockedCaseCount} blocked case(s)."));
                return manifest.Aggregate.FailedCaseCount == 0 && manifest.Aggregate.BlockedCaseCount == 0 ? 0 : 1;
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
        writer.WriteLine("  generated-exact-checkpoint --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --insertions 1000 --deletes 1000 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --allowlist broad --candidate-set selective --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2067 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-checkpoint.json");
        writer.WriteLine("  generated-exact-opened-search --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --runs 1 --warmup-queries 0 --seed 0x5EED2092 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-opened-search.json --index-directory VecNet.BenchmarkRunner.Artifacts/generated-exact-opened-search-index");
        writer.WriteLine("  generated-exact-memory-smoke --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --insertions 1000 --deletes 1000 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --allowlist broad --candidate-set selective --duplicate-ids 0 --unknown-ids 0 --warmup-queries 1 --seed 0x5EED2094 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-memory-smoke.json --save-directory VecNet.BenchmarkRunner.Artifacts/generated-exact-memory-smoke-save --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/generated-exact-memory-smoke-checkpoint");
        writer.WriteLine("  generated-exact-practical-update --metric SquaredEuclidean --dimension 128 --vectors 10000 --queries 100 --top-k 10 --insertions 1000 --deletes 1000 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --allowlist broad --candidate-set selective --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2079 --output VecNet.BenchmarkRunner.Artifacts/generated-exact-practical-update.json --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/generated-exact-practical-update-checkpoint");
        writer.WriteLine("  generated-exact-update-matrix --preset smoke|standard --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2062 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-exact-update-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-exact-update-matrix/exact-update-matrix-manifest.json");
        writer.WriteLine("  generated-exact-checkpoint-matrix --preset smoke|standard --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --duplicate-ids 0 --unknown-ids 0 --runs 1 --warmup-queries 0 --seed 0x5EED2069 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-exact-checkpoint-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-exact-checkpoint-matrix/exact-checkpoint-matrix-manifest.json");
        writer.WriteLine("  compare-generated-exact --baseline VecNet.BenchmarkRunner.Artifacts/baseline.json --current VecNet.BenchmarkRunner.Artifacts/current.json --output VecNet.BenchmarkRunner.Artifacts/comparisons/comparison.json");
        writer.WriteLine("  hnsw-generated --metric SquaredEuclidean|InnerProduct|Cosine --vector-profile uniform|norm-skewed|zero-vector --dimension 128 --vectors 10000 --queries 100 --top-k 10 --runs 1 --warmup-queries 0 --seed 0x5EED2036 --m 16 --ef-construction 200 --ef-search 50 --hnsw-seed 0x0000000564543034 --output VecNet.BenchmarkRunner.Artifacts/hnsw-generated.json");
        writer.WriteLine("  generated-hnsw-base-plus-exact-delta --metric SquaredEuclidean|InnerProduct|Cosine --vector-profile uniform|norm-skewed|zero-vector --dimension 128 --vectors 10000 --queries 100 --top-k 10 --insertions 1000 --deletes 1000 --delta-deletes 0 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 1 --warmup-queries 0 --seed 0x5EED2124 --m 16 --ef-construction 200 --ef-search 50 --workspace-ef-search 50 --hnsw-seed 0x0000000564543034 --output VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta.json");
        writer.WriteLine("  generated-hnsw-base-plus-exact-delta-checkpoint --metric SquaredEuclidean|InnerProduct|Cosine --vector-profile uniform|norm-skewed|zero-vector --dimension 128 --vectors 1024 --queries 16 --top-k 10 --insertions 128 --deletes 128 --delta-deletes 16 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 1 --warmup-queries 0 --seed 0x5EED2134 --m 16 --ef-construction 128 --ef-search 128 --workspace-ef-search 128 --hnsw-seed 0x484E535700013400 --output VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-checkpoint.json --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-checkpoint-output");
        writer.WriteLine("  generated-hnsw-allowlist-filtered --metric SquaredEuclidean|InnerProduct|Cosine --vector-profile uniform|norm-skewed|zero-vector --dimension 32 --vectors 512 --queries 8 --top-k 10 --insertions 64 --deletes 32 --delta-deletes 8 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --filter empty|very-selective|fallback-boundary|broad|all --runs 1 --warmup-queries 1 --seed 0x5EED2148 --m 8 --ef-construction 64 --ef-search 64 --hnsw-seed 0x484E535700014800 --output VecNet.BenchmarkRunner.Artifacts/generated-hnsw-allowlist-filtered.json --opened-index-directory VecNet.BenchmarkRunner.Artifacts/generated-hnsw-allowlist-filtered-opened --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/generated-hnsw-allowlist-filtered-checkpoint");
        writer.WriteLine("  generated-hnsw-allowlist-filtered-matrix --preset smoke|standard --queries 8 --runs 1 --warmup-queries 1 --metric SquaredEuclidean|InnerProduct|Cosine --seed 0x5EED2148 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-hnsw-allowlist-filtered-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-hnsw-allowlist-filtered-matrix/hnsw-allowlist-filtered-matrix-manifest.json");
        writer.WriteLine("  generated-hnsw-base-plus-exact-delta-matrix --preset smoke|standard --vectors 64 --queries 4 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 1 --warmup-queries 0 --seed 0x5EED2125 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-matrix/hnsw-base-plus-exact-delta-matrix-manifest.json");
        writer.WriteLine("  generated-hnsw-base-plus-exact-delta-checkpoint-matrix --preset smoke|standard --vectors 64 --queries 4 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 1 --warmup-queries 1 --metric SquaredEuclidean|InnerProduct|Cosine --seed 0x5EED2136 --output-dir VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-checkpoint-matrix --manifest VecNet.BenchmarkRunner.Artifacts/generated-hnsw-base-plus-exact-delta-checkpoint-matrix/hnsw-base-plus-exact-delta-checkpoint-matrix-manifest.json");
        writer.WriteLine("  hnsw-generated-durable --metric SquaredEuclidean|InnerProduct|Cosine --vector-profile uniform|norm-skewed|zero-vector --dimension 128 --vectors 1024 --queries 25 --top-k 10 --runs 1 --warmup-queries 0 --seed 0x5EED2073 --m 16 --ef-construction 200 --ef-search 50 --hnsw-seed 0x0000000000564543 --output VecNet.BenchmarkRunner.Artifacts/hnsw-generated-durable.json --snapshot-directory VecNet.BenchmarkRunner.Artifacts/hnsw-generated-durable-snapshot");
        writer.WriteLine("  generated-hnsw-memory-smoke --metric SquaredEuclidean|InnerProduct --vector-profile uniform|norm-skewed|zero-vector --dimension 128 --vectors 4096 --queries 32 --top-k 10 --warmup-queries 4 --seed 0x5EED2112 --m 8 --ef-construction 64 --ef-search 128 --hnsw-seed 0x484E535700011212 --sample-interval-ms 10 --output VecNet.BenchmarkRunner.Artifacts/generated-hnsw-memory-smoke.json --snapshot-directory VecNet.BenchmarkRunner.Artifacts/generated-hnsw-memory-smoke-snapshot");
        writer.WriteLine("  hnswlib-generated-comparison --metric SquaredEuclidean --dimension 128 --vectors 4096 --queries 100 --top-k 10 --runs 1 --warmup-queries 3 --seed 0x5EED2118 --m 8 --ef-construction 64 --ef-search 128 --hnsw-seed 0x484E535700011818 --hnswlib-python VecNet.BenchmarkRunner.Artifacts/vec-118-tools/hnswlib-venv/Scripts/python.exe --output VecNet.BenchmarkRunner.Artifacts/hnswlib-generated-comparison.json --work-directory VecNet.BenchmarkRunner.Artifacts/hnswlib-generated-comparison-work");
        writer.WriteLine("  hnswlib-generated-comparison-matrix --preset smoke|standard --vectors 256 --queries 4 --runs 1 --warmup-queries 0 --seed 0x5EED2119 --hnswlib-python VecNet.BenchmarkRunner.Artifacts/vec-118-tools/hnswlib-venv/Scripts/python.exe --output-dir VecNet.BenchmarkRunner.Artifacts/hnswlib-generated-comparison-matrix --manifest VecNet.BenchmarkRunner.Artifacts/hnswlib-generated-comparison-matrix/hnswlib-comparison-matrix-manifest.json");
        writer.WriteLine("  external-fashion-mnist-hnswlib-comparison --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison.json --work-directory VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison-work --vecnet-snapshot-directory VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison-vecnet-snapshot --hnswlib-index VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison-hnswlib.bin --hnswlib-python VecNet.BenchmarkRunner.Artifacts/vec-118-tools/hnswlib-venv/Scripts/python.exe --query-count 50 --top-k 10 --runs 1 --warmup-queries 3 --m 8 --ef-construction 64 --ef-search 100 --seed 0x484E535700012000");
        writer.WriteLine("  external-fashion-mnist-hnswlib-comparison-matrix --preset smoke|standard --cache-root VecNet.DatasetCache --query-count 50 --runs 1 --warmup-queries 3 --seed 0x484E535700012100 --hnswlib-python VecNet.BenchmarkRunner.Artifacts/vec-118-tools/hnswlib-venv/Scripts/python.exe --output-dir VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison-matrix --manifest VecNet.BenchmarkRunner.Artifacts/fashion-mnist-hnswlib-comparison-matrix/fashion-mnist-hnswlib-comparison-matrix-manifest.json");
        writer.WriteLine("  hnsw-generated-durable-matrix --preset smoke --seed 0x5EED0750 --output-dir VecNet.BenchmarkRunner.Artifacts/hnsw-generated-durable-matrix --manifest VecNet.BenchmarkRunner.Artifacts/hnsw-generated-durable-matrix/durable-hnsw-matrix-manifest.json");
        writer.WriteLine("  hnsw-generated-matrix --preset smoke|standard --vectors 128 --queries 4 --runs 1 --warmup-queries 0 --seed 0x5EED2037 --output-dir VecNet.BenchmarkRunner.Artifacts/hnsw-matrix --manifest VecNet.BenchmarkRunner.Artifacts/hnsw-matrix/hnsw-matrix-manifest.json");
        writer.WriteLine("  external-fashion-mnist --cache-root VecNet.DatasetCache --query-count 100 --truth-depth 10 --download false --metric squared-euclidean|inner-product|cosine");
        writer.WriteLine("  external-fashion-mnist-exact --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-exact.json --query-count 3 --top-k 10 --runs 3 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine");
        writer.WriteLine("  external-fashion-mnist-hnsw --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw.json --query-count 3 --top-k 10 --runs 3 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine --m 8 --ef-construction 64 --ef-search 100 --hnsw-seed 0x484E535700000039");
        writer.WriteLine("  external-fashion-mnist-hnsw-durable --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-durable.json --snapshot-directory VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-durable-snapshot --query-count 3 --top-k 10 --runs 1 --warmup-queries 0 --metric squared-euclidean|inner-product|cosine --m 8 --ef-construction 64 --ef-search 100 --hnsw-seed 0x484E535700010901");
        writer.WriteLine("  external-fashion-mnist-hnsw-base-plus-exact-delta --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-base-plus-exact-delta.json --query-count 50 --top-k 100 --base-vectors 58000 --insertions 1000 --deletes 1000 --delta-deletes 100 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 1 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine --seed 0x5EED2127 --m 16 --ef-construction 128 --ef-search 192 --hnsw-seed 0x484E535700012700");
        writer.WriteLine("  external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/vec-138-smoke/fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint.json --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/vec-138-smoke/checkpoint-output --query-count 50 --top-k 100 --base-vectors 58000 --insertions 1000 --deletes 1000 --delta-deletes 100 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --runs 2 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine --seed 0x5EED2137 --m 16 --ef-construction 128 --ef-search 192 --hnsw-seed 0x484E535700013700");
        writer.WriteLine("  external-fashion-mnist-hnsw-allowlist-filtered --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/vec-151-smoke/fashion-mnist-external-hnsw-allowlist-filtered.json --opened-index-directory VecNet.BenchmarkRunner.Artifacts/vec-151-smoke/opened-output --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/vec-151-smoke/checkpoint-output --query-count 50 --top-k 10 --base-vectors 58000 --insertions 1000 --deletes 1000 --delta-deletes 100 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --filter fallback-boundary|broad --runs 1 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine --seed 0x5EED2151 --m 16 --ef-construction 128 --ef-search 192 --hnsw-seed 0x484E535700015100");
        writer.WriteLine("  external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-memory-smoke --cache-root VecNet.DatasetCache --output VecNet.BenchmarkRunner.Artifacts/vec-142-memory-smoke/fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-memory-smoke.json --checkpoint-directory VecNet.BenchmarkRunner.Artifacts/vec-142-memory-smoke/checkpoint-output --metric squared-euclidean|inner-product|cosine --sample-interval-ms 10");
        writer.WriteLine("  external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix --preset smoke|standard --cache-root VecNet.DatasetCache --metric squared-euclidean|inner-product|cosine --output-dir VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-matrix --manifest VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-matrix/fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-matrix-manifest.json");
        writer.WriteLine("  external-fashion-mnist-hnsw-base-plus-exact-delta-matrix --preset smoke|standard --cache-root VecNet.DatasetCache --query-count 50 --runs 1 --warmup-queries 3 --metric squared-euclidean|inner-product|cosine --seed 0x5EED2128 --duplicate-inserts 1 --unknown-deletes 1 --repeated-deletes 1 --output-dir VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-base-plus-exact-delta-matrix --manifest VecNet.BenchmarkRunner.Artifacts/fashion-mnist-external-hnsw-base-plus-exact-delta-matrix/fashion-mnist-external-hnsw-base-plus-exact-delta-matrix-manifest.json");
    }
}
