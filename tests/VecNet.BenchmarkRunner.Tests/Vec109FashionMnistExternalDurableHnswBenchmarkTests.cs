using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec109FashionMnistExternalDurableHnswBenchmarkTests
{
    [Fact]
    public void ParseExternalFashionMnistDurableHnsw_UsesPrivateDefaults()
    {
        FashionMnistExternalDurableHnswBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-durable.json"), options.OutputPath);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-durable-snapshot"), options.SnapshotDirectory);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(100, options.EfSearch);
        Assert.Equal(0x484E535700010901UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--truth-depth", "10")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--seed", "0x5EED")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--vectors", "32")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--dimension", "16")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--baseline-report-id", "baseline")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--output", "")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--snapshot-directory", "")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--runs", "0")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--runs", "6")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--m", "1")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--m", "65")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--m", "8", "--ef-construction", "7")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--hnsw-seed", "0xNOTHEX")]
    public void ParseExternalFashionMnistDurableHnsw_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistDurableHnsw(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistDurableHnsw_AcceptsCosine(string metric)
    {
        FashionMnistExternalDurableHnswBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
    }

    [Fact]
    public void ParseExternalFashionMnistDurableHnsw_RejectsInnerProduct()
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() =>
            CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--metric", "InnerProduct"]));

        Assert.Contains("unsupported", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsExternalDurableHnswParityReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("report", baseCount: 40, queryCount: 6, truthDepth: 5);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "external-durable-hnsw-report.json");
        string snapshotDirectory = Path.Combine(cacheRoot, "..", "external-durable-hnsw-snapshot");
        string[] arguments =
        [
            "external-fashion-mnist-hnsw-durable",
            "--cache-root", cacheRoot,
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory,
            "--query-count", "4",
            "--top-k", "5",
            "--runs", "2",
            "--warmup-queries", "2",
            "--metric", "squared-euclidean",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000010901"
        ];
        FashionMnistExternalDurableHnswBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistDurableHnsw(arguments);

        ExternalDurableHnswBenchmarkReport report =
            FashionMnistExternalDurableHnswBenchmarkScenario.Run(options, arguments);
        FashionMnistExternalDurableHnswBenchmarkScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalDurableHnswBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-109", report.TaskId);
        Assert.Equal("external-fashion-mnist-hnsw-durable", report.ScenarioName);
        Assert.Equal("external-fashion-mnist-hnsw-durable", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("external-durable-hnsw-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.PreviewReadinessEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonArtifactEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal("VecNet.ExternalDatasetManifest", report.Dataset.AdmissionManifest.SchemaName);
        Assert.Equal("0.1", report.Dataset.AdmissionManifest.SchemaVersion);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Truth.Sha256, report.Truth.Sha256);
        Assert.Equal(40, report.Workload.BaseCount);
        Assert.Equal(6, report.Workload.QueryMatrixCount);
        Assert.Equal(4, report.Workload.MeasuredQueryCount);
        Assert.Equal(5, report.Workload.TopK);
        Assert.Equal(5, report.Truth.TopK);
        Assert.Equal(40, report.DurableWorkload.VectorCount);
        Assert.Equal(4, report.DurableWorkload.QueryCount);
        Assert.Equal("0x0000000000010901", report.DurableWorkload.HnswSeed);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(8, report.Hnsw.EfSearch);

        AssertOperation(report.Operations.Build, "build", "admitted Fashion-MNIST base vectors", 2);
        AssertOperation(report.Operations.Save, "save", "HnswIndex.Save", 2);
        AssertOperation(report.Operations.Open, "open", "HnswIndex.OpenReadOnly", 2);
        Assert.Equal("sourceSearch", report.Operations.SourceSearch.Name);
        Assert.Equal("openedSearch", report.Operations.OpenedSearch.Name);
        Assert.Equal(2, report.Operations.SourceSearch.Runs.Length);
        Assert.Equal(2, report.Operations.OpenedSearch.Runs.Length);
        Assert.Equal(4, report.Operations.SourceSearch.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal(4, report.Operations.OpenedSearch.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("notMeasured", report.Operations.ResidentProcessMemory.Status);

        Assert.Equal("measured", report.Measurement.Build.ManagedAllocations.Status);
        Assert.True(double.Parse(report.Measurement.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Equal("notMeasured", report.Measurement.Save.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Measurement.Open.ManagedAllocations.Status);
        AssertSearchMeasurement(report.Measurement.SourceSearch, "source HnswIndex.Search");
        AssertSearchMeasurement(report.Measurement.OpenedSearch, "opened HnswIndex.Search");
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Contains("manifest checksum validation", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);

        Assert.Equal("written", report.Outputs.SnapshotOutput.Status);
        Assert.True(Directory.Exists(report.Outputs.SnapshotOutput.DirectoryPath));
        Assert.Equal(5, report.Outputs.SnapshotOutput.FileCount);
        Assert.True(report.Outputs.SnapshotOutput.TotalBytes > 0);
        Assert.True(report.Outputs.SnapshotOutput.ManifestBytes > 0);
        Assert.Equal(32 + (40 * 8), report.Outputs.SnapshotOutput.IdsBytes);
        Assert.Equal(48 + (40 * 16 * 4), report.Outputs.SnapshotOutput.VectorsBytes);
        Assert.Equal(32 + (40 * 4), report.Outputs.SnapshotOutput.LevelsBytes);
        Assert.True(report.Outputs.SnapshotOutput.GraphBytes > 0);
        Assert.Equal("outsideSaveAndOpenDuration", report.Outputs.SnapshotOutput.ScanTimingScope);
        Assert.True(File.Exists(Path.Combine(report.Outputs.SnapshotOutput.DirectoryPath, "hnsw.manifest.json")));
        Assert.True(File.Exists(Path.Combine(report.Outputs.SnapshotOutput.DirectoryPath, "hnsw.graph.bin")));

        AssertMetrics(report.Metrics.SourceHnsw);
        AssertMetrics(report.Metrics.OpenedHnsw);
        Assert.True(report.Metrics.SourceAndOpenedRecallEqual);
        Assert.True(report.Metrics.SourceAndOpenedOrderedAgreementEqual);
        Assert.True(report.Metrics.SourceAndOpenedDistanceIntegrityEqual);
        Assert.True(report.Validation.SourceOpenedParity.AllResultsMatched);
        Assert.Equal(0, report.Validation.SourceOpenedParity.WrittenCountMismatchCount);
        Assert.Equal(0, report.Validation.SourceOpenedParity.IdMismatchCount);
        Assert.Equal(0, report.Validation.SourceOpenedParity.OrderMismatchCount);
        Assert.Equal(0, report.Validation.SourceOpenedParity.DistanceMismatchCount);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingTruth);
        Assert.True(report.Validation.SourceHnswBuilt);
        Assert.True(report.Validation.SourceHnswSaved);
        Assert.True(report.Validation.OpenedHnswOpened);
        Assert.True(report.Validation.OpenedIndexReadOnly);
        Assert.True(report.Validation.SourceHnswComparedToTruth);
        Assert.True(report.Validation.OpenedHnswComparedToTruth);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForSource);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForOpened);
        Assert.Equal("passed", report.Validation.OpenedReadOnlyMutation.Status);
        Assert.True(report.Validation.OutputBytesScannedOutsideSaveOpenDuration);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.PreviewReadinessEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);

        Assert.Equal("estimatedPayloadLowerBoundsAndFileFacts", report.MemoryEstimates.Status);
        Assert.Equal(40L * 16L * sizeof(float), report.MemoryEstimates.VectorPayloadBytes);
        Assert.Equal(report.Outputs.SnapshotOutput.TotalBytes, report.MemoryEstimates.DurableOutputBytes);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.PeakMemory.Status);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.PreviewReadinessEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalDurableHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("external-fashion-mnist-hnsw-durable", root.GetProperty("scenarioName").GetString());
        Assert.Equal("external-durable-hnsw-smoke", root.GetProperty("evidence").GetProperty("scope").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("sourceSearch").GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("openedSearch").GetProperty("latency").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("sourceOpenedParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    [Fact]
    public void Run_WithSyntheticCosineAdmittedCache_EmitsDurableCosineParityReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("cosine-report", baseCount: 32, queryCount: 5, truthDepth: 5, metric: VectorMetric.Cosine);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "external-durable-hnsw-cosine-report.json");
        string snapshotDirectory = Path.Combine(cacheRoot, "..", "external-durable-hnsw-cosine-snapshot");
        var options = new FashionMnistExternalDurableHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            snapshotDirectory,
            QueryCount: 3,
            TopK: 5,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.Cosine,
            M: 4,
            EfConstruction: 16,
            EfSearch: 16,
            HnswSeed: 0x0000000000012390UL);

        ExternalDurableHnswBenchmarkReport report =
            FashionMnistExternalDurableHnswBenchmarkScenario.Run(options, ["external-fashion-mnist-hnsw-durable", "--metric", "Cosine"]);

        Assert.Equal("fashion-mnist-784-cosine", report.Dataset.DatasetId);
        Assert.Equal("Cosine", report.Workload.VecNetMetric);
        Assert.Equal("Cosine", report.Index.Metric);
        Assert.Equal("Cosine", report.DurableWorkload.Metric);
        Assert.Equal("vecnet-scalar-reference-cosine", report.Truth.Kind);
        Assert.Contains("canonical cosine", report.Truth.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Metrics.SourceHnsw.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.OpenedHnsw.DistanceToleranceStatus);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.SourceOpenedParity.AllResultsMatched);
    }

    [Fact]
    public void Program_UnsupportedDownloadOptionDoesNotRunAdmissionOrWriteArtifacts()
    {
        string cacheRoot = CreateArtifactDirectory("program-no-download");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string snapshotDirectory = Path.Combine(cacheRoot, "snapshot");
        TextWriter originalOut = Console.Out;
        TextWriter originalError = Console.Error;
        using var output = new StringWriter();
        using var error = new StringWriter();
        try
        {
            Console.SetOut(output);
            Console.SetError(error);

            int exitCode = BenchmarkRunnerProgram.Run(
                [
                    "external-fashion-mnist-hnsw-durable",
                    "--cache-root", cacheRoot,
                    "--output", outputPath,
                    "--snapshot-directory", snapshotDirectory,
                    "--download", "true"
                ]);

            Assert.Equal(1, exitCode);
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
        }

        Assert.Contains("Unsupported option '--download'", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "raw")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "converted")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "truth")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "manifests")));
    }

    [Fact]
    public void ExistingExternalAndGeneratedHnswCommandsRemainCompatible()
    {
        _ = CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--query-count", "1", "--top-k", "1", "--ef-search", "1"]);
        _ = CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--query-count", "1", "--top-k", "1", "--ef-search", "1"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--snapshot-directory", "snap"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--cache-root", "VecNet.DatasetCache"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--truth-depth", "10"]));
        Assert.Equal("external-fashion-mnist-hnsw-durable", FashionMnistExternalDurableHnswBenchmarkOptions.ScenarioName);
        Assert.Equal("external-fashion-mnist-hnsw", FashionMnistExternalHnswBenchmarkOptions.ScenarioName);
        Assert.Equal("hnsw-generated-durable", DurableHnswGeneratedOptions.ScenarioName);
    }

    private static void AssertOperation(DurableHnswOperationInfo operation, string name, string timedOperationContains, int runCount)
    {
        Assert.Equal(name, operation.Name);
        Assert.Contains(timedOperationContains, operation.TimedOperation, StringComparison.Ordinal);
        Assert.Equal(runCount, operation.Runs.Length);
        Assert.Equal(runCount, operation.Aggregate.RunCount);
        Assert.True(operation.Aggregate.MeanElapsedMilliseconds >= 0);
    }

    private static void AssertSearchMeasurement(DurableHnswSearchMeasurementInfo measurement, string timedOperationContains)
    {
        Assert.Equal("measured", measurement.Latency.Status);
        Assert.Contains(timedOperationContains, measurement.Latency.TimedOperation, StringComparison.Ordinal);
        Assert.Equal("measured", measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", measurement.ManagedAllocations.Unit);
        Assert.Contains("caller-owned SearchResult[] and HnswSearchWorkspace", measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
    }

    private static void AssertMetrics(DurableHnswOperationMetricsInfo metrics)
    {
        Assert.InRange(metrics.RecallAtK, 0, 1);
        Assert.InRange(metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
        Assert.Equal("passed", metrics.ReturnedResultIntegrity.Status);
        Assert.True(metrics.ReturnedResultIntegrity.CheckedResultCount > 0);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.ResultCountViolationCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, metrics.ReturnedResultIntegrity.DistanceMismatchCount);
    }

    private static FashionMnistAdmissionResult RunSyntheticAdmission(
        string prefix,
        int baseCount,
        int queryCount,
        int truthDepth,
        VectorMetric metric = VectorMetric.SquaredEuclidean)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 4, columns: 4);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: queryCount, TruthDepth: truthDepth, DownloadRawFiles: false, metric);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 17)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 41)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount)).ToArray());

        static FashionMnistRawFileSpec Spec(string path, string fileName, string role, int expectedCount) =>
            new(fileName, role, expectedCount, FileChecksum.ComputeMd5(path), downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            datasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: baseCount,
            QueryCount: queryCount,
            ImageRows: rows,
            ImageColumns: columns,
            Dimension: checked(rows * columns),
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", baseCount),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", baseCount),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", queryCount),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", queryCount));
    }

    private static byte[] CreatePixels(int count, int dimension, int offset)
    {
        var payload = new byte[checked(count * dimension)];
        for (int row = 0; row < count; row++)
        {
            for (int column = 0; column < dimension; column++)
            {
                payload[(row * dimension) + column] = (byte)((row * 19 + column * 23 + offset) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)(i % 10);
        }

        return labels;
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2051);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2049);
        WriteInt32BigEndian(decoded, count);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream Gzip(byte[] decoded)
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            gzip.Write(decoded);
        }

        compressed.Position = 0;
        return compressed;
    }

    private static void WriteInt32BigEndian(Stream stream, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, value);
        stream.Write(buffer);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec109-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
