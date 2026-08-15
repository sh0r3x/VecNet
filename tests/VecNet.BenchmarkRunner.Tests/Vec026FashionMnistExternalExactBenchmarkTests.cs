using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec026FashionMnistExternalExactBenchmarkTests
{
    [Fact]
    public void ParseExternalFashionMnistExact_UsesPrivateDefaults()
    {
        FashionMnistExternalExactBenchmarkOptions options = CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-exact.json"), options.OutputPath);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(3, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-exact", "--download", "false")]
    [InlineData("external-fashion-mnist-exact", "--query-count", "0")]
    [InlineData("external-fashion-mnist-exact", "--top-k", "0")]
    [InlineData("external-fashion-mnist-exact", "--runs", "0")]
    [InlineData("external-fashion-mnist-exact", "--runs", "6")]
    [InlineData("external-fashion-mnist-exact", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-exact", "--cache-root", "")]
    [InlineData("external-fashion-mnist-exact", "--output", "")]
    public void ParseExternalFashionMnistExact_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistExact(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistExact_AcceptsCosine(string metric)
    {
        FashionMnistExternalExactBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
    }

    [Fact]
    public void ParseExternalFashionMnistExact_AcceptsInnerProduct()
    {
        FashionMnistExternalExactBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--metric", "InnerProduct"]);

        Assert.Equal(VectorMetric.InnerProduct, options.Metric);
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsPrivateExternalBenchmarkReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("report");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "exact-report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(
            cacheRoot,
            outputPath,
            QueryCount: 2,
            TopK: 2,
            Runs: 3,
            WarmupQueries: 2,
            VectorMetric.SquaredEuclidean);

        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-exact", "--query-count", "2", "--top-k", "2"]);
        FashionMnistExternalExactBenchmarkScenario.Write(report, options.OutputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-026", report.TaskId);
        Assert.Equal("external-fashion-mnist-exact", report.Command.Scenario);
        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal("VecNet.ExternalDatasetManifest", report.Dataset.AdmissionManifest.SchemaName);
        Assert.Equal("0.1", report.Dataset.AdmissionManifest.SchemaVersion);
        Assert.Equal("manifests/fashion-mnist-784-euclidean/dataset-manifest.json", report.Dataset.AdmissionManifest.RelativePath);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Conversion.OutputFiles.Select(file => file.Sha256), report.Dataset.ConvertedMatrices.Select(file => file.Sha256));
        Assert.Equal(admission.Manifest.Truth.Sha256, report.Truth.Sha256);
        Assert.Equal(2, report.Workload.MeasuredQueryCount);
        Assert.Equal(2, report.Workload.TopK);
        Assert.Equal(2, report.Truth.TopK);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("public ExactFlatIndex.Search(query, results)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("index build", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingTruth);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal(0, report.Metrics.MissingResultCount);
        Assert.Equal(0, report.Metrics.ExtraResultCount);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("external-exact-smoke", root.GetProperty("evidence").GetProperty("scope").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.DoesNotContain("latencyTicks", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_WithSyntheticCosineAdmittedCache_UsesCosineTruthAndDistances()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("cosine-report", VectorMetric.Cosine);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "exact-cosine-report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(
            cacheRoot,
            outputPath,
            QueryCount: 2,
            TopK: 2,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.Cosine);

        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-exact", "--metric", "Cosine"]);

        Assert.Equal("fashion-mnist-784-cosine", report.Dataset.DatasetId);
        Assert.Equal("Cosine", report.Workload.VecNetMetric);
        Assert.Equal("Cosine", report.Index.Metric);
        Assert.Equal("vecnet-scalar-reference-cosine", report.Truth.Kind);
        Assert.Contains("canonical cosine", report.Truth.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(admission.Manifest.Truth.Sha256, report.Truth.Sha256);
    }

    [Fact]
    public void Run_MissingManifest_FailsWithoutCreatingReport()
    {
        string cacheRoot = CreateArtifactDirectory("missing-manifest");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean);

        Assert.Throws<FileNotFoundException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(options, ["external-fashion-mnist-exact"]));
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_MatrixChecksumMismatch_FailsBeforeReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("checksum");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string baseMatrixPath = Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le");
        using (FileStream stream = File.Open(baseMatrixPath, FileMode.Open, FileAccess.ReadWrite))
        {
            stream.Position = stream.Length - 1;
            stream.WriteByte(123);
        }

        var options = new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(options, ["external-fashion-mnist-exact"]));

        Assert.Contains("base matrix SHA-256 mismatch", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_QueryAndTopKBounds_AreValidatedAgainstExistingTruth()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("bounds");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);

        Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, Path.Combine(cacheRoot, "too-many-queries.json"), 3, 1, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));
        Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, Path.Combine(cacheRoot, "too-large-topk.json"), 1, 3, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));
    }

    [Fact]
    public void CompareGeneratedExact_TreatsExternalBenchmarkReportAsUnsupportedSchema()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("comparison");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string reportPath = Path.Combine(cacheRoot, "external-report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(cacheRoot, reportPath, 2, 2, 1, 0, VectorMetric.SquaredEuclidean);
        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(options, ["external-fashion-mnist-exact"]);
        FashionMnistExternalExactBenchmarkScenario.Write(report, reportPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(reportPath, reportPath, Path.Combine(cacheRoot, "comparison.json")),
            ["compare-generated-exact"]);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema");
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    private static FashionMnistAdmissionResult RunSyntheticAdmission(string prefix, VectorMetric metric = VectorMetric.SquaredEuclidean)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false, metric);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist"], spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(string cacheRoot)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(4, 2, 2, [
            1, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            9, 9, 9, 9
        ]).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(4, [0, 1, 2, 3]).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(2, 2, 2, [
            1, 0, 0, 0,
            0, 1, 0, 0
        ]).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(2, [1, 2]).ToArray());

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
            BaseCount: 4,
            QueryCount: 2,
            ImageRows: 2,
            ImageColumns: 2,
            Dimension: 4,
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", 4),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", 4),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", 2),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", 2));
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
            "vec026-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
