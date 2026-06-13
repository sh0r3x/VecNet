using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec026IndependentTests
{
    [Fact]
    public void Run_ReportJsonDoesNotLeakAbsoluteCachePathAndKeepsEligibilityFalse()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("privacy");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "external-report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(
            cacheRoot,
            outputPath,
            QueryCount: 2,
            TopK: 2,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean);

        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-exact", "--cache-root", cacheRoot, "--output", outputPath]);
        FashionMnistExternalExactBenchmarkScenario.Write(report, outputPath);

        string json = File.ReadAllText(outputPath);
        string fullCacheRoot = Path.GetFullPath(cacheRoot);
        Assert.DoesNotContain(cacheRoot, json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(fullCacheRoot, json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(outputPath, json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(Path.GetFullPath(outputPath), json, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("manifests/fashion-mnist-784-euclidean/dataset-manifest.json", report.Dataset.AdmissionManifest.RelativePath);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.All(report.Dataset.ConvertedMatrices, matrix => Assert.False(Path.IsPathRooted(matrix.RelativePath)));
        Assert.False(Path.IsPathRooted(report.Truth.RelativePath));
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void Run_ReportMemoryMetadataIsNotMeasuredOrAbsent()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("memory");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "external-report.json");
        var options = new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean);

        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(options, ["external-fashion-mnist-exact"]);
        FashionMnistExternalExactBenchmarkScenario.Write(report, outputPath);

        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("bytes", report.Measurement.Memory.Unit);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement memory = document.RootElement.GetProperty("measurement").GetProperty("memory");
        Assert.Equal("notMeasured", memory.GetProperty("status").GetString());
        Assert.Equal("absent", memory.GetProperty("value").GetString());
        Assert.Equal("bytes", memory.GetProperty("unit").GetString());
        Assert.False(memory.TryGetProperty("workingSetBytes", out _));
        Assert.False(memory.TryGetProperty("residentSetBytes", out _));
        Assert.False(memory.TryGetProperty("privateBytes", out _));
        Assert.False(memory.TryGetProperty("gcHeapBytes", out _));
        Assert.False(memory.TryGetProperty("peakWorkingSetBytes", out _));
    }

    [Theory]
    [InlineData("schemaName", "Wrong.Schema", "schemaName")]
    [InlineData("schemaVersion", "9.9", "schemaVersion")]
    [InlineData("datasetId", "wrong-dataset", "datasetId")]
    public void Run_InvalidManifestIdentityFailsBeforeReport(string propertyName, string replacement, string expectedMessageFragment)
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("manifest-" + propertyName);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        ExternalDatasetManifest manifest = admission.Manifest;
        manifest = propertyName switch
        {
            "schemaName" => manifest with { SchemaName = replacement },
            "schemaVersion" => manifest with { SchemaVersion = replacement },
            "datasetId" => manifest with { DatasetId = replacement },
            _ => manifest
        };
        WriteManifest(admission.ManifestPath, manifest);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));

        Assert.Contains(expectedMessageFragment, exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Theory]
    [InlineData("converted/fashion-mnist-784-euclidean/base.f32le", "must be relative")]
    [InlineData("../converted/fashion-mnist-784-euclidean/base.f32le", "dot segments")]
    public void Run_InvalidManifestArtifactPathsAreRejected(string relativePath, string expectedMessageFragment)
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("manifest-path");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        ExternalDatasetManifest manifest = admission.Manifest;
        ExternalConvertedMatrixEntry baseEntry = manifest.Conversion.OutputFiles.Single(file => file.Role == "base");
        ExternalConvertedMatrixEntry[] outputFiles = manifest.Conversion.OutputFiles
            .Select(file => file.Role == "base" ? baseEntry with { RelativePath = Path.GetFullPath(Path.Combine(cacheRoot, relativePath)) } : file)
            .ToArray();
        if (relativePath.StartsWith("../", StringComparison.Ordinal))
        {
            outputFiles = manifest.Conversion.OutputFiles
                .Select(file => file.Role == "base" ? baseEntry with { RelativePath = relativePath } : file)
                .ToArray();
        }

        WriteManifest(admission.ManifestPath, manifest with
        {
            Conversion = manifest.Conversion with { OutputFiles = outputFiles }
        });

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));

        Assert.Contains(expectedMessageFragment, exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_ConvertedMatrixHeaderMismatchFailsEvenWhenChecksumMatchesManifest()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("matrix-header");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string baseMatrixPath = Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le");
        DenseFloat32Matrix.Write(baseMatrixPath, rowCount: 5, dimension: 4, values: new float[20]);
        string alteredSha256 = FileChecksum.ComputeSha256(baseMatrixPath);
        ExternalDatasetManifest manifest = admission.Manifest;
        WriteManifest(admission.ManifestPath, manifest with
        {
            Conversion = manifest.Conversion with
            {
                OutputFiles = manifest.Conversion.OutputFiles
                    .Select(file => file.Role == "base" ? file with { Sha256 = alteredSha256 } : file)
                    .ToArray()
            }
        });

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));

        Assert.Contains("row count mismatch", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_TruthChecksumMismatchFailsBeforeReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("truth-checksum");
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        File.AppendAllText(admission.TruthPath, " ");

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));

        Assert.Contains("truth artifact SHA-256 mismatch", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Theory]
    [InlineData("schema", "External truth schemaName")]
    [InlineData("queryOrdinal", "query ordinals")]
    [InlineData("declaredDepth", "truth depth")]
    [InlineData("shallowNeighborList", "cover requested top-k")]
    public void Run_InvalidTruthSchemaQueryAndDepthMetadataFailsBeforeReport(string mutation, string expectedMessageFragment)
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("truth-" + mutation);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        ExternalExactTruthArtifact truth = ReadTruth(admission.TruthPath);
        truth = mutation switch
        {
            "schema" => truth with { SchemaName = "Wrong.ExternalTruth" },
            "queryOrdinal" => truth with
            {
                Queries = truth.Queries
                    .Select((query, index) => index == 0 ? query with { QueryOrdinal = 7 } : query)
                    .ToArray()
            },
            "declaredDepth" => truth with { TruthDepth = 1 },
            "shallowNeighborList" => truth with
            {
                Queries = truth.Queries
                    .Select((query, index) => index == 0 ? query with { Neighbors = query.Neighbors.Take(1).ToArray() } : query)
                    .ToArray()
            },
            _ => truth
        };
        WriteTruthAndUpdateManifest(admission, truth);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalExactBenchmarkScenario.Run(
                new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, 1, 2, 1, 0, VectorMetric.SquaredEuclidean),
                ["external-fashion-mnist-exact"]));

        Assert.Contains(expectedMessageFragment, exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_QueryCountAndTopKBoundsRejectExistingTruthLimits()
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
    public void CompareGeneratedExact_RejectsExternalReportsWithoutComparisonMetrics()
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
        Assert.Empty(comparison.Metrics);
        Assert.Empty(comparison.Cases);
        Assert.Equal(1, comparison.Warnings.NotComparableCount);
        Assert.Equal(0, comparison.Warnings.CorrectnessWarningCount);
        Assert.Equal(0, comparison.Warnings.PerformanceWarningCount);
        Assert.Equal(0, comparison.Warnings.AllocationWarningCount);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Theory]
    [InlineData("--download", "true")]
    [InlineData("--truth-depth", "2")]
    [InlineData("--convert", "true")]
    [InlineData("--generate-truth", "true")]
    public void ParseExternalFashionMnistExact_RejectsPreparationOptions(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() =>
            CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Program_UnsupportedDownloadOptionDoesNotRunAdmissionOrWriteReport()
    {
        string cacheRoot = CreateArtifactDirectory("program-no-download");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        TextWriter originalOut = Console.Out;
        TextWriter originalError = Console.Error;
        using var output = new StringWriter();
        using var error = new StringWriter();
        try
        {
            Console.SetOut(output);
            Console.SetError(error);

            int exitCode = BenchmarkRunnerProgram.Run(
                ["external-fashion-mnist-exact", "--cache-root", cacheRoot, "--output", outputPath, "--download", "true"]);

            Assert.Equal(1, exitCode);
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
        }

        Assert.Contains("Unsupported option '--download'", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "raw")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "converted")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "truth")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "manifests")));
    }

    private static FashionMnistAdmissionResult RunSyntheticAdmission(string prefix)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
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
            0, 0, 0, 0,
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

    private static ExternalExactTruthArtifact ReadTruth(string truthPath) =>
        ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(truthPath))!;

    private static void WriteTruthAndUpdateManifest(FashionMnistAdmissionResult admission, ExternalExactTruthArtifact truth)
    {
        ReportWriter.WriteJson(truth, admission.TruthPath);
        ExternalDatasetManifest manifest = admission.Manifest with
        {
            Truth = admission.Manifest.Truth with { Sha256 = FileChecksum.ComputeSha256(admission.TruthPath) }
        };
        WriteManifest(admission.ManifestPath, manifest);
    }

    private static void WriteManifest(string manifestPath, ExternalDatasetManifest manifest) =>
        ReportWriter.WriteJson(manifest, manifestPath);

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
            "vec026-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
