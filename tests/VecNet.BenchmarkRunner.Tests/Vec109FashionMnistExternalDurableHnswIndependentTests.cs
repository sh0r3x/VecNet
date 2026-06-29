using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec109FashionMnistExternalDurableHnswIndependentTests
{
    [Theory]
    [InlineData("EXTERNAL-FASHION-MNIST-HNSW-DURABLE", "--CACHE-ROOT", "cache", "--OUTPUT", "out.json", "--SNAPSHOT-DIRECTORY", "snap", "--QUERY-COUNT", "1", "--TOP-K", "1", "--RUNS", "5", "--WARMUP-QUERIES", "0", "--METRIC", "SQUAREDEUCLIDEAN", "--M", "2", "--EF-CONSTRUCTION", "2", "--EF-SEARCH", "1", "--HNSW-SEED", "18446744073709551615")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--metric", "squared-euclidean", "--m", "64", "--ef-construction", "4096", "--ef-search", "4096", "--top-k", "4096", "--hnsw-seed", "0xffffffffffffffff")]
    public void ParseExternalFashionMnistDurableHnsw_AcceptsCaseInsensitiveNamesAndBoundaryValues(params string[] args)
    {
        FashionMnistExternalDurableHnswBenchmarkOptions options = CommandLine.ParseExternalFashionMnistDurableHnsw(args);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.InRange(options.M, 2, 64);
        Assert.InRange(options.Runs, 1, 5);
        Assert.InRange(options.EfConstruction, options.M, 4096);
        Assert.InRange(options.EfSearch, options.TopK, 4096);
        Assert.Equal(ulong.MaxValue, options.HnswSeed);
    }

    [Theory]
    [InlineData("external-fashion-mnist-hnsw-durable", "--query-count")]
    [InlineData("external-fashion-mnist-hnsw-durable", "bare-token")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--output", "report.json", "--snapshot-directory")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--download-raw-files", "false")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--generate-truth", "true")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--truth", "truth.json")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--matrix", "standard")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--index-directory", "index")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--candidate-set", "all")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--allowlist", "all")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--comparison", "hnswlib")]
    public void ParseExternalFashionMnistDurableHnsw_RejectsMalformedAndCrossScenarioOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistDurableHnsw(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_UsesExistingCacheAndTruthWithoutMutatingAdmittedArtifacts()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "cache-reuse",
            baseCount: 36,
            queryCount: 5,
            truthDepth: 4,
            rows: 3,
            columns: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "external-durable-independent.json");
        string snapshotRoot = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "snapshot-root");
        string occupiedRunDirectory = Path.Combine(snapshotRoot, "run-001");
        Directory.CreateDirectory(occupiedRunDirectory);
        string occupiedMarker = Path.Combine(occupiedRunDirectory, "do-not-overwrite.txt");
        File.WriteAllText(occupiedMarker, "existing snapshot content");
        Dictionary<string, string> cacheBefore = SnapshotFiles(cacheRoot);

        ExternalDurableHnswBenchmarkReport report = FashionMnistExternalDurableHnswBenchmarkScenario.Run(
            new FashionMnistExternalDurableHnswBenchmarkOptions(
                cacheRoot,
                outputPath,
                snapshotRoot,
                QueryCount: 4,
                TopK: 4,
                Runs: 2,
                WarmupQueries: 5,
                VectorMetric.SquaredEuclidean,
                M: 3,
                EfConstruction: 9,
                EfSearch: 4,
                HnswSeed: 0x10901UL),
            ["external-fashion-mnist-hnsw-durable", "--cache-root", cacheRoot]);
        FashionMnistExternalDurableHnswBenchmarkScenario.Write(report, outputPath);

        Assert.Equal(cacheBefore, SnapshotFiles(cacheRoot));
        Assert.Equal(FileChecksum.ComputeSha256(admission.TruthPath), report.Truth.Sha256);
        Assert.True(File.Exists(occupiedMarker));
        Assert.NotEqual(Path.GetFullPath(occupiedRunDirectory), Path.GetFullPath(report.Outputs.SnapshotOutput.DirectoryPath));
        Assert.True(Directory.Exists(Path.Combine(snapshotRoot, "run-002")));
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingTruth);
        Assert.True(report.Validation.SourceOpenedParity.AllResultsMatched);
        Assert.True(report.Validation.OutputBytesScannedOutsideSaveOpenDuration);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(5, report.Measurement.Warmup.WarmupCount);
    }

    [Fact]
    public void Run_JsonSchemaIsSeparatedFromOtherHnswReportsAndComparisonArtifacts()
    {
        string directory = CreateArtifactDirectory("schema-separation");
        string exactPath = Path.Combine(directory, "exact-generated.json");
        BenchmarkReport exact = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 5,
                VectorCount: 18,
                QueryCount: 3,
                TopK: 2,
                Seed: 0x5EED1090,
                exactPath,
                BaselineReportId: null,
                Runs: 2,
                WarmupQueries: 1),
            ["exact-generated"]);
        ReportWriter.Write(exact, exactPath);

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "schema-external",
            baseCount: 24,
            queryCount: 3,
            truthDepth: 3,
            rows: 2,
            columns: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string durablePath = Path.Combine(directory, "external-durable-hnsw.json");
        ExternalDurableHnswBenchmarkReport durable = FashionMnistExternalDurableHnswBenchmarkScenario.Run(
            new FashionMnistExternalDurableHnswBenchmarkOptions(
                cacheRoot,
                durablePath,
                Path.Combine(directory, "snapshot"),
                QueryCount: 3,
                TopK: 3,
                Runs: 1,
                WarmupQueries: 0,
                VectorMetric.SquaredEuclidean,
                M: 2,
                EfConstruction: 4,
                EfSearch: 3,
                HnswSeed: 0x10902UL),
            ["external-fashion-mnist-hnsw-durable"]);
        FashionMnistExternalDurableHnswBenchmarkScenario.Write(durable, durablePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(durablePath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalDurableHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.NotEqual("VecNet.ExternalHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.NotEqual("VecNet.DurableHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("external-fashion-mnist-hnsw-durable", root.GetProperty("scenarioName").GetString());
        Assert.Equal("sourceSearch", root.GetProperty("operations").GetProperty("sourceSearch").GetProperty("name").GetString());
        Assert.Equal("openedSearch", root.GetProperty("operations").GetProperty("openedSearch").GetProperty("name").GetString());
        Assert.True(root.GetProperty("metrics").GetProperty("sourceAndOpenedRecallEqual").GetBoolean());
        Assert.True(root.GetProperty("metrics").GetProperty("sourceAndOpenedDistanceIntegrityEqual").GetBoolean());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("openedReadOnlyMutation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("openedReadOnlyMutation").GetProperty("rejectedBeforeVectorValidation").GetBoolean());

        string durableBefore = File.ReadAllText(durablePath);
        string exactBefore = File.ReadAllText(exactPath);
        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(exactPath, durablePath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact"]);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "artifactKindMismatch");
        Assert.Empty(comparison.Metrics);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal(exactBefore, File.ReadAllText(exactPath));
        Assert.Equal(durableBefore, File.ReadAllText(durablePath));

        string json = File.ReadAllText(durablePath);
        Assert.DoesNotContain("VecNet.ExternalHnswBenchmarkReport", json, StringComparison.Ordinal);
        Assert.DoesNotContain("VecNet.DurableHnswBenchmarkReport", json, StringComparison.Ordinal);
        Assert.DoesNotContain("baselineCandidateEligibility", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("regressionThreshold", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Program_MissingAdmittedCacheFailsWithoutCreatingReportOrSnapshot()
    {
        string directory = CreateArtifactDirectory("missing-cache");
        string cacheRoot = Path.Combine(directory, "missing-cache-root");
        string outputPath = Path.Combine(directory, "report.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
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
                    "--query-count", "1",
                    "--top-k", "1",
                    "--ef-search", "1"
                ]);

            Assert.Equal(1, exitCode);
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
        }

        Assert.Contains("requires an existing admitted Fashion-MNIST dataset manifest", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "truth")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "converted")));
        Assert.Empty(output.ToString());
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(
        string prefix,
        int baseCount,
        int queryCount,
        int truthDepth,
        int rows,
        int columns)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows, columns);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static Dictionary<string, string> SnapshotFiles(string root) =>
        Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
            .Select(path => new
            {
                RelativePath = Path.GetRelativePath(root, path).Replace('\\', '/'),
                Sha256 = FileChecksum.ComputeSha256(path)
            })
            .OrderBy(item => item.RelativePath, StringComparer.Ordinal)
            .ToDictionary(item => item.RelativePath, item => item.Sha256, StringComparer.Ordinal);

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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 13)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 2)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 101)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 5)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 31 + column * 17 + offset + ((row * column) % 11) * 7) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count, int offset)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)((i + offset) % 10);
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
            "vec109-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
