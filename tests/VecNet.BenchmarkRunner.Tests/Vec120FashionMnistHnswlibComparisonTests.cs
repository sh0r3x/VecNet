using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec120FashionMnistHnswlibComparisonTests
{
    [Fact]
    public void ParseFashionMnistHnswlibComparison_UsesPrivatePinnedDefaults()
    {
        FashionMnistHnswlibComparisonOptions options = CommandLine.ParseFashionMnistHnswlibComparison(["external-fashion-mnist-hnswlib-comparison"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison.json"), options.OutputPath);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-work"), options.WorkDirectory);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-vecnet-snapshot"), options.VecNetSnapshotDirectory);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-hnswlib-comparison-hnswlib.bin"), options.HnswlibIndexPath);
        Assert.Equal(HnswEstablishedComparisonOptions.Default.HnswlibPythonPath, options.HnswlibPythonPath);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(100, options.EfSearch);
        Assert.Equal(0x484E535700012000UL, options.Seed);
        Assert.Equal("0.8.0", HnswEstablishedComparisonOptions.HnswlibVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256);
        Assert.Equal("Apache-2.0", HnswEstablishedComparisonOptions.HnswlibLicense);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--download", "false")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--metric", "SquaredEuclidean")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--vectors", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--queries", "3")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--hnsw-seed", "0x1200")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--output", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--work-directory", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--vecnet-snapshot-directory", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--hnswlib-index", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--hnswlib-python", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--runs", "0")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--runs", "6")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--m", "1")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--m", "65")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--m", "8", "--ef-construction", "7")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--seed", "0xNOTHEX")]
    public void ParseFashionMnistHnswlibComparison_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparison(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ProgramRun_WithMissingHnswlibPythonFailsWithoutWritingFakeReportOrWork()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("missing-tool", baseCount: 16, queryCount: 3, truthDepth: 3, rows: 2, columns: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "run");
        Directory.CreateDirectory(directory);
        string outputPath = Path.Combine(directory, "comparison.json");
        string workDirectory = Path.Combine(directory, "work");
        string snapshotDirectory = Path.Combine(directory, "vecnet-snapshot");
        string hnswlibIndexPath = Path.Combine(directory, "hnswlib-index.bin");
        string missingPythonPath = Path.Combine(directory, "missing-python.exe");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "external-fashion-mnist-hnswlib-comparison",
                "--cache-root", cacheRoot,
                "--output", outputPath,
                "--work-directory", workDirectory,
                "--vecnet-snapshot-directory", snapshotDirectory,
                "--hnswlib-index", hnswlibIndexPath,
                "--hnswlib-python", missingPythonPath,
                "--query-count", "2",
                "--top-k", "2",
                "--runs", "1",
                "--warmup-queries", "0",
                "--m", "2",
                "--ef-construction", "4",
                "--ef-search", "2",
                "--seed", "0x0000000000001200"
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(workDirectory));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.False(File.Exists(hnswlibIndexPath));
    }

    [Fact]
    public void Run_WhenPinnedHnswlibIsUsable_ProducesPrivateExternalComparisonReport()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!PinnedHnswlibIsUsable(pythonPath))
        {
            return;
        }

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("report", baseCount: 32, queryCount: 4, truthDepth: 4, rows: 2, columns: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "comparison");
        Directory.CreateDirectory(directory);
        string outputPath = Path.Combine(directory, "fashion-mnist-hnswlib-comparison.json");
        var options = new FashionMnistHnswlibComparisonOptions(
            cacheRoot,
            outputPath,
            Path.Combine(directory, "work"),
            Path.Combine(directory, "vecnet-snapshot"),
            Path.Combine(directory, "hnswlib-index.bin"),
            pythonPath,
            QueryCount: 3,
            TopK: 4,
            Runs: 1,
            WarmupQueries: 1,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            Seed: 0x0000000000001201UL);

        FashionMnistHnswlibComparisonReport report = FashionMnistHnswlibComparisonScenario.Run(
            options,
            ["external-fashion-mnist-hnswlib-comparison"]);
        FashionMnistHnswlibComparisonScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.True(File.Exists(options.HnswlibIndexPath));
        Assert.Equal("VecNet.FashionMnistHnswlibComparisonReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-120", report.TaskId);
        Assert.Equal("external-fashion-mnist-hnswlib-comparison", report.ScenarioName);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("private-fashion-mnist-hnswlib-comparison", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonPublicationEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.Equal("hnswlib", report.SourcePinning.PackageName);
        Assert.Equal("PyPI", report.SourcePinning.PackageSource);
        Assert.Equal("0.8.0", report.SourcePinning.PackageVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", report.SourcePinning.SourceDistributionSha256);
        Assert.Equal("Apache-2.0", report.SourcePinning.License);
        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Truth.Sha256, report.Truth.Sha256);
        Assert.Equal(32, report.Workload.BaseCount);
        Assert.Equal(4, report.Workload.QueryMatrixCount);
        Assert.Equal(3, report.Workload.MeasuredQueryCount);
        Assert.Equal(4, report.Workload.TopK);
        Assert.Equal(4, report.Truth.TruthDepth);
        Assert.Equal(2, report.Parameters.M);
        Assert.Equal(8, report.Parameters.EfConstruction);
        Assert.Equal(4, report.Parameters.EfSearch);
        Assert.Equal("0x0000000000001201", report.Parameters.Seed);
        Assert.Equal("VecNet", report.VecNet.Name);
        Assert.Equal("hnswlib", report.Hnswlib.Name);
        Assert.Equal("0.8.0", report.Hnswlib.Version);
        Assert.Equal("passed", report.VecNet.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal("passed", report.Hnswlib.Metrics.ReturnedResultIntegrity.Status);
        Assert.InRange(report.VecNet.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Hnswlib.Metrics.RecallAtK, 0, 1);
        Assert.Equal("measured", report.VecNet.Search.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Hnswlib.Search.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.VecNet.Memory.Status);
        Assert.Equal("notMeasured", report.Hnswlib.Memory.Status);
        Assert.Equal("fileFacts", report.VecNet.PersistedBytes.Status);
        Assert.Equal("fileFacts", report.Hnswlib.PersistedBytes.Status);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingCache);
        Assert.True(report.Validation.LoadedExistingTruth);
        Assert.True(report.Validation.IdenticalVectorsQueriesIdsAndParameters);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonPublicationEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonPublicationEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.FashionMnistHnswlibComparisonReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("dataset").GetProperty("datasetId").GetString());
        Assert.Equal("0.8.0", root.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal("passed", root.GetProperty("vecNet").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("hnswlib").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("hnswlib").GetProperty("memory").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonPublicationEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("\"publicClaimEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    private static bool PinnedHnswlibIsUsable(string pythonPath)
    {
        if (!File.Exists(pythonPath))
        {
            return false;
        }

        try
        {
            var startInfo = new ProcessStartInfo(pythonPath)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            startInfo.ArgumentList.Add("-c");
            startInfo.ArgumentList.Add("import importlib.metadata; import hnswlib; print(importlib.metadata.version('hnswlib'))");
            using Process process = Process.Start(startInfo)!;
            string output = process.StandardOutput.ReadToEnd();
            _ = process.StandardError.ReadToEnd();
            process.WaitForExit(5000);
            return process.ExitCode == 0 && output.Contains("0.8.0", StringComparison.Ordinal);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
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
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 43)).ToArray());
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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 13 + offset) % 251);
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
            string.Create(CultureInfo.InvariantCulture, $"vec120-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
