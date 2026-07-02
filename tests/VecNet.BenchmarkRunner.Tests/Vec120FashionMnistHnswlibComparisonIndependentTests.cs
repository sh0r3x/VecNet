using System.Buffers.Binary;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec120FashionMnistHnswlibComparisonIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveBoundaryCommandAndUsesOneComparisonSeed()
    {
        FashionMnistHnswlibComparisonOptions options = CommandLine.ParseFashionMnistHnswlibComparison(
            [
                "EXTERNAL-FASHION-MNIST-HNSWLIB-COMPARISON",
                "--CACHE-ROOT", "VecNet.DatasetCache",
                "--OUTPUT", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec120-independent", "report.json"),
                "--WORK-DIRECTORY", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec120-independent", "work"),
                "--VECNET-SNAPSHOT-DIRECTORY", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec120-independent", "snapshot"),
                "--HNSWLIB-INDEX", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec120-independent", "hnswlib.bin"),
                "--HNSWLIB-PYTHON", HnswEstablishedComparisonOptions.Default.HnswlibPythonPath,
                "--QUERY-COUNT", "1",
                "--TOP-K", "4096",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "100",
                "--M", "64",
                "--EF-CONSTRUCTION", "4096",
                "--EF-SEARCH", "4096",
                "--SEED", "0xFFFFFFFFFFFFFFFF"
            ]);

        Assert.Equal(FashionMnistHnswlibComparisonOptions.ScenarioName, "external-fashion-mnist-hnswlib-comparison");
        Assert.Equal(1, options.QueryCount);
        Assert.Equal(4096, options.TopK);
        Assert.Equal(5, options.Runs);
        Assert.Equal(100, options.WarmupQueries);
        Assert.Equal(64, options.M);
        Assert.Equal(4096, options.EfConstruction);
        Assert.Equal(4096, options.EfSearch);
        Assert.Equal(ulong.MaxValue, options.Seed);
        Assert.True(options.EfSearch >= options.TopK);
    }

    [Theory]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--download", "true")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--download", "false")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--truth-refresh", "true")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--dimension", "784")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--vectors", "60000")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--queries", "10000")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--metric", "squared-euclidean")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--hnsw-seed", "0x484E535700012000")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--snapshot-directory", "durable-snapshot")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--sample-interval-ms", "1")]
    [InlineData("external-fashion-mnist", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist-exact", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist-hnsw", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--hnswlib-python", "python.exe")]
    [InlineData("hnswlib-generated-comparison", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnswlib-generated-comparison-matrix", "--cache-root", "VecNet.DatasetCache")]
    public void Parsers_KeepFashionMnistComparisonScopedAwayFromDownloadTruthGeneratedAndOtherModes(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "external-fashion-mnist-hnswlib-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparison(args)),
            "external-fashion-mnist" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(args)),
            "external-fashion-mnist-exact" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistExact(args)),
            "external-fashion-mnist-hnsw" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(args)),
            "external-fashion-mnist-hnsw-durable" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistDurableHnsw(args)),
            "hnswlib-generated-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args)),
            "hnswlib-generated-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparisonMatrix(args)),
            _ => throw new InvalidOperationException("Unexpected parser fixture.")
        };

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ProgramRun_WithMissingToolAndMissingCacheStillFailsBeforeCacheOrWorkCreation()
    {
        string directory = NewArtifactDirectory("missing-tool-before-cache");
        string missingCacheRoot = Path.Combine(directory, "missing-cache");
        string outputPath = Path.Combine(directory, "report.json");
        string workDirectory = Path.Combine(directory, "work");
        string snapshotDirectory = Path.Combine(directory, "vecnet-snapshot");
        string hnswlibIndex = Path.Combine(directory, "hnswlib.bin");
        string missingPython = Path.Combine(directory, "missing", "python.exe");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "external-fashion-mnist-hnswlib-comparison",
                "--cache-root", missingCacheRoot,
                "--output", outputPath,
                "--work-directory", workDirectory,
                "--vecnet-snapshot-directory", snapshotDirectory,
                "--hnswlib-index", hnswlibIndex,
                "--hnswlib-python", missingPython,
                "--query-count", "1",
                "--top-k", "1",
                "--runs", "1",
                "--warmup-queries", "0",
                "--m", "2",
                "--ef-construction", "2",
                "--ef-search", "1"
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(workDirectory));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.False(File.Exists(hnswlibIndex));
        Assert.False(Directory.Exists(missingCacheRoot));
        Assert.Empty(Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories));
    }

    [Fact]
    public void ReportJson_WhenPinnedHnswlibIsUsable_PreservesAdmittedFashionMnistAndPrivateEligibilityInvariants()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!PinnedHnswlibIsUsable(pythonPath))
        {
            return;
        }

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "report-json",
            baseCount: 18,
            queryCount: 3,
            truthDepth: 3,
            rows: 28,
            columns: 28);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "comparison-run");
        Directory.CreateDirectory(directory);

        string outputPath = Path.Combine(directory, "fashion-mnist-hnswlib-comparison.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison",
            "--cache-root", cacheRoot,
            "--output", outputPath,
            "--work-directory", Path.Combine(directory, "work"),
            "--vecnet-snapshot-directory", Path.Combine(directory, "vecnet-snapshot"),
            "--hnswlib-index", Path.Combine(directory, "hnswlib.bin"),
            "--hnswlib-python", pythonPath,
            "--query-count", "2",
            "--top-k", "3",
            "--runs", "1",
            "--warmup-queries", "1",
            "--m", "2",
            "--ef-construction", "6",
            "--ef-search", "3",
            "--seed", "0x0000000000001202"
        ];

        FashionMnistHnswlibComparisonOptions options = CommandLine.ParseFashionMnistHnswlibComparison(args);
        FashionMnistHnswlibComparisonReport report = FashionMnistHnswlibComparisonScenario.Run(options, args);
        FashionMnistHnswlibComparisonScenario.Write(report, outputPath);

        string json = File.ReadAllText(outputPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.FashionMnistHnswlibComparisonReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-120", root.GetProperty("taskId").GetString());
        Assert.Equal(FashionMnistHnswlibComparisonOptions.ScenarioName, root.GetProperty("scenarioName").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());

        JsonElement sourcePinning = root.GetProperty("sourcePinning");
        Assert.Equal("hnswlib", sourcePinning.GetProperty("packageName").GetString());
        Assert.Equal("PyPI", sourcePinning.GetProperty("packageSource").GetString());
        Assert.Equal("0.8.0", sourcePinning.GetProperty("packageVersion").GetString());
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", sourcePinning.GetProperty("sourceDistributionSha256").GetString());
        Assert.Equal("Apache-2.0", sourcePinning.GetProperty("license").GetString());
        Assert.Contains("non-shipping", sourcePinning.GetProperty("licensePosture").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("No hnswlib", sourcePinning.GetProperty("productDependencyPosture").GetString(), StringComparison.Ordinal);

        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("dataset").GetProperty("datasetId").GetString());
        Assert.Equal("VecNet.ExternalDatasetManifest", root.GetProperty("dataset").GetProperty("admissionManifest").GetProperty("schemaName").GetString());
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), root.GetProperty("dataset").GetProperty("admissionManifest").GetProperty("sha256").GetString());
        Assert.Equal(admission.Manifest.Truth.Sha256, root.GetProperty("truth").GetProperty("sha256").GetString());
        Assert.Equal(18, root.GetProperty("workload").GetProperty("baseCount").GetInt32());
        Assert.Equal(3, root.GetProperty("workload").GetProperty("queryMatrixCount").GetInt32());
        Assert.Equal(2, root.GetProperty("workload").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(784, root.GetProperty("workload").GetProperty("dimension").GetInt32());
        Assert.Equal(3, root.GetProperty("truth").GetProperty("truthDepth").GetInt32());
        Assert.Equal("first N query vectors from the admitted query matrix", root.GetProperty("truth").GetProperty("querySelection").GetString());

        JsonElement parameters = root.GetProperty("parameters");
        Assert.Equal("SquaredEuclidean", parameters.GetProperty("metric").GetString());
        Assert.Equal(784, parameters.GetProperty("dimension").GetInt32());
        Assert.Equal(18, parameters.GetProperty("baseVectorCount").GetInt32());
        Assert.Equal(3, parameters.GetProperty("queryMatrixCount").GetInt32());
        Assert.Equal(2, parameters.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(3, parameters.GetProperty("topK").GetInt32());
        Assert.Equal(2, parameters.GetProperty("m").GetInt32());
        Assert.Equal(6, parameters.GetProperty("efConstruction").GetInt32());
        Assert.Equal(6, parameters.GetProperty("hnswlibEfConstruction").GetInt32());
        Assert.Equal(3, parameters.GetProperty("efSearch").GetInt32());
        Assert.Equal(3, parameters.GetProperty("hnswlibEf").GetInt32());
        Assert.Equal("0x0000000000001202", parameters.GetProperty("seed").GetString());
        Assert.Equal(1, parameters.GetProperty("threadCount").GetInt32());

        Assert.Contains("never downloads", root.GetProperty("methodology").GetProperty("datasetPolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("out-of-process", root.GetProperty("methodology").GetProperty("pythonBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("same binary inputs", root.GetProperty("methodology").GetProperty("identicalInputsPolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("loadedExistingCache").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("loadedExistingTruth").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("identicalVectorsQueriesIdsAndParameters").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("vecNetComparedToTruth").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("hnswlibComparedToTruth").GetBoolean());

        AssertImplementation(root.GetProperty("vecNet"), "VecNet", expectedVersion: null, expectedSearchAllocationStatus: "measured");
        AssertImplementation(root.GetProperty("hnswlib"), "hnswlib", expectedVersion: "0.8.0", expectedSearchAllocationStatus: "notMeasured");
        Assert.Equal("fileFacts", root.GetProperty("vecNet").GetProperty("persistedBytes").GetProperty("status").GetString());
        Assert.Equal("fileFacts", root.GetProperty("hnswlib").GetProperty("persistedBytes").GetProperty("status").GetString());

        AssertNoTrueEligibilityFields(root);
        AssertNoPropertyNamed(
            root,
            "downloadRawFiles",
            "truthRefresh",
            "generatedData",
            "generatedVectorCount",
            "generatedQueryCount",
            "baseline",
            "baselineReportId",
            "candidateEligibility",
            "comparisonArtifactEligible",
            "publicClaimStatus",
            "regressionDecision",
            "regressionThreshold",
            "packageMetadata",
            "packageProjectUrl",
            "nugetPublication");
        Assert.DoesNotContain("\"publicClaimEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("README.md", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("PackageReference", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NuGet", json, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertImplementation(JsonElement implementation, string expectedName, string? expectedVersion, string expectedSearchAllocationStatus)
    {
        Assert.Equal(expectedName, implementation.GetProperty("name").GetString());
        if (expectedVersion is not null)
        {
            Assert.Equal(expectedVersion, implementation.GetProperty("version").GetString());
        }

        JsonElement metrics = implementation.GetProperty("metrics");
        Assert.Equal("measured", implementation.GetProperty("build").GetProperty("status").GetString());
        Assert.Equal("measured", implementation.GetProperty("search").GetProperty("status").GetString());
        Assert.Equal(2, implementation.GetProperty("search").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(expectedSearchAllocationStatus, implementation.GetProperty("search").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("notMeasured", implementation.GetProperty("memory").GetProperty("status").GetString());
        Assert.InRange(metrics.GetProperty("recallAtK").GetDouble(), 0, 1);
        Assert.InRange(metrics.GetProperty("orderedAgreement").GetDouble(), 0, 1);
        Assert.Equal("passed", metrics.GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("duplicateIdCount").GetInt32());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("unknownIdCount").GetInt32());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("nonFiniteDistanceCount").GetInt32());
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
            if (!process.WaitForExit(5000))
            {
                process.Kill(entireProcessTree: true);
                return false;
            }

            return process.ExitCode == 0 && output.Trim().Equals("0.8.0", StringComparison.Ordinal);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or Win32Exception)
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
        string cacheRoot = NewArtifactDirectory(prefix);
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 31)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 71)).ToArray());
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
                payload[(row * dimension) + column] = (byte)((row * 17 + column * 19 + offset) % 251);
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

    private static void AssertNoTrueEligibilityFields(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Name.EndsWith("Eligible", StringComparison.OrdinalIgnoreCase))
                {
                    Assert.Equal(JsonValueKind.False, property.Value.ValueKind);
                }

                AssertNoTrueEligibilityFields(property.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoTrueEligibilityFields(item);
            }
        }
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                AssertNoPropertyNamed(property.Value, disallowedNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, disallowedNames);
            }
        }
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec120-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
