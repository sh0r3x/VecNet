using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec121FashionMnistHnswlibComparisonMatrixTests
{
    [Fact]
    public void ParseFashionMnistHnswlibComparisonMatrix_UsesPrivateSmokeDefaultsAndExpandsProfiles()
    {
        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(
            ["external-fashion-mnist-hnswlib-comparison-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(0x484E535700012100UL, options.Seed);
        Assert.EndsWith(Path.Combine("vec-118-tools", "hnswlib-venv", "Scripts", "python.exe"), options.HnswlibPythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("fashion-mnist-hnswlib-comparison-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);

        FashionMnistHnswlibComparisonMatrixScenario.FashionMnistComparisonMatrixCase[] cases =
            FashionMnistHnswlibComparisonMatrixScenario.ExpandCases(options);

        Assert.Equal(3, cases.Length);
        Assert.Equal([10], cases.Select(c => c.Options.TopK).Distinct().ToArray());
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(c => c.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.Contains(cases, c => c.ProfileName == "balanced-m8" && c.Options.M == 8 && c.Options.EfConstruction == 64 && c.Options.EfSearch == 128);
        Assert.Contains(cases, c => c.ProfileName == "wide-m16" && c.Options.M == 16 && c.Options.EfConstruction == 128 && c.Options.EfSearch == 192);
        Assert.Contains(cases, c => c.ProfileName == "default-m16" && c.Options.M == 16 && c.Options.EfConstruction == 200 && c.Options.EfSearch == 200);
        Assert.All(cases, matrixCase =>
        {
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.Equal(options.CacheRoot, matrixCase.Options.CacheRoot);
            Assert.Equal(options.HnswlibPythonPath, matrixCase.Options.HnswlibPythonPath);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.WorkDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.VecNetSnapshotDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.HnswlibIndexPath, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void ExpandCases_StandardPresetIncludesProfilesAndTopKOneHundred()
    {
        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(
            [
                "external-fashion-mnist-hnswlib-comparison-matrix",
                "--preset", "standard",
                "--query-count", "2",
                "--runs", "5",
                "--warmup-queries", "1",
                "--seed", "0xFFFFFFFFFFFFFFFF"
            ]);

        FashionMnistHnswlibComparisonMatrixScenario.FashionMnistComparisonMatrixCase[] cases =
            FashionMnistHnswlibComparisonMatrixScenario.ExpandCases(options);

        Assert.Equal(6, cases.Length);
        Assert.Equal([10, 100], cases.Select(c => c.Options.TopK).Distinct().ToArray());
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(c => c.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(6, cases.Select(c => c.CaseId).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(6, cases.Select(c => c.Options.Seed).Distinct().Count());
        Assert.All(cases, matrixCase =>
        {
            Assert.Equal("VecNet.DatasetCache", matrixCase.Options.CacheRoot);
            Assert.Equal(2, matrixCase.Options.QueryCount);
            Assert.Equal(5, matrixCase.Options.Runs);
            Assert.Equal(1, matrixCase.Options.WarmupQueries);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
        });
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--preset", "unknown")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--runs", "0")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--runs", "6")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--seed", "0xNOTHEX")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--output-dir", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--manifest", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--hnswlib-python", "")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--top-k", "10")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--m", "8")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--ef-search", "128")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--download", "false")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--output", "report.json")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--output-dir", "matrix")]
    [InlineData("hnswlib-generated-comparison-matrix", "--cache-root", "VecNet.DatasetCache")]
    public void Parsers_KeepFashionMnistMatrixSingleCaseGeneratedAndDatasetAdmissionModesIsolated(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "external-fashion-mnist-hnswlib-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args)),
            "external-fashion-mnist-hnswlib-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparison(args)),
            "hnswlib-generated-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparisonMatrix(args)),
            _ => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args))
        };

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_WithMissingCacheWritesBlockedManifestWithoutLinkedReports()
    {
        string directory = NewArtifactDirectory("missing-cache");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "standard",
            "--cache-root", Path.Combine(directory, "missing-cache-root"),
            "--query-count", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--hnswlib-python", Path.Combine(directory, "missing-python.exe"),
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args);
        FashionMnistHnswlibComparisonMatrixManifest manifest = FashionMnistHnswlibComparisonMatrixScenario.Run(options, args);
        FashionMnistHnswlibComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.FashionMnistHnswlibComparisonMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-121", manifest.TaskId);
        Assert.Equal("external-fashion-mnist-hnswlib-comparison-matrix", manifest.ScenarioName);
        Assert.Equal("unavailable", manifest.CacheTruth.Status);
        Assert.Equal(6, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.SkippedCaseCount);
        Assert.Equal(6, manifest.Aggregate.BlockedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("blocked", matrixCase.Status);
            Assert.Equal("blocked", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.False(File.Exists(matrixCase.LinkedReportPath));
            Assert.Equal("fashion-mnist-784-euclidean", matrixCase.DatasetId);
            Assert.Equal(784, matrixCase.Dimension);
            Assert.True(matrixCase.EfSearch >= matrixCase.TopK);
        });

        string json = File.ReadAllText(manifestPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VEC-121", root.GetProperty("taskId").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("cacheTruth").GetProperty("datasetId").GetString());
        Assert.Equal("hnswlib", root.GetProperty("sourcePinning").GetProperty("packageName").GetString());
        Assert.Equal("0.8.0", root.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal([10, 100], ToIntArray(root.GetProperty("design").GetProperty("topKValues")));
        AssertNoTrueEligibilityFields(root);
        Assert.DoesNotContain("\"taskId\": \"VEC-120\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"status\": \"passed\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"linkedReportId\": \"", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_WithAdmittedCacheButMissingHnswlibWritesBlockedManifestWithoutFakeReports()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("missing-hnswlib", baseCount: 18, queryCount: 2, truthDepth: 10);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix");
        Directory.CreateDirectory(directory);
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "smoke",
            "--cache-root", cacheRoot,
            "--query-count", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--hnswlib-python", Path.Combine(directory, "missing-python.exe"),
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        int exitCode = BenchmarkRunnerProgram.Run(args);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("available", root.GetProperty("cacheTruth").GetProperty("status").GetString());
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), root.GetProperty("cacheTruth").GetProperty("admissionManifestSha256").GetString());
        Assert.Equal(admission.Manifest.Truth.Sha256, root.GetProperty("cacheTruth").GetProperty("truthSha256").GetString());
        Assert.Equal(3, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.All(root.GetProperty("cases").EnumerateArray(), matrixCase =>
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.False(File.Exists(matrixCase.GetProperty("linkedReportPath").GetString()!));
            Assert.Contains("unavailable", matrixCase.GetProperty("errorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void Run_WhenPinnedHnswlibIsUsableWritesManifestAndVec120LinkedReports()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!PinnedHnswlibIsUsable(pythonPath))
        {
            return;
        }

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("linked-report", baseCount: 18, queryCount: 2, truthDepth: 10);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix-success");
        Directory.CreateDirectory(directory);
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "smoke",
            "--cache-root", cacheRoot,
            "--query-count", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x0000000000012100",
            "--hnswlib-python", pythonPath,
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args);
        FashionMnistHnswlibComparisonMatrixManifest manifest = FashionMnistHnswlibComparisonMatrixScenario.Run(options, args);
        FashionMnistHnswlibComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal(3, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal("available", manifest.CacheTruth.Status);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonPublicationEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        FashionMnistHnswlibComparisonMatrixCaseManifest selectedCase = manifest.Cases.Single(c => c.ProfileName == "balanced-m8");
        Assert.Equal("passed", selectedCase.Status);
        Assert.NotNull(selectedCase.LinkedReportId);
        Assert.True(File.Exists(selectedCase.LinkedReportPath));

        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(selectedCase.LinkedReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.FashionMnistHnswlibComparisonReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-120", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("external-fashion-mnist-hnswlib-comparison", reportRoot.GetProperty("scenarioName").GetString());
        Assert.Equal(selectedCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
        Assert.Equal("0.8.0", reportRoot.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("vecNet").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("hnswlib").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        AssertNoTrueEligibilityFields(reportRoot);
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
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = NewArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(string cacheRoot, int baseCount, int queryCount)
    {
        const int rows = 28;
        const int columns = 28;
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 53)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 89)).ToArray());
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
                payload[(row * dimension) + column] = (byte)((row * 23 + column * 11 + offset) % 251);
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

    private static int[] ToIntArray(JsonElement array) =>
        array.EnumerateArray().Select(value => value.GetInt32()).ToArray();

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

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec121-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
