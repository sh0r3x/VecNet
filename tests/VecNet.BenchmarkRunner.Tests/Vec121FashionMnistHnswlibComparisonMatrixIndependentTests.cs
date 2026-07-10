using System.Buffers.Binary;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec121FashionMnistHnswlibComparisonMatrixIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveBoundariesAndExpandsStandardCasesWithoutSingleCaseControls()
    {
        string outputDirectory = Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec121-independent-standard");
        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(
            [
                "EXTERNAL-FASHION-MNIST-HNSWLIB-COMPARISON-MATRIX",
                "--PRESET", "STANDARD",
                "--CACHE-ROOT", "VecNet.DatasetCache",
                "--QUERY-COUNT", "1",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "100",
                "--SEED", "0xFFFFFFFFFFFFFFFF",
                "--OUTPUT-DIR", outputDirectory,
                "--MANIFEST", Path.Combine(outputDirectory, "manifest.json"),
                "--HNSWLIB-PYTHON", HnswEstablishedComparisonOptions.Default.HnswlibPythonPath
            ]);

        FashionMnistHnswlibComparisonMatrixScenario.FashionMnistComparisonMatrixCase[] cases =
            FashionMnistHnswlibComparisonMatrixScenario.ExpandCases(options);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(1, options.QueryCount);
        Assert.Equal(5, options.Runs);
        Assert.Equal(100, options.WarmupQueries);
        Assert.Equal(ulong.MaxValue, options.Seed);
        Assert.Equal(6, cases.Length);
        Assert.Equal([10, 100], cases.Select(matrixCase => matrixCase.Options.TopK).Distinct().ToArray());
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(matrixCase => matrixCase.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(["case-01-balanced-m8-10k", "case-02-wide-m16-10k", "case-03-default-m16-10k", "case-04-balanced-m8-100k", "case-05-wide-m16-100k", "case-06-default-m16-100k"], cases.Select(matrixCase => matrixCase.CaseId).ToArray());
        Assert.Equal(6, cases.Select(matrixCase => matrixCase.Options.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal(6, cases.Select(matrixCase => matrixCase.Options.WorkDirectory).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal(6, cases.Select(matrixCase => matrixCase.Options.HnswlibIndexPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal([ulong.MaxValue, 0UL, 1UL, 2UL, 3UL, 4UL], cases.Select(matrixCase => matrixCase.Options.Seed).ToArray());

        Assert.Contains(cases, matrixCase => matrixCase.ProfileName == "balanced-m8" && matrixCase.Options.M == 8 && matrixCase.Options.EfConstruction == 64 && matrixCase.Options.EfSearch == 128);
        Assert.Contains(cases, matrixCase => matrixCase.ProfileName == "wide-m16" && matrixCase.Options.M == 16 && matrixCase.Options.EfConstruction == 128 && matrixCase.Options.EfSearch == 192);
        Assert.Contains(cases, matrixCase => matrixCase.ProfileName == "default-m16" && matrixCase.Options.M == 16 && matrixCase.Options.EfConstruction == 200 && matrixCase.Options.EfSearch == 200);
        Assert.All(cases, matrixCase =>
        {
            Assert.Equal(FashionMnistHnswlibComparisonOptions.ScenarioName, FashionMnistHnswlibComparisonOptions.ScenarioName);
            Assert.Equal("VecNet.DatasetCache", matrixCase.Options.CacheRoot);
            Assert.Equal(1, matrixCase.Options.QueryCount);
            Assert.Equal(5, matrixCase.Options.Runs);
            Assert.Equal(100, matrixCase.Options.WarmupQueries);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.StartsWith(outputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(outputDirectory, matrixCase.Options.WorkDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(outputDirectory, matrixCase.Options.VecNetSnapshotDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(outputDirectory, matrixCase.Options.HnswlibIndexPath, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Theory]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--download", "true")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--download", "false")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--truth-refresh", "true")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--dimension", "784")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--vectors", "60000")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--queries", "10000")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--top-k", "100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--m", "8")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--ef-construction", "64")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--ef-search", "128")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--hnsw-seed", "0x484E535700012100")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--output", "case.json")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--work-directory", "work")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--vecnet-snapshot-directory", "snapshot")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--hnswlib-index", "index.bin")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--snapshot-directory", "durable")]
    [InlineData("external-fashion-mnist-hnswlib-comparison-matrix", "--sample-interval-ms", "1")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnswlib-comparison", "--manifest", "manifest.json")]
    [InlineData("hnswlib-generated-comparison-matrix", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnswlib-generated-comparison-matrix", "--truth-depth", "100")]
    [InlineData("hnswlib-generated-comparison-matrix", "--download", "false")]
    [InlineData("hnswlib-generated-comparison", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnsw-generated", "--hnswlib-python", "python.exe")]
    [InlineData("hnsw-generated-matrix", "--hnswlib-python", "python.exe")]
    [InlineData("durable-hnsw-generated", "--hnswlib-python", "python.exe")]
    [InlineData("durable-hnsw-generated-matrix", "--hnswlib-python", "python.exe")]
    [InlineData("generated-hnsw-memory-smoke", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist-hnsw", "--hnswlib-python", "python.exe")]
    [InlineData("external-fashion-mnist-hnsw-durable", "--hnswlib-python", "python.exe")]
    public void Parsers_KeepFashionMnistMatrixIsolatedFromRefreshSingleCaseGeneratedDurableAndMemoryModes(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "external-fashion-mnist-hnswlib-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args)),
            "external-fashion-mnist-hnswlib-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseFashionMnistHnswlibComparison(args)),
            "hnswlib-generated-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparisonMatrix(args)),
            "hnswlib-generated-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args)),
            "hnsw-generated" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(args)),
            "hnsw-generated-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(args)),
            "durable-hnsw-generated" => Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(args)),
            "durable-hnsw-generated-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGeneratedMatrix(args)),
            "generated-hnsw-memory-smoke" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(args)),
            "external-fashion-mnist" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(args)),
            "external-fashion-mnist-hnsw" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(args)),
            "external-fashion-mnist-hnsw-durable" => Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistDurableHnsw(args)),
            _ => throw new InvalidOperationException("Unexpected parser fixture.")
        };

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void MissingCacheStandardManifestJson_PreservesPinProfilesCountsEligibilityAndNoFakeReports()
    {
        string directory = NewArtifactDirectory("missing-cache-standard");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "standard",
            "--cache-root", Path.Combine(directory, "cache-does-not-exist"),
            "--query-count", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x0000000000012101",
            "--hnswlib-python", Path.Combine(directory, "missing-python.exe"),
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        FashionMnistHnswlibComparisonMatrixOptions options = CommandLine.ParseFashionMnistHnswlibComparisonMatrix(args);
        FashionMnistHnswlibComparisonMatrixManifest manifest = FashionMnistHnswlibComparisonMatrixScenario.Run(options, args);
        FashionMnistHnswlibComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        string json = File.ReadAllText(manifestPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.FashionMnistHnswlibComparisonMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-121", root.GetProperty("taskId").GetString());
        Assert.Equal(FashionMnistHnswlibComparisonMatrixOptions.ScenarioName, root.GetProperty("scenarioName").GetString());
        Assert.Equal("standard", root.GetProperty("presetName").GetString());
        Assert.Equal(6, root.GetProperty("caseCount").GetInt32());

        JsonElement cacheTruth = root.GetProperty("cacheTruth");
        Assert.Equal("unavailable", cacheTruth.GetProperty("status").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", cacheTruth.GetProperty("datasetId").GetString());
        Assert.Equal(784, cacheTruth.GetProperty("expectedDimension").GetInt32());
        Assert.Equal("SquaredEuclidean", cacheTruth.GetProperty("metric").GetString());
        Assert.Equal(JsonValueKind.Null, cacheTruth.GetProperty("admissionManifestPath").ValueKind);
        Assert.Equal(JsonValueKind.Null, cacheTruth.GetProperty("admissionManifestSha256").ValueKind);
        Assert.Equal(JsonValueKind.Null, cacheTruth.GetProperty("truthRelativePath").ValueKind);
        Assert.Equal(JsonValueKind.Null, cacheTruth.GetProperty("truthSha256").ValueKind);
        Assert.Contains("must not download", cacheTruth.GetProperty("cachePolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("must not refresh truth", cacheTruth.GetProperty("truthPolicy").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement sourcePinning = root.GetProperty("sourcePinning");
        Assert.Equal("hnswlib", sourcePinning.GetProperty("packageName").GetString());
        Assert.Equal("PyPI", sourcePinning.GetProperty("packageSource").GetString());
        Assert.Equal("0.8.0", sourcePinning.GetProperty("packageVersion").GetString());
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", sourcePinning.GetProperty("sourceDistributionSha256").GetString());
        Assert.Equal("Apache-2.0", sourcePinning.GetProperty("license").GetString());
        Assert.Contains("non-shipping", sourcePinning.GetProperty("licensePosture").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Python/native", sourcePinning.GetProperty("nativeBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("No hnswlib", sourcePinning.GetProperty("productDependencyPosture").GetString(), StringComparison.Ordinal);

        JsonElement design = root.GetProperty("design");
        Assert.Equal("fashion-mnist-784-euclidean", design.GetProperty("datasetId").GetString());
        Assert.Equal(784, design.GetProperty("dimension").GetInt32());
        Assert.Equal("SquaredEuclidean", design.GetProperty("metric").GetString());
        Assert.Equal([10, 100], ToIntArray(design.GetProperty("topKValues")));
        AssertProfiles(design.GetProperty("profiles"));
        Assert.Contains("admitted base matrix row order", design.GetProperty("workloadPolicy").GetString(), StringComparison.OrdinalIgnoreCase);

        AssertStatusCountsMatchAggregate(root);
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(6, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal("blocked", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal("fashion-mnist-784-euclidean", matrixCase.GetProperty("datasetId").GetString());
            Assert.Equal("SquaredEuclidean", matrixCase.GetProperty("metric").GetString());
            Assert.Equal(784, matrixCase.GetProperty("dimension").GetInt32());
            Assert.True(matrixCase.GetProperty("efSearch").GetInt32() >= matrixCase.GetProperty("topK").GetInt32());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.False(File.Exists(matrixCase.GetProperty("linkedReportPath").GetString()!));
        }

        AssertNoTrueEligibilityFields(root);
        AssertNoPropertyNamed(
            root,
            "downloadRawFiles",
            "truthRefresh",
            "generatedVectorCount",
            "generatedQueryCount",
            "baselineReportId",
            "candidateEligibility",
            "comparisonArtifactEligible",
            "publicClaimStatus",
            "regressionDecision",
            "regressionThreshold",
            "packageMetadata",
            "packageProjectUrl",
            "nugetPublication");
        Assert.DoesNotContain("\"taskId\": \"VEC-120\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"status\": \"passed\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"linkedReportId\": \"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("README.md", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("PackageReference", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NuGet", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AdmittedCacheButMissingHnswlib_CommandWritesBlockedManifestWithoutPythonWorkOrLinkedReports()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("missing-tool", baseCount: 20, queryCount: 2, truthDepth: 10);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix-missing-tool");
        Directory.CreateDirectory(directory);
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "smoke",
            "--cache-root", cacheRoot,
            "--query-count", "2",
            "--runs", "1",
            "--warmup-queries", "0",
            "--hnswlib-python", Path.Combine(directory, "does-not-exist", "python.exe"),
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        int exitCode = BenchmarkRunnerProgram.Run(args);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));

        string json = File.ReadAllText(manifestPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("available", root.GetProperty("cacheTruth").GetProperty("status").GetString());
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), root.GetProperty("cacheTruth").GetProperty("admissionManifestSha256").GetString());
        Assert.Equal(admission.Manifest.Truth.Sha256, root.GetProperty("cacheTruth").GetProperty("truthSha256").GetString());
        Assert.Equal(3, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        AssertStatusCountsMatchAggregate(root);

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.False(File.Exists(matrixCase.GetProperty("linkedReportPath").GetString()!));
            Assert.Contains("unavailable", matrixCase.GetProperty("errorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        Assert.Empty(Directory.EnumerateFiles(directory, "hnswlib-driver.py", SearchOption.AllDirectories));
        Assert.Empty(Directory.EnumerateFiles(directory, "hnswlib-index.bin", SearchOption.AllDirectories));
        Assert.Empty(Directory.EnumerateFiles(directory, "query-vectors.f32le", SearchOption.AllDirectories));
        Assert.DoesNotContain("\"schemaName\": \"VecNet.FashionMnistHnswlibComparisonReport\"", json, StringComparison.OrdinalIgnoreCase);
        AssertNoTrueEligibilityFields(root);
    }

    [Fact]
    public void SuccessfulLinkedReports_WhenPinnedHnswlibIsUsable_AreVec120ReportsWithSharedMatrixParameters()
    {
        string? pythonPath = FindUsablePinnedHnswlibPython();
        if (pythonPath is null)
        {
            return;
        }

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("success", baseCount: 20, queryCount: 2, truthDepth: 10);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix-success");
        Directory.CreateDirectory(directory);
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "external-fashion-mnist-hnswlib-comparison-matrix",
            "--preset", "smoke",
            "--cache-root", cacheRoot,
            "--query-count", "2",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x0000000000012102",
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

        string manifestJson = File.ReadAllText(manifestPath);
        using JsonDocument manifestDocument = JsonDocument.Parse(manifestJson);
        JsonElement manifestRoot = manifestDocument.RootElement;
        AssertStatusCountsMatchAggregate(manifestRoot);
        AssertNoTrueEligibilityFields(manifestRoot);

        foreach (FashionMnistHnswlibComparisonMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.LinkedReportId);
            Assert.True(File.Exists(matrixCase.LinkedReportPath));

            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(matrixCase.LinkedReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            Assert.Equal("VecNet.FashionMnistHnswlibComparisonReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-120", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal(FashionMnistHnswlibComparisonOptions.ScenarioName, reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
            Assert.Equal("fashion-mnist-784-euclidean", reportRoot.GetProperty("dataset").GetProperty("datasetId").GetString());
            Assert.Equal(admission.Manifest.Truth.Sha256, reportRoot.GetProperty("truth").GetProperty("sha256").GetString());
            Assert.Equal(784, reportRoot.GetProperty("parameters").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("parameters").GetProperty("topK").GetInt32());
            Assert.Equal(matrixCase.M, reportRoot.GetProperty("parameters").GetProperty("m").GetInt32());
            Assert.Equal(matrixCase.EfConstruction, reportRoot.GetProperty("parameters").GetProperty("efConstruction").GetInt32());
            Assert.Equal(matrixCase.EfSearch, reportRoot.GetProperty("parameters").GetProperty("efSearch").GetInt32());
            Assert.Equal(matrixCase.EfSearch, reportRoot.GetProperty("parameters").GetProperty("hnswlibEf").GetInt32());
            Assert.Equal(2, reportRoot.GetProperty("parameters").GetProperty("measuredQueryCount").GetInt32());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
            Assert.True(reportRoot.GetProperty("validation").GetProperty("loadedExistingCache").GetBoolean());
            Assert.True(reportRoot.GetProperty("validation").GetProperty("loadedExistingTruth").GetBoolean());
            Assert.True(reportRoot.GetProperty("validation").GetProperty("identicalVectorsQueriesIdsAndParameters").GetBoolean());
            Assert.Equal("0.8.0", reportRoot.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("vecNet").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("hnswlib").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
            Assert.InRange(reportRoot.GetProperty("vecNet").GetProperty("metrics").GetProperty("recallAtK").GetDouble(), 0, 1);
            Assert.InRange(reportRoot.GetProperty("hnswlib").GetProperty("metrics").GetProperty("recallAtK").GetDouble(), 0, 1);
            Assert.Equal("notMeasured", reportRoot.GetProperty("hnswlib").GetProperty("memory").GetProperty("status").GetString());
            AssertNoTrueEligibilityFields(reportRoot);
        }
    }

    private static string? FindUsablePinnedHnswlibPython()
    {
        List<string> candidates =
        [
            HnswEstablishedComparisonOptions.Default.HnswlibPythonPath,
            Path.Combine("VecNet.BenchmarkRunner.Artifacts", "performance-agent-vec-119-tools", "python-3.11.9-embed-amd64", "python.exe")
        ];

        string artifactRoot = "VecNet.BenchmarkRunner.Artifacts";
        if (Directory.Exists(artifactRoot))
        {
            candidates.AddRange(Directory.EnumerateFiles(artifactRoot, "python.exe", SearchOption.AllDirectories).Take(20));
        }

        return candidates.Distinct(StringComparer.OrdinalIgnoreCase).FirstOrDefault(PinnedHnswlibIsUsable);
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 41)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 83)).ToArray());
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

    private static int[] ToIntArray(JsonElement array) =>
        array.EnumerateArray().Select(value => value.GetInt32()).ToArray();

    private static void AssertProfiles(JsonElement profiles)
    {
        Dictionary<string, JsonElement> byName = profiles
            .EnumerateArray()
            .ToDictionary(profile => profile.GetProperty("name").GetString()!, StringComparer.Ordinal);

        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], byName.Keys.Order(StringComparer.Ordinal).ToArray());
        AssertProfile(byName["balanced-m8"], m: 8, efConstruction: 64, efSearch: 128);
        AssertProfile(byName["wide-m16"], m: 16, efConstruction: 128, efSearch: 192);
        AssertProfile(byName["default-m16"], m: 16, efConstruction: 200, efSearch: 200);
    }

    private static void AssertProfile(JsonElement profile, int m, int efConstruction, int efSearch)
    {
        Assert.Equal(m, profile.GetProperty("m").GetInt32());
        Assert.Equal(efConstruction, profile.GetProperty("efConstruction").GetInt32());
        Assert.Equal(efSearch, profile.GetProperty("efSearch").GetInt32());
    }

    private static void AssertStatusCountsMatchAggregate(JsonElement root)
    {
        JsonElement cases = root.GetProperty("cases");
        int passed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "passed");
        int failed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "failed");
        int skipped = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "skipped");
        int blocked = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "blocked");
        JsonElement aggregate = root.GetProperty("aggregate");

        Assert.Equal(root.GetProperty("caseCount").GetInt32(), passed + failed + skipped + blocked);
        Assert.Equal(passed, aggregate.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(failed, aggregate.GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(skipped, aggregate.GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(blocked, aggregate.GetProperty("blockedCaseCount").GetInt32());
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
            string.Create(CultureInfo.InvariantCulture, $"vec121-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
