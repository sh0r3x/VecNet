using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec119HnswEstablishedComparisonMatrixIndependentTests
{
    [Fact]
    public void StandardPresetExpansion_IsCartesianRepresentativeMatrixWithTopKOneHundredAndEfSearchSafety()
    {
        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(
            [
                "HNSWLIB-GENERATED-COMPARISON-MATRIX",
                "--PRESET", "STANDARD",
                "--VECTORS", "100",
                "--QUERIES", "2",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "3",
                "--SEED", "0xFFFFFFFF",
                "--OUTPUT-DIR", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec119-independent-standard"),
                "--MANIFEST", Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec119-independent-standard", "manifest.json"),
                "--HNSWLIB-PYTHON", HnswEstablishedComparisonOptions.Default.HnswlibPythonPath
            ]);

        HnswEstablishedComparisonMatrixScenario.HnswComparisonMatrixCase[] cases =
            HnswEstablishedComparisonMatrixScenario.ExpandCases(options);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(18, cases.Length);
        Assert.Equal([128, 384, 768], cases.Select(c => c.Options.Dimension).Distinct().ToArray());
        Assert.Equal([10, 100], cases.Select(c => c.Options.TopK).Distinct().ToArray());
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(c => c.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.DoesNotContain(386, cases.Select(c => c.Options.Dimension));
        Assert.Equal(18, cases.Select(c => c.CaseId).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(18, cases.Select(c => c.Options.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal(18, cases.Select(c => c.Options.Seed).Distinct().Count());
        Assert.Equal(18, cases.Select(c => c.Options.HnswSeed).Distinct().Count());

        foreach (HnswEstablishedComparisonMatrixScenario.HnswComparisonMatrixCase matrixCase in cases)
        {
            Assert.Equal(VectorMetric.SquaredEuclidean, matrixCase.Options.Metric);
            Assert.Equal(100, matrixCase.Options.VectorCount);
            Assert.Equal(2, matrixCase.Options.QueryCount);
            Assert.Equal(5, matrixCase.Options.Runs);
            Assert.Equal(3, matrixCase.Options.WarmupQueries);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.WorkDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.VecNetSnapshotDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(options.OutputDirectory, matrixCase.Options.HnswlibIndexPath, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains(cases, c => c.ProfileName == "balanced-m8" && c.Options.M == 8 && c.Options.EfConstruction == 64 && c.Options.EfSearch == 128);
        Assert.Contains(cases, c => c.ProfileName == "wide-m16" && c.Options.M == 16 && c.Options.EfConstruction == 128 && c.Options.EfSearch == 192);
        Assert.Contains(cases, c => c.ProfileName == "default-m16" && c.Options.M == 16 && c.Options.EfConstruction == 200 && c.Options.EfSearch == 200);
    }

    [Theory]
    [InlineData("hnswlib-generated-comparison-matrix", "--dimension", "386")]
    [InlineData("hnswlib-generated-comparison-matrix", "--top-k", "100")]
    [InlineData("hnswlib-generated-comparison-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("hnswlib-generated-comparison-matrix", "--m", "8")]
    [InlineData("hnswlib-generated-comparison-matrix", "--ef-construction", "64")]
    [InlineData("hnswlib-generated-comparison-matrix", "--ef-search", "128")]
    [InlineData("hnswlib-generated-comparison-matrix", "--hnsw-seed", "0x484E535700011900")]
    [InlineData("hnswlib-generated-comparison-matrix", "--work-directory", "work")]
    [InlineData("hnswlib-generated-comparison-matrix", "--vecnet-snapshot-directory", "snapshot")]
    [InlineData("hnswlib-generated-comparison-matrix", "--hnswlib-index", "index.bin")]
    [InlineData("hnswlib-generated-comparison-matrix", "--snapshot-directory", "snapshot")]
    [InlineData("hnswlib-generated-comparison-matrix", "--sample-interval-ms", "1")]
    [InlineData("hnswlib-generated-comparison", "--preset", "smoke")]
    [InlineData("hnswlib-generated-comparison", "--output-dir", "matrix")]
    [InlineData("hnswlib-generated-comparison", "--manifest", "matrix.json")]
    [InlineData("hnsw-generated", "--hnswlib-python", "python.exe")]
    [InlineData("hnsw-generated-matrix", "--hnswlib-python", "python.exe")]
    [InlineData("generated-hnsw-memory-smoke", "--hnswlib-python", "python.exe")]
    public void Parsers_KeepMatrixSingleCaseAndExistingHnswModesIsolated(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "hnswlib-generated-comparison-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparisonMatrix(args)),
            "hnswlib-generated-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args)),
            "hnsw-generated" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(args)),
            "hnsw-generated-matrix" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(args)),
            "generated-hnsw-memory-smoke" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(args)),
            _ => throw new InvalidOperationException("Unexpected parser fixture.")
        };

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void BlockedManifestJson_PreservesSchemaPinDesignCountsAndNoFakeLinkedReports()
    {
        string directory = NewArtifactDirectory("blocked-json");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string missingPython = Path.Combine(directory, "missing", "python.exe");
        string[] args =
        [
            "hnswlib-generated-comparison-matrix",
            "--preset", "standard",
            "--vectors", "100",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED119B",
            "--hnswlib-python", missingPython,
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(args);
        HnswEstablishedComparisonMatrixManifest manifest = HnswEstablishedComparisonMatrixScenario.Run(options, args);
        HnswEstablishedComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        string json = File.ReadAllText(manifestPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.HnswEstablishedComparisonMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-119", root.GetProperty("taskId").GetString());
        Assert.Equal("hnswlib-generated-comparison-matrix", root.GetProperty("scenarioName").GetString());
        Assert.Equal("standard", root.GetProperty("presetName").GetString());
        Assert.Equal(18, root.GetProperty("caseCount").GetInt32());

        JsonElement sourcePinning = root.GetProperty("sourcePinning");
        Assert.Equal("hnswlib", sourcePinning.GetProperty("packageName").GetString());
        Assert.Equal("PyPI", sourcePinning.GetProperty("packageSource").GetString());
        Assert.Equal("0.8.0", sourcePinning.GetProperty("packageVersion").GetString());
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", sourcePinning.GetProperty("sourceDistributionSha256").GetString());
        Assert.Equal("Apache-2.0", sourcePinning.GetProperty("license").GetString());
        Assert.Contains("non-shipping", sourcePinning.GetProperty("licensePosture").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not distributed", sourcePinning.GetProperty("licensePosture").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Python/native", sourcePinning.GetProperty("nativeBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("No hnswlib", sourcePinning.GetProperty("productDependencyPosture").GetString(), StringComparison.Ordinal);

        JsonElement design = root.GetProperty("design");
        Assert.Equal([128, 384, 768], ToIntArray(design.GetProperty("representativeGeneratedDimensions")));
        Assert.Equal([386], ToIntArray(design.GetProperty("optionalAdversarialTailDimensions")));
        Assert.Equal([10, 100], ToIntArray(design.GetProperty("topKValues")));
        string[] profileNames = design
            .GetProperty("profiles")
            .EnumerateArray()
            .Select(profile => profile.GetProperty("name").GetString() ?? string.Empty)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], profileNames);
        Assert.Contains("must not replace", design.GetProperty("tailDimensionPolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("standard adds top-k 100", design.GetProperty("presetPolicy").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement aggregate = root.GetProperty("aggregate");
        Assert.Equal(0, aggregate.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, aggregate.GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, aggregate.GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(18, aggregate.GetProperty("blockedCaseCount").GetInt32());
        AssertStatusCountsMatchAggregate(root);

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal("blocked", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.False(File.Exists(matrixCase.GetProperty("linkedReportPath").GetString()!));
            Assert.Contains("unavailable", matrixCase.GetProperty("errorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Equal("representative", matrixCase.GetProperty("dimensionRole").GetString());
            Assert.NotEqual(386, matrixCase.GetProperty("dimension").GetInt32());
            Assert.True(matrixCase.GetProperty("efSearch").GetInt32() >= matrixCase.GetProperty("topK").GetInt32());
        }

        AssertNoTrueEligibilityFields(root);
        AssertNoPropertyNamed(
            root,
            "previewReadinessEligible",
            "comparisonArtifactEligible",
            "packageMetadata",
            "packageProjectUrl",
            "readme",
            "licenseFile",
            "nugetPublication",
            "packagePublication",
            "publicClaimStatus",
            "regressionDecision",
            "regressionThreshold");
        Assert.DoesNotContain("\"taskId\": \"VEC-118\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"status\": \"passed\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"linkedReportId\": \"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("README.md", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("PackageReference", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NuGet", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LinkedReport_WhenPinnedVenvExists_RemainsVec118CompatibleAndFalseEligible()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!File.Exists(pythonPath))
        {
            return;
        }

        string directory = NewArtifactDirectory("linked-report");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "hnswlib-generated-comparison-matrix",
            "--preset", "smoke",
            "--vectors", "16",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED119C",
            "--hnswlib-python", pythonPath,
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(args);
        HnswEstablishedComparisonMatrixManifest manifest = HnswEstablishedComparisonMatrixScenario.Run(options, args);
        HnswEstablishedComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        string manifestJson = File.ReadAllText(manifestPath);
        using JsonDocument manifestDocument = JsonDocument.Parse(manifestJson);
        JsonElement manifestRoot = manifestDocument.RootElement;
        AssertStatusCountsMatchAggregate(manifestRoot);
        Assert.Equal(9, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        AssertNoTrueEligibilityFields(manifestRoot);

        HnswEstablishedComparisonMatrixCaseManifest selectedCase = manifest.Cases.Single(
            matrixCase => matrixCase.Dimension == 384 && matrixCase.ProfileName == "default-m16" && matrixCase.TopK == 10);
        Assert.True(File.Exists(selectedCase.LinkedReportPath));
        Assert.NotNull(selectedCase.LinkedReportId);

        string reportJson = File.ReadAllText(selectedCase.LinkedReportPath);
        using JsonDocument reportDocument = JsonDocument.Parse(reportJson);
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.HnswEstablishedComparisonReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-118", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal(selectedCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
        Assert.Equal("hnswlib-generated-comparison", reportRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(reportRoot.GetProperty("validation").GetProperty("identicalVectorsQueriesIdsAndParameters").GetBoolean());
        Assert.Equal(384, reportRoot.GetProperty("parameters").GetProperty("dimension").GetInt32());
        Assert.Equal(10, reportRoot.GetProperty("parameters").GetProperty("topK").GetInt32());
        Assert.Equal(16, reportRoot.GetProperty("parameters").GetProperty("vectorCount").GetInt32());
        Assert.Equal(1, reportRoot.GetProperty("parameters").GetProperty("queryCount").GetInt32());
        Assert.Equal(16, reportRoot.GetProperty("parameters").GetProperty("m").GetInt32());
        Assert.Equal(200, reportRoot.GetProperty("parameters").GetProperty("efConstruction").GetInt32());
        Assert.Equal(200, reportRoot.GetProperty("parameters").GetProperty("efSearch").GetInt32());
        Assert.Equal("hnswlib", reportRoot.GetProperty("sourcePinning").GetProperty("packageName").GetString());
        Assert.Equal("0.8.0", reportRoot.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("vecNet").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("hnswlib").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        AssertNoTrueEligibilityFields(reportRoot);
        Assert.DoesNotContain("\"publicClaimEligible\": true", reportJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", reportJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", reportJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", reportJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("README.md", reportJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NuGet", reportJson, StringComparison.OrdinalIgnoreCase);
    }

    private static int[] ToIntArray(JsonElement array) =>
        array.EnumerateArray().Select(value => value.GetInt32()).ToArray();

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
            string.Create(CultureInfo.InvariantCulture, $"vec119-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
