using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec118HnswEstablishedComparisonIndependentTests
{
    [Fact]
    public void Defaults_PinHnswlibAndKeepRepresentativeDimensionsSeparateFromTailDimension()
    {
        HnswEstablishedComparisonOptions options = CommandLine.ParseHnswEstablishedComparison([]);

        Assert.Equal("hnswlib-generated-comparison", HnswEstablishedComparisonOptions.ScenarioName);
        Assert.Equal("hnswlib", HnswEstablishedComparisonOptions.HnswlibPackageName);
        Assert.Equal("PyPI", HnswEstablishedComparisonOptions.HnswlibPackageSource);
        Assert.Equal("0.8.0", HnswEstablishedComparisonOptions.HnswlibVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256);
        Assert.Equal("Apache-2.0", HnswEstablishedComparisonOptions.HnswlibLicense);
        Assert.Equal([128, 384, 768], HnswEstablishedComparisonOptions.RepresentativeDimensions);
        Assert.DoesNotContain(386, HnswEstablishedComparisonOptions.RepresentativeDimensions);
        Assert.Equal([386], HnswEstablishedComparisonOptions.OptionalAdversarialDimensions);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.EndsWith(Path.Combine("vec-118-tools", "hnswlib-venv", "Scripts", "python.exe"), options.HnswlibPythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.WorkDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.VecNetSnapshotDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.HnswlibIndexPath, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("HNSWLIB-GENERATED-COMPARISON", "--METRIC", "SQUAREDEUCLIDEAN", "--DIMENSION", "384", "--VECTORS", "4096", "--QUERIES", "100", "--TOP-K", "100", "--RUNS", "5", "--WARMUP-QUERIES", "0", "--SEED", "4294967295", "--M", "64", "--EF-CONSTRUCTION", "4096", "--EF-SEARCH", "4096", "--HNSW-SEED", "18446744073709551615")]
    [InlineData("hnswlib-generated-comparison", "--dimension", "386", "--vectors", "100", "--queries", "1", "--top-k", "100", "--runs", "1", "--warmup-queries", "100", "--m", "2", "--ef-construction", "2", "--ef-search", "100", "--hnsw-seed", "0xFFFFFFFFFFFFFFFF")]
    public void Parser_AcceptsRepresentativeAndOptionalTailBoundaryCommands(params string[] args)
    {
        HnswEstablishedComparisonOptions options = CommandLine.ParseHnswEstablishedComparison(args);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.InRange(options.Dimension, 1, int.MaxValue);
        Assert.InRange(options.VectorCount, 1, int.MaxValue);
        Assert.InRange(options.QueryCount, 1, int.MaxValue);
        Assert.InRange(options.TopK, 1, options.VectorCount);
        Assert.InRange(options.Runs, 1, 5);
        Assert.InRange(options.WarmupQueries, 0, int.MaxValue);
        Assert.InRange(options.M, 2, 64);
        Assert.InRange(options.EfConstruction, options.M, 4096);
        Assert.InRange(options.EfSearch, options.TopK, 4096);
    }

    [Theory]
    [InlineData("hnsw-generated", "--hnswlib-python", "python.exe")]
    [InlineData("hnsw-generated", "--hnswlib-index", "index.bin")]
    [InlineData("hnsw-generated", "--work-directory", "work")]
    [InlineData("generated-hnsw-memory-smoke", "--hnswlib-python", "python.exe")]
    [InlineData("generated-hnsw-memory-smoke", "--hnswlib-index", "index.bin")]
    [InlineData("hnswlib-generated-comparison", "--snapshot-directory", "snapshot")]
    [InlineData("hnswlib-generated-comparison", "--sample-interval-ms", "1")]
    [InlineData("hnswlib-generated-comparison", "--output-dir", "matrix")]
    [InlineData("hnswlib-generated-comparison", "--manifest", "manifest.json")]
    [InlineData("hnswlib-generated-comparison", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnswlib-generated-comparison", "--download", "false")]
    [InlineData("hnswlib-generated-comparison", "--query-count", "50")]
    [InlineData("hnswlib-generated-comparison", "--truth-depth", "100")]
    public void Parsers_RejectCrossScenarioOptionsThatWouldCollideWithExistingHnswModes(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "hnsw-generated" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(args)),
            "generated-hnsw-memory-smoke" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(args)),
            _ => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args))
        };

        Assert.Contains("option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ProgramRun_WithUnavailableHnswlibPythonFailsWithoutReportOrGeneratedWork()
    {
        string directory = NewArtifactDirectory("missing-tool-program");
        string outputPath = Path.Combine(directory, "missing-tool.json");
        string workDirectory = Path.Combine(directory, "work");
        string snapshotDirectory = Path.Combine(directory, "vecnet-snapshot");
        string hnswlibIndexPath = Path.Combine(directory, "hnswlib-index.bin");
        string missingPythonPath = Path.Combine(directory, "tools", "python.exe");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnswlib-generated-comparison",
                "--dimension", "8",
                "--vectors", "24",
                "--queries", "2",
                "--top-k", "3",
                "--runs", "1",
                "--warmup-queries", "0",
                "--m", "2",
                "--ef-construction", "4",
                "--ef-search", "3",
                "--hnswlib-python", missingPythonPath,
                "--output", outputPath,
                "--work-directory", workDirectory,
                "--vecnet-snapshot-directory", snapshotDirectory,
                "--hnswlib-index", hnswlibIndexPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(outputPath));
        Assert.False(File.Exists(hnswlibIndexPath));
        Assert.False(Directory.Exists(workDirectory));
        Assert.False(Directory.Exists(snapshotDirectory));
        Assert.DoesNotContain(
            Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories),
            path => string.Equals(Path.GetFullPath(path), Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ReportJson_WhenPinnedVenvExistsKeepsNativeManagedBoundaryFairnessAndNoPublicationLeakage()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!File.Exists(pythonPath))
        {
            return;
        }

        string directory = NewArtifactDirectory("schema");
        string outputPath = Path.Combine(directory, "hnswlib-generated-comparison.json");
        string[] args =
        [
            "hnswlib-generated-comparison",
            "--dimension", "386",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED2186",
            "--m", "2",
            "--ef-construction", "8",
            "--ef-search", "6",
            "--hnsw-seed", "0x0000000000002186",
            "--hnswlib-python", pythonPath,
            "--output", outputPath,
            "--work-directory", Path.Combine(directory, "work"),
            "--vecnet-snapshot-directory", Path.Combine(directory, "vecnet-snapshot"),
            "--hnswlib-index", Path.Combine(directory, "hnswlib-index.bin")
        ];
        HnswEstablishedComparisonOptions options = CommandLine.ParseHnswEstablishedComparison(args);

        HnswEstablishedComparisonReport report = HnswEstablishedComparisonScenario.Run(options, args);
        HnswEstablishedComparisonScenario.Write(report, outputPath);

        string json = File.ReadAllText(outputPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.HnswEstablishedComparisonReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-118", root.GetProperty("taskId").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("hnswlib", root.GetProperty("sourcePinning").GetProperty("packageName").GetString());
        Assert.Equal("0.8.0", root.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", root.GetProperty("sourcePinning").GetProperty("sourceDistributionSha256").GetString());
        Assert.Equal("Apache-2.0", root.GetProperty("sourcePinning").GetProperty("license").GetString());
        Assert.Contains("non-shipping", root.GetProperty("sourcePinning").GetProperty("licensePosture").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Python/native", root.GetProperty("sourcePinning").GetProperty("nativeBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("No hnswlib", root.GetProperty("sourcePinning").GetProperty("productDependencyPosture").GetString(), StringComparison.Ordinal);

        Assert.Equal(new[] { 128, 384, 768 }, root.GetProperty("design").GetProperty("representativeGeneratedDimensions").EnumerateArray().Select(value => value.GetInt32()).ToArray());
        Assert.Equal(new[] { 386 }, root.GetProperty("design").GetProperty("optionalAdversarialTailDimensions").EnumerateArray().Select(value => value.GetInt32()).ToArray());
        Assert.Equal(386, root.GetProperty("design").GetProperty("currentDimension").GetInt32());
        Assert.Equal("optional-adversarial-tail", root.GetProperty("design").GetProperty("currentDimensionRole").GetString());
        Assert.Contains("must not replace", root.GetProperty("design").GetProperty("tailDimensionPolicy").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Equal("generated-no-external-source", root.GetProperty("dataset").GetProperty("sourceVerificationStatus").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("identicalVectorsQueriesIdsAndParameters").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("vecNetComparedToTruth").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("hnswlibComparedToTruth").GetBoolean());
        Assert.Contains("same binary inputs", root.GetProperty("methodology").GetProperty("identicalInputsPolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("num_threads=1", root.GetProperty("methodology").GetProperty("threadingPolicy").GetString(), StringComparison.Ordinal);
        Assert.Contains("out-of-process", root.GetProperty("methodology").GetProperty("pythonBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        AssertImplementationSection(root.GetProperty("vecNet"), "VecNet", expectedVersion: null, expectedMemoryStatus: "notMeasured", expectedSearchAllocationStatus: "measured");
        AssertImplementationSection(root.GetProperty("hnswlib"), "hnswlib", "0.8.0", expectedMemoryStatus: "notMeasured", expectedSearchAllocationStatus: "notMeasured");
        Assert.Equal("fileFacts", root.GetProperty("vecNet").GetProperty("persistedBytes").GetProperty("status").GetString());
        Assert.Equal("fileFacts", root.GetProperty("hnswlib").GetProperty("persistedBytes").GetProperty("status").GetString());
        Assert.True(long.Parse(root.GetProperty("vecNet").GetProperty("persistedBytes").GetProperty("value").GetString()!, CultureInfo.InvariantCulture) > 0);
        Assert.True(long.Parse(root.GetProperty("hnswlib").GetProperty("persistedBytes").GetProperty("value").GetString()!, CultureInfo.InvariantCulture) > 0);

        AssertNoTrueEligibilityFields(root);
        AssertNoPropertyNamed(
            root,
            "previewReadinessEligible",
            "comparisonArtifactEligible",
            "baseline",
            "baselineReportId",
            "candidateEligibility",
            "regressionDecision",
            "regressionThreshold",
            "publicClaimStatus",
            "packageMetadata",
            "packageProjectUrl",
            "readme",
            "licenseFile",
            "nugetPublication");
        Assert.DoesNotContain("\"publicClaimEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("README.md", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NuGet", json, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertImplementationSection(
        JsonElement implementation,
        string expectedName,
        string? expectedVersion,
        string expectedMemoryStatus,
        string expectedSearchAllocationStatus)
    {
        Assert.Equal(expectedName, implementation.GetProperty("name").GetString());
        if (expectedVersion is not null)
        {
            Assert.Equal(expectedVersion, implementation.GetProperty("version").GetString());
        }

        JsonElement search = implementation.GetProperty("search");
        JsonElement metrics = implementation.GetProperty("metrics");
        Assert.Equal("measured", implementation.GetProperty("build").GetProperty("status").GetString());
        Assert.True(implementation.GetProperty("build").GetProperty("elapsedMilliseconds").GetDouble() >= 0);
        Assert.Equal("measured", search.GetProperty("status").GetString());
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.True(search.GetProperty("latencyP50Milliseconds").GetDouble() >= 0);
        Assert.True(search.GetProperty("latencyP95Milliseconds").GetDouble() >= search.GetProperty("latencyP50Milliseconds").GetDouble());
        Assert.True(search.GetProperty("latencyP99Milliseconds").GetDouble() >= search.GetProperty("latencyP95Milliseconds").GetDouble());
        Assert.True(search.GetProperty("qps").GetDouble() > 0);
        Assert.Equal(expectedSearchAllocationStatus, search.GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal(expectedMemoryStatus, implementation.GetProperty("memory").GetProperty("status").GetString());
        Assert.InRange(metrics.GetProperty("recallAtK").GetDouble(), 0, 1);
        Assert.InRange(metrics.GetProperty("orderedAgreement").GetDouble(), 0, 1);
        Assert.Equal("passed", metrics.GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("duplicateIdCount").GetInt32());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("unknownIdCount").GetInt32());
        Assert.Equal(0, metrics.GetProperty("returnedResultIntegrity").GetProperty("nonFiniteDistanceCount").GetInt32());
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
            string.Create(CultureInfo.InvariantCulture, $"vec118-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
