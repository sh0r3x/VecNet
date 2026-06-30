using System.Globalization;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec113HnswMemorySmokeIndependentTests
{
    [Theory]
    [InlineData("generated-hnsw-memory-smoke", "--dimension", "1", "--vectors", "1", "--queries", "1", "--top-k", "1", "--warmup-queries", "0", "--m", "2", "--ef-construction", "2", "--ef-search", "1", "--sample-interval-ms", "1")]
    [InlineData("GENERATED-HNSW-MEMORY-SMOKE", "--METRIC", "SQUAREDEUCLIDEAN", "--DIMENSION", "1", "--VECTORS", "4096", "--QUERIES", "1", "--TOP-K", "4096", "--WARMUP-QUERIES", "4096", "--SEED", "4294967295", "--M", "64", "--EF-CONSTRUCTION", "4096", "--EF-SEARCH", "4096", "--HNSW-SEED", "18446744073709551615", "--SAMPLE-INTERVAL-MS", "1000")]
    public void Parser_AcceptsBoundaryCombinationsAndCaseInsensitiveOptionNames(params string[] args)
    {
        HnswMemorySmokeOptions options = CommandLine.ParseHnswMemorySmoke(args);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.InRange(options.Dimension, 1, int.MaxValue);
        Assert.InRange(options.VectorCount, 1, int.MaxValue);
        Assert.InRange(options.QueryCount, 1, int.MaxValue);
        Assert.InRange(options.TopK, 1, options.VectorCount);
        Assert.InRange(options.WarmupQueries, 0, int.MaxValue);
        Assert.InRange(options.M, 2, 64);
        Assert.InRange(options.EfConstruction, options.M, 4096);
        Assert.InRange(options.EfSearch, options.TopK, 4096);
        Assert.InRange(options.SampleIntervalMilliseconds, 1, 1000);
    }

    [Theory]
    [InlineData("generated-hnsw-memory-smoke", "--dimension")]
    [InlineData("generated-hnsw-memory-smoke", "dimension", "8")]
    [InlineData("generated-hnsw-memory-smoke", "--snapshot-directory", "--output")]
    [InlineData("generated-hnsw-memory-smoke", "--comparison", "candidate.json")]
    [InlineData("generated-hnsw-memory-smoke", "--baseline", "baseline.json")]
    [InlineData("generated-hnsw-memory-smoke", "--current", "current.json")]
    [InlineData("generated-hnsw-memory-smoke", "--truth-depth", "100")]
    [InlineData("generated-hnsw-memory-smoke", "--query-count", "50")]
    [InlineData("generated-hnsw-memory-smoke", "--download", "false")]
    [InlineData("generated-hnsw-memory-smoke", "--filter", "all")]
    [InlineData("generated-hnsw-memory-smoke", "--allowlist", "all")]
    [InlineData("generated-hnsw-memory-smoke", "--candidate-set", "all")]
    public void Parser_RejectsMalformedDuplicateAndCrossScenarioOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ProgramRun_OccupiedSnapshotDirectoryFailsWithoutWritingReportOrDeletingExistingFiles()
    {
        string directory = NewArtifactDirectory("occupied-snapshot");
        string outputPath = Path.Combine(directory, "should-not-exist.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        Directory.CreateDirectory(snapshotDirectory);
        string markerPath = Path.Combine(snapshotDirectory, "existing.txt");
        File.WriteAllText(markerPath, "existing snapshot guard");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "generated-hnsw-memory-smoke",
                "--dimension", "8",
                "--vectors", "16",
                "--queries", "2",
                "--top-k", "2",
                "--warmup-queries", "0",
                "--m", "2",
                "--ef-construction", "4",
                "--ef-search", "2",
                "--sample-interval-ms", "1",
                "--output", outputPath,
                "--snapshot-directory", snapshotDirectory
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(outputPath));
        Assert.True(File.Exists(markerPath));
        Assert.Equal("existing snapshot guard", File.ReadAllText(markerPath));
        Assert.DoesNotContain(
            Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories),
            path => string.Equals(Path.GetFullPath(path), Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ReportJson_KeepsMemorySectionContractsUnsupportedPostureAndFalseEligibilityStable()
    {
        HnswMemorySmokeReport report = CreateSmallReport("json-contract", out string outputPath);
        string json = ReportWriter.Serialize(report);
        HnswMemorySmokeScenario.Write(report, outputPath);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswMemorySmokeReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("generated-hnsw-memory-smoke", root.GetProperty("scenarioName").GetString());
        Assert.Equal("VEC-113", root.GetProperty("taskId").GetString());

        AssertPropertyNames(
            root.GetProperty("actualMemory"),
            "status",
            "scope",
            "measurementMethod",
            "claimBoundary",
            "baselineProcess",
            "postDatasetGeneration",
            "postSourceBuildRetained",
            "postSourceWarmSearchRetained",
            "postSaveRetained",
            "postOpenReadOnlyRetained",
            "postOpenedWarmSearchRetained",
            "postValidationRetained",
            "unsupported",
            "limitations");
        AssertPropertyNames(
            root.GetProperty("peakMemory"),
            "status",
            "scope",
            "claimBoundary",
            "build",
            "save",
            "open",
            "sourceSearchWarmupPeakMemory",
            "openedSearchWarmupPeakMemory",
            "peakTemporaryDiskBytes",
            "limitations");
        AssertPropertyNames(
            root.GetProperty("layoutLowerBounds"),
            "status",
            "claimBoundary",
            "vectorPayloadLowerBoundBytes",
            "idPayloadLowerBoundBytes",
            "levelPayloadLowerBoundBytes",
            "graphCountPayloadLowerBoundBytes",
            "graphNeighborPayloadLowerBoundBytes",
            "graphPayloadLowerBoundBytes",
            "idMapEntryPayloadLowerBoundBytes",
            "searchWorkspacePayloadLowerBoundBytes",
            "buildScratchPayloadLowerBoundBytes",
            "sourceRetainedPayloadLowerBoundBytes",
            "openedRetainedPayloadLowerBoundBytes",
            "layers",
            "exclusions");
        AssertPropertyNames(
            root.GetProperty("storageSize"),
            "status",
            "boundary",
            "snapshotDirectoryPathPolicy",
            "snapshotDirectory",
            "fileCount",
            "totalBytes",
            "manifestBytes",
            "idsBytes",
            "vectorsBytes",
            "levelsBytes",
            "graphBytes",
            "bytesPerVector",
            "scanTimingScope",
            "peakObservedOutputDirectoryBytes",
            "peakTemporaryDiskBytes");

        AssertAllSamplesMeasuredAndWorkingSetContextOnly(root.GetProperty("actualMemory"));
        AssertPeakOperation(root.GetProperty("peakMemory").GetProperty("build"), "build", sampleIntervalMilliseconds: 1);
        AssertPeakOperation(root.GetProperty("peakMemory").GetProperty("save"), "save", sampleIntervalMilliseconds: 1);
        AssertPeakOperation(root.GetProperty("peakMemory").GetProperty("open"), "open", sampleIntervalMilliseconds: 1);
        AssertUnsupportedStatus(root.GetProperty("actualMemory").GetProperty("unsupported"));
        Assert.Equal("notMeasured", root.GetProperty("peakMemory").GetProperty("sourceSearchWarmupPeakMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("peakMemory").GetProperty("openedSearchWarmupPeakMemory").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("peakMemory").GetProperty("peakTemporaryDiskBytes").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("storageSize").GetProperty("peakObservedOutputDirectoryBytes").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("storageSize").GetProperty("peakTemporaryDiskBytes").GetProperty("status").GetString());

        JsonElement validation = root.GetProperty("validation");
        Assert.Equal("passed", validation.GetProperty("status").GetString());
        Assert.True(validation.GetProperty("sourceOpenedParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.Equal("passed", validation.GetProperty("sourceReturnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", validation.GetProperty("openedReturnedResultIntegrity").GetProperty("status").GetString());

        AssertFalseEligibility(root.GetProperty("evidence"));
        AssertFalseEligibility(validation);
        AssertFalseEligibility(root.GetProperty("eligibility"));
        Assert.DoesNotContain("\"publicClaimEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"previewReadinessEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonArtifactEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.False(root.TryGetProperty("baseline", out _));
        Assert.False(root.TryGetProperty("comparison", out _));
        Assert.False(root.TryGetProperty("regression", out _));

        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public void SourceOpenedParityComparison_ReportsWrittenIdOrderAndDistanceMismatches()
    {
        SearchResult[][] source =
        [
            [new SearchResult(1, 0.5f), new SearchResult(2, 1.5f)],
            [new SearchResult(3, 2.5f)],
            [new SearchResult(4, 4.5f)]
        ];
        SearchResult[][] opened =
        [
            [new SearchResult(1, 0.5f)],
            [new SearchResult(5, 2.5f)],
            [new SearchResult(4, 4.75f), new SearchResult(6, 9.0f)]
        ];

        DurableHnswParityInfo parity = InvokeMemorySmokeParity(source, opened);

        Assert.Equal(3, parity.QueryCount);
        Assert.Equal(2, parity.WrittenCountMismatchCount);
        Assert.Equal(1, parity.IdMismatchCount);
        Assert.Equal(2, parity.OrderMismatchCount);
        Assert.Equal(1, parity.DistanceMismatchCount);
        Assert.False(parity.AllResultsMatched);
        Assert.Contains("exact ID order", parity.Policy, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExistingHnswRunnerReports_DoNotGainMemorySmokeSchemaSections()
    {
        string directory = NewArtifactDirectory("schema-isolation");
        HnswBenchmarkReport generated = HnswGeneratedScenario.Run(
            new HnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 2,
                TopK: 3,
                Seed: 0x5EED1131,
                OutputPath: Path.Combine(directory, "hnsw-generated.json"),
                Runs: 1,
                WarmupQueries: 0,
                M: 2,
                EfConstruction: 4,
                EfSearch: 3,
                HnswSeed: 0x1131UL),
            ["hnsw-generated"]);
        DurableHnswBenchmarkReport durable = DurableHnswGeneratedScenario.Run(
            new DurableHnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 2,
                TopK: 3,
                Seed: 0x5EED1132,
                OutputPath: Path.Combine(directory, "hnsw-generated-durable.json"),
                SnapshotDirectory: Path.Combine(directory, "snapshot"),
                Runs: 1,
                WarmupQueries: 0,
                M: 2,
                EfConstruction: 4,
                EfSearch: 3,
                HnswSeed: 0x1132UL),
            ["hnsw-generated-durable"]);

        using JsonDocument generatedDocument = JsonDocument.Parse(ReportWriter.Serialize(generated));
        JsonElement generatedRoot = generatedDocument.RootElement;
        Assert.Equal("VecNet.HnswBenchmarkReport", generatedRoot.GetProperty("schemaName").GetString());
        Assert.True(generatedRoot.TryGetProperty("memoryEstimate", out _));
        Assert.False(generatedRoot.TryGetProperty("actualMemory", out _));
        Assert.False(generatedRoot.TryGetProperty("peakMemory", out _));
        Assert.False(generatedRoot.TryGetProperty("layoutLowerBounds", out _));
        Assert.False(generatedRoot.TryGetProperty("storageSize", out _));
        AssertNoTrueEligibilityFields(generatedRoot.GetProperty("evidence"));
        AssertNoTrueEligibilityFields(generatedRoot.GetProperty("validation"));

        using JsonDocument durableDocument = JsonDocument.Parse(ReportWriter.Serialize(durable));
        JsonElement durableRoot = durableDocument.RootElement;
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", durableRoot.GetProperty("schemaName").GetString());
        Assert.True(durableRoot.TryGetProperty("memoryEstimates", out _));
        Assert.True(durableRoot.TryGetProperty("outputs", out _));
        Assert.False(durableRoot.TryGetProperty("actualMemory", out _));
        Assert.False(durableRoot.TryGetProperty("peakMemory", out _));
        Assert.False(durableRoot.TryGetProperty("layoutLowerBounds", out _));
        Assert.False(durableRoot.TryGetProperty("storageSize", out _));
        AssertNoTrueEligibilityFields(durableRoot.GetProperty("evidence"));
        AssertNoTrueEligibilityFields(durableRoot.GetProperty("validation"));
        AssertNoTrueEligibilityFields(durableRoot.GetProperty("eligibility"));
    }

    private static HnswMemorySmokeReport CreateSmallReport(string prefix, out string outputPath)
    {
        string directory = NewArtifactDirectory(prefix);
        outputPath = Path.Combine(directory, "generated-hnsw-memory-smoke.json");
        string snapshotDirectory = Path.Combine(directory, "snapshot");
        string[] args =
        [
            "generated-hnsw-memory-smoke",
            "--dimension", "8",
            "--vectors", "24",
            "--queries", "2",
            "--top-k", "3",
            "--warmup-queries", "1",
            "--seed", "0x5EED1133",
            "--m", "2",
            "--ef-construction", "4",
            "--ef-search", "3",
            "--hnsw-seed", "0x0000000000001133",
            "--sample-interval-ms", "1",
            "--output", outputPath,
            "--snapshot-directory", snapshotDirectory
        ];

        HnswMemorySmokeOptions options = CommandLine.ParseHnswMemorySmoke(args);
        return HnswMemorySmokeScenario.Run(options, args);
    }

    private static DurableHnswParityInfo InvokeMemorySmokeParity(SearchResult[][] source, SearchResult[][] opened)
    {
        MethodInfo? method = typeof(HnswMemorySmokeScenario).GetMethod(
            "CompareSourceOpenedParity",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        object? result = method.Invoke(null, [source, opened]);
        DurableHnswParityInfo parity = Assert.IsType<DurableHnswParityInfo>(result);
        return parity;
    }

    private static void AssertAllSamplesMeasuredAndWorkingSetContextOnly(JsonElement actualMemory)
    {
        foreach (string sampleName in new[]
        {
            "baselineProcess",
            "postDatasetGeneration",
            "postSourceBuildRetained",
            "postSourceWarmSearchRetained",
            "postSaveRetained",
            "postOpenReadOnlyRetained",
            "postOpenedWarmSearchRetained",
            "postValidationRetained"
        })
        {
            JsonElement sample = actualMemory.GetProperty(sampleName);
            Assert.Equal("measured", sample.GetProperty("managedHeapSizeBytes").GetProperty("status").GetString());
            Assert.Equal("measured", sample.GetProperty("gcCommittedBytes").GetProperty("status").GetString());
            Assert.Equal("measured", sample.GetProperty("gcFragmentedBytes").GetProperty("status").GetString());
            Assert.Equal("measured", sample.GetProperty("processPrivateBytes").GetProperty("status").GetString());
            Assert.Equal("measured", sample.GetProperty("processWorkingSetBytes").GetProperty("status").GetString());
            Assert.Equal("measured", sample.GetProperty("processPeakWorkingSetBytes").GetProperty("status").GetString());
            Assert.False(sample.GetProperty("processPrivateBytes").GetProperty("contextOnly").GetBoolean());
            Assert.True(sample.GetProperty("processWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
            Assert.True(sample.GetProperty("processPeakWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        }
    }

    private static void AssertPeakOperation(JsonElement operation, string name, int sampleIntervalMilliseconds)
    {
        Assert.Equal(name, operation.GetProperty("name").GetString());
        Assert.Equal("sampled", operation.GetProperty("status").GetString());
        Assert.Equal(sampleIntervalMilliseconds, operation.GetProperty("sampleIntervalMilliseconds").GetInt32());
        Assert.True(operation.GetProperty("sampleCount").GetInt32() >= 2);
        Assert.Equal("sampled", operation.GetProperty("peakObservedManagedHeapSizeBytes").GetProperty("status").GetString());
        Assert.Equal("sampled", operation.GetProperty("peakObservedGcCommittedBytes").GetProperty("status").GetString());
        Assert.Equal("sampled", operation.GetProperty("peakObservedPrivateBytes").GetProperty("status").GetString());
        Assert.False(operation.GetProperty("peakObservedPrivateBytes").GetProperty("contextOnly").GetBoolean());
        Assert.Equal("sampled", operation.GetProperty("peakObservedWorkingSetBytes").GetProperty("status").GetString());
        Assert.True(operation.GetProperty("peakObservedWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        Assert.Contains("observed sampled peak", operation.GetProperty("missedShortPeakCaveat").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("whole-process", operation.GetProperty("wholeProcessCaveat").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(operation.GetProperty("timedScope").GetString()!);
        Assert.NotEmpty(operation.GetProperty("excludedOperations").GetString()!);
    }

    private static void AssertUnsupportedStatus(JsonElement unsupported)
    {
        foreach (string propertyName in new[]
        {
            "objectAccurateIdMapRetainedMemory",
            "objectAccurateGraphLayerObjectMemory",
            "objectHeadersArrayHeadersAlignmentAndSlack",
            "neighborCandidateRetainedLayout"
        })
        {
            JsonElement field = unsupported.GetProperty(propertyName);
            Assert.Equal("notAvailable", field.GetProperty("status").GetString());
            Assert.Equal("absent", field.GetProperty("value").GetString());
            Assert.NotEmpty(field.GetProperty("reason").GetString()!);
        }

        foreach (string propertyName in new[]
        {
            "indexOnlyPrivateBytes",
            "openedOnlyRetainedMemory",
            "saveManagedAllocations",
            "openManagedAllocations",
            "trueProcessPeakMemory",
            "peakTemporaryDisk"
        })
        {
            JsonElement field = unsupported.GetProperty(propertyName);
            Assert.Equal("notMeasured", field.GetProperty("status").GetString());
            Assert.Equal("absent", field.GetProperty("value").GetString());
            Assert.NotEmpty(field.GetProperty("reason").GetString()!);
        }
    }

    private static void AssertFalseEligibility(JsonElement section)
    {
        Assert.False(section.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(section.GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(section.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(section.GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(section.GetProperty("regressionGateEligible").GetBoolean());
    }

    private static void AssertNoTrueEligibilityFields(JsonElement section)
    {
        foreach (string propertyName in new[]
        {
            "publicClaimEligible",
            "previewReadinessEligible",
            "baselineCandidateEligible",
            "comparisonArtifactEligible",
            "regressionGateEligible"
        })
        {
            if (section.TryGetProperty(propertyName, out JsonElement value))
            {
                Assert.False(value.GetBoolean());
            }
        }
    }

    private static void AssertPropertyNames(JsonElement element, params string[] expectedNames)
    {
        string[] actualNames = element.EnumerateObject().Select(property => property.Name).ToArray();
        Assert.Equal(expectedNames, actualNames);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec113-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
