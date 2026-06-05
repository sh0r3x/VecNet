using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec016IndependentTests
{
    [Theory]
    [InlineData(new long[] { 7 }, 0.00, 7)]
    [InlineData(new long[] { 7 }, 0.50, 7)]
    [InlineData(new long[] { 7 }, 0.99, 7)]
    [InlineData(new long[] { 10, 20 }, 0.50, 10)]
    [InlineData(new long[] { 10, 20 }, 0.51, 20)]
    [InlineData(new long[] { 10, 20 }, 1.00, 20)]
    [InlineData(new long[] { 1, 2, 3 }, 0.34, 2)]
    [InlineData(new long[] { 1, 2, 3 }, 0.67, 3)]
    [InlineData(new long[] { 1, 2, 3, 4 }, 0.25, 1)]
    [InlineData(new long[] { 1, 2, 3, 4 }, 0.50, 2)]
    [InlineData(new long[] { 1, 2, 3, 4 }, 0.75, 3)]
    [InlineData(new long[] { 1, 2, 3, 4 }, 1.00, 4)]
    public void NearestRankMilliseconds_CoversEdgeCountsAndPercentileBoundaries(
        long[] sortedTicks,
        double percentile,
        double expectedMilliseconds)
    {
        double actualMilliseconds = LatencyPercentiles.NearestRankMilliseconds(
            sortedTicks,
            percentile,
            ticksPerSecond: 1000);

        Assert.Equal(expectedMilliseconds, actualMilliseconds);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NearestRankMilliseconds_RejectsInvalidTickFrequency(long ticksPerSecond)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LatencyPercentiles.NearestRankMilliseconds([1], 0.50, ticksPerSecond));
    }

    [Fact]
    public void Run_SerializedJsonKeepsLatencyFieldsAtEverySearchLevelAndAggregateMeans()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.InnerProduct,
            Dimension: 9,
            VectorCount: 27,
            QueryCount: 4,
            TopK: 6,
            Seed: 0x5EED0162,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-016-search-fields.json",
            BaselineReportId: "metadata-only-baseline",
            Runs: 3,
            WarmupQueries: 7);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--metric", "InnerProduct",
                "--queries", "4",
                "--runs", "3",
                "--warmup-queries", "7",
                "--baseline-report-id", "metadata-only-baseline"
            ]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        JsonElement search = root.GetProperty("search");
        JsonElement runs = search.GetProperty("runs");
        JsonElement aggregate = search.GetProperty("aggregate");

        Assert.Equal(4, root.GetProperty("scenario").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(4, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(4, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal(3, runs.GetArrayLength());
        Assert.Equal(3, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(7, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.DoesNotContain("\"measuredQueryCount\":11", ReportWriter.Serialize(report), StringComparison.Ordinal);
        Assert.DoesNotContain("\"measuredQueryCountPerRun\":11", ReportWriter.Serialize(report), StringComparison.Ordinal);

        AssertPositiveLatencyFields(search);
        Assert.True(aggregate.TryGetProperty("meanLatencyP50Milliseconds", out _));
        Assert.True(aggregate.TryGetProperty("meanLatencyP95Milliseconds", out _));
        Assert.True(aggregate.TryGetProperty("meanLatencyP99Milliseconds", out _));

        double[] p50 = new double[runs.GetArrayLength()];
        double[] p95 = new double[runs.GetArrayLength()];
        double[] p99 = new double[runs.GetArrayLength()];
        for (int index = 0; index < runs.GetArrayLength(); index++)
        {
            JsonElement run = runs[index];
            Assert.Equal(index + 1, run.GetProperty("runNumber").GetInt32());
            Assert.Equal(4, run.GetProperty("measuredQueryCount").GetInt32());
            AssertPositiveLatencyFields(run);
            p50[index] = run.GetProperty("latencyP50Milliseconds").GetDouble();
            p95[index] = run.GetProperty("latencyP95Milliseconds").GetDouble();
            p99[index] = run.GetProperty("latencyP99Milliseconds").GetDouble();
            Assert.True(p95[index] >= p50[index]);
            Assert.True(p99[index] >= p95[index]);
        }

        Assert.Equal(p50.Average(), aggregate.GetProperty("meanLatencyP50Milliseconds").GetDouble(), precision: 12);
        Assert.Equal(p95.Average(), aggregate.GetProperty("meanLatencyP95Milliseconds").GetDouble(), precision: 12);
        Assert.Equal(p99.Average(), aggregate.GetProperty("meanLatencyP99Milliseconds").GetDouble(), precision: 12);
        Assert.Equal(aggregate.GetProperty("meanLatencyP50Milliseconds").GetDouble(), search.GetProperty("latencyP50Milliseconds").GetDouble(), precision: 12);
        Assert.Equal(aggregate.GetProperty("meanLatencyP95Milliseconds").GetDouble(), search.GetProperty("latencyP95Milliseconds").GetDouble(), precision: 12);
        Assert.Equal(aggregate.GetProperty("meanLatencyP99Milliseconds").GetDouble(), search.GetProperty("latencyP99Milliseconds").GetDouble(), precision: 12);

        AssertManagedAllocationMetadataStillPresent(root);
        AssertFalseEligibility(root);
        AssertNoRawLatencyOrComparisonFields(root);
    }

    [Fact]
    public void Run_LatencyMetadataStatesMeasuredSearchBoundaryAndExcludesComparisonWork()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.Cosine,
            Dimension: 8,
            VectorCount: 20,
            QueryCount: 3,
            TopK: 5,
            Seed: 0x5EED0163,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-016-metadata.json",
            BaselineReportId: null,
            Runs: 1,
            WarmupQueries: 5);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        JsonElement latency = root.GetProperty("measurement").GetProperty("latency");

        Assert.Equal("measured", latency.GetProperty("status").GetString());
        Assert.Equal("milliseconds", latency.GetProperty("unit").GetString());
        Assert.Equal("perMeasuredQuery", latency.GetProperty("sampleScope").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, results)", latency.GetProperty("timedOperation").GetString());
        Assert.DoesNotContain("comparison", latency.GetProperty("timedOperation").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("capture", latency.GetProperty("timedOperation").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertContainsAll(
            latency.GetProperty("excludedOperations").GetString() ?? string.Empty,
            "setup",
            "index build",
            "scalar-reference truth",
            "warmup",
            "result capture/comparison",
            "report writing");
        AssertContainsAll(
            latency.GetProperty("percentileEstimator").GetString() ?? string.Empty,
            "nearest-rank",
            "sorted per-run query latency samples",
            "ceil(sampleCount * percentile) - 1",
            "clamped");
        AssertContainsAll(
            latency.GetProperty("aggregateSemantics").GetString() ?? string.Empty,
            "Top-level search latency percentile fields",
            "search.aggregate",
            "arithmetic means",
            "not BenchmarkDotNet statistics");
        AssertContainsAll(
            latency.GetProperty("rawSampleDisclosure").GetString() ?? string.Empty,
            "Raw per-query latency samples",
            "not emitted");

        Assert.Contains(
            "result capture/comparison",
            root.GetProperty("scenario").GetProperty("excludedFromSearchTiming").GetString(),
            StringComparison.OrdinalIgnoreCase);
        Assert.Contains(
            "result capture/comparison",
            string.Join(" ", root.GetProperty("notes").EnumerateArray().Select(item => item.GetString())),
            StringComparison.OrdinalIgnoreCase);
        AssertManagedAllocationMetadataStillPresent(root);
        AssertNoRawLatencyOrComparisonFields(root);
    }

    [Fact]
    public void MatrixRun_AllSmokePerCaseReportsCarryLatencyMetadataAndFalseEligibility()
    {
        string outputDirectory = NewArtifactDirectory("vec016-independent-matrix");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 2,
            Seed: 0x5EED0164,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(
            options,
            ["exact-generated-matrix", "--queries", "1", "--warmup-queries", "2"]);

        Assert.Equal("VEC-015", manifest.TaskId);
        Assert.Equal(18, manifest.CaseCount);
        Assert.Equal(18, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        foreach (GeneratedExactMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(matrixCase.ReportPath));
            JsonElement root = document.RootElement;
            JsonElement latency = root.GetProperty("measurement").GetProperty("latency");

            Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
            Assert.Equal("measured", latency.GetProperty("status").GetString());
            Assert.Equal("milliseconds", latency.GetProperty("unit").GetString());
            Assert.Equal("perMeasuredQuery", latency.GetProperty("sampleScope").GetString());
            Assert.Contains("ExactFlatIndex.Search", latency.GetProperty("timedOperation").GetString(), StringComparison.Ordinal);
            Assert.Contains("nearest-rank", latency.GetProperty("percentileEstimator").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not BenchmarkDotNet statistics", latency.GetProperty("aggregateSemantics").GetString(), StringComparison.Ordinal);
            Assert.Equal(1, root.GetProperty("search").GetProperty("runs").GetArrayLength());
            Assert.Equal(1, root.GetProperty("search").GetProperty("aggregate").GetProperty("runCount").GetInt32());
            Assert.Equal(1, root.GetProperty("search").GetProperty("aggregate").GetProperty("measuredQueryCountPerRun").GetInt32());
            Assert.Equal(2, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
            AssertManagedAllocationMetadataStillPresent(root);
            AssertFalseEligibility(root);
            AssertNoRawLatencyOrComparisonFields(root);
        }
    }

    private static string NewArtifactDirectory(string prefix) =>
        Path.Combine("VecNet.BenchmarkRunner.Artifacts", prefix + "-" + Guid.NewGuid().ToString("N"));

    private static void AssertPositiveLatencyFields(JsonElement searchLike)
    {
        Assert.True(searchLike.TryGetProperty("latencyP50Milliseconds", out JsonElement p50));
        Assert.True(searchLike.TryGetProperty("latencyP95Milliseconds", out JsonElement p95));
        Assert.True(searchLike.TryGetProperty("latencyP99Milliseconds", out JsonElement p99));
        Assert.True(p50.GetDouble() >= 0);
        Assert.True(p95.GetDouble() >= 0);
        Assert.True(p99.GetDouble() >= 0);
    }

    private static void AssertManagedAllocationMetadataStillPresent(JsonElement root)
    {
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement managedAllocations = measurement.GetProperty("managedAllocations");
        JsonElement aggregate = root.GetProperty("search").GetProperty("aggregate");

        Assert.Equal("measured", managedAllocations.GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", managedAllocations.GetProperty("unit").GetString());
        Assert.Contains("ExactFlatIndex.Search", managedAllocations.GetProperty("reason").GetString(), StringComparison.Ordinal);
        Assert.True(double.TryParse(
            managedAllocations.GetProperty("value").GetString(),
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out double measuredValue));
        Assert.Equal(aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble(), measuredValue);

        foreach (JsonElement run in root.GetProperty("search").GetProperty("runs").EnumerateArray())
        {
            Assert.True(run.TryGetProperty("managedAllocatedBytes", out _));
            Assert.True(run.TryGetProperty("managedAllocatedBytesPerQuery", out _));
        }
    }

    private static void AssertFalseEligibility(JsonElement root)
    {
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
    }

    private static void AssertContainsAll(string value, params string[] expectedParts)
    {
        foreach (string expectedPart in expectedParts)
        {
            Assert.Contains(expectedPart, value, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void AssertNoRawLatencyOrComparisonFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "latencySamples",
            "latencyTicks",
            "latencySampleTicks",
            "queryLatencies",
            "queryLatencyMilliseconds",
            "perQueryLatencyMilliseconds",
            "rawLatencySamples",
            "rawQueryLatencySamples",
            "baselineReportPath",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "latencyDeltaPercent",
            "qpsRatio",
            "allocationDeltaBytes",
            "allocationRatio",
            "regressionPassed",
            "regressionThreshold",
            "threshold",
            "delta",
            "ratio");
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, string.Create(CultureInfo.InvariantCulture, $"Unexpected field '{property.Name}' was present."));
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
}
