using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec016LatencyPercentileTests
{
    [Fact]
    public void NearestRankMilliseconds_UsesDocumentedEstimatorOverSortedSamples()
    {
        long[] sortedTicks = [1, 2, 3, 4, 5];

        Assert.Equal(0, LatencyPercentiles.NearestRankMilliseconds([], 0.50, ticksPerSecond: 1000));
        Assert.Equal(1, LatencyPercentiles.NearestRankMilliseconds(sortedTicks, 0.00, ticksPerSecond: 1000));
        Assert.Equal(3, LatencyPercentiles.NearestRankMilliseconds(sortedTicks, 0.50, ticksPerSecond: 1000));
        Assert.Equal(5, LatencyPercentiles.NearestRankMilliseconds(sortedTicks, 0.95, ticksPerSecond: 1000));
        Assert.Equal(5, LatencyPercentiles.NearestRankMilliseconds(sortedTicks, 0.99, ticksPerSecond: 1000));
    }

    [Theory]
    [InlineData(-0.01)]
    [InlineData(1.01)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NearestRankMilliseconds_RejectsInvalidPercentile(double percentile)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LatencyPercentiles.NearestRankMilliseconds([1], percentile, ticksPerSecond: 1000));
    }

    [Fact]
    public void Run_SerializesLatencyMetadataAndAggregateMeanPercentileSemantics()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 10,
            VectorCount: 24,
            QueryCount: 5,
            TopK: 6,
            Seed: 0x5EED0160,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec016-latency-metadata.json",
            BaselineReportId: null,
            Runs: 3,
            WarmupQueries: 4);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--queries", "5",
                "--runs", "3",
                "--warmup-queries", "4"
            ]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;

        JsonElement latency = root.GetProperty("measurement").GetProperty("latency");
        Assert.Equal("measured", latency.GetProperty("status").GetString());
        Assert.Equal("milliseconds", latency.GetProperty("unit").GetString());
        Assert.Equal("perMeasuredQuery", latency.GetProperty("sampleScope").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, results)", latency.GetProperty("timedOperation").GetString());
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
            "ceil(sampleCount * percentile) - 1");
        AssertContainsAll(
            latency.GetProperty("aggregateSemantics").GetString() ?? string.Empty,
            "means across per-run percentile values",
            "not BenchmarkDotNet statistics");
        Assert.Contains("not emitted", latency.GetProperty("rawSampleDisclosure").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP50Milliseconds), report.Search.LatencyP50Milliseconds);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP95Milliseconds), report.Search.LatencyP95Milliseconds);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP99Milliseconds), report.Search.LatencyP99Milliseconds);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP50Milliseconds), report.Search.Aggregate.MeanLatencyP50Milliseconds);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP95Milliseconds), report.Search.Aggregate.MeanLatencyP95Milliseconds);
        Assert.Equal(report.Search.Runs.Average(run => run.LatencyP99Milliseconds), report.Search.Aggregate.MeanLatencyP99Milliseconds);

        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(4, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoRawLatencyArrays(root);
    }

    [Fact]
    public void MatrixPerCaseReportsCarryLatencyMetadataThroughExistingReportReuse()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec016-matrix-latency-" + Guid.NewGuid().ToString("N"));
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED0161,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(
            options,
            ["exact-generated-matrix", "--queries", "2", "--runs", "2", "--warmup-queries", "1"]);

        GeneratedExactMatrixCaseManifest firstCase = manifest.Cases[0];
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(firstCase.ReportPath));
        JsonElement root = document.RootElement;

        JsonElement latency = root.GetProperty("measurement").GetProperty("latency");
        Assert.Equal("measured", latency.GetProperty("status").GetString());
        Assert.Equal("milliseconds", latency.GetProperty("unit").GetString());
        Assert.Contains("ExactFlatIndex.Search", latency.GetProperty("timedOperation").GetString(), StringComparison.Ordinal);
        Assert.Contains("nearest-rank", latency.GetProperty("percentileEstimator").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not BenchmarkDotNet", latency.GetProperty("aggregateSemantics").GetString(), StringComparison.Ordinal);
        Assert.Equal(2, root.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoRawLatencyArrays(root);
    }

    private static void AssertContainsAll(string value, params string[] expectedParts)
    {
        foreach (string expectedPart in expectedParts)
        {
            Assert.Contains(expectedPart, value, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void AssertNoRawLatencyArrays(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "latencySamples",
            "latencyTicks",
            "queryLatencies",
            "rawLatencySamples",
            "rawQueryLatencySamples");
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, $"Unexpected raw latency field '{property.Name}' was present.");
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
