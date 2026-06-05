using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec017RunToRunNoiseTests
{
    [Fact]
    public void Calculate_UsesDocumentedSampleStatisticsForSyntheticValues()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([2, 4, 6]);

        Assert.Equal(3, statistics.Count);
        Assert.Equal(4, statistics.Mean);
        Assert.Equal(2, statistics.SampleStandardDeviation);
        Assert.Equal(0.5, statistics.CoefficientOfVariation);
        Assert.Equal(2, statistics.Min);
        Assert.Equal(6, statistics.Max);
        Assert.Equal(4, statistics.Spread);
    }

    [Fact]
    public void Calculate_WhenMeanIsZero_DoesNotReportCoefficientOfVariation()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([-1, 0, 1]);

        Assert.Equal(0, statistics.Mean);
        Assert.Equal(1, statistics.SampleStandardDeviation);
        Assert.Null(statistics.CoefficientOfVariation);
        Assert.Equal(2, statistics.Spread);
    }

    [Fact]
    public void Calculate_WhenOnlyOneValue_DoesNotReportRunToRunStandardDeviation()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([12]);

        Assert.Equal(1, statistics.Count);
        Assert.Equal(12, statistics.Mean);
        Assert.Null(statistics.SampleStandardDeviation);
        Assert.Null(statistics.CoefficientOfVariation);
        Assert.Equal(12, statistics.Min);
        Assert.Equal(12, statistics.Max);
        Assert.Equal(0, statistics.Spread);
    }

    [Fact]
    public void Calculate_RejectsEmptyValues()
    {
        Assert.Throws<ArgumentException>(() => RunToRunNoiseStatistics.Calculate([]));
    }

    [Fact]
    public void Run_SingleRunSerializesNoiseAsUnavailable()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 8,
            VectorCount: 16,
            QueryCount: 4,
            TopK: 5,
            Seed: 0x5EED0170,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec017-single-run.json",
            BaselineReportId: null,
            Runs: 1,
            WarmupQueries: 0);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated", "--runs", "1"]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        JsonElement noise = root.GetProperty("measurement").GetProperty("runToRunNoise");

        Assert.Equal("singleRun", root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.False(root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("notMeasured", noise.GetProperty("status").GetString());
        Assert.False(noise.GetProperty("noiseMeasured").GetBoolean());
        Assert.Equal(1, noise.GetProperty("runCount").GetInt32());
        Assert.Contains("Only one measured run", noise.GetProperty("reason").GetString(), StringComparison.Ordinal);
        Assert.Contains("not BenchmarkDotNet", noise.GetProperty("nonGoals").GetString(), StringComparison.Ordinal);

        AssertNoiseMetricUnavailable(noise.GetProperty("elapsedMilliseconds"), "milliseconds");
        AssertNoiseMetricUnavailable(noise.GetProperty("qps"), "queriesPerSecond");
        AssertNoiseMetricUnavailable(noise.GetProperty("latencyP50Milliseconds"), "milliseconds");
        AssertNoiseMetricUnavailable(noise.GetProperty("latencyP95Milliseconds"), "milliseconds");
        AssertNoiseMetricUnavailable(noise.GetProperty("latencyP99Milliseconds"), "milliseconds");
        AssertNoiseMetricUnavailable(noise.GetProperty("managedAllocatedBytesPerQuery"), "bytesPerQuery");
        AssertFalseEligibility(root);
    }

    [Fact]
    public void Run_MultipleRunsSerializesNoiseSummariesForRequiredFields()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.InnerProduct,
            Dimension: 9,
            VectorCount: 24,
            QueryCount: 5,
            TopK: 6,
            Seed: 0x5EED0171,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec017-multi-run.json",
            BaselineReportId: "metadata-only-baseline",
            Runs: 4,
            WarmupQueries: 3);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            ["exact-generated", "--runs", "4", "--warmup-queries", "3"]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        JsonElement noise = root.GetProperty("measurement").GetProperty("runToRunNoise");

        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.True(root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("measured", noise.GetProperty("status").GetString());
        Assert.True(noise.GetProperty("noiseMeasured").GetBoolean());
        Assert.Equal(4, noise.GetProperty("runCount").GetInt32());
        AssertContainsAll(
            noise.GetProperty("statistics").GetString() ?? string.Empty,
            "mean",
            "sample standard deviation",
            "coefficient of variation",
            "min/max spread");
        AssertContainsAll(
            noise.GetProperty("scope").GetString() ?? string.Empty,
            "ExactFlatIndex.Search",
            "warmup",
            "excluded");
        Assert.Contains("not BenchmarkDotNet", noise.GetProperty("nonGoals").GetString(), StringComparison.Ordinal);

        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("elapsedMilliseconds"),
            "milliseconds",
            report.Search.Runs.Select(run => run.ElapsedMilliseconds).ToArray());
        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("qps"),
            "queriesPerSecond",
            report.Search.Runs.Select(run => run.Qps).ToArray());
        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("latencyP50Milliseconds"),
            "milliseconds",
            report.Search.Runs.Select(run => run.LatencyP50Milliseconds).ToArray());
        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("latencyP95Milliseconds"),
            "milliseconds",
            report.Search.Runs.Select(run => run.LatencyP95Milliseconds).ToArray());
        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("latencyP99Milliseconds"),
            "milliseconds",
            report.Search.Runs.Select(run => run.LatencyP99Milliseconds).ToArray());
        AssertMeasuredMetricMatchesRuns(
            noise.GetProperty("managedAllocatedBytesPerQuery"),
            "bytesPerQuery",
            report.Search.Runs.Select(run => run.ManagedAllocatedBytesPerQuery).ToArray());

        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoComparisonOrThresholdFields(root);
    }

    [Fact]
    public void MatrixPerCaseReportsCarryRunToRunNoiseMetadataThroughExistingReportReuse()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec017-matrix-noise-" + Guid.NewGuid().ToString("N"));
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED0172,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(
            options,
            ["exact-generated-matrix", "--queries", "2", "--runs", "2", "--warmup-queries", "1"]);

        Assert.Equal("VEC-015", manifest.TaskId);
        Assert.Equal(18, manifest.CaseCount);
        Assert.Equal(18, manifest.Aggregate.PassedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifest.Cases[0].ReportPath));
        JsonElement root = document.RootElement;
        JsonElement noise = root.GetProperty("measurement").GetProperty("runToRunNoise");

        Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
        Assert.Equal("measured", noise.GetProperty("status").GetString());
        Assert.True(noise.GetProperty("noiseMeasured").GetBoolean());
        Assert.Equal(2, noise.GetProperty("runCount").GetInt32());
        Assert.Equal("measured", noise.GetProperty("elapsedMilliseconds").GetProperty("status").GetString());
        Assert.Equal("measured", noise.GetProperty("managedAllocatedBytesPerQuery").GetProperty("status").GetString());
        Assert.Equal(2, root.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        AssertFalseEligibility(root);
        AssertNoComparisonOrThresholdFields(root);
    }

    private static void AssertNoiseMetricUnavailable(JsonElement metric, string expectedUnit)
    {
        Assert.Equal("notMeasured", metric.GetProperty("status").GetString());
        Assert.Equal(expectedUnit, metric.GetProperty("unit").GetString());
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("mean").ValueKind);
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("sampleStandardDeviation").ValueKind);
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("coefficientOfVariation").ValueKind);
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("min").ValueKind);
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("max").ValueKind);
        Assert.Equal(JsonValueKind.Null, metric.GetProperty("spread").ValueKind);
        Assert.Contains("one measured run", metric.GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertMeasuredMetricMatchesRuns(JsonElement metric, string expectedUnit, double[] values)
    {
        DescriptiveStatistics expected = RunToRunNoiseStatistics.Calculate(values);

        Assert.Equal("measured", metric.GetProperty("status").GetString());
        Assert.Equal(expectedUnit, metric.GetProperty("unit").GetString());
        Assert.Equal(expected.Mean, metric.GetProperty("mean").GetDouble(), precision: 12);
        Assert.Equal(expected.SampleStandardDeviation!.Value, metric.GetProperty("sampleStandardDeviation").GetDouble(), precision: 12);
        if (expected.CoefficientOfVariation.HasValue)
        {
            Assert.Equal(expected.CoefficientOfVariation.Value, metric.GetProperty("coefficientOfVariation").GetDouble(), precision: 12);
        }
        else
        {
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("coefficientOfVariation").ValueKind);
        }

        Assert.Equal(expected.Min, metric.GetProperty("min").GetDouble(), precision: 12);
        Assert.Equal(expected.Max, metric.GetProperty("max").GetDouble(), precision: 12);
        Assert.Equal(expected.Spread, metric.GetProperty("spread").GetDouble(), precision: 12);
        Assert.Contains("documented private descriptive-statistics formula", metric.GetProperty("reason").GetString(), StringComparison.Ordinal);
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

    private static void AssertNoComparisonOrThresholdFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baselineReportPath",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "latencyDeltaPercent",
            "qpsRatio",
            "allocationDeltaBytes",
            "allocationRatio",
            "regressionPassed",
            "regressionThreshold",
            "acceptableNoiseThreshold",
            "noiseThreshold",
            "threshold",
            "delta",
            "ratio",
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
                Assert.False(disallowed, $"Unexpected field '{property.Name}' was present.");
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
