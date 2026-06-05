using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec017IndependentTests
{
    [Fact]
    public void Calculate_UsesSampleStandardDeviationAndMinMaxSpreadForIndependentValues()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([10, 10, 14, 18]);

        Assert.Equal(4, statistics.Count);
        Assert.Equal(13, statistics.Mean);
        Assert.Equal(Math.Sqrt(44.0 / 3), statistics.SampleStandardDeviation!.Value, precision: 12);
        Assert.Equal(statistics.SampleStandardDeviation.Value / 13, statistics.CoefficientOfVariation!.Value, precision: 12);
        Assert.Equal(10, statistics.Min);
        Assert.Equal(18, statistics.Max);
        Assert.Equal(8, statistics.Spread);
    }

    [Fact]
    public void Calculate_WhenValuesContainInfinity_DisablesDeviationAndCoefficientOfVariation()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([1, double.PositiveInfinity, 3]);

        Assert.Equal(3, statistics.Count);
        Assert.Equal(double.PositiveInfinity, statistics.Mean);
        Assert.Null(statistics.SampleStandardDeviation);
        Assert.Null(statistics.CoefficientOfVariation);
        Assert.Equal(1, statistics.Min);
        Assert.Equal(double.PositiveInfinity, statistics.Max);
        Assert.Equal(double.PositiveInfinity, statistics.Spread);
    }

    [Fact]
    public void Calculate_WhenValuesContainNaN_DisablesDeviationAndKeepsNonFiniteSummaryVisibleToCaller()
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate([1, double.NaN, 3]);

        Assert.Equal(3, statistics.Count);
        Assert.True(double.IsNaN(statistics.Mean));
        Assert.Null(statistics.SampleStandardDeviation);
        Assert.Null(statistics.CoefficientOfVariation);
        Assert.True(double.IsNaN(statistics.Min));
        Assert.True(double.IsNaN(statistics.Max));
        Assert.True(double.IsNaN(statistics.Spread));
    }

    [Fact]
    public void Run_SingleRunNoiseJsonHasExpectedShapeAndNullMetricStatistics()
    {
        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.Cosine,
                Dimension: 11,
                VectorCount: 24,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED1701,
                OutputPath: NewArtifactPath("vec017-independent-single.json"),
                BaselineReportId: null,
                Runs: 1,
                WarmupQueries: 2),
            ["exact-generated", "--metric", "Cosine", "--runs", "1", "--warmup-queries", "2"]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement noise = measurement.GetProperty("runToRunNoise");

        Assert.Equal(
            ["latency", "managedAllocations", "memory", "repeatedRuns", "runToRunNoise", "warmup"],
            measurement.EnumerateObject().Select(property => property.Name).ToArray());
        Assert.Equal(
            [
                "status",
                "runCount",
                "noiseMeasured",
                "scope",
                "statistics",
                "reason",
                "nonGoals",
                "elapsedMilliseconds",
                "qps",
                "latencyP50Milliseconds",
                "latencyP95Milliseconds",
                "latencyP99Milliseconds",
                "managedAllocatedBytesPerQuery"
            ],
            noise.EnumerateObject().Select(property => property.Name).ToArray());

        Assert.Equal("notMeasured", noise.GetProperty("status").GetString());
        Assert.Equal(1, noise.GetProperty("runCount").GetInt32());
        Assert.False(noise.GetProperty("noiseMeasured").GetBoolean());
        Assert.Contains("one measured run", noise.GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("private local descriptive", noise.GetProperty("nonGoals").GetString(), StringComparison.OrdinalIgnoreCase);

        foreach (string metricName in RequiredNoiseMetricNames)
        {
            JsonElement metric = noise.GetProperty(metricName);
            Assert.Equal("notMeasured", metric.GetProperty("status").GetString());
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("mean").ValueKind);
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("sampleStandardDeviation").ValueKind);
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("coefficientOfVariation").ValueKind);
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("min").ValueKind);
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("max").ValueKind);
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("spread").ValueKind);
        }

        AssertExistingMeasurementMetadataPresent(root, expectedRuns: 1, expectedWarmupQueries: 2);
        AssertFalseEligibility(root);
        AssertNoForbiddenReportFields(root);
        Assert.DoesNotContain("NaN", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Infinity", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_MultipleRunNoiseMetricsMatchSerializedRunsAndKeepMetadataWordingBounded()
    {
        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 13,
                VectorCount: 32,
                QueryCount: 5,
                TopK: 7,
                Seed: 0x5EED1702,
                OutputPath: NewArtifactPath("vec017-independent-multi.json"),
                BaselineReportId: "metadata-only-baseline",
                Runs: 3,
                WarmupQueries: 6),
            [
                "exact-generated",
                "--metric", "SquaredEuclidean",
                "--runs", "3",
                "--warmup-queries", "6",
                "--baseline-report-id", "metadata-only-baseline"
            ]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement noise = root.GetProperty("measurement").GetProperty("runToRunNoise");

        Assert.Equal("measured", noise.GetProperty("status").GetString());
        Assert.Equal(3, noise.GetProperty("runCount").GetInt32());
        Assert.True(noise.GetProperty("noiseMeasured").GetBoolean());
        AssertContainsAll(
            noise.GetProperty("scope").GetString() ?? string.Empty,
            "measured generated exact-search runs",
            "ExactFlatIndex.Search",
            "warmup",
            "excluded");
        AssertContainsAll(
            noise.GetProperty("statistics").GetString() ?? string.Empty,
            "sample standard deviation",
            "coefficient of variation",
            "finite and non-zero",
            "min/max spread");
        AssertContainsAll(
            noise.GetProperty("nonGoals").GetString() ?? string.Empty,
            "Private local descriptive metadata only",
            "not BenchmarkDotNet statistics",
            "not baseline comparison math",
            "not an acceptable-noise threshold",
            "not a regression decision");

        AssertMetricMatchesRuns(noise.GetProperty("elapsedMilliseconds"), "milliseconds", report.Search.Runs.Select(run => run.ElapsedMilliseconds).ToArray());
        AssertMetricMatchesRuns(noise.GetProperty("qps"), "queriesPerSecond", report.Search.Runs.Select(run => run.Qps).ToArray());
        AssertMetricMatchesRuns(noise.GetProperty("latencyP50Milliseconds"), "milliseconds", report.Search.Runs.Select(run => run.LatencyP50Milliseconds).ToArray());
        AssertMetricMatchesRuns(noise.GetProperty("latencyP95Milliseconds"), "milliseconds", report.Search.Runs.Select(run => run.LatencyP95Milliseconds).ToArray());
        AssertMetricMatchesRuns(noise.GetProperty("latencyP99Milliseconds"), "milliseconds", report.Search.Runs.Select(run => run.LatencyP99Milliseconds).ToArray());
        AssertMetricMatchesRuns(noise.GetProperty("managedAllocatedBytesPerQuery"), "bytesPerQuery", report.Search.Runs.Select(run => run.ManagedAllocatedBytesPerQuery).ToArray());

        Assert.Equal(
            report.Search.Aggregate.MeanManagedAllocatedBytesPerQuery,
            double.Parse(root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("value").GetString()!, System.Globalization.CultureInfo.InvariantCulture));
        AssertExistingMeasurementMetadataPresent(root, expectedRuns: 3, expectedWarmupQueries: 6);
        AssertFalseEligibility(root);
        AssertNoForbiddenReportFields(root);
        Assert.DoesNotContain("NaN", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Infinity", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MatrixRun_AllSmokePerCaseReportsCarryRunToRunNoiseAndNoForbiddenFields()
    {
        string outputDirectory = NewArtifactDirectory("vec017-independent-matrix");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED1703,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(
            options,
            ["exact-generated-matrix", "--runs", "2", "--warmup-queries", "1"]);

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

            string json = File.ReadAllText(matrixCase.ReportPath);
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            JsonElement noise = root.GetProperty("measurement").GetProperty("runToRunNoise");

            Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
            Assert.Equal(matrixCase.Metric, root.GetProperty("dataset").GetProperty("metric").GetString());
            Assert.Equal(matrixCase.Dimension, root.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.TopK, root.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal("measured", noise.GetProperty("status").GetString());
            Assert.True(noise.GetProperty("noiseMeasured").GetBoolean());
            Assert.Equal(2, noise.GetProperty("runCount").GetInt32());
            foreach (string metricName in RequiredNoiseMetricNames)
            {
                Assert.Equal("measured", noise.GetProperty(metricName).GetProperty("status").GetString());
            }

            AssertExistingMeasurementMetadataPresent(root, expectedRuns: 2, expectedWarmupQueries: 1);
            AssertFalseEligibility(root);
            AssertNoForbiddenReportFields(root);
            Assert.DoesNotContain("NaN", json, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("Infinity", json, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static readonly string[] RequiredNoiseMetricNames =
    [
        "elapsedMilliseconds",
        "qps",
        "latencyP50Milliseconds",
        "latencyP95Milliseconds",
        "latencyP99Milliseconds",
        "managedAllocatedBytesPerQuery"
    ];

    private static string NewArtifactDirectory(string prefix) =>
        Path.Combine("VecNet.BenchmarkRunner.Artifacts", prefix + "-" + Guid.NewGuid().ToString("N"));

    private static string NewArtifactPath(string fileName) =>
        Path.Combine(NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName)), fileName);

    private static void AssertMetricMatchesRuns(JsonElement metric, string expectedUnit, double[] values)
    {
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate(values);

        Assert.Equal("measured", metric.GetProperty("status").GetString());
        Assert.Equal(expectedUnit, metric.GetProperty("unit").GetString());
        Assert.Equal(statistics.Mean, metric.GetProperty("mean").GetDouble(), precision: 12);
        Assert.Equal(statistics.SampleStandardDeviation!.Value, metric.GetProperty("sampleStandardDeviation").GetDouble(), precision: 12);
        if (statistics.CoefficientOfVariation.HasValue)
        {
            Assert.Equal(statistics.CoefficientOfVariation.Value, metric.GetProperty("coefficientOfVariation").GetDouble(), precision: 12);
        }
        else
        {
            Assert.Equal(JsonValueKind.Null, metric.GetProperty("coefficientOfVariation").ValueKind);
        }

        Assert.Equal(statistics.Min, metric.GetProperty("min").GetDouble(), precision: 12);
        Assert.Equal(statistics.Max, metric.GetProperty("max").GetDouble(), precision: 12);
        Assert.Equal(statistics.Spread, metric.GetProperty("spread").GetDouble(), precision: 12);
        Assert.Equal(statistics.Max - statistics.Min, metric.GetProperty("spread").GetDouble(), precision: 12);
    }

    private static void AssertExistingMeasurementMetadataPresent(JsonElement root, int expectedRuns, int expectedWarmupQueries)
    {
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement search = root.GetProperty("search");

        Assert.Equal(expectedRuns, search.GetProperty("runs").GetArrayLength());
        Assert.Equal(expectedRuns, search.GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal(expectedRuns, measurement.GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
        Assert.Equal(expectedRuns > 1, measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal(expectedWarmupQueries, measurement.GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal(expectedWarmupQueries > 0 ? "executed" : "absent", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal("measured", measurement.GetProperty("latency").GetProperty("status").GetString());
        Assert.Contains("nearest-rank", measurement.GetProperty("latency").GetProperty("percentileEstimator").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", measurement.GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", measurement.GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Equal("notMeasured", measurement.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", measurement.GetProperty("memory").GetProperty("value").GetString());
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

    private static void AssertNoForbiddenReportFields(JsonElement element)
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
            "baselineComparison",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "latencyDeltaPercent",
            "qpsRatio",
            "allocationDeltaBytes",
            "allocationRatio",
            "regressionPassed",
            "regressionDecision",
            "regressionThreshold",
            "acceptableNoiseThreshold",
            "noiseThreshold",
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
                Assert.False(disallowed, $"Unexpected report field '{property.Name}' was present.");
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
