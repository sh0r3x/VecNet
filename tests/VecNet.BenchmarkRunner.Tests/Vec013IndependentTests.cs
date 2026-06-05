using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec013IndependentTests
{
    [Fact]
    public void Run_DefaultReportKeepsCompatibilityAndRecordsMeasuredAllocationMetadata()
    {
        GeneratedExactSearchOptions options = CommandLine.Parse(
            [
                "exact-generated",
                "--dimension", "7",
                "--vectors", "18",
                "--queries", "4",
                "--top-k", "5",
                "--seed", "0x5EED0130"
            ]);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;

        Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());

        JsonElement search = root.GetProperty("search");
        JsonElement runs = search.GetProperty("runs");
        JsonElement aggregate = search.GetProperty("aggregate");
        Assert.Equal(4, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(1, runs.GetArrayLength());
        Assert.Equal(1, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(4, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());

        JsonElement run = runs[0];
        Assert.Equal(1, run.GetProperty("runNumber").GetInt32());
        Assert.Equal(4, run.GetProperty("measuredQueryCount").GetInt32());
        AssertAllocationNormalization(run);
        AssertSingleRunAllocationAggregate(run, aggregate);

        JsonElement measurement = root.GetProperty("measurement");
        AssertManagedAllocationMeasurementMatchesAggregate(measurement, aggregate);
        AssertMemoryNotMeasured(measurement.GetProperty("memory"));
        Assert.Equal("singleRun", measurement.GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.Equal(1, measurement.GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
        Assert.False(measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("absent", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(0, measurement.GetProperty("warmup").GetProperty("warmupCount").GetInt32());

        AssertFalseEligibility(root);
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
    }

    [Fact]
    public void Run_RepeatedRunsAllocationAggregateMatchesEveryMeasuredRun()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.InnerProduct,
            Dimension: 13,
            VectorCount: 31,
            QueryCount: 6,
            TopK: 7,
            Seed: 0x5EED0131,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-013-repeated.json",
            BaselineReportId: "metadata-only-baseline",
            Runs: 5,
            WarmupQueries: 3);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--metric", "InnerProduct",
                "--runs", "5",
                "--warmup-queries", "3",
                "--baseline-report-id", "metadata-only-baseline"
            ]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        JsonElement search = root.GetProperty("search");
        JsonElement runs = search.GetProperty("runs");
        JsonElement aggregate = search.GetProperty("aggregate");

        Assert.Equal(5, runs.GetArrayLength());
        Assert.Equal(5, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(6, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());

        long[] allocatedBytes = new long[5];
        double[] allocatedBytesPerQuery = new double[5];
        for (int index = 0; index < runs.GetArrayLength(); index++)
        {
            JsonElement run = runs[index];
            Assert.Equal(index + 1, run.GetProperty("runNumber").GetInt32());
            Assert.Equal(6, run.GetProperty("measuredQueryCount").GetInt32());
            AssertAllocationNormalization(run);
            allocatedBytes[index] = run.GetProperty("managedAllocatedBytes").GetInt64();
            allocatedBytesPerQuery[index] = run.GetProperty("managedAllocatedBytesPerQuery").GetDouble();
        }

        Assert.Equal(allocatedBytes.Average(), aggregate.GetProperty("meanManagedAllocatedBytes").GetDouble());
        Assert.Equal(allocatedBytes.Min(), aggregate.GetProperty("minManagedAllocatedBytes").GetInt64());
        Assert.Equal(allocatedBytes.Max(), aggregate.GetProperty("maxManagedAllocatedBytes").GetInt64());
        Assert.Equal(allocatedBytesPerQuery.Average(), aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble());
        Assert.Equal(allocatedBytesPerQuery.Min(), aggregate.GetProperty("minManagedAllocatedBytesPerQuery").GetDouble());
        Assert.Equal(allocatedBytesPerQuery.Max(), aggregate.GetProperty("maxManagedAllocatedBytesPerQuery").GetDouble());
        AssertManagedAllocationMeasurementMatchesAggregate(root.GetProperty("measurement"), aggregate);

        JsonElement repeatedRuns = root.GetProperty("measurement").GetProperty("repeatedRuns");
        Assert.Equal("measured", repeatedRuns.GetProperty("status").GetString());
        Assert.Equal(5, repeatedRuns.GetProperty("runCount").GetInt32());
        Assert.True(repeatedRuns.GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(3, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal("metadata-only-baseline", root.GetProperty("baseline").GetProperty("baselineReportId").GetString());
        AssertFalseEligibility(root);
    }

    [Fact]
    public void Run_ResultCaptureAfterMeasuredAllocationDoesNotInflateLastRun()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 16,
            VectorCount: 64,
            QueryCount: 8,
            TopK: 32,
            Seed: 0x5EED0132,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-013-capture.json",
            BaselineReportId: null,
            Runs: 3,
            WarmupQueries: 0);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);

        long[] runAllocations = report.Search.Runs
            .Select(run => run.ManagedAllocatedBytes)
            .ToArray();

        Assert.Equal(3, runAllocations.Length);
        Assert.All(runAllocations, allocatedBytes => Assert.True(allocatedBytes >= 0));
        Assert.Equal(runAllocations[0], runAllocations[2]);
        Assert.Equal(runAllocations[1], runAllocations[2]);
        Assert.Equal(runAllocations.Min(), report.Search.Aggregate.MinManagedAllocatedBytes);
        Assert.Equal(runAllocations.Max(), report.Search.Aggregate.MaxManagedAllocatedBytes);
    }

    [Fact]
    public void Run_WarmupQueriesDoNotChangeMeasuredAllocationTotalsOrMeasuredCounts()
    {
        var withoutWarmup = new GeneratedExactSearchOptions(
            VectorMetric.Cosine,
            Dimension: 12,
            VectorCount: 40,
            QueryCount: 5,
            TopK: 6,
            Seed: 0x5EED0133,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-013-no-warmup.json",
            BaselineReportId: null,
            Runs: 2,
            WarmupQueries: 0);
        var withWarmup = withoutWarmup with
        {
            OutputPath = "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-013-heavy-warmup.json",
            WarmupQueries = 41
        };

        BenchmarkReport coldReport = GeneratedExactSearchScenario.Run(withoutWarmup, ["exact-generated"]);
        BenchmarkReport warmedReport = GeneratedExactSearchScenario.Run(withWarmup, ["exact-generated"]);

        Assert.Equal(5, warmedReport.Scenario.MeasuredQueryCount);
        Assert.Equal(5, warmedReport.Search.MeasuredQueryCount);
        Assert.Equal(5, warmedReport.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("executed", warmedReport.Measurement.Warmup.Status);
        Assert.Equal(41, warmedReport.Measurement.Warmup.WarmupCount);

        Assert.Equal(
            coldReport.Search.Runs.Select(run => run.ManagedAllocatedBytes).ToArray(),
            warmedReport.Search.Runs.Select(run => run.ManagedAllocatedBytes).ToArray());
        Assert.Equal(
            coldReport.Search.Runs.Select(run => run.ManagedAllocatedBytesPerQuery).ToArray(),
            warmedReport.Search.Runs.Select(run => run.ManagedAllocatedBytesPerQuery).ToArray());
        Assert.Equal(
            coldReport.Search.Aggregate.MeanManagedAllocatedBytesPerQuery,
            warmedReport.Search.Aggregate.MeanManagedAllocatedBytesPerQuery);
        Assert.NotEqual(46, warmedReport.Search.Aggregate.MeasuredQueryCountPerRun);
    }

    [Fact]
    public void Run_ReportDoesNotEmitMemoryEvidenceOrClaimEligibilityFieldsAsTrue()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 10,
            VectorCount: 25,
            QueryCount: 4,
            TopK: 5,
            Seed: 0x5EED0134,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-013-posture.json",
            BaselineReportId: "baseline-is-metadata-only",
            Runs: 4,
            WarmupQueries: 9);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.Equal("smoke", root.GetProperty("baseline").GetProperty("suitability").GetString());
        AssertMemoryNotMeasured(root.GetProperty("measurement").GetProperty("memory"));
        AssertFalseEligibility(root);
        AssertNoPropertyNamed(
            root,
            "residentMemoryBytes",
            "processMemoryBytes",
            "workingSetBytes",
            "heapSizeBytes",
            "memoryBytes",
            "baselineReportPath",
            "comparisonResult",
            "regressionPassed",
            "regressionThreshold");
    }

    private static void AssertAllocationNormalization(JsonElement run)
    {
        int measuredQueryCount = run.GetProperty("measuredQueryCount").GetInt32();
        long managedAllocatedBytes = run.GetProperty("managedAllocatedBytes").GetInt64();
        double managedAllocatedBytesPerQuery = run.GetProperty("managedAllocatedBytesPerQuery").GetDouble();

        Assert.True(managedAllocatedBytes >= 0);
        Assert.True(managedAllocatedBytesPerQuery >= 0);
        Assert.Equal((double)managedAllocatedBytes / measuredQueryCount, managedAllocatedBytesPerQuery);
    }

    private static void AssertSingleRunAllocationAggregate(JsonElement run, JsonElement aggregate)
    {
        long managedAllocatedBytes = run.GetProperty("managedAllocatedBytes").GetInt64();
        double managedAllocatedBytesPerQuery = run.GetProperty("managedAllocatedBytesPerQuery").GetDouble();

        Assert.Equal(managedAllocatedBytes, aggregate.GetProperty("meanManagedAllocatedBytes").GetDouble());
        Assert.Equal(managedAllocatedBytes, aggregate.GetProperty("minManagedAllocatedBytes").GetInt64());
        Assert.Equal(managedAllocatedBytes, aggregate.GetProperty("maxManagedAllocatedBytes").GetInt64());
        Assert.Equal(managedAllocatedBytesPerQuery, aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble());
        Assert.Equal(managedAllocatedBytesPerQuery, aggregate.GetProperty("minManagedAllocatedBytesPerQuery").GetDouble());
        Assert.Equal(managedAllocatedBytesPerQuery, aggregate.GetProperty("maxManagedAllocatedBytesPerQuery").GetDouble());
    }

    private static void AssertManagedAllocationMeasurementMatchesAggregate(JsonElement measurement, JsonElement aggregate)
    {
        JsonElement managedAllocations = measurement.GetProperty("managedAllocations");
        Assert.Equal("measured", managedAllocations.GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", managedAllocations.GetProperty("unit").GetString());
        Assert.Contains(
            "ExactFlatIndex.Search",
            managedAllocations.GetProperty("reason").GetString(),
            StringComparison.Ordinal);
        Assert.Contains(
            "warmup",
            managedAllocations.GetProperty("reason").GetString(),
            StringComparison.OrdinalIgnoreCase);

        double value = double.Parse(
            managedAllocations.GetProperty("value").GetString() ?? string.Empty,
            CultureInfo.InvariantCulture);
        Assert.Equal(aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble(), value);
    }

    private static void AssertMemoryNotMeasured(JsonElement memory)
    {
        Assert.Equal("notMeasured", memory.GetProperty("status").GetString());
        Assert.Equal("absent", memory.GetProperty("value").GetString());
        Assert.Equal("bytes", memory.GetProperty("unit").GetString());
    }

    private static void AssertFalseEligibility(JsonElement root)
    {
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
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
