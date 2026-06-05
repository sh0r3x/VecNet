using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec012IndependentTests
{
    [Theory]
    [InlineData("exact-generated", "--runs")]
    [InlineData("exact-generated", "--warmup-queries")]
    [InlineData("exact-generated", "--output")]
    [InlineData("exact-generated", "--baseline-report-id")]
    [InlineData("exact-generated", "--runs-extra", "2")]
    [InlineData("exact-generated", "--warmup-query", "1")]
    [InlineData("exact-generated", "--runs", "--warmup-queries")]
    public void Parse_RejectsMissingValuesAndUnsupportedOptionNames(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.Parse(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Parse_WithScenarioOnlyPreservesOneRunZeroWarmupCompatibility()
    {
        GeneratedExactSearchOptions options = CommandLine.Parse(["exact-generated"]);

        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
    }

    [Fact]
    public void Run_DefaultReportSerializesSingleRunZeroWarmupAndFalseEligibility()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 6,
            VectorCount: 12,
            QueryCount: 3,
            TopK: 4,
            Seed: 0x5EED012C,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-012-default.json",
            BaselineReportId: "private-baseline-id");

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;

        Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());

        JsonElement search = root.GetProperty("search");
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(1, search.GetProperty("runs").GetArrayLength());
        Assert.Equal(1, search.GetProperty("runs")[0].GetProperty("runNumber").GetInt32());
        Assert.Equal(3, search.GetProperty("runs")[0].GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(1, search.GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal(3, search.GetProperty("aggregate").GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal(
            search.GetProperty("runs")[0].GetProperty("elapsedMilliseconds").GetDouble(),
            search.GetProperty("aggregate").GetProperty("meanElapsedMilliseconds").GetDouble());

        JsonElement repeatedRuns = root.GetProperty("measurement").GetProperty("repeatedRuns");
        Assert.Equal("singleRun", repeatedRuns.GetProperty("status").GetString());
        Assert.Equal(1, repeatedRuns.GetProperty("runCount").GetInt32());
        Assert.False(repeatedRuns.GetProperty("varianceMeasured").GetBoolean());

        JsonElement warmup = root.GetProperty("measurement").GetProperty("warmup");
        Assert.Equal("absent", warmup.GetProperty("status").GetString());
        Assert.Equal(0, warmup.GetProperty("warmupCount").GetInt32());
        AssertMeasuredAllocationsAndUnmeasuredMemory(root);

        AssertFalseEligibility(root);
    }

    [Fact]
    public void Run_MultipleRunsWithWarmupBeyondQueryCountKeepsWarmupOutsideMeasuredCounts()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.Cosine,
            Dimension: 10,
            VectorCount: 28,
            QueryCount: 3,
            TopK: 5,
            Seed: 0x5EED012D,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/test-agent-vec-012-multi.json",
            BaselineReportId: null,
            Runs: 4,
            WarmupQueries: 8);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--metric", "Cosine",
                "--queries", "3",
                "--runs", "4",
                "--warmup-queries", "8"
            ]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement search = root.GetProperty("search");
        JsonElement runs = search.GetProperty("runs");
        JsonElement aggregate = search.GetProperty("aggregate");

        Assert.Equal(3, root.GetProperty("scenario").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(4, runs.GetArrayLength());
        Assert.Equal(4, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(3, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Contains(
            "warmup",
            root.GetProperty("scenario").GetProperty("excludedFromSearchTiming").GetString(),
            StringComparison.OrdinalIgnoreCase);

        double[] elapsed = runs.EnumerateArray()
            .Select((run, index) =>
            {
                Assert.Equal(index + 1, run.GetProperty("runNumber").GetInt32());
                Assert.Equal(3, run.GetProperty("measuredQueryCount").GetInt32());
                Assert.True(run.GetProperty("qps").GetDouble() > 0);
                Assert.True(run.GetProperty("latencyP95Milliseconds").GetDouble() >= run.GetProperty("latencyP50Milliseconds").GetDouble());
                Assert.True(run.GetProperty("latencyP99Milliseconds").GetDouble() >= run.GetProperty("latencyP95Milliseconds").GetDouble());
                return run.GetProperty("elapsedMilliseconds").GetDouble();
            })
            .ToArray();

        Assert.Equal(elapsed.Average(), aggregate.GetProperty("meanElapsedMilliseconds").GetDouble());
        Assert.Equal(elapsed.Min(), aggregate.GetProperty("minElapsedMilliseconds").GetDouble());
        Assert.Equal(elapsed.Max(), aggregate.GetProperty("maxElapsedMilliseconds").GetDouble());
        Assert.Equal(
            aggregate.GetProperty("meanElapsedMilliseconds").GetDouble(),
            search.GetProperty("elapsedMilliseconds").GetDouble());

        JsonElement repeatedRuns = root.GetProperty("measurement").GetProperty("repeatedRuns");
        Assert.Equal("measured", repeatedRuns.GetProperty("status").GetString());
        Assert.Equal(4, repeatedRuns.GetProperty("runCount").GetInt32());
        Assert.True(repeatedRuns.GetProperty("varianceMeasured").GetBoolean());

        JsonElement warmup = root.GetProperty("measurement").GetProperty("warmup");
        Assert.Equal("executed", warmup.GetProperty("status").GetString());
        Assert.Equal(8, warmup.GetProperty("warmupCount").GetInt32());
        Assert.DoesNotContain("\"measuredQueryCount\":11", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"measuredQueryCountPerRun\":11", json, StringComparison.Ordinal);
        AssertMeasuredAllocationsAndUnmeasuredMemory(root);

        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal(1.0, root.GetProperty("metrics").GetProperty("recallAtK").GetDouble());
        Assert.Equal(1.0, root.GetProperty("metrics").GetProperty("orderedAgreement").GetDouble());
        AssertFalseEligibility(root);
    }

    private static void AssertFalseEligibility(JsonElement root)
    {
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
    }

    private static void AssertMeasuredAllocationsAndUnmeasuredMemory(JsonElement root)
    {
        JsonElement measurement = root.GetProperty("measurement");
        Assert.Equal("measured", measurement.GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", measurement.GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Equal("notMeasured", measurement.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", measurement.GetProperty("memory").GetProperty("value").GetString());

        JsonElement search = root.GetProperty("search");
        foreach (JsonElement run in search.GetProperty("runs").EnumerateArray())
        {
            Assert.True(run.GetProperty("managedAllocatedBytes").GetInt64() >= 0);
            Assert.True(run.GetProperty("managedAllocatedBytesPerQuery").GetDouble() >= 0);
        }

        JsonElement aggregate = search.GetProperty("aggregate");
        Assert.True(aggregate.GetProperty("meanManagedAllocatedBytes").GetDouble() >= 0);
        Assert.True(aggregate.GetProperty("minManagedAllocatedBytes").GetInt64() >= 0);
        Assert.True(aggregate.GetProperty("maxManagedAllocatedBytes").GetInt64() >= 0);
        Assert.True(aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble() >= 0);
        Assert.True(aggregate.GetProperty("minManagedAllocatedBytesPerQuery").GetDouble() >= 0);
        Assert.True(aggregate.GetProperty("maxManagedAllocatedBytesPerQuery").GetDouble() >= 0);
    }
}
