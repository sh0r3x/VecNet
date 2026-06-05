using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class ReportWriterTests
{
    [Fact]
    public void Serialize_UsesCanonicalCamelCaseReportShape()
    {
        var report = new BenchmarkReport(
            SchemaName: "VecNet.BenchmarkReport",
            SchemaVersion: "0.1",
            ReportId: "test-report",
            GeneratedAtUtc: DateTimeOffset.UnixEpoch,
            TaskId: "VEC-014",
            ClaimClass: "local-evidence",
            PrivacyClass: "private-raw",
            Evidence: new EvidenceInfo(
                "smoke",
                "local-evidence",
                false,
                "not public evidence",
                ["allocation not measured"]),
            Repository: new RepositoryInfo("abc123", "main", Dirty: true),
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", ["exact-generated"]),
            Command: new CommandInfo("exact-generated", ["exact-generated"]),
            Environment: new EnvironmentInfo(".NET test OS", "X64", ".NET 10", "win-x64", 8, false, 8),
            Dataset: new DatasetInfo("generated-uniform", "generated-no-external-source", "uniform[-1,1)", "0x5EED2009", "SquaredEuclidean", 2, 3, 1),
            Truth: new TruthInfo("scalar-reference-generated", 2, ScalarGroundTruth.TiePolicy),
            Scenario: new ScenarioInfo("exact-generated", 2, 1, 1, "setup excluded"),
            Index: new IndexInfo("Exact", "ExactFlatIndex", "SquaredEuclidean", 2, 3, "public default"),
            Search: new SearchInfo(
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                2000,
                [new SearchRunInfo(1, 1, 0.5, 0.5, 0.5, 0.5, 2000, 0, 0)],
                new AggregateTimingInfo(1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2000, 2000, 2000, 0, 0, 0, 0, 0, 0)),
            Measurement: new MeasurementInfo(
                new MeasurementStatusInfo("measured", "0", "bytesPerQuery", "measured"),
                new MeasurementStatusInfo("notMeasured", "absent", "bytes", "not measured"),
                new RepeatedRunInfo("notMeasured", 1, false, "not measured"),
                new WarmupInfo("notMeasured", 0, "not measured")),
            Metrics: new MetricsInfo(1, 1, "passed", 0, 0),
            Baseline: new BaselineInfo(
                "baseline-report",
                "smoke",
                false,
                false,
                "not eligible"),
            Validation: new ValidationInfo("passed", "smoke", true, true, false, false, true),
            Notes: ["test"]);

        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.BenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.Equal("generated-uniform", root.GetProperty("dataset").GetProperty("kind").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Equal("absent", root.GetProperty("measurement").GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal(1, root.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal(0, root.GetProperty("search").GetProperty("runs")[0].GetProperty("managedAllocatedBytes").GetInt64());
        Assert.Equal(0, root.GetProperty("search").GetProperty("runs")[0].GetProperty("managedAllocatedBytesPerQuery").GetDouble());
        Assert.Equal(1, root.GetProperty("search").GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal(0, root.GetProperty("search").GetProperty("aggregate").GetProperty("meanManagedAllocatedBytes").GetDouble());
        Assert.Equal(0, root.GetProperty("search").GetProperty("aggregate").GetProperty("minManagedAllocatedBytes").GetInt64());
        Assert.Equal(0, root.GetProperty("search").GetProperty("aggregate").GetProperty("maxManagedAllocatedBytes").GetInt64());
        Assert.Equal(1.0, root.GetProperty("metrics").GetProperty("recallAtK").GetDouble());
        Assert.Equal("baseline-report", root.GetProperty("baseline").GetProperty("baselineReportId").GetString());
        Assert.Equal("smoke", root.GetProperty("baseline").GetProperty("suitability").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
    }
}
