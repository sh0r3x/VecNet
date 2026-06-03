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
            TaskId: "VEC-010",
            ClaimClass: "local-evidence",
            PrivacyClass: "private-raw",
            Repository: new RepositoryInfo("abc123", "main", Dirty: true),
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", ["exact-generated"]),
            Command: new CommandInfo("exact-generated", ["exact-generated"]),
            Environment: new EnvironmentInfo(".NET test OS", "X64", ".NET 10", "win-x64", 8, false, 8),
            Dataset: new DatasetInfo("generated-uniform", "generated-no-external-source", "uniform[-1,1)", "0x5EED2009", "SquaredEuclidean", 2, 3, 1),
            Truth: new TruthInfo("scalar-reference-generated", 2, ScalarGroundTruth.TiePolicy),
            Scenario: new ScenarioInfo("exact-generated", 2, 1, 1, "setup excluded"),
            Index: new IndexInfo("Exact", "ExactFlatIndex", "SquaredEuclidean", 2, 3, "public default"),
            Search: new SearchInfo(1, 0.5, 0.5, 0.5, 0.5, 2000),
            Metrics: new MetricsInfo(1, 1, "passed", 0, 0),
            Validation: new ValidationInfo("passed", true, true, true),
            Notes: ["test"]);

        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.BenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-010", root.GetProperty("taskId").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("generated-uniform", root.GetProperty("dataset").GetProperty("kind").GetString());
        Assert.Equal(1.0, root.GetProperty("metrics").GetProperty("recallAtK").GetDouble());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
    }
}
