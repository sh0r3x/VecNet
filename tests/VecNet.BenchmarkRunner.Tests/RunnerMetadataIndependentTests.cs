using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class RunnerMetadataIndependentTests
{
    [Fact]
    public void Run_WhenBaselineReportIdOmitted_SerializesNullBaselineAndIneligibleSmokeMetadata()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 9,
            VectorCount: 13,
            QueryCount: 4,
            TopK: 5,
            Seed: 0x5EED011A,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/independent-omitted.json",
            BaselineReportId: null);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VEC-012", root.GetProperty("taskId").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("baseline").GetProperty("baselineReportId").ValueKind);
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());

        AssertMeasurementAbsence(root);
    }

    [Fact]
    public void Run_WhenBaselineReportIdProvided_RecordsMetadataOnlyWithoutComparisonFields()
    {
        const string baselineReportId = "baseline-private-20260603";
        var options = new GeneratedExactSearchOptions(
            VectorMetric.InnerProduct,
            Dimension: 7,
            VectorCount: 19,
            QueryCount: 6,
            TopK: 3,
            Seed: 0x5EED011B,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/independent-provided.json",
            BaselineReportId: baselineReportId);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--baseline-report-id",
                baselineReportId
            ]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        JsonElement baseline = root.GetProperty("baseline");
        Assert.Equal(baselineReportId, baseline.GetProperty("baselineReportId").GetString());
        Assert.Equal("smoke", baseline.GetProperty("suitability").GetString());
        Assert.False(baseline.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(baseline.GetProperty("regressionGateEligible").GetBoolean());
        Assert.Contains("not implemented", baseline.GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);

        AssertNoPropertyNamed(
            root,
            "baselineReportPath",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "qpsRatio",
            "regressionPassed",
            "regressionThreshold");
    }

    [Fact]
    public void Run_DoesNotCopyOutputPathIntoDerivedMetadataFields()
    {
        const string sensitiveOutputPath = @"C:\private\owner\runner-output\secret-report.json";
        var options = new GeneratedExactSearchOptions(
            VectorMetric.Cosine,
            Dimension: 5,
            VectorCount: 11,
            QueryCount: 3,
            TopK: 4,
            Seed: 0x5EED011C,
            OutputPath: sensitiveOutputPath,
            BaselineReportId: "baseline-id");

        BenchmarkReport report = GeneratedExactSearchScenario.Run(
            options,
            [
                "exact-generated",
                "--output",
                sensitiveOutputPath
            ]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.DoesNotContain("secret-report", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("runner-output", report.ReportId, StringComparison.OrdinalIgnoreCase);
        AssertDoesNotContainText(root.GetProperty("evidence"), sensitiveOutputPath);
        AssertDoesNotContainText(root.GetProperty("measurement"), sensitiveOutputPath);
        AssertDoesNotContainText(root.GetProperty("baseline"), sensitiveOutputPath);
        AssertDoesNotContainText(root.GetProperty("validation"), sensitiveOutputPath);
        AssertDoesNotContainText(root.GetProperty("notes"), sensitiveOutputPath);
    }

    [Theory]
    [InlineData("exact-generated", "--baseline-report-id")]
    [InlineData("exact-generated", "--baseline-report-id", " ")]
    [InlineData("exact-generated", "--baseline-report-id", "   ")]
    public void Parse_RejectsBaselineReportIdWithoutUsableValue(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.Parse(args));

        Assert.Contains("baseline-report-id", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_WithMultipleRuns_SerializesMeasuredVarianceAndWarmupMetadata()
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 8,
            VectorCount: 16,
            QueryCount: 5,
            TopK: 4,
            Seed: 0x5EED012B,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/independent-multi-run.json",
            BaselineReportId: null,
            Runs: 2,
            WarmupQueries: 3);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement repeatedRuns = measurement.GetProperty("repeatedRuns");
        JsonElement warmup = measurement.GetProperty("warmup");

        Assert.Equal("measured", repeatedRuns.GetProperty("status").GetString());
        Assert.Equal(2, repeatedRuns.GetProperty("runCount").GetInt32());
        Assert.True(repeatedRuns.GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("executed", warmup.GetProperty("status").GetString());
        Assert.Equal(3, warmup.GetProperty("warmupCount").GetInt32());

        JsonElement search = root.GetProperty("search");
        Assert.Equal(2, search.GetProperty("runs").GetArrayLength());
        Assert.Equal(2, search.GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal(5, search.GetProperty("aggregate").GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
    }

    private static void AssertMeasurementAbsence(JsonElement root)
    {
        JsonElement measurement = root.GetProperty("measurement");
        AssertMeasurementStatus(
            measurement.GetProperty("managedAllocations"),
            expectedUnit: "bytesPerOperation");
        AssertMeasurementStatus(
            measurement.GetProperty("memory"),
            expectedUnit: "bytes");

        JsonElement repeatedRuns = measurement.GetProperty("repeatedRuns");
        Assert.Equal("singleRun", repeatedRuns.GetProperty("status").GetString());
        Assert.Equal(1, repeatedRuns.GetProperty("runCount").GetInt32());
        Assert.False(repeatedRuns.GetProperty("varianceMeasured").GetBoolean());
        Assert.Contains("variance", repeatedRuns.GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement warmup = measurement.GetProperty("warmup");
        Assert.Equal("absent", warmup.GetProperty("status").GetString());
        Assert.Equal(0, warmup.GetProperty("warmupCount").GetInt32());
        Assert.Contains("warmup", warmup.GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertMeasurementStatus(JsonElement status, string expectedUnit)
    {
        Assert.Equal("notMeasured", status.GetProperty("status").GetString());
        Assert.Equal("absent", status.GetProperty("value").GetString());
        Assert.Equal(expectedUnit, status.GetProperty("unit").GetString());
        Assert.False(string.IsNullOrEmpty(status.GetProperty("reason").GetString()));
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, $"Unexpected comparison field '{property.Name}' was present.");
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

    private static void AssertDoesNotContainText(JsonElement element, string sensitiveText)
    {
        if (element.ValueKind == JsonValueKind.String)
        {
            Assert.DoesNotContain(
                sensitiveText,
                element.GetString() ?? string.Empty,
                StringComparison.OrdinalIgnoreCase);
        }
        else if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                AssertDoesNotContainText(property.Value, sensitiveText);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertDoesNotContainText(item, sensitiveText);
            }
        }
    }
}
