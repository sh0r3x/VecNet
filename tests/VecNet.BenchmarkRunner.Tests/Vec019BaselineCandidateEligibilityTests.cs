using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec019BaselineCandidateEligibilityTests
{
    [Fact]
    public void GeneratedExactReport_WhenD038MinimumSatisfied_MarksPrivateBaselineCandidateOnly()
    {
        BenchmarkReport report = CreateReport(VectorMetric.SquaredEuclidean, queryCount: 100, runs: 3);
        report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(
            report with { Repository = CleanRepository() });

        Assert.True(report.Baseline.BaselineCandidateEligible);
        Assert.True(report.Validation.BaselineCandidateEligible);
        Assert.Equal("private-baseline-candidate", report.Baseline.Suitability);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Baseline.RegressionGateEligible);
        Assert.True(report.Baseline.CandidateEligibility?.Eligible);
        Assert.Empty(report.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        Assert.Contains(
            "run count is at least 3",
            report.Baseline.CandidateEligibility.SatisfiedConditions);
        Assert.Contains(
            "measured query count is at least 100",
            report.Baseline.CandidateEligibility.SatisfiedConditions);

        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(report));
        JsonElement root = document.RootElement;
        Assert.True(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.True(root.GetProperty("baseline").GetProperty("candidateEligibility").GetProperty("eligible").GetBoolean());
    }

    [Fact]
    public void GeneratedExactReport_WhenSingleRunAndSmallQuery_RemainsIneligibleWithReasons()
    {
        BenchmarkReport report = CreateReport(VectorMetric.InnerProduct, queryCount: 99, runs: 1);
        report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(
            report with { Repository = CleanRepository() });

        Assert.False(report.Baseline.BaselineCandidateEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.Equal("smoke", report.Baseline.Suitability);
        Assert.False(report.Baseline.RegressionGateEligible);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.Contains(
            "run count is at least 3",
            report.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        Assert.Contains(
            "measured query count is at least 100",
            report.Baseline.CandidateEligibility.UnsatisfiedConditions);
    }

    [Fact]
    public void GeneratedExactReport_WhenRepositoryIsDirty_RemainsIneligible()
    {
        BenchmarkReport report = CreateReport(VectorMetric.Cosine, queryCount: 100, runs: 3);
        report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(
            report with { Repository = CleanRepository() with { Dirty = true } });

        Assert.False(report.Baseline.BaselineCandidateEligible);
        Assert.Contains(
            "repository working tree is clean",
            report.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        Assert.False(report.Baseline.RegressionGateEligible);
        Assert.False(report.Evidence.PublicClaimEligible);
    }

    [Fact]
    public void StandardMatrixManifest_WhenAllLinkedReportsAreEligible_MarksPrivateBaselineCandidateOnly()
    {
        string outputDirectory = CreateArtifactDirectory();
        GeneratedExactMatrixManifest manifest = CreateMatrixManifest(
            outputDirectory,
            GeneratedExactMatrixOptions.StandardPresetName);

        manifest = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);

        Assert.True(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.True(manifest.Eligibility.CandidateEligibility?.Eligible);
        Assert.Empty(manifest.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
        Assert.Contains(
            "preset is standard",
            manifest.Eligibility.CandidateEligibility.SatisfiedConditions);
        Assert.Contains(
            "standard matrix case set is canonical",
            manifest.Eligibility.CandidateEligibility.SatisfiedConditions);

        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(manifest));
        JsonElement eligibility = document.RootElement.GetProperty("eligibility");
        Assert.True(eligibility.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(eligibility.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(eligibility.GetProperty("regressionGateEligible").GetBoolean());
        Assert.True(eligibility.GetProperty("candidateEligibility").GetProperty("eligible").GetBoolean());
    }

    [Fact]
    public void SmokeMatrixManifest_RemainsBaselineCandidateIneligible()
    {
        string outputDirectory = CreateArtifactDirectory();
        GeneratedExactMatrixManifest manifest = CreateMatrixManifest(
            outputDirectory,
            GeneratedExactMatrixOptions.SmokePresetName);

        manifest = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);

        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Contains(
            "preset is standard",
            manifest.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
    }

    [Fact]
    public void StandardMatrixManifest_WhenLinkedReportIsMissing_RemainsIneligible()
    {
        string outputDirectory = CreateArtifactDirectory();
        GeneratedExactMatrixManifest manifest = CreateMatrixManifest(
            outputDirectory,
            GeneratedExactMatrixOptions.StandardPresetName);
        File.Delete(manifest.Cases[0].ReportPath);

        manifest = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);

        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Contains(
            "case 01 linked report exists",
            manifest.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
    }

    [Fact]
    public void StandardMatrixManifest_ReevaluatesLinkedReportInsteadOfTrustingStaleEligibilityFlags()
    {
        string outputDirectory = CreateArtifactDirectory();
        GeneratedExactMatrixManifest manifest = CreateMatrixManifest(
            outputDirectory,
            GeneratedExactMatrixOptions.StandardPresetName);

        BenchmarkReport staleReport = ReportWriter.Deserialize<BenchmarkReport>(
            File.ReadAllText(manifest.Cases[0].ReportPath))!;
        Assert.True(staleReport.Baseline.BaselineCandidateEligible);
        Assert.True(staleReport.Validation.BaselineCandidateEligible);
        Assert.True(staleReport.Baseline.CandidateEligibility?.Eligible);

        staleReport = staleReport with { Repository = CleanRepository() with { Dirty = true } };
        ReportWriter.Write(staleReport, manifest.Cases[0].ReportPath);

        manifest = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);

        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Contains(
            "case 01 linked report is private-baseline-candidate eligible",
            manifest.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
    }

    private static BenchmarkReport CreateReport(VectorMetric metric, int queryCount, int runs)
    {
        var options = new GeneratedExactSearchOptions(
            metric,
            Dimension: 32,
            VectorCount: 100,
            QueryCount: queryCount,
            TopK: 10,
            Seed: 0x5EED0190,
            OutputPath: Path.Combine(CreateArtifactDirectory(), "report.json"),
            BaselineReportId: null,
            Runs: runs,
            WarmupQueries: 1);

        return GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
    }

    private static GeneratedExactMatrixManifest CreateMatrixManifest(string outputDirectory, string presetName)
    {
        var options = new GeneratedExactMatrixOptions(
            presetName,
            VectorCount: 100,
            QueryCount: 100,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0191,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));
        GeneratedExactSearchOptions[] expandedCases = GeneratedExactMatrixScenario.ExpandCases(options);
        BenchmarkReport baseReport = CreateReport(VectorMetric.SquaredEuclidean, queryCount: 100, runs: 3);

        var cases = new GeneratedExactMatrixCaseManifest[expandedCases.Length];
        for (int i = 0; i < expandedCases.Length; i++)
        {
            GeneratedExactSearchOptions caseOptions = expandedCases[i];
            string reportId = string.Create(CultureInfo.InvariantCulture, $"vec019-case-{i + 1:D2}");
            BenchmarkReport caseReport = CreateEligibleCaseReport(baseReport, caseOptions, reportId);
            ReportWriter.Write(caseReport, caseOptions.OutputPath);

            cases[i] = new GeneratedExactMatrixCaseManifest(
                i + 1,
                caseOptions.Metric.ToString(),
                caseOptions.Dimension,
                caseOptions.VectorCount,
                caseOptions.QueryCount,
                caseOptions.TopK,
                caseOptions.Runs,
                caseOptions.WarmupQueries,
                string.Create(CultureInfo.InvariantCulture, $"0x{caseOptions.Seed:X8}"),
                caseOptions.OutputPath,
                reportId,
                "passed",
                "passed",
                ErrorMessage: null);
        }

        return new GeneratedExactMatrixManifest(
            SchemaName: "VecNet.BenchmarkMatrixManifest",
            SchemaVersion: "0.1",
            TaskId: "VEC-015",
            ScenarioName: GeneratedExactMatrixOptions.ScenarioName,
            PresetName: presetName,
            GeneratedAtUtc: DateTimeOffset.UnixEpoch,
            Repository: CleanRepository(),
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", ["exact-generated-matrix"]),
            OutputDirectory: outputDirectory,
            CaseCount: cases.Length,
            Cases: cases,
            Aggregate: new GeneratedExactMatrixAggregate(cases.Length, FailedCaseCount: 0),
            Eligibility: new GeneratedExactMatrixEligibility(
                "local-evidence",
                "private-raw",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "pending VEC-019 eligibility evaluation"),
            Notes: ["test manifest"]);
    }

    private static BenchmarkReport CreateEligibleCaseReport(
        BenchmarkReport baseReport,
        GeneratedExactSearchOptions caseOptions,
        string reportId)
    {
        SearchRunInfo[] runs = baseReport.Search.Runs
            .Select(run => run with { MeasuredQueryCount = caseOptions.QueryCount })
            .ToArray();
        AggregateTimingInfo aggregate = baseReport.Search.Aggregate with
        {
            MeasuredQueryCountPerRun = caseOptions.QueryCount
        };

        BenchmarkReport report = baseReport with
        {
            ReportId = reportId,
            Repository = CleanRepository(),
            Dataset = baseReport.Dataset with
            {
                Metric = caseOptions.Metric.ToString(),
                Dimension = caseOptions.Dimension,
                VectorCount = caseOptions.VectorCount,
                QueryCount = caseOptions.QueryCount
            },
            Truth = baseReport.Truth with { Depth = caseOptions.TopK },
            Scenario = baseReport.Scenario with
            {
                TopK = caseOptions.TopK,
                MeasuredQueryCount = caseOptions.QueryCount
            },
            Index = baseReport.Index with
            {
                Metric = caseOptions.Metric.ToString(),
                Dimension = caseOptions.Dimension,
                VectorCount = caseOptions.VectorCount
            },
            Search = baseReport.Search with
            {
                MeasuredQueryCount = caseOptions.QueryCount,
                Runs = runs,
                Aggregate = aggregate
            },
            Measurement = baseReport.Measurement with
            {
                Warmup = new WarmupInfo("executed", caseOptions.WarmupQueries, "test warmup")
            }
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static RepositoryInfo CleanRepository() => new("abc123", "main", Dirty: false);

    private static string CreateArtifactDirectory()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec019-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }
}
