using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec019IndependentTests
{
    [Fact]
    public void GeneratedExactReport_EligibleJsonIncludesCandidateMetadataAndNoComparisonFields()
    {
        BenchmarkReport report = CreateEligibleReport();
        string json = ReportWriter.Serialize(report);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement baseline = root.GetProperty("baseline");
        JsonElement eligibility = baseline.GetProperty("candidateEligibility");

        Assert.Equal("private-baseline-candidate", baseline.GetProperty("suitability").GetString());
        Assert.True(baseline.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(baseline.GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());

        Assert.True(eligibility.GetProperty("eligible").GetBoolean());
        Assert.Equal("generated-exact-report", eligibility.GetProperty("artifactKind").GetString());
        Assert.Equal("D-038 private generated exact baseline candidate policy", eligibility.GetProperty("policy").GetString());
        Assert.Equal(3, eligibility.GetProperty("minimumRuns").GetInt32());
        Assert.Equal(100, eligibility.GetProperty("minimumMeasuredQueries").GetInt32());
        Assert.Equal(JsonValueKind.Array, eligibility.GetProperty("satisfiedConditions").ValueKind);
        Assert.Empty(eligibility.GetProperty("unsatisfiedConditions").EnumerateArray());
        AssertContainsStringElement(eligibility.GetProperty("satisfiedConditions"), "run count is at least 3");
        AssertContainsStringElement(eligibility.GetProperty("satisfiedConditions"), "measured query count is at least 100");

        AssertNoForbiddenComparisonFields(root);
    }

    [Fact]
    public void GeneratedExactReport_MinimumRunAndQueryCountsAreRequired()
    {
        BenchmarkReport twoRunReport = ApplyCleanEligibility(CreateReport(queryCount: 100, runs: 2));
        BenchmarkReport smallQueryReport = ApplyCleanEligibility(CreateReport(queryCount: 99, runs: 3));

        Assert.False(twoRunReport.Baseline.BaselineCandidateEligible);
        Assert.Contains("run count is at least 3", twoRunReport.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        Assert.DoesNotContain("measured query count is at least 100", twoRunReport.Baseline.CandidateEligibility.UnsatisfiedConditions);

        Assert.False(smallQueryReport.Baseline.BaselineCandidateEligible);
        Assert.Contains("measured query count is at least 100", smallQueryReport.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        Assert.DoesNotContain("run count is at least 3", smallQueryReport.Baseline.CandidateEligibility.UnsatisfiedConditions);
    }

    [Fact]
    public void GeneratedExactReport_InvalidRepositoryAndValidationFieldsRemainIneligible()
    {
        var cases = new (string ExpectedCondition, Func<BenchmarkReport, BenchmarkReport> Mutate)[]
        {
            ("repository commit is present", report => report with { Repository = CleanRepository() with { Commit = string.Empty } }),
            ("repository working tree is clean", report => report with { Repository = CleanRepository() with { Dirty = true } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Validation = report.Validation with { Status = "failed" } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Metrics = report.Metrics with { RecallAtK = 0.99 } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Metrics = report.Metrics with { OrderedAgreement = 0.99 } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Metrics = report.Metrics with { DistanceToleranceStatus = "failed" } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Metrics = report.Metrics with { DistanceMismatchCount = 1 } }),
            ("exact validation passed with perfect recall, ordering and distance agreement", report => report with { Metrics = report.Metrics with { MissingResultCount = 1 } })
        };

        foreach ((string expectedCondition, Func<BenchmarkReport, BenchmarkReport> mutate) in cases)
        {
            BenchmarkReport report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(mutate(CreateEligibleReport()));

            Assert.False(report.Baseline.BaselineCandidateEligible);
            Assert.False(report.Validation.BaselineCandidateEligible);
            Assert.False(report.Baseline.RegressionGateEligible);
            Assert.False(report.Evidence.PublicClaimEligible);
            Assert.Contains(expectedCondition, report.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        }
    }

    [Fact]
    public void GeneratedExactReport_MissingMeasurementMetadataRemainsIneligible()
    {
        var cases = new (string ExpectedCondition, Func<BenchmarkReport, BenchmarkReport> Mutate)[]
        {
            ("latency metadata is present", report => report with { Measurement = report.Measurement with { Latency = null! } }),
            ("repeated-run metadata is present", report => report with { Measurement = report.Measurement with { RepeatedRuns = null! } }),
            ("managed allocations are measured in bytes per query", report => report with { Measurement = report.Measurement with { ManagedAllocations = null! } }),
            ("managed allocations are measured in bytes per query", report => report with { Measurement = report.Measurement with { ManagedAllocations = report.Measurement.ManagedAllocations with { Unit = "bytes" } } }),
            ("resident/process memory is explicitly not measured", report => report with { Measurement = report.Measurement with { Memory = null! } }),
            ("resident/process memory is explicitly not measured", report => report with { Measurement = report.Measurement with { Memory = report.Measurement.Memory with { Status = "measured", Value = "1" } } }),
            ("run-to-run noise metadata is present", report => report with { Measurement = report.Measurement with { RunToRunNoise = null! } })
        };

        foreach ((string expectedCondition, Func<BenchmarkReport, BenchmarkReport> mutate) in cases)
        {
            BenchmarkReport report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(mutate(CreateEligibleReport()));

            Assert.False(report.Baseline.BaselineCandidateEligible);
            Assert.False(report.Validation.BaselineCandidateEligible);
            Assert.Contains(expectedCondition, report.Baseline.CandidateEligibility!.UnsatisfiedConditions);
        }
    }

    [Fact]
    public void StandardMatrixManifest_AllLinkedReportsEligibleAndAllCasesPassedAreRequired()
    {
        GeneratedExactMatrixManifest eligible = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(CreateStandardManifest());

        Assert.True(eligible.Eligibility.BaselineCandidateEligible);
        Assert.False(eligible.Eligibility.PublicClaimEligible);
        Assert.False(eligible.Eligibility.RegressionGateEligible);
        Assert.Empty(eligible.Eligibility.CandidateEligibility!.UnsatisfiedConditions);

        GeneratedExactMatrixManifest failedCase = eligible with
        {
            Aggregate = new GeneratedExactMatrixAggregate(eligible.CaseCount - 1, FailedCaseCount: 1),
            Cases = ReplaceCase(
                eligible.Cases,
                0,
                eligible.Cases[0] with { Status = "failed", ValidationStatus = "failed", ErrorMessage = "synthetic failure" }),
            Eligibility = IneligibleMatrixEligibility()
        };

        failedCase = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(failedCase);

        Assert.False(failedCase.Eligibility.BaselineCandidateEligible);
        Assert.Contains("failed case count is zero", failedCase.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
        Assert.Contains("all cases passed and link to per-case reports", failedCase.Eligibility.CandidateEligibility.UnsatisfiedConditions);
    }

    [Fact]
    public void MatrixManifest_LinkedReportsThatAreUnreadableOrIneligibleKeepManifestIneligible()
    {
        GeneratedExactMatrixManifest unreadable = CreateStandardManifest();
        File.WriteAllText(unreadable.Cases[0].ReportPath, "{not-json");
        unreadable = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(unreadable);

        Assert.False(unreadable.Eligibility.BaselineCandidateEligible);
        Assert.Contains("case 01 linked report is readable schema 0.1 JSON", unreadable.Eligibility.CandidateEligibility!.UnsatisfiedConditions);

        GeneratedExactMatrixManifest ineligible = CreateStandardManifest();
        BenchmarkReport report = ReportWriter.Deserialize<BenchmarkReport>(File.ReadAllText(ineligible.Cases[1].ReportPath))!;
        report = BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report with { Repository = CleanRepository() with { Dirty = true } });
        ReportWriter.Write(report, ineligible.Cases[1].ReportPath);
        ineligible = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(ineligible);

        Assert.False(ineligible.Eligibility.BaselineCandidateEligible);
        Assert.Contains("case 02 linked report is private-baseline-candidate eligible", ineligible.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
    }

    [Fact]
    public void MatrixManifest_ReevaluatesLinkedReportEligibilityInsteadOfTrustingStaleFlags()
    {
        GeneratedExactMatrixManifest manifest = CreateStandardManifest();
        BenchmarkReport staleReport = ReportWriter.Deserialize<BenchmarkReport>(File.ReadAllText(manifest.Cases[0].ReportPath))!;
        Assert.True(staleReport.Baseline.BaselineCandidateEligible);

        staleReport = staleReport with { Repository = CleanRepository() with { Dirty = true } };
        ReportWriter.Write(staleReport, manifest.Cases[0].ReportPath);

        manifest = BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);

        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.Contains("case 01 linked report is private-baseline-candidate eligible", manifest.Eligibility.CandidateEligibility!.UnsatisfiedConditions);
    }

    private static BenchmarkReport CreateEligibleReport() => ApplyCleanEligibility(CreateReport(queryCount: 100, runs: 3));

    private static BenchmarkReport ApplyCleanEligibility(BenchmarkReport report) =>
        BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report with { Repository = CleanRepository() });

    private static BenchmarkReport CreateReport(int queryCount, int runs)
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 32,
            VectorCount: 100,
            QueryCount: queryCount,
            TopK: 10,
            Seed: 0x5EED019A,
            OutputPath: Path.Combine(CreateArtifactDirectory(), "report.json"),
            BaselineReportId: null,
            Runs: runs,
            WarmupQueries: 1);

        return GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
    }

    private static GeneratedExactMatrixManifest CreateStandardManifest()
    {
        string outputDirectory = CreateArtifactDirectory();
        var options = new GeneratedExactMatrixOptions(
            GeneratedExactMatrixOptions.StandardPresetName,
            VectorCount: 100,
            QueryCount: 100,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED019B,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactSearchOptions[] expandedCases = GeneratedExactMatrixScenario.ExpandCases(options);
        BenchmarkReport baseReport = CreateEligibleReport();
        var cases = new GeneratedExactMatrixCaseManifest[expandedCases.Length];

        for (int i = 0; i < expandedCases.Length; i++)
        {
            GeneratedExactSearchOptions caseOptions = expandedCases[i];
            string reportId = string.Create(CultureInfo.InvariantCulture, $"vec019-independent-case-{i + 1:D2}");
            BenchmarkReport report = CreateEligibleCaseReport(baseReport, caseOptions, reportId);
            ReportWriter.Write(report, caseOptions.OutputPath);

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
            PresetName: GeneratedExactMatrixOptions.StandardPresetName,
            GeneratedAtUtc: DateTimeOffset.UnixEpoch,
            Repository: CleanRepository(),
            Runner: new RunnerInfo("VecNet.BenchmarkRunner", "0.1", ["exact-generated-matrix"]),
            OutputDirectory: outputDirectory,
            CaseCount: cases.Length,
            Cases: cases,
            Aggregate: new GeneratedExactMatrixAggregate(cases.Length, FailedCaseCount: 0),
            Eligibility: IneligibleMatrixEligibility(),
            Notes: ["independent test manifest"]);
    }

    private static BenchmarkReport CreateEligibleCaseReport(
        BenchmarkReport baseReport,
        GeneratedExactSearchOptions caseOptions,
        string reportId)
    {
        SearchRunInfo[] runs = baseReport.Search.Runs
            .Select(run => run with { MeasuredQueryCount = caseOptions.QueryCount })
            .ToArray();

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
                Aggregate = baseReport.Search.Aggregate with { MeasuredQueryCountPerRun = caseOptions.QueryCount }
            },
            Measurement = baseReport.Measurement with
            {
                Warmup = new WarmupInfo("executed", caseOptions.WarmupQueries, "independent test warmup")
            }
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static GeneratedExactMatrixCaseManifest[] ReplaceCase(
        GeneratedExactMatrixCaseManifest[] cases,
        int index,
        GeneratedExactMatrixCaseManifest replacement)
    {
        GeneratedExactMatrixCaseManifest[] copy = cases.ToArray();
        copy[index] = replacement;
        return copy;
    }

    private static GeneratedExactMatrixEligibility IneligibleMatrixEligibility() =>
        new(
            "local-evidence",
            "private-raw",
            "smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "pending independent eligibility evaluation");

    private static RepositoryInfo CleanRepository() => new("abc123", "main", Dirty: false);

    private static string CreateArtifactDirectory()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec019-independent-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static void AssertContainsStringElement(JsonElement array, string expected)
    {
        Assert.Contains(
            array.EnumerateArray(),
            item => string.Equals(item.GetString(), expected, StringComparison.Ordinal));
    }

    private static void AssertNoForbiddenComparisonFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baselineReportPath",
            "baselineComparison",
            "comparisonArtifact",
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
            "warning",
            "warningClassification",
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
                Assert.False(disallowed, string.Create(CultureInfo.InvariantCulture, $"Unexpected comparison/regression field '{property.Name}' was present."));
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
