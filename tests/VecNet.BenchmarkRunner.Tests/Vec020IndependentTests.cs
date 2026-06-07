using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec020IndependentTests
{
    [Fact]
    public void CompatibleReportComparisonJson_HasPrivateWarningOnlyShapeAndFalseArtifactEligibility()
    {
        string directory = CreateArtifactDirectory("report-json");
        string baselinePath = Path.Combine(directory, "baseline.json");
        string currentPath = Path.Combine(directory, "current.json");
        string outputPath = Path.Combine(directory, "comparison.json");
        ReportWriter.Write(WithSearchMetrics(CreateEligibleReport("baseline"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.02)), baselinePath);
        ReportWriter.Write(WithSearchMetrics(CreateEligibleReport("current"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.02)), currentPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselinePath, currentPath, outputPath),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(comparison, outputPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.BenchmarkComparison", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("generated-exact-report-comparison", root.GetProperty("artifactKind").GetString());
        Assert.Equal("comparable", root.GetProperty("compatibility").GetProperty("status").GetString());
        Assert.Equal("warning-only", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("regressionGateEligible").GetBoolean());
        Assert.Contains(root.GetProperty("metrics").EnumerateArray(), metric => metric.GetProperty("name").GetString() == "search.aggregate.meanElapsedMilliseconds");
        AssertNoForbiddenGateProperties(root);
    }

    [Fact]
    public void IncompatibleReportComparison_EmitsPreciseNotComparableDimensionReason()
    {
        BenchmarkReport baseline = CreateEligibleReport("baseline");
        BenchmarkReport current = CreateEligibleReport("current") with
        {
            Dataset = baseline.Dataset with { Dimension = 386 },
            Index = baseline.Index with { Dimension = 386 }
        };

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Metrics);
        CompatibilityReason dimensionReason = Assert.Single(
            comparison.Compatibility.Reasons,
            reason => reason.Code == "fieldMismatch" && reason.Field == "dataset.dimension");
        Assert.Equal("32", dimensionReason.Expected);
        Assert.Equal("386", dimensionReason.Actual);
        Assert.Contains("notComparable", comparison.Warnings.Labels);
    }

    [Fact]
    public void CompatibleStandardMatrixComparison_HasStablePerCaseCountsAndSortedCaseKeys()
    {
        string directory = CreateArtifactDirectory("matrix-stable");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current");

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("generated-exact-standard-matrix-comparison", comparison.ArtifactKind);
        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Equal(36, comparison.Cases.Length);
        Assert.Equal(36, comparison.MatrixSummary!.ComparableCaseCount);
        Assert.Equal(0, comparison.MatrixSummary.NotComparableCaseCount);
        Assert.Equal(comparison.Cases.Select(item => item.CaseKey).Order(StringComparer.Ordinal), comparison.Cases.Select(item => item.CaseKey));
        Assert.Equal(36, comparison.Cases.Select(item => item.CaseKey).Distinct(StringComparer.Ordinal).Count());
        Assert.Contains(comparison.Cases, item => item.CaseKey.StartsWith("Cosine:768:100:100:100:3:1:", StringComparison.Ordinal));
        Assert.Contains(comparison.Cases, item => item.CaseKey.StartsWith("SquaredEuclidean:32:100:100:1:3:1:", StringComparison.Ordinal));
        Assert.All(comparison.Cases, item => Assert.Equal("comparable", item.Compatibility.Status));
    }

    [Fact]
    public void MatrixComparison_WithMissingAndInvalidLinkedCurrentReports_IsPartiallyComparable()
    {
        string directory = CreateArtifactDirectory("matrix-partial");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current");
        File.Delete(current.Cases[0].ReportPath);
        File.WriteAllText(current.Cases[1].ReportPath, "{not-json");

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("partiallyComparable", comparison.Compatibility.Status);
        Assert.Equal(34, comparison.MatrixSummary!.ComparableCaseCount);
        Assert.Equal(2, comparison.MatrixSummary.NotComparableCaseCount);
        Assert.Contains(comparison.Cases, item => item.Compatibility.Reasons.Any(reason => reason.Code == "missingLinkedReport"));
        Assert.Contains(comparison.Cases, item => item.Compatibility.Reasons.Any(reason => reason.Code == "unreadableLinkedReport"));
    }

    [Fact]
    public void CurrentMatrixWithInvalidPrivateLocalPosture_IsNotComparableBeforeLinkedCases()
    {
        string directory = CreateArtifactDirectory("matrix-posture");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current") with
        {
            Eligibility = new GeneratedExactMatrixEligibility(
                "local-evidence",
                "public-summary",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: true,
                RegressionGateEligible: false,
                "synthetic stale current posture")
        };

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Cases);
        CompatibilityReason reason = Assert.Single(comparison.Compatibility.Reasons, item => item.Field == "current.matrixCandidateEligibility");
        Assert.Contains("manifest is private local evidence", reason.Actual);
    }

    [Fact]
    public void DirtyCurrentReportAndMatrix_StayWarningOnlyAndDoNotEnableGateEligibility()
    {
        BenchmarkReport dirtyCurrentReport = CreateEligibleReport("current") with
        {
            Repository = CleanRepository() with { Dirty = true }
        };
        BenchmarkComparisonArtifact reportComparison = CompareReports(CreateEligibleReport("baseline"), dirtyCurrentReport);

        string directory = CreateArtifactDirectory("dirty-matrix");
        GeneratedExactMatrixManifest dirtyCurrentMatrix = CreateStandardManifest(Path.Combine(directory, "current"), "current") with
        {
            Repository = CleanRepository() with { Dirty = true }
        };
        BenchmarkComparisonArtifact matrixComparison = CompareMatrices(
            CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline"),
            dirtyCurrentMatrix);

        Assert.Equal("comparable", reportComparison.Compatibility.Status);
        Assert.Equal("comparable", matrixComparison.Compatibility.Status);
        Assert.Contains("dirtyCurrentWarning", reportComparison.Warnings.Labels);
        Assert.Contains("dirtyCurrentWarning", matrixComparison.Warnings.Labels);
        Assert.False(reportComparison.PublicClaimEligible);
        Assert.False(reportComparison.BaselineCandidateEligible);
        Assert.False(reportComparison.RegressionGateEligible);
        Assert.False(matrixComparison.PublicClaimEligible);
        Assert.False(matrixComparison.BaselineCandidateEligible);
        Assert.False(matrixComparison.RegressionGateEligible);
    }

    [Fact]
    public void AvailableNoiseUsesCvStddevSpreadMaxSideNoiseAndDenominatorFloors()
    {
        NoiseProfile baselineNoise = CustomNoise(
            elapsed: MetricNoise("milliseconds", 0, stddev: 1e-13, cv: null, spread: 2e-14),
            qps: MetricNoise("queriesPerSecond", 1000, stddev: 5, cv: 0.02, spread: 10),
            p50: MetricNoise("milliseconds", 1, stddev: 0.01, cv: 0.01, spread: 0.60),
            p95: MetricNoise("milliseconds", 2, stddev: 0.02, cv: 0.02, spread: 0.04),
            p99: MetricNoise("milliseconds", 3, stddev: 0.03, cv: 0.03, spread: 0.06),
            allocation: MetricNoise("bytesPerQuery", 0, stddev: 0.4, cv: null, spread: 0.2));
        NoiseProfile currentNoise = CustomNoise(
            elapsed: MetricNoise("milliseconds", 0, stddev: 2e-13, cv: null, spread: 1e-14),
            qps: MetricNoise("queriesPerSecond", 1000, stddev: 250, cv: 0.01, spread: 30),
            p50: MetricNoise("milliseconds", 1, stddev: 0.02, cv: 0.02, spread: 0.03),
            p95: MetricNoise("milliseconds", 2, stddev: 0.04, cv: 0.04, spread: 0.80),
            p99: MetricNoise("milliseconds", 3, stddev: 0.03, cv: 0.03, spread: 0.06),
            allocation: MetricNoise("bytesPerQuery", 0, stddev: 0.7, cv: null, spread: 0.1));

        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), 100, 1000, 1, 2, 3, 0, baselineNoise);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), 112, 880, 1.12, 2.24, 3.36, 0, currentNoise);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        AssertClose(0.20, Metric(comparison, "search.aggregate.meanElapsedMilliseconds").AvailableNoise.Fraction);
        AssertClose(0.25, Metric(comparison, "search.aggregate.meanQps").AvailableNoise.Fraction);
        AssertClose(0.60, Metric(comparison, "search.aggregate.meanLatencyP50Milliseconds").AvailableNoise.Fraction);
        AssertClose(0.40, Metric(comparison, "search.aggregate.meanLatencyP95Milliseconds").AvailableNoise.Fraction);
        AssertClose(0.70, Metric(comparison, "search.aggregate.meanManagedAllocatedBytesPerQuery").AvailableNoise.Fraction);
    }

    [Fact]
    public void WorseBeyondThresholdNoiseDominatedAndNoiseUnavailable_AreDistinctLabels()
    {
        BenchmarkComparisonArtifact beyondThreshold = CompareReports(
            WithSearchMetrics(CreateEligibleReport("baseline-a"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.01)),
            WithSearchMetrics(CreateEligibleReport("current-a"), 112, 880, 1, 2, 3, 0, UniformNoise(0.01)));
        BenchmarkComparisonArtifact noiseDominated = CompareReports(
            WithSearchMetrics(CreateEligibleReport("baseline-b"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.20)),
            WithSearchMetrics(CreateEligibleReport("current-b"), 112, 880, 1, 2, 3, 0, UniformNoise(0.20)));
        BenchmarkComparisonArtifact noiseUnavailable = CompareReports(
            WithNoiseStatus(WithSearchMetrics(CreateEligibleReport("baseline-c"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.01)), "notMeasured"),
            WithNoiseStatus(WithSearchMetrics(CreateEligibleReport("current-c"), 112, 880, 1, 2, 3, 0, UniformNoise(0.01)), "notMeasured"));

        Assert.Equal("performanceWarning", Metric(beyondThreshold, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.Equal("noiseDominated", Metric(noiseDominated, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.Equal("noiseUnavailable", Metric(noiseUnavailable, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.DoesNotContain("performanceWarning", noiseUnavailable.Warnings.Labels);
    }

    [Fact]
    public void ZeroBaselineNumericHandling_CoversAllocationAndHigherIsBetterRatios()
    {
        BenchmarkComparisonArtifact allocationComparison = CompareReports(
            WithSearchMetrics(CreateEligibleReport("baseline-allocation"), 100, 1000, 1, 2, 3, 0, UniformNoise(0.01)),
            WithSearchMetrics(CreateEligibleReport("current-allocation"), 100, 1000, 1, 2, 3, 5, UniformNoise(0.01)));
        BenchmarkComparisonArtifact qpsComparison = CompareReports(
            WithSearchMetrics(CreateEligibleReport("baseline-qps"), 100, 0, 1, 2, 3, 0, UniformNoise(0.01)),
            WithSearchMetrics(CreateEligibleReport("current-qps"), 100, 10, 1, 2, 3, 0, UniformNoise(0.01)));

        MetricComparisonEntry allocation = Metric(allocationComparison, "search.aggregate.meanManagedAllocatedBytesPerQuery");
        MetricComparisonEntry qps = Metric(qpsComparison, "search.aggregate.meanQps");

        Assert.Equal("allocationWarning", allocation.WarningLabel);
        Assert.True(double.IsPositiveInfinity(allocation.Ratio!.Value));
        Assert.True(double.IsPositiveInfinity(allocation.PercentChange!.Value));
        Assert.Equal("notComparable", qps.WarningLabel);
        Assert.Null(qps.Ratio);
        Assert.Null(qps.PercentChange);
    }

    [Fact]
    public void ComparisonRun_DoesNotMutateReportOrManifestInputs()
    {
        string directory = CreateArtifactDirectory("immutability");
        string baselineReportPath = Path.Combine(directory, "baseline-report.json");
        string currentReportPath = Path.Combine(directory, "current-report.json");
        string reportOutputPath = Path.Combine(directory, "report-comparison.json");
        ReportWriter.Write(CreateEligibleReport("baseline"), baselineReportPath);
        ReportWriter.Write(CreateEligibleReport("current"), currentReportPath);
        string baselineReportBefore = File.ReadAllText(baselineReportPath);
        string currentReportBefore = File.ReadAllText(currentReportPath);

        BenchmarkComparisonArtifact reportComparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselineReportPath, currentReportPath, reportOutputPath),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(reportComparison, reportOutputPath);

        string matrixBaselineDir = Path.Combine(directory, "baseline-matrix");
        string matrixCurrentDir = Path.Combine(directory, "current-matrix");
        GeneratedExactMatrixManifest baselineMatrix = CreateStandardManifest(matrixBaselineDir, "baseline");
        GeneratedExactMatrixManifest currentMatrix = CreateStandardManifest(matrixCurrentDir, "current");
        string baselineMatrixPath = Path.Combine(directory, "baseline-manifest.json");
        string currentMatrixPath = Path.Combine(directory, "current-manifest.json");
        string matrixOutputPath = Path.Combine(directory, "matrix-comparison.json");
        GeneratedExactMatrixScenario.WriteManifest(baselineMatrix, baselineMatrixPath);
        GeneratedExactMatrixScenario.WriteManifest(currentMatrix, currentMatrixPath);
        string baselineManifestBefore = File.ReadAllText(baselineMatrixPath);
        string currentManifestBefore = File.ReadAllText(currentMatrixPath);
        string linkedReportBefore = File.ReadAllText(currentMatrix.Cases[0].ReportPath);

        BenchmarkComparisonArtifact matrixComparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselineMatrixPath, currentMatrixPath, matrixOutputPath),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(matrixComparison, matrixOutputPath);

        Assert.Equal(baselineReportBefore, File.ReadAllText(baselineReportPath));
        Assert.Equal(currentReportBefore, File.ReadAllText(currentReportPath));
        Assert.Equal(baselineManifestBefore, File.ReadAllText(baselineMatrixPath));
        Assert.Equal(currentManifestBefore, File.ReadAllText(currentMatrixPath));
        Assert.Equal(linkedReportBefore, File.ReadAllText(currentMatrix.Cases[0].ReportPath));
    }

    private static BenchmarkComparisonArtifact CompareReports(BenchmarkReport baseline, BenchmarkReport current)
    {
        string directory = CreateArtifactDirectory("report");
        string baselinePath = Path.Combine(directory, "baseline.json");
        string currentPath = Path.Combine(directory, "current.json");
        ReportWriter.Write(baseline, baselinePath);
        ReportWriter.Write(current, currentPath);

        return BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselinePath, currentPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact"]);
    }

    private static BenchmarkComparisonArtifact CompareMatrices(GeneratedExactMatrixManifest baseline, GeneratedExactMatrixManifest current)
    {
        string directory = CreateArtifactDirectory("matrix");
        string baselinePath = Path.Combine(directory, "baseline-manifest.json");
        string currentPath = Path.Combine(directory, "current-manifest.json");
        GeneratedExactMatrixScenario.WriteManifest(baseline, baselinePath);
        GeneratedExactMatrixScenario.WriteManifest(current, currentPath);

        return BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselinePath, currentPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact"]);
    }

    private static BenchmarkReport CreateEligibleReport(string reportId)
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 32,
            VectorCount: 100,
            QueryCount: 100,
            TopK: 10,
            Seed: 0x5EED0202,
            OutputPath: Path.Combine(CreateArtifactDirectory("source"), reportId + ".json"),
            BaselineReportId: null,
            Runs: 3,
            WarmupQueries: 1);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        report = report with
        {
            ReportId = reportId,
            Repository = CleanRepository(),
            GeneratedAtUtc = DateTimeOffset.UnixEpoch
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static BenchmarkReport WithSearchMetrics(
        BenchmarkReport report,
        double elapsed,
        double qps,
        double p50,
        double p95,
        double p99,
        double allocation,
        NoiseProfile noiseProfile)
    {
        SearchRunInfo[] runs = Enumerable.Range(1, report.Search.Aggregate.RunCount)
            .Select(run => new SearchRunInfo(run, report.Search.MeasuredQueryCount, elapsed, p50, p95, p99, qps, (long)allocation, allocation))
            .ToArray();

        report = report with
        {
            Search = report.Search with
            {
                ElapsedMilliseconds = elapsed,
                LatencyP50Milliseconds = p50,
                LatencyP95Milliseconds = p95,
                LatencyP99Milliseconds = p99,
                Qps = qps,
                Runs = runs,
                Aggregate = report.Search.Aggregate with
                {
                    MeanElapsedMilliseconds = elapsed,
                    MinElapsedMilliseconds = elapsed,
                    MaxElapsedMilliseconds = elapsed,
                    MeanLatencyP50Milliseconds = p50,
                    MeanLatencyP95Milliseconds = p95,
                    MeanLatencyP99Milliseconds = p99,
                    MeanQps = qps,
                    MinQps = qps,
                    MaxQps = qps,
                    MeanManagedAllocatedBytes = allocation * report.Search.MeasuredQueryCount,
                    MinManagedAllocatedBytes = (long)(allocation * report.Search.MeasuredQueryCount),
                    MaxManagedAllocatedBytes = (long)(allocation * report.Search.MeasuredQueryCount),
                    MeanManagedAllocatedBytesPerQuery = allocation,
                    MinManagedAllocatedBytesPerQuery = allocation,
                    MaxManagedAllocatedBytesPerQuery = allocation
                }
            },
            Measurement = report.Measurement with
            {
                ManagedAllocations = report.Measurement.ManagedAllocations with
                {
                    Value = allocation.ToString(CultureInfo.InvariantCulture)
                },
                RunToRunNoise = new RunToRunNoiseInfo(
                    "measured",
                    report.Search.Aggregate.RunCount,
                    NoiseMeasured: true,
                    "independent synthetic scope",
                    "independent synthetic statistics",
                    "independent synthetic measured noise",
                    "independent synthetic non-goals",
                    noiseProfile.Elapsed,
                    noiseProfile.Qps,
                    noiseProfile.P50,
                    noiseProfile.P95,
                    noiseProfile.P99,
                    noiseProfile.Allocation)
            }
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static BenchmarkReport WithNoiseStatus(BenchmarkReport report, string status)
    {
        RunToRunNoiseInfo noise = report.Measurement.RunToRunNoise;
        return report with
        {
            Measurement = report.Measurement with
            {
                RunToRunNoise = noise with
                {
                    Status = status,
                    NoiseMeasured = status == "measured",
                    ElapsedMilliseconds = noise.ElapsedMilliseconds with { Status = status },
                    Qps = noise.Qps with { Status = status },
                    LatencyP50Milliseconds = noise.LatencyP50Milliseconds with { Status = status },
                    LatencyP95Milliseconds = noise.LatencyP95Milliseconds with { Status = status },
                    LatencyP99Milliseconds = noise.LatencyP99Milliseconds with { Status = status },
                    ManagedAllocatedBytesPerQuery = noise.ManagedAllocatedBytesPerQuery with { Status = status }
                }
            }
        };
    }

    private static NoiseProfile UniformNoise(double fraction) =>
        CustomNoise(
            MetricNoise("milliseconds", 100, 100 * fraction, fraction, 100 * fraction),
            MetricNoise("queriesPerSecond", 1000, 1000 * fraction, fraction, 1000 * fraction),
            MetricNoise("milliseconds", 1, fraction, fraction, fraction),
            MetricNoise("milliseconds", 2, 2 * fraction, fraction, 2 * fraction),
            MetricNoise("milliseconds", 3, 3 * fraction, fraction, 3 * fraction),
            MetricNoise("bytesPerQuery", 10, 10 * fraction, fraction, 10 * fraction));

    private static NoiseProfile CustomNoise(
        RunToRunMetricNoiseInfo elapsed,
        RunToRunMetricNoiseInfo qps,
        RunToRunMetricNoiseInfo p50,
        RunToRunMetricNoiseInfo p95,
        RunToRunMetricNoiseInfo p99,
        RunToRunMetricNoiseInfo allocation) =>
        new(elapsed, qps, p50, p95, p99, allocation);

    private static RunToRunMetricNoiseInfo MetricNoise(
        string unit,
        double mean,
        double? stddev,
        double? cv,
        double? spread) =>
        new(
            "measured",
            unit,
            mean,
            stddev,
            cv,
            mean,
            mean,
            spread,
            "independent synthetic metric noise");

    private static GeneratedExactMatrixManifest CreateStandardManifest(string outputDirectory, string idPrefix)
    {
        Directory.CreateDirectory(outputDirectory);
        var options = new GeneratedExactMatrixOptions(
            GeneratedExactMatrixOptions.StandardPresetName,
            VectorCount: 100,
            QueryCount: 100,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0202,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactSearchOptions[] expandedCases = GeneratedExactMatrixScenario.ExpandCases(options);
        BenchmarkReport baseReport = CreateEligibleReport(idPrefix + "-base");
        var cases = new GeneratedExactMatrixCaseManifest[expandedCases.Length];

        for (int i = 0; i < expandedCases.Length; i++)
        {
            GeneratedExactSearchOptions caseOptions = expandedCases[i];
            string reportId = string.Create(CultureInfo.InvariantCulture, $"{idPrefix}-case-{i + 1:D2}");
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

        GeneratedExactMatrixManifest manifest = new(
            "VecNet.BenchmarkMatrixManifest",
            "0.1",
            "VEC-015",
            GeneratedExactMatrixOptions.ScenarioName,
            GeneratedExactMatrixOptions.StandardPresetName,
            DateTimeOffset.UnixEpoch,
            CleanRepository(),
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", ["exact-generated-matrix"]),
            outputDirectory,
            cases.Length,
            cases,
            new GeneratedExactMatrixAggregate(cases.Length, FailedCaseCount: 0),
            new GeneratedExactMatrixEligibility("local-evidence", "private-raw", "smoke", false, false, false, "independent test"),
            ["independent test matrix"]);

        return BaselineCandidateEligibility.ApplyGeneratedExactMatrixEligibility(manifest);
    }

    private static BenchmarkReport CreateEligibleCaseReport(BenchmarkReport baseReport, GeneratedExactSearchOptions caseOptions, string reportId)
    {
        BenchmarkReport report = baseReport with
        {
            ReportId = reportId,
            Dataset = baseReport.Dataset with
            {
                Metric = caseOptions.Metric.ToString(),
                Dimension = caseOptions.Dimension,
                VectorCount = caseOptions.VectorCount,
                QueryCount = caseOptions.QueryCount,
                Seed = string.Create(CultureInfo.InvariantCulture, $"0x{caseOptions.Seed:X8}")
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
                Runs = baseReport.Search.Runs.Select(run => run with { MeasuredQueryCount = caseOptions.QueryCount }).ToArray(),
                Aggregate = baseReport.Search.Aggregate with { MeasuredQueryCountPerRun = caseOptions.QueryCount }
            },
            Measurement = baseReport.Measurement with
            {
                Warmup = new WarmupInfo("executed", caseOptions.WarmupQueries, "independent test warmup")
            }
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static MetricComparisonEntry Metric(BenchmarkComparisonArtifact comparison, string name) =>
        comparison.Metrics.Single(metric => metric.Name == name);

    private static RepositoryInfo CleanRepository() => new("abc123", "main", Dirty: false);

    private static string CreateArtifactDirectory(string prefix)
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec020-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static void AssertClose(double expected, double? actual)
    {
        Assert.True(actual.HasValue);
        Assert.Equal(expected, actual.Value, precision: 12);
    }

    private static void AssertNoForbiddenGateProperties(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "publicClaimPassed",
            "publicClaimStatus",
            "baselineCandidatePassed",
            "baselineCandidateStatus",
            "regressionPassed",
            "regressionDecision",
            "regressionFailure",
            "hardGate",
            "hardGateEligible",
            "hardGatePassed",
            "buildFailed",
            "taskFailed");
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
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

    private sealed record NoiseProfile(
        RunToRunMetricNoiseInfo Elapsed,
        RunToRunMetricNoiseInfo Qps,
        RunToRunMetricNoiseInfo P50,
        RunToRunMetricNoiseInfo P95,
        RunToRunMetricNoiseInfo P99,
        RunToRunMetricNoiseInfo Allocation);
}
