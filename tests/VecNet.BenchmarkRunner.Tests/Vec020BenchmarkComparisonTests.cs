using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec020BenchmarkComparisonTests
{
    [Fact]
    public void ReportComparison_CompatibleReports_EmitsWarningOnlyPrivateArtifact()
    {
        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 0, noiseFraction: 0.01);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 0, noiseFraction: 0.01);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("VecNet.BenchmarkComparison", comparison.SchemaName);
        Assert.Equal("0.1", comparison.SchemaVersion);
        Assert.Equal("generated-exact-report-comparison", comparison.ArtifactKind);
        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Compatibility.Reasons);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Contains(comparison.Metrics, metric => metric.Name == "search.aggregate.meanElapsedMilliseconds");
        Assert.Equal("noWarnings", comparison.Warnings.Status);
    }

    [Fact]
    public void ReportComparison_IncompatibleReports_EmitsNotComparableReasons()
    {
        BenchmarkReport baseline = CreateEligibleReport("baseline");
        BenchmarkReport current = CreateEligibleReport("current") with
        {
            Dataset = baseline.Dataset with { Dimension = baseline.Dataset.Dimension + 1 }
        };

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Metrics);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Field == "dataset.dimension");
        Assert.Contains("notComparable", comparison.Warnings.Labels);
    }

    [Fact]
    public void ReportComparison_ImprovedMetrics_AreClassifiedAsImproved()
    {
        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 10, noiseFraction: 0.01);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), elapsed: 80, qps: 1250, p50: 0.8, p95: 1.5, p99: 2.4, allocation: 5, noiseFraction: 0.01);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("improved", Metric(comparison, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.Equal("improved", Metric(comparison, "search.aggregate.meanQps").WarningLabel);
        Assert.Equal("improved", Metric(comparison, "search.aggregate.meanLatencyP50Milliseconds").WarningLabel);
        Assert.Equal("improved", Metric(comparison, "search.aggregate.meanManagedAllocatedBytesPerQuery").WarningLabel);
        Assert.Equal("noWarnings", comparison.Warnings.Status);
    }

    [Fact]
    public void ReportComparison_WorseMetricsBeyondNoise_AreWarningOnlyLabels()
    {
        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 10, noiseFraction: 0.01);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), elapsed: 120, qps: 800, p50: 1.2, p95: 2.3, p99: 3.6, allocation: 12, noiseFraction: 0.01);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("performanceWarning", Metric(comparison, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.Equal("performanceWarning", Metric(comparison, "search.aggregate.meanQps").WarningLabel);
        Assert.Equal("performanceWarning", Metric(comparison, "search.aggregate.meanLatencyP50Milliseconds").WarningLabel);
        Assert.Equal("performanceWarning", Metric(comparison, "search.aggregate.meanLatencyP95Milliseconds").WarningLabel);
        Assert.Equal("performanceWarning", Metric(comparison, "search.aggregate.meanLatencyP99Milliseconds").WarningLabel);
        Assert.Equal("allocationWarning", Metric(comparison, "search.aggregate.meanManagedAllocatedBytesPerQuery").WarningLabel);
        Assert.Equal("warningsPresent", comparison.Warnings.Status);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Fact]
    public void ReportComparison_WorseMetricsWithinAvailableNoise_AreNoiseDominated()
    {
        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 10, noiseFraction: 0.30);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), elapsed: 120, qps: 800, p50: 1.2, p95: 2.4, p99: 3.6, allocation: 12, noiseFraction: 0.20);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        MetricComparisonEntry elapsed = Metric(comparison, "search.aggregate.meanElapsedMilliseconds");
        Assert.Equal("noiseDominated", elapsed.WarningLabel);
        Assert.Equal(0.30, elapsed.AvailableNoise.Fraction);
        Assert.Equal("inconclusive", comparison.Warnings.Status);
    }

    [Fact]
    public void ReportComparison_WhenNoiseUnavailable_DoesNotEmitPerformanceThresholdWarning()
    {
        BenchmarkReport baseline = WithNoiseStatus(WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 10, noiseFraction: 0.01), "notMeasured");
        BenchmarkReport current = WithNoiseStatus(WithSearchMetrics(CreateEligibleReport("current"), elapsed: 130, qps: 700, p50: 1.3, p95: 2.6, p99: 3.9, allocation: 13, noiseFraction: 0.01), "notMeasured");

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("noiseUnavailable", Metric(comparison, "search.aggregate.meanElapsedMilliseconds").WarningLabel);
        Assert.Equal("unavailable", Metric(comparison, "search.aggregate.meanElapsedMilliseconds").AvailableNoise.Status);
        Assert.DoesNotContain("performanceWarning", comparison.Warnings.Labels);
    }

    [Fact]
    public void ReportComparison_CorrectnessRegression_EmitsCorrectnessWarning()
    {
        BenchmarkReport baseline = CreateEligibleReport("baseline");
        BenchmarkReport current = CreateEligibleReport("current") with
        {
            Metrics = baseline.Metrics with { MissingResultCount = 1 }
        };

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Equal("correctnessWarning", Metric(comparison, "correctness.missingResultCount").WarningLabel);
        Assert.Equal("warningsPresent", comparison.Warnings.Status);
    }

    [Fact]
    public void ReportComparison_ZeroToNonZeroAllocation_EmitsAllocationWarningAndInfinityRatio()
    {
        BenchmarkReport baseline = WithSearchMetrics(CreateEligibleReport("baseline"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 0, noiseFraction: 0.01);
        BenchmarkReport current = WithSearchMetrics(CreateEligibleReport("current"), elapsed: 100, qps: 1000, p50: 1, p95: 2, p99: 3, allocation: 1, noiseFraction: 0.01);

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);
        MetricComparisonEntry allocation = Metric(comparison, "search.aggregate.meanManagedAllocatedBytesPerQuery");

        Assert.Equal("allocationWarning", allocation.WarningLabel);
        Assert.True(double.IsPositiveInfinity(allocation.Ratio!.Value));
        Assert.True(double.IsPositiveInfinity(allocation.PercentChange!.Value));
    }

    [Fact]
    public void ReportComparison_DirtyCurrent_EmitsDirtyCurrentWarningOnlyMetadata()
    {
        BenchmarkReport baseline = CreateEligibleReport("baseline");
        BenchmarkReport current = CreateEligibleReport("current") with
        {
            Repository = CleanRepository() with { Dirty = true }
        };

        BenchmarkComparisonArtifact comparison = CompareReports(baseline, current);

        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Contains("dirtyCurrentWarning", comparison.Warnings.Labels);
        Assert.Equal(1, comparison.Warnings.DirtyCurrentWarningCount);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Fact]
    public void ReportComparison_DoesNotMutateInputReportFiles()
    {
        string directory = CreateArtifactDirectory("immutability");
        string baselinePath = Path.Combine(directory, "baseline.json");
        string currentPath = Path.Combine(directory, "current.json");
        string outputPath = Path.Combine(directory, "comparison.json");
        ReportWriter.Write(CreateEligibleReport("baseline"), baselinePath);
        ReportWriter.Write(CreateEligibleReport("current"), currentPath);
        string beforeBaseline = File.ReadAllText(baselinePath);
        string beforeCurrent = File.ReadAllText(currentPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselinePath, currentPath, outputPath),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(comparison, outputPath);

        Assert.Equal(beforeBaseline, File.ReadAllText(baselinePath));
        Assert.Equal(beforeCurrent, File.ReadAllText(currentPath));
        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public void MatrixComparison_CompatibleStandardMatrices_EmitsPerCaseComparisonsAndSummary()
    {
        string directory = CreateArtifactDirectory("matrix-compatible");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current");

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("generated-exact-standard-matrix-comparison", comparison.ArtifactKind);
        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Equal(36, comparison.Cases.Length);
        Assert.NotNull(comparison.MatrixSummary);
        Assert.Equal(36, comparison.MatrixSummary!.ComparableCaseCount);
        Assert.Equal(0, comparison.MatrixSummary.NotComparableCaseCount);
        Assert.All(comparison.Cases, item => Assert.Equal("comparable", item.Compatibility.Status));
    }

    [Fact]
    public void MatrixComparison_CurrentMatrixWithOnlyDirtyRepository_RemainsComparableAndWarns()
    {
        string directory = CreateArtifactDirectory("matrix-dirty-current");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current") with
        {
            Repository = CleanRepository() with { Dirty = true }
        };

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("comparable", comparison.Compatibility.Status);
        Assert.Contains("dirtyCurrentWarning", comparison.Warnings.Labels);
        Assert.Equal(1, comparison.Warnings.DirtyCurrentWarningCount);
        Assert.Equal(36, comparison.MatrixSummary!.ComparableCaseCount);
        Assert.Equal(0, comparison.MatrixSummary.NotComparableCaseCount);
    }

    [Fact]
    public void MatrixComparison_CurrentMatrixWithInvalidPrivateLocalPosture_IsNotComparable()
    {
        string directory = CreateArtifactDirectory("matrix-invalid-current-posture");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current") with
        {
            Eligibility = new GeneratedExactMatrixEligibility(
                "public-evidence",
                "public-summary",
                "smoke",
                PublicClaimEligible: false,
                BaselineCandidateEligible: true,
                RegressionGateEligible: false,
                "stale invalid current posture")
        };

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Cases);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Field == "current.matrixCandidateEligibility");
    }

    [Fact]
    public void MatrixComparison_IncompatibleMatrices_EmitsNotComparableCaseReasons()
    {
        string directory = CreateArtifactDirectory("matrix-incompatible");
        GeneratedExactMatrixManifest baseline = CreateStandardManifest(Path.Combine(directory, "baseline"), "baseline");
        GeneratedExactMatrixManifest current = CreateStandardManifest(Path.Combine(directory, "current"), "current");
        File.Delete(current.Cases[0].ReportPath);

        BenchmarkComparisonArtifact comparison = CompareMatrices(baseline, current);

        Assert.Equal("partiallyComparable", comparison.Compatibility.Status);
        Assert.NotNull(comparison.MatrixSummary);
        Assert.Equal(35, comparison.MatrixSummary!.ComparableCaseCount);
        Assert.Equal(1, comparison.MatrixSummary.NotComparableCaseCount);
        Assert.Contains(comparison.Cases, item => item.Compatibility.Status == "notComparable");
    }

    [Fact]
    public void CompareCommand_ParsesAndWritesPrivateComparisonArtifact()
    {
        string directory = CreateArtifactDirectory("command");
        string baselinePath = Path.Combine(directory, "baseline.json");
        string currentPath = Path.Combine(directory, "current.json");
        string outputPath = Path.Combine(directory, "comparison.json");
        ReportWriter.Write(CreateEligibleReport("baseline"), baselinePath);
        ReportWriter.Write(CreateEligibleReport("current"), currentPath);

        BenchmarkComparisonOptions options = CommandLine.ParseComparison(
            [
                "compare-generated-exact",
                "--baseline", baselinePath,
                "--current", currentPath,
                "--output", outputPath
            ]);
        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(options, ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(comparison, options.OutputPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.BenchmarkComparison", root.GetProperty("schemaName").GetString());
        Assert.Equal("warning-only", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.False(root.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("regressionGateEligible").GetBoolean());
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

    private static MetricComparisonEntry Metric(BenchmarkComparisonArtifact comparison, string name) =>
        comparison.Metrics.Single(metric => metric.Name == name);

    private static BenchmarkReport CreateEligibleReport(string reportId)
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 32,
            VectorCount: 100,
            QueryCount: 100,
            TopK: 10,
            Seed: 0x5EED0200,
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
        double noiseFraction)
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
                RunToRunNoise = CreateNoise(report.Search.Aggregate.RunCount, elapsed, qps, p50, p95, p99, allocation, noiseFraction)
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

    private static RunToRunNoiseInfo CreateNoise(
        int runs,
        double elapsed,
        double qps,
        double p50,
        double p95,
        double p99,
        double allocation,
        double fraction) =>
        new(
            "measured",
            runs,
            NoiseMeasured: true,
            "test scope",
            "test statistics",
            "test measured noise",
            "test non-goals",
            MetricNoise("milliseconds", elapsed, fraction),
            MetricNoise("queriesPerSecond", qps, fraction),
            MetricNoise("milliseconds", p50, fraction),
            MetricNoise("milliseconds", p95, fraction),
            MetricNoise("milliseconds", p99, fraction),
            MetricNoise("bytesPerQuery", allocation, fraction));

    private static RunToRunMetricNoiseInfo MetricNoise(string unit, double mean, double fraction) =>
        new(
            "measured",
            unit,
            mean,
            Math.Abs(mean) * fraction,
            fraction,
            mean,
            mean,
            Math.Abs(mean) * fraction,
            "synthetic measured noise");

    private static GeneratedExactMatrixManifest CreateStandardManifest(string outputDirectory, string idPrefix)
    {
        Directory.CreateDirectory(outputDirectory);
        var options = new GeneratedExactMatrixOptions(
            GeneratedExactMatrixOptions.StandardPresetName,
            VectorCount: 100,
            QueryCount: 100,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0201,
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
            new GeneratedExactMatrixEligibility("local-evidence", "private-raw", "smoke", false, false, false, "test"),
            ["test matrix"]);

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
                Warmup = new WarmupInfo("executed", caseOptions.WarmupQueries, "test warmup")
            }
        };

        return BaselineCandidateEligibility.ApplyGeneratedExactReportEligibility(report);
    }

    private static RepositoryInfo CleanRepository() => new("abc123", "main", Dirty: false);

    private static string CreateArtifactDirectory(string prefix)
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec020-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }
}
