using System.Globalization;
using System.Text.Json;

namespace VecNet.BenchmarkRunner;

public static class BenchmarkComparisonScenario
{
    private const string SchemaName = "VecNet.BenchmarkComparison";
    private const string SchemaVersion = "0.1";
    private const string TaskId = "VEC-020";
    private const string ReportComparisonKind = "generated-exact-report-comparison";
    private const string MatrixComparisonKind = "generated-exact-standard-matrix-comparison";
    private const string UnknownKind = "unknown";
    private static readonly string[] StandardMetrics =
    [
        VectorMetric.SquaredEuclidean.ToString(),
        VectorMetric.InnerProduct.ToString(),
        VectorMetric.Cosine.ToString()
    ];

    private static readonly int[] StandardDimensions = [32, 128, 386, 768];
    private static readonly int[] StandardTopKValues = [1, 10, 100];

    public static BenchmarkComparisonArtifact Run(BenchmarkComparisonOptions options, IReadOnlyList<string> commandArguments)
    {
        LoadedArtifact baseline = Load(options.BaselinePath);
        LoadedArtifact current = Load(options.CurrentPath);

        ComparisonBuildResult result =
            baseline.Report is not null && current.Report is not null
                ? CompareReports(options.BaselinePath, baseline.Report, options.CurrentPath, current.Report)
                : baseline.Matrix is not null && current.Matrix is not null
                    ? CompareMatrices(options.BaselinePath, baseline.Matrix, options.CurrentPath, current.Matrix)
                    : CreateKindMismatchResult(baseline, current);

        if (baseline.LoadFailure is not null || current.LoadFailure is not null)
        {
            var reasons = new List<CompatibilityReason>(result.Compatibility.Reasons);
            if (baseline.LoadFailure is not null)
            {
                reasons.Add(baseline.LoadFailure);
            }

            if (current.LoadFailure is not null)
            {
                reasons.Add(current.LoadFailure);
            }

            result = result with
            {
                Compatibility = new ComparisonCompatibility("notComparable", reasons.ToArray()),
                Metrics = [],
                Cases = [],
                MatrixSummary = null,
                Warnings = CreateWarningSummary([], isNotComparable: true, dirtyCurrent: false)
            };
        }

        return new BenchmarkComparisonArtifact(
            SchemaName,
            SchemaVersion,
            CreateComparisonId(result.ArtifactKind, baseline.Identity.Id, current.Identity.Id),
            DateTimeOffset.UtcNow,
            TaskId,
            "local-evidence",
            "private-raw",
            new ComparisonEvidenceInfo(
                "warning-only",
                "local-evidence",
                "D-039 warning-only private baseline comparison policy",
                "Private local derived comparison artifact only; warnings are review signals, not public claims, hard gates or regression decisions."),
            result.ArtifactKind,
            baseline.Identity,
            current.Identity,
            result.Compatibility,
            result.Metrics,
            result.Cases,
            result.MatrixSummary,
            result.Warnings,
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            Notes:
            [
                "Private raw local warning-only comparison evidence; not a public benchmark claim.",
                "Comparison artifacts are derived outputs and do not mutate input reports or manifests.",
                "Warning labels are not hard gates, build failures, CI decisions or regression pass/fail decisions.",
                "Run-to-run noise is descriptive local context only, not BenchmarkDotNet statistics or a confidence interval.",
                "Resident/process memory, persisted bytes, disk I/O, build time, training time, external datasets and ANN artifacts are not compared."
            ]);
    }

    public static void Write(BenchmarkComparisonArtifact artifact, string outputPath) =>
        ReportWriter.WriteComparison(artifact, outputPath);

    private static ComparisonBuildResult CompareReports(
        string baselinePath,
        BenchmarkReport baseline,
        string currentPath,
        BenchmarkReport current)
    {
        var reasons = new List<CompatibilityReason>();
        AddReportCompatibilityReasons(reasons, baseline, current, requireBaselineEligible: true);
        bool dirtyCurrent = current.Repository.Dirty;

        MetricComparisonEntry[] metrics = reasons.Count == 0
            ? CreateReportMetrics(baseline, current)
            : [];
        ComparisonWarningSummary warnings = CreateWarningSummary(
            metrics,
            isNotComparable: reasons.Count > 0,
            dirtyCurrent);

        return new ComparisonBuildResult(
            ReportComparisonKind,
            new ComparisonCompatibility(reasons.Count == 0 ? "comparable" : "notComparable", reasons.ToArray()),
            metrics,
            [],
            null,
            warnings);
    }

    private static ComparisonBuildResult CompareMatrices(
        string baselinePath,
        GeneratedExactMatrixManifest baseline,
        string currentPath,
        GeneratedExactMatrixManifest current)
    {
        var manifestReasons = new List<CompatibilityReason>();
        AddMatrixCompatibilityReasons(manifestReasons, baseline, current);
        bool dirtyCurrent = current.Repository.Dirty;

        if (manifestReasons.Count > 0)
        {
            return new ComparisonBuildResult(
                MatrixComparisonKind,
                new ComparisonCompatibility("notComparable", manifestReasons.ToArray()),
                [],
                [],
                new MatrixComparisonSummary(0, 0, 0, 0, 0, 0, 0, 0),
                CreateWarningSummary([], isNotComparable: true, dirtyCurrent));
        }

        Dictionary<string, GeneratedExactMatrixCaseManifest> baselineCases = CreateCaseMap(baseline.Cases);
        Dictionary<string, GeneratedExactMatrixCaseManifest> currentCases = CreateCaseMap(current.Cases);
        var caseComparisons = new List<MatrixCaseComparisonEntry>();

        foreach (string key in baselineCases.Keys.Order(StringComparer.Ordinal))
        {
            GeneratedExactMatrixCaseManifest baselineCase = baselineCases[key];
            GeneratedExactMatrixCaseManifest currentCase = currentCases[key];
            MatrixCaseComparisonEntry comparison = CompareMatrixCase(key, baselineCase, currentCase);
            caseComparisons.Add(comparison);
        }

        int comparable = caseComparisons.Count(item => item.Compatibility.Status == "comparable");
        int notComparable = caseComparisons.Count - comparable;
        string status = comparable == 0 ? "notComparable" : notComparable == 0 ? "comparable" : "partiallyComparable";
        ComparisonWarningSummary summaryWarnings = CreateMatrixWarningSummary(caseComparisons, dirtyCurrent);

        var matrixSummary = new MatrixComparisonSummary(
            comparable,
            notComparable,
            caseComparisons.Count(item => item.Warnings.ImprovedCount > 0 && HasNoWarnings(item.Warnings)),
            caseComparisons.Count(item => item.Warnings.UnchangedCount > 0 && HasNoWarnings(item.Warnings)),
            caseComparisons.Count(item => item.Warnings.NoiseDominatedCount > 0 || item.Warnings.NoiseUnavailableCount > 0),
            caseComparisons.Sum(item => item.Warnings.CorrectnessWarningCount),
            caseComparisons.Sum(item => item.Warnings.PerformanceWarningCount),
            caseComparisons.Sum(item => item.Warnings.AllocationWarningCount));

        return new ComparisonBuildResult(
            MatrixComparisonKind,
            new ComparisonCompatibility(status, []),
            [],
            caseComparisons.ToArray(),
            matrixSummary,
            summaryWarnings);
    }

    private static MatrixCaseComparisonEntry CompareMatrixCase(
        string key,
        GeneratedExactMatrixCaseManifest baselineCase,
        GeneratedExactMatrixCaseManifest currentCase)
    {
        var reasons = new List<CompatibilityReason>();
        Require(reasons, baselineCase.Status == "passed", "baselineCase.status", "passed", baselineCase.Status, "baseline matrix case must be passed");
        Require(reasons, currentCase.Status == "passed", "currentCase.status", "passed", currentCase.Status, "current matrix case must be passed");
        Require(reasons, baselineCase.ValidationStatus == "passed", "baselineCase.validationStatus", "passed", baselineCase.ValidationStatus, "baseline matrix case validation must be passed");
        Require(reasons, currentCase.ValidationStatus == "passed", "currentCase.validationStatus", "passed", currentCase.ValidationStatus, "current matrix case validation must be passed");
        Require(reasons, !string.IsNullOrWhiteSpace(baselineCase.ReportId), "baselineCase.reportId", "present", baselineCase.ReportId, "baseline matrix case report ID must be present");
        Require(reasons, !string.IsNullOrWhiteSpace(currentCase.ReportId), "currentCase.reportId", "present", currentCase.ReportId, "current matrix case report ID must be present");
        Require(reasons, !string.IsNullOrWhiteSpace(baselineCase.ReportPath), "baselineCase.reportPath", "present", baselineCase.ReportPath, "baseline matrix case report path must be present");
        Require(reasons, !string.IsNullOrWhiteSpace(currentCase.ReportPath), "currentCase.reportPath", "present", currentCase.ReportPath, "current matrix case report path must be present");

        BenchmarkReport? baselineReport = null;
        BenchmarkReport? currentReport = null;
        if (reasons.Count == 0)
        {
            baselineReport = LoadReportForCase(reasons, baselineCase.ReportPath, "baselineCase.reportPath");
            currentReport = LoadReportForCase(reasons, currentCase.ReportPath, "currentCase.reportPath");
        }

        if (baselineReport is not null)
        {
            AddCaseReportMatchReasons(reasons, baselineCase, baselineReport, "baselineCase");
        }

        if (currentReport is not null)
        {
            AddCaseReportMatchReasons(reasons, currentCase, currentReport, "currentCase");
        }

        MetricComparisonEntry[] metrics = [];
        if (reasons.Count == 0 && baselineReport is not null && currentReport is not null)
        {
            AddReportCompatibilityReasons(reasons, baselineReport, currentReport, requireBaselineEligible: true);
            if (reasons.Count == 0)
            {
                metrics = CreateReportMetrics(baselineReport, currentReport);
            }
        }

        bool dirtyCurrent = currentReport?.Repository.Dirty ?? false;
        return new MatrixCaseComparisonEntry(
            key,
            baselineCase.ReportId,
            currentCase.ReportId,
            new ComparisonCompatibility(reasons.Count == 0 ? "comparable" : "notComparable", reasons.ToArray()),
            metrics,
            CreateWarningSummary(metrics, isNotComparable: reasons.Count > 0, dirtyCurrent));
    }

    private static void AddReportCompatibilityReasons(
        List<CompatibilityReason> reasons,
        BenchmarkReport baseline,
        BenchmarkReport current,
        bool requireBaselineEligible)
    {
        BaselineCandidateEligibilityInfo baselineEligibility = BaselineCandidateEligibility.EvaluateGeneratedExactReport(baseline);
        BaselineCandidateEligibilityInfo currentEligibility = BaselineCandidateEligibility.EvaluateGeneratedExactReport(current);

        if (requireBaselineEligible)
        {
            Require(reasons, baselineEligibility.Eligible, "baseline.baselineCandidateEligibility", "eligible", string.Join("; ", baselineEligibility.UnsatisfiedConditions), "baseline report must recalculate as a D-038 private baseline candidate");
        }

        bool currentEligible = currentEligibility.Eligible || HasOnlyAllowedCurrentUnsatisfiedConditions(currentEligibility);
        Require(reasons, currentEligible, "current.baselineCandidateEligibility", "eligible, dirty-current-only or correctness-warning-only", string.Join("; ", currentEligibility.UnsatisfiedConditions), "current report must satisfy generated exact comparison eligibility except dirty repository identity and correctness warning signals");

        Compare(reasons, "schemaName", baseline.SchemaName, current.SchemaName);
        Compare(reasons, "schemaVersion", baseline.SchemaVersion, current.SchemaVersion);
        Compare(reasons, "command.scenario", baseline.Command.Scenario, current.Command.Scenario);
        Compare(reasons, "scenario.name", baseline.Scenario.Name, current.Scenario.Name);
        Compare(reasons, "runner.name", baseline.Runner.Name, current.Runner.Name);
        Compare(reasons, "runner.version", baseline.Runner.Version, current.Runner.Version);
        Compare(reasons, "dataset.kind", baseline.Dataset.Kind, current.Dataset.Kind);
        Compare(reasons, "dataset.sourceVerificationStatus", baseline.Dataset.SourceVerificationStatus, current.Dataset.SourceVerificationStatus);
        Compare(reasons, "dataset.distribution", baseline.Dataset.Distribution, current.Dataset.Distribution);
        Compare(reasons, "dataset.metric", baseline.Dataset.Metric, current.Dataset.Metric);
        Compare(reasons, "dataset.dimension", baseline.Dataset.Dimension, current.Dataset.Dimension);
        Compare(reasons, "dataset.vectorCount", baseline.Dataset.VectorCount, current.Dataset.VectorCount);
        Compare(reasons, "dataset.queryCount", baseline.Dataset.QueryCount, current.Dataset.QueryCount);
        Compare(reasons, "dataset.seed", baseline.Dataset.Seed, current.Dataset.Seed);
        Compare(reasons, "truth.kind", baseline.Truth.Kind, current.Truth.Kind);
        Compare(reasons, "truth.depth", baseline.Truth.Depth, current.Truth.Depth);
        Compare(reasons, "truth.tiePolicy", baseline.Truth.TiePolicy, current.Truth.TiePolicy);
        Compare(reasons, "scenario.topK", baseline.Scenario.TopK, current.Scenario.TopK);
        Compare(reasons, "scenario.measuredQueryCount", baseline.Scenario.MeasuredQueryCount, current.Scenario.MeasuredQueryCount);
        Compare(reasons, "scenario.concurrency", baseline.Scenario.Concurrency, current.Scenario.Concurrency);
        Compare(reasons, "index.profile", baseline.Index.Profile, current.Index.Profile);
        Compare(reasons, "index.type", baseline.Index.Type, current.Index.Type);
        Compare(reasons, "index.metric", baseline.Index.Metric, current.Index.Metric);
        Compare(reasons, "index.dimension", baseline.Index.Dimension, current.Index.Dimension);
        Compare(reasons, "index.vectorCount", baseline.Index.VectorCount, current.Index.VectorCount);
        Compare(reasons, "index.configuration", baseline.Index.Configuration, current.Index.Configuration);
        Compare(reasons, "measurement.latency.status", baseline.Measurement.Latency.Status, current.Measurement.Latency.Status);
        Compare(reasons, "measurement.latency.unit", baseline.Measurement.Latency.Unit, current.Measurement.Latency.Unit);
        Compare(reasons, "measurement.latency.sampleScope", baseline.Measurement.Latency.SampleScope, current.Measurement.Latency.SampleScope);
        Compare(reasons, "measurement.latency.timedOperation", baseline.Measurement.Latency.TimedOperation, current.Measurement.Latency.TimedOperation);
        Compare(reasons, "measurement.latency.percentileEstimator", baseline.Measurement.Latency.PercentileEstimator, current.Measurement.Latency.PercentileEstimator);
        Compare(reasons, "measurement.warmup.warmupCount", baseline.Measurement.Warmup.WarmupCount, current.Measurement.Warmup.WarmupCount);
        Compare(reasons, "measurement.repeatedRuns.runCount", baseline.Measurement.RepeatedRuns.RunCount, current.Measurement.RepeatedRuns.RunCount);
        Compare(reasons, "measurement.managedAllocations.status", baseline.Measurement.ManagedAllocations.Status, current.Measurement.ManagedAllocations.Status);
        Compare(reasons, "measurement.managedAllocations.unit", baseline.Measurement.ManagedAllocations.Unit, current.Measurement.ManagedAllocations.Unit);
        Compare(reasons, "measurement.memory.status", baseline.Measurement.Memory.Status, current.Measurement.Memory.Status);
        Compare(reasons, "measurement.memory.value", baseline.Measurement.Memory.Value, current.Measurement.Memory.Value);
        Compare(reasons, "measurement.runToRunNoise.status", baseline.Measurement.RunToRunNoise.Status, current.Measurement.RunToRunNoise.Status);
        Compare(reasons, "measurement.runToRunNoise.runCount", baseline.Measurement.RunToRunNoise.RunCount, current.Measurement.RunToRunNoise.RunCount);
        Compare(reasons, "search.aggregate.runCount", baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount);
        Compare(reasons, "search.aggregate.measuredQueryCountPerRun", baseline.Search.Aggregate.MeasuredQueryCountPerRun, current.Search.Aggregate.MeasuredQueryCountPerRun);
        Compare(reasons, "environment.osDescription", baseline.Environment.OsDescription, current.Environment.OsDescription);
        Compare(reasons, "environment.processArchitecture", baseline.Environment.ProcessArchitecture, current.Environment.ProcessArchitecture);
        Compare(reasons, "environment.frameworkDescription", baseline.Environment.FrameworkDescription, current.Environment.FrameworkDescription);
        Compare(reasons, "environment.runtimeIdentifier", baseline.Environment.RuntimeIdentifier, current.Environment.RuntimeIdentifier);
        Compare(reasons, "environment.processorCount", baseline.Environment.ProcessorCount, current.Environment.ProcessorCount);
        Compare(reasons, "environment.serverGc", baseline.Environment.ServerGc, current.Environment.ServerGc);
        Compare(reasons, "environment.vectorFloatCount", baseline.Environment.VectorFloatCount, current.Environment.VectorFloatCount);
        Require(reasons, baseline.Search.Aggregate.RunCount >= 3 && current.Search.Aggregate.RunCount >= 3, "search.aggregate.runCount", ">=3 on both sides", $"{baseline.Search.Aggregate.RunCount}/{current.Search.Aggregate.RunCount}", "single-run timing or allocation comparison is not comparable");
        Require(reasons, baseline.Measurement.Memory.Status == "notMeasured" && current.Measurement.Memory.Status == "notMeasured", "measurement.memory.status", "notMeasured", $"{baseline.Measurement.Memory.Status}/{current.Measurement.Memory.Status}", "resident/process memory must remain explicitly not measured");
    }

    private static void AddMatrixCompatibilityReasons(
        List<CompatibilityReason> reasons,
        GeneratedExactMatrixManifest baseline,
        GeneratedExactMatrixManifest current)
    {
        BaselineCandidateEligibilityInfo baselineEligibility = BaselineCandidateEligibility.EvaluateGeneratedExactMatrix(baseline);
        string[] currentUnsatisfiedConditions = EvaluateCurrentMatrixManifestConditions(current);
        Require(reasons, baselineEligibility.Eligible, "baseline.matrixCandidateEligibility", "eligible", string.Join("; ", baselineEligibility.UnsatisfiedConditions), "baseline matrix must recalculate as a D-038 private baseline candidate");
        bool currentEligible = currentUnsatisfiedConditions.Length == 0 || HasOnlyAllowedDirtyCurrentMatrixUnsatisfiedConditions(currentUnsatisfiedConditions);
        Require(reasons, currentEligible, "current.matrixCandidateEligibility", "eligible or dirty-current-only", string.Join("; ", currentUnsatisfiedConditions), "current matrix must satisfy generated exact standard matrix comparison eligibility except dirty repository identity");

        Compare(reasons, "schemaName", baseline.SchemaName, current.SchemaName);
        Compare(reasons, "schemaVersion", baseline.SchemaVersion, current.SchemaVersion);
        Compare(reasons, "scenarioName", baseline.ScenarioName, current.ScenarioName);
        Compare(reasons, "presetName", baseline.PresetName, current.PresetName);
        Require(reasons, baseline.PresetName == GeneratedExactMatrixOptions.StandardPresetName && current.PresetName == GeneratedExactMatrixOptions.StandardPresetName, "presetName", "standard", $"{baseline.PresetName}/{current.PresetName}", "matrix comparison supports only standard generated exact matrix manifests");
        Compare(reasons, "runner.name", baseline.Runner.Name, current.Runner.Name);
        Compare(reasons, "runner.version", baseline.Runner.Version, current.Runner.Version);
        Compare(reasons, "caseCount", baseline.CaseCount, current.CaseCount);
        Require(reasons, baseline.Aggregate.FailedCaseCount == 0 && current.Aggregate.FailedCaseCount == 0, "aggregate.failedCaseCount", "0", $"{baseline.Aggregate.FailedCaseCount}/{current.Aggregate.FailedCaseCount}", "both matrix manifests must have zero failed cases");
        Require(reasons, !baseline.Eligibility.PublicClaimEligible && !current.Eligibility.PublicClaimEligible, "eligibility.publicClaimEligible", "false", $"{baseline.Eligibility.PublicClaimEligible}/{current.Eligibility.PublicClaimEligible}", "public-claim eligibility must be false");
        Require(reasons, !baseline.Eligibility.RegressionGateEligible && !current.Eligibility.RegressionGateEligible, "eligibility.regressionGateEligible", "false", $"{baseline.Eligibility.RegressionGateEligible}/{current.Eligibility.RegressionGateEligible}", "regression-gate eligibility must be false");
        Require(reasons, !string.IsNullOrWhiteSpace(baseline.Repository.Commit), "baseline.repository.commit", "present", baseline.Repository.Commit, "baseline repository commit must be present");
        Require(reasons, !baseline.Repository.Dirty, "baseline.repository.dirty", "false", baseline.Repository.Dirty.ToString(CultureInfo.InvariantCulture), "baseline repository must be clean");
        Require(reasons, !string.IsNullOrWhiteSpace(current.Repository.Commit), "current.repository.commit", "present", current.Repository.Commit, "current repository commit must be present");

        Dictionary<string, GeneratedExactMatrixCaseManifest> baselineCases = CreateCaseMap(baseline.Cases);
        Dictionary<string, GeneratedExactMatrixCaseManifest> currentCases = CreateCaseMap(current.Cases);
        Require(reasons, baselineCases.Count == baseline.Cases.Length, "baseline.cases", "unique case keys", baseline.Cases.Length.ToString(CultureInfo.InvariantCulture), "baseline matrix case keys must be unique");
        Require(reasons, currentCases.Count == current.Cases.Length, "current.cases", "unique case keys", current.Cases.Length.ToString(CultureInfo.InvariantCulture), "current matrix case keys must be unique");
        Require(reasons, baselineCases.Keys.Order(StringComparer.Ordinal).SequenceEqual(currentCases.Keys.Order(StringComparer.Ordinal)), "cases.caseKeySet", "exact match", "different", "matrix case key set must match exactly");
    }

    private static MetricComparisonEntry[] CreateReportMetrics(BenchmarkReport baseline, BenchmarkReport current)
    {
        var entries = new List<MetricComparisonEntry>
        {
            CompareCorrectnessMetric("correctness.recallAtK", baseline.Metrics.RecallAtK, current.Metrics.RecallAtK, higherIsBetter: true),
            CompareCorrectnessMetric("correctness.orderedAgreement", baseline.Metrics.OrderedAgreement, current.Metrics.OrderedAgreement, higherIsBetter: true),
            CompareCorrectnessMetric("correctness.missingResultCount", baseline.Metrics.MissingResultCount, current.Metrics.MissingResultCount, higherIsBetter: false),
            CompareCorrectnessMetric("correctness.distanceMismatchCount", baseline.Metrics.DistanceMismatchCount, current.Metrics.DistanceMismatchCount, higherIsBetter: false),
            CompareStatusMetric("correctness.validationStatus", baseline.Validation.Status, current.Validation.Status),
            CompareStatusMetric("correctness.distanceToleranceStatus", baseline.Metrics.DistanceToleranceStatus, current.Metrics.DistanceToleranceStatus),
            ComparePerformanceMetric("search.elapsedMilliseconds", "milliseconds", "lowerIsBetter", baseline.Search.ElapsedMilliseconds, current.Search.ElapsedMilliseconds, baseline.Measurement.RunToRunNoise.ElapsedMilliseconds, current.Measurement.RunToRunNoise.ElapsedMilliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanElapsedMilliseconds", "milliseconds", "lowerIsBetter", baseline.Search.Aggregate.MeanElapsedMilliseconds, current.Search.Aggregate.MeanElapsedMilliseconds, baseline.Measurement.RunToRunNoise.ElapsedMilliseconds, current.Measurement.RunToRunNoise.ElapsedMilliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.qps", "queriesPerSecond", "higherIsBetter", baseline.Search.Qps, current.Search.Qps, baseline.Measurement.RunToRunNoise.Qps, current.Measurement.RunToRunNoise.Qps, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanQps", "queriesPerSecond", "higherIsBetter", baseline.Search.Aggregate.MeanQps, current.Search.Aggregate.MeanQps, baseline.Measurement.RunToRunNoise.Qps, current.Measurement.RunToRunNoise.Qps, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.latencyP50Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.LatencyP50Milliseconds, current.Search.LatencyP50Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP50Milliseconds, current.Measurement.RunToRunNoise.LatencyP50Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.latencyP95Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.LatencyP95Milliseconds, current.Search.LatencyP95Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP95Milliseconds, current.Measurement.RunToRunNoise.LatencyP95Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.latencyP99Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.LatencyP99Milliseconds, current.Search.LatencyP99Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP99Milliseconds, current.Measurement.RunToRunNoise.LatencyP99Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.15, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanLatencyP50Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.Aggregate.MeanLatencyP50Milliseconds, current.Search.Aggregate.MeanLatencyP50Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP50Milliseconds, current.Measurement.RunToRunNoise.LatencyP50Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanLatencyP95Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.Aggregate.MeanLatencyP95Milliseconds, current.Search.Aggregate.MeanLatencyP95Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP95Milliseconds, current.Measurement.RunToRunNoise.LatencyP95Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanLatencyP99Milliseconds", "milliseconds", "lowerIsBetter", baseline.Search.Aggregate.MeanLatencyP99Milliseconds, current.Search.Aggregate.MeanLatencyP99Milliseconds, baseline.Measurement.RunToRunNoise.LatencyP99Milliseconds, current.Measurement.RunToRunNoise.LatencyP99Milliseconds, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.15, 1e-12, false),
            ComparePerformanceMetric("search.aggregate.meanManagedAllocatedBytesPerQuery", "bytesPerQuery", "lowerIsBetter", baseline.Search.Aggregate.MeanManagedAllocatedBytesPerQuery, current.Search.Aggregate.MeanManagedAllocatedBytesPerQuery, baseline.Measurement.RunToRunNoise.ManagedAllocatedBytesPerQuery, current.Measurement.RunToRunNoise.ManagedAllocatedBytesPerQuery, baseline.Search.Aggregate.RunCount, current.Search.Aggregate.RunCount, 0.10, 1.0, true),
            CompareContextMetric("search.aggregate.minManagedAllocatedBytesPerQuery", "bytesPerQuery", baseline.Search.Aggregate.MinManagedAllocatedBytesPerQuery, current.Search.Aggregate.MinManagedAllocatedBytesPerQuery),
            CompareContextMetric("search.aggregate.maxManagedAllocatedBytesPerQuery", "bytesPerQuery", baseline.Search.Aggregate.MaxManagedAllocatedBytesPerQuery, current.Search.Aggregate.MaxManagedAllocatedBytesPerQuery)
        };

        for (int i = 0; i < baseline.Search.Runs.Length && i < current.Search.Runs.Length; i++)
        {
            entries.Add(ComparePerformanceMetric(
                string.Create(CultureInfo.InvariantCulture, $"search.runs.{i + 1}.managedAllocatedBytesPerQuery"),
                "bytesPerQuery",
                "lowerIsBetter",
                baseline.Search.Runs[i].ManagedAllocatedBytesPerQuery,
                current.Search.Runs[i].ManagedAllocatedBytesPerQuery,
                baseline.Measurement.RunToRunNoise.ManagedAllocatedBytesPerQuery,
                current.Measurement.RunToRunNoise.ManagedAllocatedBytesPerQuery,
                baseline.Search.Aggregate.RunCount,
                current.Search.Aggregate.RunCount,
                0.10,
                1.0,
                true));
        }

        return entries.ToArray();
    }

    private static MetricComparisonEntry CompareCorrectnessMetric(string name, double baseline, double current, bool higherIsBetter)
    {
        MetricComparisonEntry entry = CreateNumericEntry(
            name,
            "countOrFraction",
            higherIsBetter ? "higherIsBetter" : "lowerIsBetter",
            baseline,
            current,
            new AvailableNoiseInfo("notApplicable", null, null, "Correctness warnings are semantic review signals and do not use noise context."));
        bool worsened = higherIsBetter ? current < baseline : current > baseline;
        return entry with { WarningLabel = worsened ? "correctnessWarning" : "unchanged" };
    }

    private static MetricComparisonEntry CompareStatusMetric(string name, string baseline, string current)
    {
        string label = current == "passed" && baseline == current ? "unchanged" : "correctnessWarning";
        return new MetricComparisonEntry(
            name,
            "status",
            "status",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            new AvailableNoiseInfo("notApplicable", null, null, "Status fields compare accepted values, not numeric ratios."),
            label);
    }

    private static MetricComparisonEntry CompareContextMetric(string name, string unit, double baseline, double current) =>
        CreateNumericEntry(
            name,
            unit,
            "context",
            baseline,
            current,
            new AvailableNoiseInfo("notApplicable", null, null, "Context metric only; warning classification is based on aggregate mean managed allocated bytes per query.")) with { WarningLabel = "unchanged" };

    private static MetricComparisonEntry ComparePerformanceMetric(
        string name,
        string unit,
        string direction,
        double baseline,
        double current,
        RunToRunMetricNoiseInfo baselineNoise,
        RunToRunMetricNoiseInfo currentNoise,
        int baselineRuns,
        int currentRuns,
        double thresholdFraction,
        double denominatorFloor,
        bool allocationMetric)
    {
        AvailableNoiseInfo noise = ComputeAvailableNoise(baselineNoise, currentNoise, baselineRuns, currentRuns, denominatorFloor);
        MetricComparisonEntry entry = CreateNumericEntry(name, unit, direction, baseline, current, noise);
        if (entry.WarningLabel == "notComparable")
        {
            return entry;
        }

        if (allocationMetric && baseline == 0 && current > 0)
        {
            return entry with { WarningLabel = "allocationWarning" };
        }

        double worseFraction = entry.WorseFraction ?? 0;
        double improvementFraction = entry.ImprovementFraction ?? 0;
        if (worseFraction <= 0 && improvementFraction <= 0)
        {
            return entry with { WarningLabel = "unchanged" };
        }

        if (noise.Status == "unavailable")
        {
            return entry with { WarningLabel = "noiseUnavailable" };
        }

        double availableNoise = noise.Fraction ?? 0;
        if (worseFraction > 0)
        {
            if (worseFraction <= availableNoise)
            {
                return entry with { WarningLabel = "noiseDominated" };
            }

            if (worseFraction > thresholdFraction)
            {
                return entry with { WarningLabel = allocationMetric ? "allocationWarning" : "performanceWarning" };
            }

            return entry with { WarningLabel = "unchanged" };
        }

        return entry with { WarningLabel = improvementFraction > availableNoise ? "improved" : "improvedWithinNoise" };
    }

    private static MetricComparisonEntry CreateNumericEntry(
        string name,
        string unit,
        string direction,
        double baseline,
        double current,
        AvailableNoiseInfo noise)
    {
        double delta = current - baseline;
        double? ratio;
        double? percentChange;
        double? worseFraction = null;
        double? improvementFraction = null;
        string label = "unchanged";

        if (baseline != 0)
        {
            ratio = current / baseline;
            percentChange = delta / Math.Abs(baseline) * 100;
            if (direction == "lowerIsBetter")
            {
                worseFraction = (current - baseline) / baseline;
                improvementFraction = (baseline - current) / baseline;
            }
            else if (direction == "higherIsBetter")
            {
                worseFraction = (baseline - current) / baseline;
                improvementFraction = (current - baseline) / baseline;
            }
        }
        else if (current == 0)
        {
            ratio = 1;
            percentChange = 0;
            worseFraction = 0;
            improvementFraction = 0;
        }
        else if (direction == "lowerIsBetter")
        {
            ratio = double.PositiveInfinity;
            percentChange = double.PositiveInfinity;
            worseFraction = double.PositiveInfinity;
            improvementFraction = null;
        }
        else if (direction == "higherIsBetter" && name.StartsWith("correctness.", StringComparison.Ordinal))
        {
            ratio = null;
            percentChange = null;
        }
        else
        {
            ratio = null;
            percentChange = null;
            label = "notComparable";
        }

        return new MetricComparisonEntry(
            name,
            unit,
            direction,
            baseline,
            current,
            delta,
            ratio,
            percentChange,
            worseFraction,
            improvementFraction,
            noise,
            label);
    }

    private static AvailableNoiseInfo ComputeAvailableNoise(
        RunToRunMetricNoiseInfo baseline,
        RunToRunMetricNoiseInfo current,
        int baselineRuns,
        int currentRuns,
        double denominatorFloor)
    {
        if (baselineRuns < 3 || currentRuns < 3 || baseline.Status != "measured" || current.Status != "measured")
        {
            return new AvailableNoiseInfo("unavailable", null, null, "Noise context requires measured noise metadata and at least three measured runs on both sides.");
        }

        double? baselineFraction = SideNoiseFraction(baseline, denominatorFloor);
        double? currentFraction = SideNoiseFraction(current, denominatorFloor);
        if (baselineFraction is null || currentFraction is null)
        {
            return new AvailableNoiseInfo("unavailable", null, null, "Noise context requires finite mean, coefficient-of-variation, standard-deviation or spread fields.");
        }

        double fraction = Math.Max(baselineFraction.Value, currentFraction.Value);
        return new AvailableNoiseInfo(
            "available",
            fraction,
            fraction * 100,
            "max(baseline side noise fraction, current side noise fraction) using finite CV, stddev/mean denominator and spread/mean denominator terms.");
    }

    private static double? SideNoiseFraction(RunToRunMetricNoiseInfo noise, double denominatorFloor)
    {
        var terms = new List<double>();
        if (Finite(noise.CoefficientOfVariation))
        {
            terms.Add(Math.Abs(noise.CoefficientOfVariation!.Value));
        }

        if (Finite(noise.Mean) && Finite(noise.SampleStandardDeviation))
        {
            double denominator = Math.Max(Math.Abs(noise.Mean!.Value), denominatorFloor);
            terms.Add(Math.Abs(noise.SampleStandardDeviation!.Value) / denominator);
        }

        if (Finite(noise.Mean) && Finite(noise.Spread))
        {
            double denominator = Math.Max(Math.Abs(noise.Mean!.Value), denominatorFloor);
            terms.Add(Math.Abs(noise.Spread!.Value) / denominator);
        }

        return terms.Count == 0 ? null : terms.Max();
    }

    private static bool Finite(double? value) => value.HasValue && double.IsFinite(value.Value);

    private static ComparisonWarningSummary CreateWarningSummary(
        MetricComparisonEntry[] metrics,
        bool isNotComparable,
        bool dirtyCurrent)
    {
        int correctness = metrics.Count(item => item.WarningLabel == "correctnessWarning");
        int performance = metrics.Count(item => item.WarningLabel == "performanceWarning");
        int allocation = metrics.Count(item => item.WarningLabel == "allocationWarning");
        int dirty = dirtyCurrent ? 1 : 0;
        int noiseDominated = metrics.Count(item => item.WarningLabel == "noiseDominated");
        int noiseUnavailable = metrics.Count(item => item.WarningLabel == "noiseUnavailable");
        int improved = metrics.Count(item => item.WarningLabel == "improved");
        int improvedWithinNoise = metrics.Count(item => item.WarningLabel == "improvedWithinNoise");
        int unchanged = metrics.Count(item => item.WarningLabel == "unchanged");
        int notComparable = isNotComparable ? 1 : metrics.Count(item => item.WarningLabel == "notComparable");

        var labels = metrics.Select(item => item.WarningLabel)
            .Where(label => label != "unchanged")
            .Concat(dirtyCurrent ? ["dirtyCurrentWarning"] : Array.Empty<string>())
            .Concat(isNotComparable ? ["notComparable"] : Array.Empty<string>())
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();

        bool warningsPresent = correctness + performance + allocation + dirty > 0;
        bool inconclusive = !warningsPresent &&
            metrics.Length > 0 &&
            metrics.Where(IsPerformanceOrAllocation)
                .All(item => item.WarningLabel is "noiseUnavailable" or "noiseDominated" or "unchanged" or "improvedWithinNoise") &&
            metrics.Any(item => item.WarningLabel is "noiseUnavailable" or "noiseDominated");

        return new ComparisonWarningSummary(
            warningsPresent ? "warningsPresent" : inconclusive ? "inconclusive" : "noWarnings",
            labels,
            correctness,
            performance,
            allocation,
            dirty,
            noiseDominated,
            noiseUnavailable,
            improved,
            improvedWithinNoise,
            unchanged,
            notComparable);
    }

    private static ComparisonWarningSummary CreateMatrixWarningSummary(
        IReadOnlyList<MatrixCaseComparisonEntry> cases,
        bool dirtyCurrent)
    {
        MetricComparisonEntry[] metrics = cases.SelectMany(item => item.Metrics).ToArray();
        ComparisonWarningSummary metricSummary = CreateWarningSummary(metrics, isNotComparable: false, dirtyCurrent);
        int notComparable = cases.Count(item => item.Compatibility.Status != "comparable");

        return metricSummary with
        {
            NotComparableCount = notComparable,
            Labels = metricSummary.Labels
                .Concat(notComparable > 0 ? ["notComparable"] : Array.Empty<string>())
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal)
                .ToArray()
        };
    }

    private static bool IsPerformanceOrAllocation(MetricComparisonEntry metric) =>
        metric.Name.StartsWith("search.", StringComparison.Ordinal) &&
        metric.Direction is "lowerIsBetter" or "higherIsBetter";

    private static bool HasNoWarnings(ComparisonWarningSummary summary) =>
        summary.CorrectnessWarningCount == 0 &&
        summary.PerformanceWarningCount == 0 &&
        summary.AllocationWarningCount == 0 &&
        summary.DirtyCurrentWarningCount == 0 &&
        summary.NotComparableCount == 0;

    private static void AddCaseReportMatchReasons(List<CompatibilityReason> reasons, GeneratedExactMatrixCaseManifest matrixCase, BenchmarkReport report, string prefix)
    {
        Compare(reasons, $"{prefix}.reportId", matrixCase.ReportId, report.ReportId);
        Compare(reasons, $"{prefix}.metric", matrixCase.Metric, report.Dataset.Metric);
        Compare(reasons, $"{prefix}.indexMetric", matrixCase.Metric, report.Index.Metric);
        Compare(reasons, $"{prefix}.dimension", matrixCase.Dimension, report.Dataset.Dimension);
        Compare(reasons, $"{prefix}.indexDimension", matrixCase.Dimension, report.Index.Dimension);
        Compare(reasons, $"{prefix}.vectorCount", matrixCase.VectorCount, report.Dataset.VectorCount);
        Compare(reasons, $"{prefix}.indexVectorCount", matrixCase.VectorCount, report.Index.VectorCount);
        Compare(reasons, $"{prefix}.queryCount", matrixCase.QueryCount, report.Dataset.QueryCount);
        Compare(reasons, $"{prefix}.measuredQueryCount", matrixCase.QueryCount, report.Search.MeasuredQueryCount);
        Compare(reasons, $"{prefix}.topK", matrixCase.TopK, report.Scenario.TopK);
        Compare(reasons, $"{prefix}.runs", matrixCase.Runs, report.Search.Aggregate.RunCount);
        Compare(reasons, $"{prefix}.warmupQueries", matrixCase.WarmupQueries, report.Measurement.Warmup.WarmupCount);
    }

    private static BenchmarkReport? LoadReportForCase(List<CompatibilityReason> reasons, string path, string field)
    {
        try
        {
            if (!File.Exists(path))
            {
                reasons.Add(new CompatibilityReason("missingLinkedReport", field, "existing file", path, "linked matrix case report is missing"));
                return null;
            }

            BenchmarkReport? report = ReportWriter.Deserialize<BenchmarkReport>(File.ReadAllText(path));
            if (report is null)
            {
                reasons.Add(new CompatibilityReason("invalidLinkedReport", field, "schema 0.1 report JSON", path, "linked matrix case report is invalid"));
            }

            return report;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            reasons.Add(new CompatibilityReason("unreadableLinkedReport", field, "readable schema 0.1 report JSON", path, ex.Message));
            return null;
        }
    }

    private static LoadedArtifact Load(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return LoadedArtifact.Failed(path, new CompatibilityReason("missingArtifact", "path", "existing file", path, "input artifact is missing"));
            }

            string json = File.ReadAllText(path);
            using JsonDocument document = JsonDocument.Parse(json);
            string? schemaName = document.RootElement.TryGetProperty("schemaName", out JsonElement schema)
                ? schema.GetString()
                : null;

            if (schemaName == "VecNet.BenchmarkReport")
            {
                BenchmarkReport? report = ReportWriter.Deserialize<BenchmarkReport>(json);
                return report is null
                    ? LoadedArtifact.Failed(path, new CompatibilityReason("invalidJson", "schemaName", "VecNet.BenchmarkReport", schemaName, "report JSON could not be deserialized"))
                    : LoadedArtifact.ForReport(path, report);
            }

            if (schemaName == "VecNet.BenchmarkMatrixManifest")
            {
                GeneratedExactMatrixManifest? manifest = ReportWriter.Deserialize<GeneratedExactMatrixManifest>(json);
                return manifest is null
                    ? LoadedArtifact.Failed(path, new CompatibilityReason("invalidJson", "schemaName", "VecNet.BenchmarkMatrixManifest", schemaName, "matrix manifest JSON could not be deserialized"))
                    : LoadedArtifact.ForMatrix(path, manifest);
            }

            return LoadedArtifact.Failed(path, new CompatibilityReason("unsupportedSchema", "schemaName", "VecNet.BenchmarkReport or VecNet.BenchmarkMatrixManifest", schemaName, "input artifact schema is unsupported"));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return LoadedArtifact.Failed(path, new CompatibilityReason("unreadableOrInvalidJson", "path", "readable JSON artifact", path, ex.Message));
        }
    }

    private static ComparisonBuildResult CreateKindMismatchResult(LoadedArtifact baseline, LoadedArtifact current)
    {
        string baselineKind = baseline.Identity.Kind;
        string currentKind = current.Identity.Kind;
        bool sameUnknown = baselineKind == UnknownKind && currentKind == UnknownKind;
        var reasons = sameUnknown
            ? Array.Empty<CompatibilityReason>()
            :
            [
                new CompatibilityReason(
                    "artifactKindMismatch",
                    "artifactKind",
                    baselineKind,
                    currentKind,
                    "report/manifest kind mixing is not comparable")
            ];

        return new ComparisonBuildResult(
            UnknownKind,
            new ComparisonCompatibility("notComparable", reasons),
            [],
            [],
            null,
            CreateWarningSummary([], isNotComparable: true, dirtyCurrent: false));
    }

    private static Dictionary<string, GeneratedExactMatrixCaseManifest> CreateCaseMap(IEnumerable<GeneratedExactMatrixCaseManifest> cases)
    {
        var map = new Dictionary<string, GeneratedExactMatrixCaseManifest>(StringComparer.Ordinal);
        foreach (GeneratedExactMatrixCaseManifest matrixCase in cases)
        {
            map.TryAdd(CreateCaseKey(matrixCase), matrixCase);
        }

        return map;
    }

    private static string CreateCaseKey(GeneratedExactMatrixCaseManifest matrixCase) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"{matrixCase.Metric}:{matrixCase.Dimension}:{matrixCase.VectorCount}:{matrixCase.QueryCount}:{matrixCase.TopK}:{matrixCase.Runs}:{matrixCase.WarmupQueries}:{matrixCase.Seed}");

    private static bool HasOnlyAllowedCurrentUnsatisfiedConditions(BaselineCandidateEligibilityInfo eligibility) =>
        eligibility.UnsatisfiedConditions.Length > 0 &&
        eligibility.UnsatisfiedConditions.All(
            condition =>
                condition == "repository working tree is clean" ||
                condition == "exact validation passed with perfect recall, ordering and distance agreement");

    private static string[] EvaluateCurrentMatrixManifestConditions(GeneratedExactMatrixManifest manifest)
    {
        var unsatisfied = new List<string>();
        AddCondition(
            manifest.SchemaName == "VecNet.BenchmarkMatrixManifest" && manifest.SchemaVersion == "0.1",
            "matrix manifest schema is VecNet.BenchmarkMatrixManifest 0.1");
        AddCondition(manifest.ScenarioName == GeneratedExactMatrixOptions.ScenarioName, "scenario is exact-generated-matrix");
        AddCondition(manifest.PresetName == GeneratedExactMatrixOptions.StandardPresetName, "preset is standard");
        AddCondition(
            manifest.Eligibility.ClaimClass == "local-evidence" && manifest.Eligibility.PrivacyClass == "private-raw",
            "manifest is private local evidence");
        AddCondition(!manifest.Eligibility.PublicClaimEligible, "public-claim eligibility is false");
        AddCondition(!manifest.Eligibility.RegressionGateEligible, "regression-gate eligibility is false");
        AddCondition(!string.IsNullOrWhiteSpace(manifest.Repository.Commit), "repository commit is present");
        AddCondition(!manifest.Repository.Dirty, "repository working tree is clean");
        AddCondition(manifest.Aggregate.FailedCaseCount == 0, "failed case count is zero");
        AddCondition(
            manifest.CaseCount == manifest.Cases.Length &&
            manifest.CaseCount == StandardMetrics.Length * StandardDimensions.Length * StandardTopKValues.Length,
            "all standard matrix cases are present");
        AddCondition(HasCanonicalStandardMatrixCases(manifest.Cases), "standard matrix case set is canonical");
        AddCondition(
            manifest.Cases.All(
                matrixCase =>
                    matrixCase.Status == "passed" &&
                    matrixCase.ValidationStatus == "passed" &&
                    !string.IsNullOrWhiteSpace(matrixCase.ReportId) &&
                    !string.IsNullOrWhiteSpace(matrixCase.ReportPath)),
            "all cases passed and link to per-case reports");

        return unsatisfied.ToArray();

        void AddCondition(bool condition, string conditionName)
        {
            if (!condition)
            {
                unsatisfied.Add(conditionName);
            }
        }
    }

    private static bool HasCanonicalStandardMatrixCases(IEnumerable<GeneratedExactMatrixCaseManifest> cases)
    {
        var expected = new HashSet<string>(StringComparer.Ordinal);
        foreach (string metric in StandardMetrics)
        {
            foreach (int dimension in StandardDimensions)
            {
                foreach (int topK in StandardTopKValues)
                {
                    expected.Add(CreateStandardCaseKey(metric, dimension, topK));
                }
            }
        }

        var actual = new HashSet<string>(StringComparer.Ordinal);
        foreach (GeneratedExactMatrixCaseManifest matrixCase in cases)
        {
            if (!actual.Add(CreateStandardCaseKey(matrixCase.Metric, matrixCase.Dimension, matrixCase.TopK)))
            {
                return false;
            }
        }

        return actual.SetEquals(expected);
    }

    private static string CreateStandardCaseKey(string metric, int dimension, int topK) => $"{metric}:{dimension}:{topK}";

    private static bool HasOnlyAllowedDirtyCurrentMatrixUnsatisfiedConditions(string[] unsatisfiedConditions) =>
        unsatisfiedConditions.Length > 0 &&
        unsatisfiedConditions.All(condition => condition == "repository working tree is clean");

    private static void Compare<T>(List<CompatibilityReason> reasons, string field, T expected, T actual)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            reasons.Add(new CompatibilityReason(
                "fieldMismatch",
                field,
                Convert.ToString(expected, CultureInfo.InvariantCulture),
                Convert.ToString(actual, CultureInfo.InvariantCulture),
                "required compatibility field does not match"));
        }
    }

    private static void Require(List<CompatibilityReason> reasons, bool condition, string field, string? expected, string? actual, string message)
    {
        if (!condition)
        {
            reasons.Add(new CompatibilityReason("requirementNotSatisfied", field, expected, actual, message));
        }
    }

    private static string CreateComparisonId(string artifactKind, string? baselineId, string? currentId)
    {
        string baseline = string.IsNullOrWhiteSpace(baselineId) ? "unknown-baseline" : baselineId;
        string current = string.IsNullOrWhiteSpace(currentId) ? "unknown-current" : currentId;
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{BenchmarkComparisonOptions.ScenarioName}-{artifactKind}-{baseline}-to-{current}-{DateTimeOffset.UtcNow:yyyyMMddHHmmss}");
    }

    private sealed record ComparisonBuildResult(
        string ArtifactKind,
        ComparisonCompatibility Compatibility,
        MetricComparisonEntry[] Metrics,
        MatrixCaseComparisonEntry[] Cases,
        MatrixComparisonSummary? MatrixSummary,
        ComparisonWarningSummary Warnings);

    private sealed record LoadedArtifact(
        BenchmarkReport? Report,
        GeneratedExactMatrixManifest? Matrix,
        ComparisonArtifactIdentity Identity,
        CompatibilityReason? LoadFailure)
    {
        public static LoadedArtifact ForReport(string path, BenchmarkReport report) =>
            new(
                report,
                null,
                new ComparisonArtifactIdentity(
                    "generated-exact-report",
                    path,
                    report.ReportId,
                    report.SchemaName,
                    report.SchemaVersion,
                    report.GeneratedAtUtc,
                    report.Repository,
                    BaselineCandidateEligibility.EvaluateGeneratedExactReport(report).Eligible,
                    report.Repository.Dirty),
                null);

        public static LoadedArtifact ForMatrix(string path, GeneratedExactMatrixManifest manifest) =>
            new(
                null,
                manifest,
                new ComparisonArtifactIdentity(
                    "generated-exact-standard-matrix-manifest",
                    path,
                    string.Create(CultureInfo.InvariantCulture, $"{manifest.ScenarioName}:{manifest.PresetName}:{manifest.CaseCount}"),
                    manifest.SchemaName,
                    manifest.SchemaVersion,
                    manifest.GeneratedAtUtc,
                    manifest.Repository,
                    BaselineCandidateEligibility.EvaluateGeneratedExactMatrix(manifest).Eligible,
                    manifest.Repository.Dirty),
                null);

        public static LoadedArtifact Failed(string path, CompatibilityReason reason) =>
            new(
                null,
                null,
                new ComparisonArtifactIdentity(UnknownKind, path, null, null, null, null, null, BaselineCandidateEligible: false, Dirty: false),
                reason);
    }
}
