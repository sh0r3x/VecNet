namespace VecNet.BenchmarkRunner;

public static class BaselineCandidateEligibility
{
    public const int MinimumRuns = 3;
    public const int MinimumMeasuredQueries = 100;

    private const string Policy = "D-038 private generated exact baseline candidate policy";
    private const string ReportKind = "generated-exact-report";
    private const string MatrixKind = "generated-exact-standard-matrix-manifest";

    private static readonly string[] StandardMetrics =
    [
        VectorMetric.SquaredEuclidean.ToString(),
        VectorMetric.InnerProduct.ToString(),
        VectorMetric.Cosine.ToString()
    ];

    private static readonly int[] StandardDimensions = [32, 128, 386, 768];
    private static readonly int[] StandardTopKValues = [1, 10, 100];

    public static BenchmarkReport ApplyGeneratedExactReportEligibility(BenchmarkReport report)
    {
        BaselineCandidateEligibilityInfo eligibility = EvaluateGeneratedExactReport(report);
        string suitability = eligibility.Eligible ? "private-baseline-candidate" : "smoke";

        return report with
        {
            Baseline = report.Baseline with
            {
                Suitability = suitability,
                BaselineCandidateEligible = eligibility.Eligible,
                RegressionGateEligible = false,
                Reason = eligibility.Reason,
                CandidateEligibility = eligibility
            },
            Validation = report.Validation with
            {
                BaselineCandidateEligible = eligibility.Eligible,
                PublicClaimEligible = false
            }
        };
    }

    public static GeneratedExactMatrixManifest ApplyGeneratedExactMatrixEligibility(GeneratedExactMatrixManifest manifest)
    {
        BaselineCandidateEligibilityInfo eligibility = EvaluateGeneratedExactMatrix(manifest);
        return manifest with
        {
            Eligibility = manifest.Eligibility with
            {
                PublicClaimEligible = false,
                BaselineCandidateEligible = eligibility.Eligible,
                RegressionGateEligible = false,
                Reason = eligibility.Reason,
                CandidateEligibility = eligibility
            }
        };
    }

    public static BaselineCandidateEligibilityInfo EvaluateGeneratedExactReport(BenchmarkReport report)
    {
        var builder = new ConditionBuilder();

        builder.Require(
            report.SchemaName == "VecNet.BenchmarkReport" && report.SchemaVersion == "0.1",
            "report schema is VecNet.BenchmarkReport 0.1");
        builder.Require(
            report.Command.Scenario == GeneratedExactSearchOptions.ScenarioName &&
            report.Scenario.Name == GeneratedExactSearchOptions.ScenarioName,
            "scenario is exact-generated");
        builder.Require(
            report.ClaimClass == "local-evidence" &&
            report.PrivacyClass == "private-raw" &&
            report.Evidence.Scope == "local-evidence",
            "artifact is private local evidence");
        builder.Require(!report.Evidence.PublicClaimEligible, "public-claim eligibility is false");
        builder.Require(!report.Baseline.RegressionGateEligible, "regression-gate eligibility is false");
        builder.Require(!string.IsNullOrWhiteSpace(report.Repository.Commit), "repository commit is present");
        builder.Require(!report.Repository.Dirty, "repository working tree is clean");
        builder.Require(
            report.Dataset.Kind == GeneratedDataset.Kind &&
            report.Dataset.SourceVerificationStatus == "generated-no-external-source",
            "dataset is generated with no external source");
        builder.Require(
            report.Truth.Kind == ScalarGroundTruth.Kind &&
            report.Truth.Depth >= report.Scenario.TopK,
            "scalar-reference truth metadata is complete");
        builder.Require(
            report.Index.Profile == "Exact" &&
            report.Index.Type == nameof(ExactFlatIndex) &&
            report.Index.Metric == report.Dataset.Metric &&
            report.Index.Dimension == report.Dataset.Dimension &&
            report.Index.VectorCount == report.Dataset.VectorCount,
            "exact index metadata matches the generated workload");
        builder.Require(
            report.Validation.Status == "passed" &&
            report.Metrics.RecallAtK == 1 &&
            report.Metrics.OrderedAgreement == 1 &&
            report.Metrics.DistanceToleranceStatus == "passed" &&
            report.Metrics.MissingResultCount == 0 &&
            report.Metrics.DistanceMismatchCount == 0,
            "exact validation passed with perfect recall, ordering and distance agreement");
        builder.Require(
            report.Measurement.Latency is not null &&
            report.Measurement.Latency.Status == "measured" &&
            report.Measurement.Latency.SampleScope == "perMeasuredQuery",
            "latency metadata is present");
        builder.Require(
            report.Measurement.RepeatedRuns is not null &&
            report.Measurement.RepeatedRuns.RunCount == report.Search.Aggregate.RunCount,
            "repeated-run metadata is present");
        builder.Require(
            report.Measurement.ManagedAllocations is not null &&
            report.Measurement.ManagedAllocations.Status == "measured" &&
            report.Measurement.ManagedAllocations.Unit == "bytesPerQuery",
            "managed allocations are measured in bytes per query");
        builder.Require(
            report.Measurement.Memory is not null &&
            report.Measurement.Memory.Status == "notMeasured" &&
            report.Measurement.Memory.Value == "absent",
            "resident/process memory is explicitly not measured");
        builder.Require(
            report.Measurement.RunToRunNoise is not null &&
            report.Measurement.RunToRunNoise.RunCount == report.Search.Aggregate.RunCount,
            "run-to-run noise metadata is present");
        builder.Require(report.Search.Aggregate.RunCount >= MinimumRuns, "run count is at least 3");
        builder.Require(report.Search.MeasuredQueryCount >= MinimumMeasuredQueries, "measured query count is at least 100");

        return builder.ToEligibility(ReportKind);
    }

    public static BaselineCandidateEligibilityInfo EvaluateGeneratedExactMatrix(GeneratedExactMatrixManifest manifest)
    {
        var builder = new ConditionBuilder();

        builder.Require(
            manifest.SchemaName == "VecNet.BenchmarkMatrixManifest" && manifest.SchemaVersion == "0.1",
            "matrix manifest schema is VecNet.BenchmarkMatrixManifest 0.1");
        builder.Require(
            manifest.ScenarioName == GeneratedExactMatrixOptions.ScenarioName,
            "scenario is exact-generated-matrix");
        builder.Require(
            manifest.PresetName == GeneratedExactMatrixOptions.StandardPresetName,
            "preset is standard");
        builder.Require(
            manifest.Eligibility.ClaimClass == "local-evidence" &&
            manifest.Eligibility.PrivacyClass == "private-raw",
            "manifest is private local evidence");
        builder.Require(!manifest.Eligibility.PublicClaimEligible, "public-claim eligibility is false");
        builder.Require(!manifest.Eligibility.RegressionGateEligible, "regression-gate eligibility is false");
        builder.Require(!string.IsNullOrWhiteSpace(manifest.Repository.Commit), "repository commit is present");
        builder.Require(!manifest.Repository.Dirty, "repository working tree is clean");
        builder.Require(manifest.Aggregate.FailedCaseCount == 0, "failed case count is zero");
        builder.Require(
            manifest.CaseCount == manifest.Cases.Length &&
            manifest.CaseCount == StandardMetrics.Length * StandardDimensions.Length * StandardTopKValues.Length,
            "all standard matrix cases are present");
        builder.Require(HasCanonicalStandardCases(manifest.Cases), "standard matrix case set is canonical");

        bool allCasesPassedAndLinked = manifest.Cases.All(
            matrixCase =>
                matrixCase.Status == "passed" &&
                matrixCase.ValidationStatus == "passed" &&
                !string.IsNullOrWhiteSpace(matrixCase.ReportId) &&
                !string.IsNullOrWhiteSpace(matrixCase.ReportPath));
        builder.Require(allCasesPassedAndLinked, "all cases passed and link to per-case reports");

        foreach (GeneratedExactMatrixCaseManifest matrixCase in manifest.Cases)
        {
            EvaluateLinkedCaseReport(builder, matrixCase);
        }

        return builder.ToEligibility(MatrixKind);
    }

    private static void EvaluateLinkedCaseReport(ConditionBuilder builder, GeneratedExactMatrixCaseManifest matrixCase)
    {
        string label = $"case {matrixCase.CaseNumber:D2}";
        if (string.IsNullOrWhiteSpace(matrixCase.ReportPath))
        {
            builder.Require(false, $"{label} report path is present");
            return;
        }

        if (!File.Exists(matrixCase.ReportPath))
        {
            builder.Require(false, $"{label} linked report exists");
            return;
        }

        BenchmarkReport? report;
        try
        {
            report = ReportWriter.Deserialize<BenchmarkReport>(File.ReadAllText(matrixCase.ReportPath));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            builder.Require(false, $"{label} linked report is readable schema 0.1 JSON");
            return;
        }

        if (report is null)
        {
            builder.Require(false, $"{label} linked report is readable schema 0.1 JSON");
            return;
        }

        bool linkedReportMatchesCase =
            matrixCase.ReportId == report.ReportId &&
            matrixCase.Metric == report.Dataset.Metric &&
            matrixCase.Metric == report.Index.Metric &&
            matrixCase.Dimension == report.Dataset.Dimension &&
            matrixCase.Dimension == report.Index.Dimension &&
            matrixCase.VectorCount == report.Dataset.VectorCount &&
            matrixCase.VectorCount == report.Index.VectorCount &&
            matrixCase.QueryCount == report.Dataset.QueryCount &&
            matrixCase.QueryCount == report.Search.MeasuredQueryCount &&
            matrixCase.TopK == report.Scenario.TopK &&
            matrixCase.Runs == report.Search.Aggregate.RunCount &&
            matrixCase.WarmupQueries == report.Measurement.Warmup.WarmupCount;
        builder.Require(linkedReportMatchesCase, $"{label} linked report metadata matches the manifest case");

        BaselineCandidateEligibilityInfo reportEligibility = EvaluateGeneratedExactReport(report);
        builder.Require(reportEligibility.Eligible, $"{label} linked report is private-baseline-candidate eligible");
    }

    private static bool HasCanonicalStandardCases(GeneratedExactMatrixCaseManifest[] cases)
    {
        var expected = new HashSet<string>(StringComparer.Ordinal);
        foreach (string metric in StandardMetrics)
        {
            foreach (int dimension in StandardDimensions)
            {
                foreach (int topK in StandardTopKValues)
                {
                    expected.Add(CreateCaseKey(metric, dimension, topK));
                }
            }
        }

        var actual = new HashSet<string>(StringComparer.Ordinal);
        foreach (GeneratedExactMatrixCaseManifest matrixCase in cases)
        {
            if (!actual.Add(CreateCaseKey(matrixCase.Metric, matrixCase.Dimension, matrixCase.TopK)))
            {
                return false;
            }
        }

        return actual.SetEquals(expected);
    }

    private static string CreateCaseKey(string metric, int dimension, int topK) => $"{metric}:{dimension}:{topK}";

    private sealed class ConditionBuilder
    {
        private readonly List<string> _satisfied = [];
        private readonly List<string> _unsatisfied = [];

        public void Require(bool condition, string conditionName)
        {
            if (condition)
            {
                _satisfied.Add(conditionName);
            }
            else
            {
                _unsatisfied.Add(conditionName);
            }
        }

        public BaselineCandidateEligibilityInfo ToEligibility(string artifactKind)
        {
            bool eligible = _unsatisfied.Count == 0;
            return new BaselineCandidateEligibilityInfo(
                eligible,
                artifactKind,
                Policy,
                MinimumRuns,
                MinimumMeasuredQueries,
                _satisfied.ToArray(),
                _unsatisfied.ToArray(),
                eligible
                    ? "Artifact satisfies D-038 private generated exact baseline-candidate metadata conditions; it remains private/local and is not public-claim or regression-gate eligible."
                    : "Artifact is not a private generated exact baseline candidate because one or more D-038 conditions are unsatisfied; baseline comparison math, regression decisions and variance thresholds are not implemented.");
        }
    }
}
