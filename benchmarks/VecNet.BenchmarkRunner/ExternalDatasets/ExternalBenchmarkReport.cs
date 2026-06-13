namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record ExternalBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    ExternalBenchmarkEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    ExternalBenchmarkDatasetInfo Dataset,
    ExternalBenchmarkWorkloadInfo Workload,
    ExternalBenchmarkTruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    SearchInfo Search,
    MeasurementInfo Measurement,
    ExternalBenchmarkMetricsInfo Metrics,
    ExternalBenchmarkValidationInfo Validation,
    ExternalBenchmarkEligibilityInfo Eligibility,
    string[] Notes);

public sealed record ExternalBenchmarkEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record ExternalBenchmarkDatasetInfo(
    string DatasetId,
    ExternalDatasetSource Source,
    ExternalDatasetLicense License,
    ExternalDatasetPrivacy Privacy,
    ExternalDatasetShape Shape,
    ExternalDatasetMetric Metric,
    ExternalBenchmarkAdmissionManifestInfo AdmissionManifest,
    ExternalRawFileManifestEntry[] RawFiles,
    ExternalConvertedMatrixEntry[] ConvertedMatrices,
    ConversionManifestSummary Conversion,
    FashionMnistLabelMetadata Labels);

public sealed record ExternalBenchmarkAdmissionManifestInfo(
    string SchemaName,
    string SchemaVersion,
    string RelativePath,
    string Sha256);

public sealed record ExternalBenchmarkWorkloadInfo(
    int BaseCount,
    int QueryMatrixCount,
    int MeasuredQueryCount,
    string QuerySubsetPolicy,
    int Dimension,
    string SourceDataType,
    string ConvertedDataType,
    string UpstreamMetric,
    string VecNetMetric,
    string MetricMappingNote,
    int TopK,
    int TruthDepth,
    string TiePolicy);

public sealed record ExternalBenchmarkTruthInfo(
    string SchemaName,
    string SchemaVersion,
    string Kind,
    string RelativePath,
    string Sha256,
    string QuerySubsetPolicy,
    int QuerySubsetCount,
    int TruthDepth,
    int TopK,
    string TiePolicy,
    string DistanceSemantics,
    string[] SourceRawSha256);

public sealed record ExternalBenchmarkMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount);

public sealed record ExternalBenchmarkValidationInfo(
    string Status,
    string EvidenceStatus,
    bool LoadedExistingTruth,
    bool FinalRunComparedToTruth,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record ExternalBenchmarkEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
