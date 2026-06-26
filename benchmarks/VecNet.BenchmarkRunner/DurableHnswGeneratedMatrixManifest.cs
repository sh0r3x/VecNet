namespace VecNet.BenchmarkRunner;

public sealed record DurableHnswGeneratedMatrixManifest(
    string SchemaName,
    string SchemaVersion,
    string TaskId,
    string ScenarioName,
    string PresetName,
    DateTimeOffset GeneratedAtUtc,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    string OutputDirectory,
    int CaseCount,
    DurableHnswGeneratedMatrixCaseManifest[] Cases,
    DurableHnswGeneratedMatrixAggregate Aggregate,
    DurableHnswGeneratedMatrixValidation Validation,
    DurableHnswGeneratedMatrixEligibility Eligibility,
    string[] Notes);

public sealed record DurableHnswGeneratedMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string ProfileName,
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    string ReportPath,
    string SnapshotDirectory,
    string[] CommandArguments,
    string? ReportId,
    string? LinkedReportSchemaName,
    string? LinkedReportSchemaVersion,
    string? LinkedReportTaskId,
    string? LinkedReportScenarioName,
    string Status,
    string ValidationStatus,
    string? ErrorType,
    string? ErrorMessage);

public sealed record DurableHnswGeneratedMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record DurableHnswGeneratedMatrixValidation(
    string Status,
    int PassedCaseCount,
    int FailedCaseCount,
    string LinkedReportSchemaName,
    string LinkedReportSchemaVersion,
    string LinkedReportScenarioName,
    bool AllLinkedReportsValidationPassed,
    bool AllLinkedReportsPrivateRaw,
    bool AllLinkedReportsEligibilityFalse);

public sealed record DurableHnswGeneratedMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string PreviewReadinessReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason);
