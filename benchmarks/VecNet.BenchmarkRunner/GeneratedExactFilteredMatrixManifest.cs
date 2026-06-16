namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactFilteredMatrixManifest(
    string SchemaName,
    string SchemaVersion,
    string TaskId,
    string ScenarioName,
    string PresetName,
    DateTimeOffset GeneratedAtUtc,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    string OutputDirectory,
    int CaseCount,
    GeneratedExactFilteredMatrixCaseManifest[] Cases,
    GeneratedExactFilteredMatrixAggregate Aggregate,
    GeneratedExactFilteredMatrixEligibility Eligibility,
    string[] Notes);

public sealed record GeneratedExactFilteredMatrixCaseManifest(
    int CaseNumber,
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    string FilterKind,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    string ReportPath,
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record GeneratedExactFilteredMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record GeneratedExactFilteredMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
