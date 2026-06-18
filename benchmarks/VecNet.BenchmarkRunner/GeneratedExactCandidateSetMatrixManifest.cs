namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCandidateSetMatrixManifest(
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
    GeneratedExactCandidateSetMatrixCaseManifest[] Cases,
    GeneratedExactCandidateSetMatrixAggregate Aggregate,
    GeneratedExactCandidateSetMatrixEligibility Eligibility,
    string[] Notes);

public sealed record GeneratedExactCandidateSetMatrixCaseManifest(
    int CaseNumber,
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    string CandidateSetKind,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    string ReportPath,
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record GeneratedExactCandidateSetMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record GeneratedExactCandidateSetMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
