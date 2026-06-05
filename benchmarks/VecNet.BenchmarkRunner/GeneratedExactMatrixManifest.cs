namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactMatrixManifest(
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
    GeneratedExactMatrixCaseManifest[] Cases,
    GeneratedExactMatrixAggregate Aggregate,
    GeneratedExactMatrixEligibility Eligibility,
    string[] Notes);

public sealed record GeneratedExactMatrixCaseManifest(
    int CaseNumber,
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    string ReportPath,
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record GeneratedExactMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record GeneratedExactMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string Reason);
