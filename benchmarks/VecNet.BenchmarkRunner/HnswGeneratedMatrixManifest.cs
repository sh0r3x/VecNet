namespace VecNet.BenchmarkRunner;

public sealed record HnswGeneratedMatrixManifest(
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
    HnswGeneratedMatrixCaseManifest[] Cases,
    HnswGeneratedMatrixAggregate Aggregate,
    HnswGeneratedMatrixEligibility Eligibility,
    string[] Notes);

public sealed record HnswGeneratedMatrixCaseManifest(
    int CaseNumber,
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
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record HnswGeneratedMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record HnswGeneratedMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
