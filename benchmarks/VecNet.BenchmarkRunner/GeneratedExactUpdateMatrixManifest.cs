namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactUpdateMatrixManifest(
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
    GeneratedExactUpdateMatrixCaseManifest[] Cases,
    GeneratedExactUpdateMatrixAggregate Aggregate,
    GeneratedExactUpdateMatrixEligibility Eligibility,
    string[] Notes);

public sealed record GeneratedExactUpdateMatrixCaseManifest(
    int CaseNumber,
    string Metric,
    int Dimension,
    int BaseVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    double ExpectedTombstoneRatio,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string AllowlistKind,
    string CandidateSetKind,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    string ReportPath,
    string[] CommandArguments,
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record GeneratedExactUpdateMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record GeneratedExactUpdateMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
