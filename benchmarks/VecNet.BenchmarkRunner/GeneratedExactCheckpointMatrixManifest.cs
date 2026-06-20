namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCheckpointMatrixManifest(
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
    string ValidationStatus,
    GeneratedExactCheckpointMatrixCaseManifest[] Cases,
    GeneratedExactCheckpointMatrixAggregate Aggregate,
    GeneratedExactCheckpointMatrixEligibility Eligibility,
    string[] Notes);

public sealed record GeneratedExactCheckpointMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string Metric,
    int Dimension,
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    double ExpectedTombstoneRatio,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    string AllowlistKind,
    string CandidateSetKind,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    string CheckpointMode,
    string CheckpointTargetPolicy,
    string ReportPath,
    string[] CommandArguments,
    string? ReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record GeneratedExactCheckpointMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount);

public sealed record GeneratedExactCheckpointMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool PreviewReadinessEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string PreviewReadinessReason);
