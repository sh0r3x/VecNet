namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaMatrixManifest(
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
    HnswBasePlusExactDeltaMatrixDesignInfo Design,
    int CaseCount,
    HnswBasePlusExactDeltaMatrixCaseManifest[] Cases,
    HnswBasePlusExactDeltaMatrixAggregate Aggregate,
    HnswBasePlusExactDeltaMatrixEligibility Eligibility,
    string[] Notes);

public sealed record HnswBasePlusExactDeltaMatrixDesignInfo(
    string Metric,
    int[] Dimensions,
    int[] TopKValues,
    HnswBasePlusExactDeltaMatrixHnswProfileInfo[] HnswProfiles,
    HnswBasePlusExactDeltaMatrixUpdateProfileInfo[] UpdateProfiles,
    string WorkloadPolicy,
    string PresetPolicy,
    string ScopePolicy);

public sealed record HnswBasePlusExactDeltaMatrixHnswProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record HnswBasePlusExactDeltaMatrixUpdateProfileInfo(
    string Name,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    string Description);

public sealed record HnswBasePlusExactDeltaMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string HnswProfileName,
    string UpdateProfileName,
    string Metric,
    int Dimension,
    int BaseVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string LinkedReportPath,
    string[] CommandArguments,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    HnswBasePlusExactDeltaMatrixRecallSummary RecallSummary,
    HnswBasePlusExactDeltaMatrixUnderfillSummary UnderfillSummary,
    HnswBasePlusExactDeltaMatrixMutationSummary MutationSummary,
    HnswBasePlusExactDeltaMatrixCountSummary CountSummary,
    string? ErrorMessage);

public sealed record HnswBasePlusExactDeltaMatrixRecallSummary(
    string Status,
    double? RecallAtK,
    double? OrderedAgreement,
    string? DistanceToleranceStatus,
    int? MissingResultCount,
    int? ExtraResultCount,
    string? ReturnedResultIntegrityStatus);

public sealed record HnswBasePlusExactDeltaMatrixUnderfillSummary(
    string Status,
    int QueryCount,
    int RequestedResultCountPerQuery,
    int TotalRequestedResultSlots,
    int? TotalReturnedResults,
    int? UnderfilledQueryCount,
    int? UnderfilledSlotCount);

public sealed record HnswBasePlusExactDeltaMatrixMutationSummary(
    string Status,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int? CommittedMutationCount,
    int? StatusCommitted,
    int? StatusDuplicateId,
    int? StatusUnknownId,
    int? StatusAlreadyDeleted,
    bool? GenerationDeltaMatchesCommittedMutations,
    long? GenerationAfterMutations);

public sealed record HnswBasePlusExactDeltaMatrixCountSummary(
    string Status,
    int BasePhysicalVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int? BaseLiveVectorCount,
    int? DeltaPhysicalVectorCount,
    int? DeltaLiveVectorCount,
    int? BaseTombstoneCount,
    int? DeltaTombstoneCount,
    int? TombstoneCount,
    int? LiveVectorCount,
    int? DeletedReservedIdCount,
    long? Generation,
    double? TombstoneRatio,
    double? DeltaInsertRatio);

public sealed record HnswBasePlusExactDeltaMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount);

public sealed record HnswBasePlusExactDeltaMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
