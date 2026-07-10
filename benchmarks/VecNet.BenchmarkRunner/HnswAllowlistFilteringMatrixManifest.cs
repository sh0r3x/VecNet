namespace VecNet.BenchmarkRunner;

public sealed record HnswAllowlistFilteringMatrixManifest(
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
    HnswAllowlistFilteringMatrixDesignInfo Design,
    int CaseCount,
    string ValidationStatus,
    HnswAllowlistFilteringMatrixCaseManifest[] Cases,
    HnswAllowlistFilteringMatrixAggregate Aggregate,
    HnswAllowlistFilteringMatrixEligibility Eligibility,
    string[] Notes);

public sealed record HnswAllowlistFilteringMatrixDesignInfo(
    string Metric,
    int[] Dimensions,
    int[] TopKValues,
    string[] FilterProfiles,
    HnswAllowlistFilteringMatrixUpdateProfileInfo[] UpdateProfiles,
    HnswAllowlistFilteringMatrixHnswProfileInfo HnswProfile,
    string MatrixSeed,
    string HnswMatrixSeedBase,
    string SeedPolicy,
    string WorkloadPolicy,
    string PresetPolicy,
    string ScopePolicy);

public sealed record HnswAllowlistFilteringMatrixUpdateProfileInfo(
    string Name,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    string Description);

public sealed record HnswAllowlistFilteringMatrixHnswProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record HnswAllowlistFilteringMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string FilterProfile,
    string UpdateProfileName,
    string BranchFocus,
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
    string RelativeReportPath,
    string RelativeOpenedIndexDirectoryPath,
    string RelativeCheckpointDirectoryPath,
    string[] CommandArguments,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    HnswAllowlistFilteringMatrixBranchSummary BranchSummary,
    HnswAllowlistFilteringMatrixFallbackSummary ExactFallbackParity,
    HnswAllowlistFilteringMatrixBroadEmissionSummary BroadEmission,
    HnswAllowlistFilteringMatrixUnderfillSummary Underfill,
    HnswAllowlistFilteringMatrixAllowlistSummary Allowlist,
    HnswAllowlistFilteringMatrixTombstoneSummary Tombstones,
    HnswAllowlistFilteringMatrixDeltaScanSummary ExactFilteredDeltaScan,
    HnswAllowlistFilteringMatrixMutationSummary Mutations,
    HnswAllowlistFilteringMatrixCountSummary Counts,
    HnswAllowlistFilteringMatrixIntegritySummary ReturnedResultIntegrity,
    HnswAllowlistFilteringMatrixAllocationSummary Allocations,
    HnswAllowlistFilteringMatrixEligibilitySummary RecursiveEligibility,
    string? ErrorMessage);

public sealed record HnswAllowlistFilteringMatrixBranchSummary(
    string Status,
    int ExactFallbackQueryCount,
    int BroadEmissionQueryCount,
    int BranchThresholdEfSearch,
    string? ExpectedBranch,
    string? BranchConsistencyStatus,
    int? BranchMismatchCount);

public sealed record HnswAllowlistFilteringMatrixFallbackSummary(
    string Status,
    int ExactFallbackQueryCount,
    bool? AllSearchesPassed,
    int? CountMismatchCount,
    int? IdOrOrderMismatchCount,
    int? DistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixBroadEmissionSummary(
    string Status,
    int BroadEmissionQueryCount,
    double? MinRecallAtK,
    double? MaxRecallAtK,
    double? MinOrderedAgreement,
    double? MaxOrderedAgreement,
    int? MissingResultCount,
    int? ExtraResultCount,
    int? DistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixUnderfillSummary(
    string Status,
    int QueryCount,
    int RequestedResultCountPerQuery,
    int TotalRequestedResultSlots,
    int? TotalReturnedResults,
    int? TotalExactTruthAvailableResults,
    int? UnderfilledQueryCount,
    int? UnderfilledSlotCount);

public sealed record HnswAllowlistFilteringMatrixAllowlistSummary(
    string Status,
    string? Profile,
    int? InputIdCountPerQuery,
    int? DistinctInputIdCountPerQuery,
    int? KnownIdCountPerQuery,
    int? UnknownIdCountPerQuery,
    int? DuplicateInputIdCountPerQuery,
    int? TombstonedInputIdCountPerQuery,
    int? KnownLiveAllowedCountPerQuery,
    int? LiveBaseAllowedCountPerQuery,
    int? LiveDeltaAllowedCountPerQuery,
    int? KnownLiveAllowedMin,
    double? KnownLiveAllowedMean,
    int? KnownLiveAllowedMax);

public sealed record HnswAllowlistFilteringMatrixTombstoneSummary(
    string Status,
    int? BaseTombstoneCount,
    int? DeltaTombstoneCount,
    int? TombstoneCount,
    int? TombstonedInputIdCountPerQuery,
    int? ReturnedBaseTombstoneCount,
    int? ReturnedDeltaTombstoneCount,
    bool? SuppressionPassedForAllSearches);

public sealed record HnswAllowlistFilteringMatrixDeltaScanSummary(
    string Status,
    int? SourceLiveDeltaScannedCountPerQuery,
    int? SourceAllowedLiveDeltaCountPerQuery,
    int? SourceTotalEmittedDeltaResultCount,
    int? RebuiltLiveDeltaScannedCountPerQuery,
    int? RebuiltAllowedLiveDeltaCountPerQuery,
    int? RebuiltTotalEmittedDeltaResultCount);

public sealed record HnswAllowlistFilteringMatrixMutationSummary(
    string Status,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int? CommittedMutationCount,
    bool? GenerationDeltaMatchesCommittedMutations,
    long? GenerationAfterMutations);

public sealed record HnswAllowlistFilteringMatrixCountSummary(
    string Status,
    int BasePhysicalVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int? PreCheckpointLiveVectorCount,
    int? PreCheckpointTombstoneCount,
    int? PostCheckpointLiveVectorCount,
    int? PostCheckpointTombstoneCount,
    int? DeletedReservedIdCount,
    double? PreCheckpointTombstoneRatio,
    double? PreCheckpointDeltaInsertRatio);

public sealed record HnswAllowlistFilteringMatrixIntegritySummary(
    string Status,
    bool? PassedForAllSearches,
    int? CheckedResultCount,
    int? UnknownIdCount,
    int? TombstonedIdCount,
    int? NotAllowedIdCount,
    int? DuplicateIdCount,
    int? NonFiniteDistanceCount,
    int? DistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixAllocationSummary(
    string Status,
    double? MaxMeanManagedAllocatedBytesPerSearchCall,
    double? MaxManagedAllocatedBytesPerSearchCall,
    long? MaxManagedAllocatedBytesPerRun);

public sealed record HnswAllowlistFilteringMatrixEligibilitySummary(
    string Status,
    bool LinkedReportInspected,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool AllEligibilityFlagsFalse);

public sealed record HnswAllowlistFilteringMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount,
    int LinkedReportCount,
    HnswAllowlistFilteringMatrixAggregateBranchCoverage BranchCoverage,
    HnswAllowlistFilteringMatrixAggregateFallbackSummary ExactFallbackParity,
    HnswAllowlistFilteringMatrixAggregateBroadEmissionSummary BroadEmission,
    HnswAllowlistFilteringMatrixAggregateUnderfillSummary Underfill,
    HnswAllowlistFilteringMatrixAggregateAllowlistSummary Allowlist,
    HnswAllowlistFilteringMatrixAggregateMutationCountSummary MutationCounts,
    HnswAllowlistFilteringMatrixAggregateIntegritySummary ReturnedResultIntegrity,
    HnswAllowlistFilteringMatrixAggregateAllocationSummary Allocations,
    HnswAllowlistFilteringMatrixEligibilitySummary RecursiveEligibility);

public sealed record HnswAllowlistFilteringMatrixAggregateBranchCoverage(
    string Status,
    int ExactFallbackCaseCount,
    int BroadEmissionCaseCount,
    int MixedBranchCaseCount,
    int BranchMismatchCaseCount,
    string[] CoveredFilterProfiles);

public sealed record HnswAllowlistFilteringMatrixAggregateFallbackSummary(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int TotalCountMismatchCount,
    int TotalIdOrOrderMismatchCount,
    int TotalDistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixAggregateBroadEmissionSummary(
    string Status,
    int RecordedCaseCount,
    double? MinRecallAtK,
    double? MaxRecallAtK,
    double? MinOrderedAgreement,
    double? MaxOrderedAgreement,
    int TotalMissingResultCount,
    int TotalExtraResultCount,
    int TotalDistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixAggregateUnderfillSummary(
    string Status,
    int RecordedCaseCount,
    int TotalRequestedResultSlots,
    int TotalReturnedResults,
    int TotalExactTruthAvailableResults,
    int TotalUnderfilledQueryCount,
    int TotalUnderfilledSlotCount);

public sealed record HnswAllowlistFilteringMatrixAggregateAllowlistSummary(
    string Status,
    int RecordedCaseCount,
    int MinKnownLiveAllowedPerQuery,
    double MeanKnownLiveAllowedPerQuery,
    int MaxKnownLiveAllowedPerQuery,
    int TotalUnknownInputIds,
    int TotalDuplicateInputIds,
    int TotalTombstonedInputIds);

public sealed record HnswAllowlistFilteringMatrixAggregateMutationCountSummary(
    string Status,
    int RecordedCaseCount,
    int TotalInsertedDeltaVectorCount,
    int TotalDeletedBaseVectorCount,
    int TotalDeletedDeltaVectorCount,
    int TotalTombstoneCount,
    int MinLiveVectorCount,
    int MaxLiveVectorCount);

public sealed record HnswAllowlistFilteringMatrixAggregateIntegritySummary(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int TotalCheckedResultCount,
    int TotalUnknownIdCount,
    int TotalTombstonedIdCount,
    int TotalNotAllowedIdCount,
    int TotalDuplicateIdCount,
    int TotalNonFiniteDistanceCount,
    int TotalDistanceMismatchCount);

public sealed record HnswAllowlistFilteringMatrixAggregateAllocationSummary(
    string Status,
    int RecordedCaseCount,
    double? MaxMeanManagedAllocatedBytesPerSearchCall,
    double? MaxManagedAllocatedBytesPerSearchCall,
    long? MaxManagedAllocatedBytesPerRun);

public sealed record HnswAllowlistFilteringMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason);
