namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record ExternalHnswBasePlusExactDeltaMatrixManifest(
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
    ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo CacheTruth,
    ExternalHnswBasePlusExactDeltaMatrixDesignInfo Design,
    int CaseCount,
    ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] Cases,
    ExternalHnswBasePlusExactDeltaMatrixAggregate Aggregate,
    ExternalHnswBasePlusExactDeltaMatrixEligibility Eligibility,
    string[] Notes);

public sealed record ExternalHnswBasePlusExactDeltaMatrixCacheTruthInfo(
    string Status,
    string CacheRoot,
    string DatasetId,
    int ExpectedDimension,
    string Metric,
    string CachePolicy,
    string TruthPolicy,
    string? AdmissionManifestPath,
    string? AdmissionManifestSha256,
    string? TruthRelativePath,
    string? TruthSha256,
    int? BaseVectorCount,
    int? QueryMatrixCount,
    int? TruthQuerySubsetCount,
    int? TruthDepth,
    string? ErrorMessage);

public sealed record ExternalHnswBasePlusExactDeltaMatrixDesignInfo(
    string DatasetId,
    int Dimension,
    string Metric,
    int QueryCount,
    int[] TopKValues,
    ExternalHnswBasePlusExactDeltaMatrixUpdateProfileInfo[] UpdateProfiles,
    ExternalHnswBasePlusExactDeltaMatrixHnswProfileInfo[] HnswProfiles,
    string MatrixSeed,
    string WorkloadSeedPolicy,
    string HnswSeedPolicy,
    string RowSelectionPolicy,
    string ExactUpdatedTruthPolicy,
    string MeasurementBoundaryPolicy,
    string ScopePolicy);

public sealed record ExternalHnswBasePlusExactDeltaMatrixUpdateProfileInfo(
    string Name,
    int BaseStartRow,
    int BaseEndRowInclusive,
    int BaseRowCount,
    int DeltaStartRow,
    int DeltaEndRowInclusive,
    int DeltaRowCount,
    int UnusedCandidateRowCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int ExpectedLiveVectorCount,
    string Description);

public sealed record ExternalHnswBasePlusExactDeltaMatrixHnswProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record ExternalHnswBasePlusExactDeltaMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string UpdateProfileName,
    string HnswProfileName,
    string DatasetId,
    string Metric,
    int Dimension,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string WorkloadSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    int ImmutableBaseStartRow,
    int ImmutableBaseEndRowInclusive,
    int ImmutableBaseRowCount,
    int DeltaStartRow,
    int DeltaEndRowInclusive,
    int DeltaRowCount,
    int UnusedCandidateRowCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int ExpectedPhysicalVectorCount,
    int ExpectedLiveVectorCount,
    string LinkedReportPath,
    string[] CommandArguments,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary RecallOrderSummary,
    ExternalHnswBasePlusExactDeltaMatrixIntegritySummary IntegritySummary,
    ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary UnderfillSummary,
    ExternalHnswBasePlusExactDeltaMatrixAllocationSummary AllocationSummary,
    ExternalHnswBasePlusExactDeltaMatrixMutationSummary MutationSummary,
    ExternalHnswBasePlusExactDeltaMatrixCountSummary CountSummary,
    ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary EligibilitySummary,
    string? ErrorMessage);

public sealed record ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary(
    string Status,
    double? RecallAtK,
    double? OrderedAgreement,
    string? DistanceToleranceStatus,
    int? DistanceMismatchCount,
    int? MissingResultCount,
    int? ExtraResultCount);

public sealed record ExternalHnswBasePlusExactDeltaMatrixIntegritySummary(
    string Status,
    int? CheckedResultCount,
    int? QueryCountMismatchCount,
    int? ResultCountViolationCount,
    int? NonFiniteDistanceCount,
    int? DuplicateIdCount,
    int? UnknownIdCount,
    int? TombstonedIdCount,
    int? DistanceMismatchCount);

public sealed record ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary(
    string Status,
    int QueryCount,
    int RequestedResultCountPerQuery,
    int TotalRequestedResultSlots,
    int? TotalReturnedResults,
    int? UnderfilledQueryCount,
    int? UnderfilledSlotCount);

public sealed record ExternalHnswBasePlusExactDeltaMatrixAllocationSummary(
    string Status,
    double? MeanElapsedMilliseconds,
    double? LatencyP50Milliseconds,
    double? LatencyP95Milliseconds,
    double? LatencyP99Milliseconds,
    double? Qps,
    double? MeanManagedAllocatedBytesPerSearchCall,
    string? ManagedAllocationStatus,
    string? MemoryStatus);

public sealed record ExternalHnswBasePlusExactDeltaMatrixMutationSummary(
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
    long? GenerationDelta,
    long? GenerationAfterMutations);

public sealed record ExternalHnswBasePlusExactDeltaMatrixCountSummary(
    string Status,
    int ExpectedBasePhysicalVectorCount,
    int ExpectedDeltaPhysicalVectorCount,
    int ExpectedPhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int? BasePhysicalVectorCount,
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

public sealed record ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ValidationPublicClaimEligible,
    bool ValidationBaselineCandidateEligible,
    bool ValidationRegressionGateEligible);

public sealed record ExternalHnswBasePlusExactDeltaMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount,
    int ReturnedResultIntegrityNotPassedCaseCount,
    int DistanceToleranceNotPassedCaseCount,
    ExternalHnswBasePlusExactDeltaMatrixRecallAggregate Recall,
    ExternalHnswBasePlusExactDeltaMatrixOrderAggregate Order,
    ExternalHnswBasePlusExactDeltaMatrixUnderfillAggregate Underfill,
    ExternalHnswBasePlusExactDeltaMatrixAllocationAggregate Allocation,
    ExternalHnswBasePlusExactDeltaMatrixMutationAggregate Mutations,
    ExternalHnswBasePlusExactDeltaMatrixCountAggregate Counts,
    ExternalHnswBasePlusExactDeltaMatrixEligibilityAggregate Eligibility);

public sealed record ExternalHnswBasePlusExactDeltaMatrixRecallAggregate(
    double? MinimumRecallAtK,
    double? MaximumRecallAtK,
    ExternalHnswBasePlusExactDeltaMatrixGroupedDoubleSummary[] ByTopK);

public sealed record ExternalHnswBasePlusExactDeltaMatrixOrderAggregate(
    double? MinimumOrderedAgreement,
    double? MaximumOrderedAgreement,
    ExternalHnswBasePlusExactDeltaMatrixGroupedDoubleSummary[] ByTopK);

public sealed record ExternalHnswBasePlusExactDeltaMatrixGroupedDoubleSummary(
    string Group,
    double? Minimum,
    double? Maximum);

public sealed record ExternalHnswBasePlusExactDeltaMatrixUnderfillAggregate(
    int CaseCountWithAnyUnderfill,
    int TotalUnderfilledQueryCount,
    int TotalUnderfilledSlotCount,
    ExternalHnswBasePlusExactDeltaMatrixWorstUnderfillSummary[] WorstByTopKAndUpdateProfile);

public sealed record ExternalHnswBasePlusExactDeltaMatrixWorstUnderfillSummary(
    string Group,
    int WorstUnderfilledSlotCount);

public sealed record ExternalHnswBasePlusExactDeltaMatrixAllocationAggregate(
    double? MaximumMeanManagedAllocatedBytesPerSearchCall,
    int CaseCountWithAllocationGreaterThanZero);

public sealed record ExternalHnswBasePlusExactDeltaMatrixMutationAggregate(
    int CaseCountWithMutationOrGenerationMismatch,
    int TotalCommittedMutationCount,
    int TotalDuplicateIdStatusCount,
    int TotalUnknownIdStatusCount,
    int TotalAlreadyDeletedStatusCount);

public sealed record ExternalHnswBasePlusExactDeltaMatrixCountAggregate(
    int? MinimumLiveVectorCount,
    int? MaximumLiveVectorCount,
    int? MaximumTombstoneCount,
    double? MaximumTombstoneRatio,
    double? MaximumDeltaInsertRatio);

public sealed record ExternalHnswBasePlusExactDeltaMatrixEligibilityAggregate(
    int LinkedReportNonFalseEligibilityCount,
    bool ManifestPublicClaimEligible,
    bool ManifestBaselineCandidateEligible,
    bool ManifestRegressionGateEligible,
    bool ComparisonPublicationEligible);

public sealed record ExternalHnswBasePlusExactDeltaMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ComparisonPublicationEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string ComparisonPublicationReason);
