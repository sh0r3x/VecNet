namespace VecNet.BenchmarkRunner;

public sealed record HnswAllowlistFilteringBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    HnswEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    HnswConfigurationInfo Hnsw,
    HnswAllowlistFilteringWorkloadInfo Workload,
    HnswBasePlusExactDeltaCheckpointCountInfo PreCheckpointCounts,
    HnswBasePlusExactDeltaCheckpointMutationInfo Mutations,
    HnswBasePlusExactDeltaCheckpointResultInfo CheckpointResult,
    HnswBasePlusExactDeltaCheckpointCountInfo PostCheckpointCounts,
    HnswAllowlistFilteringInfo Allowlist,
    HnswAllowlistBranchInfo Branches,
    HnswAllowlistSearchSectionsInfo Searches,
    HnswAllowlistParityInfo Parity,
    MeasurementStatusInfo Memory,
    HnswAllowlistValidationInfo Validation,
    HnswAllowlistEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswAllowlistFilteringWorkloadInfo(
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    string OpenedIndexDirectory,
    string CheckpointDirectory,
    string MutationOrder,
    string IdPolicy);

public sealed record HnswAllowlistFilteringInfo(
    string Profile,
    int QueryCount,
    int InputIdCountPerQuery,
    int DistinctInputIdCountPerQuery,
    int KnownIdCountPerQuery,
    int UnknownIdCountPerQuery,
    int DuplicateInputIdCountPerQuery,
    int TombstonedInputIdCountPerQuery,
    int KnownLiveAllowedCountPerQuery,
    int LiveBaseAllowedCountPerQuery,
    int LiveDeltaAllowedCountPerQuery,
    int KnownLiveAllowedMin,
    double KnownLiveAllowedMean,
    int KnownLiveAllowedMax,
    int TotalInputIdCount,
    int TotalDistinctInputIdCount,
    int TotalKnownIdCount,
    int TotalUnknownIdCount,
    int TotalDuplicateInputIdCount,
    int TotalTombstonedInputIdCount,
    int TotalKnownLiveAllowedCount,
    string GenerationFormula,
    string DuplicatePolicy,
    string UnknownIdPolicy,
    string TombstoneProbePolicy);

public sealed record HnswAllowlistBranchInfo(
    int ExactFallbackQueryCount,
    int BroadEmissionQueryCount,
    int BranchThresholdEfSearch,
    string ExpectedBranch,
    string BranchConsistencyStatus,
    int BranchMismatchCount,
    string Policy);

public sealed record HnswAllowlistSearchSectionsInfo(
    HnswAllowlistSearchSectionInfo ImmutableHnsw,
    HnswAllowlistSearchSectionInfo OpenedHnsw,
    HnswAllowlistSearchSectionInfo SourceComposite,
    HnswAllowlistSearchSectionInfo RebuiltComposite,
    HnswAllowlistSearchSectionInfo CheckpointOpenedHnsw);

public sealed record HnswAllowlistSearchSectionInfo(
    string Name,
    string TimedOperation,
    HnswAllowlistBranchInfo Branches,
    SearchInfo Search,
    MeasurementInfo Measurement,
    HnswAllowlistExactFallbackValidationInfo ExactFallbackValidation,
    HnswAllowlistBroadEmissionValidationInfo BroadEmissionValidation,
    HnswAllowlistReturnedResultIntegrityInfo ReturnedResultIntegrity,
    HnswAllowlistUnderfillInfo Underfill,
    HnswAllowlistDeltaScanInfo ExactFilteredDeltaScan,
    HnswAllowlistTombstoneSuppressionInfo TombstoneSuppression);

public sealed record HnswAllowlistExactFallbackValidationInfo(
    string Status,
    int QueryCount,
    int CountMismatchCount,
    int IdOrOrderMismatchCount,
    int DistanceMismatchCount,
    string Policy);

public sealed record HnswAllowlistBroadEmissionValidationInfo(
    string Status,
    double RecallAtK,
    double OrderedAgreement,
    int MissingResultCount,
    int ExtraResultCount,
    int DistanceMismatchCount,
    string Policy);

public sealed record HnswAllowlistReturnedResultIntegrityInfo(
    string Status,
    int CheckedResultCount,
    int QueryCountMismatchCount,
    int ResultCountViolationCount,
    int NonFiniteDistanceCount,
    int DuplicateIdCount,
    int UnknownIdCount,
    int TombstonedIdCount,
    int NotAllowedIdCount,
    int DistanceMismatchCount,
    string Policy,
    string Reason);

public sealed record HnswAllowlistUnderfillInfo(
    int QueryCount,
    int RequestedResultCountPerQuery,
    int TotalRequestedResultSlots,
    int TotalReturnedResults,
    int TotalExactTruthAvailableResults,
    int UnderfilledQueryCount,
    int UnderfilledSlotCount,
    string Policy);

public sealed record HnswAllowlistDeltaScanInfo(
    string Status,
    int LiveDeltaScannedCountPerQuery,
    int AllowedLiveDeltaCountPerQuery,
    int TotalLiveDeltaScannedCount,
    int TotalAllowedLiveDeltaCount,
    int TotalEmittedDeltaResultCount,
    string Policy);

public sealed record HnswAllowlistTombstoneSuppressionInfo(
    string Status,
    int BaseTombstoneInputCountPerQuery,
    int DeltaTombstoneInputCountPerQuery,
    int ReturnedBaseTombstoneCount,
    int ReturnedDeltaTombstoneCount,
    string Policy);

public sealed record HnswAllowlistParityInfo(
    HnswBasePlusExactDeltaCheckpointParityInfo ImmutableOpenedHnsw,
    HnswBasePlusExactDeltaCheckpointParityInfo RebuiltCompositeCheckpointOpenedHnsw,
    HnswBasePlusExactDeltaCheckpointParityInfo SourceRebuiltComposite,
    string Policy);

public sealed record HnswAllowlistValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool ExactLiveViewTruthGenerated,
    bool ImmutableHnswComparedToTruth,
    bool OpenedHnswComparedToTruth,
    bool SourceCompositeComparedToTruth,
    bool RebuiltCompositeComparedToTruth,
    bool CheckpointOpenedHnswComparedToTruth,
    bool ExactFallbackParityPassedForAllSearches,
    bool BroadEmissionIntegrityPassedForAllSearches,
    bool BranchConsistencyPassed,
    bool TombstoneSuppressionPassed,
    bool ReturnedResultIntegrityPassedForAllSearches,
    bool MemoryNotMeasured,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record HnswAllowlistEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason);
