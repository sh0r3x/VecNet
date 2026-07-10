namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaBenchmarkReport(
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
    HnswBuildInfo Build,
    HnswBasePlusExactDeltaWorkloadInfo Workload,
    HnswBasePlusExactDeltaCountInfo Counts,
    HnswBasePlusExactDeltaMutationInfo Mutations,
    SearchInfo Search,
    MeasurementInfo Measurement,
    HnswBasePlusExactDeltaMetricsInfo Metrics,
    HnswBasePlusExactDeltaUnderfillInfo Underfill,
    HnswBasePlusExactDeltaValidationInfo Validation,
    HnswEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswBasePlusExactDeltaWorkloadInfo(
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    string Seed,
    string MutationOrder,
    string IdPolicy);

public sealed record HnswBasePlusExactDeltaCountInfo(
    int BasePhysicalVectorCount,
    int BaseLiveVectorCount,
    int DeltaPhysicalVectorCount,
    int DeltaLiveVectorCount,
    int BaseTombstoneCount,
    int DeltaTombstoneCount,
    int TombstoneCount,
    int LiveVectorCount,
    int DeletedReservedIdCount,
    long Generation,
    double TombstoneRatio,
    double DeltaInsertRatio,
    string CountSemantics);

public sealed record HnswBasePlusExactDeltaMutationInfo(
    int InsertedCount,
    int DeletedBaseCount,
    int DeletedDeltaCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int CommittedMutationCount,
    long GenerationBeforeMutations,
    long GenerationAfterMutations,
    long GenerationDelta,
    bool GenerationDeltaMatchesCommittedMutations,
    GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

public sealed record HnswBasePlusExactDeltaMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    HnswBasePlusExactDeltaReturnedResultIntegrityInfo ReturnedResultIntegrity,
    string RecallDefinition,
    string DistanceValidationScope);

public sealed record HnswBasePlusExactDeltaReturnedResultIntegrityInfo(
    string Status,
    int CheckedResultCount,
    int QueryCountMismatchCount,
    int ResultCountViolationCount,
    int NonFiniteDistanceCount,
    int DuplicateIdCount,
    int UnknownIdCount,
    int TombstonedIdCount,
    int DistanceMismatchCount,
    string Policy,
    string Reason);

public sealed record HnswBasePlusExactDeltaUnderfillInfo(
    int QueryCount,
    int RequestedResultCountPerQuery,
    int TotalRequestedResultSlots,
    int TotalReturnedResults,
    int UnderfilledQueryCount,
    int UnderfilledSlotCount,
    string Policy);

public sealed record HnswBasePlusExactDeltaValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool LiveTruthGenerated,
    bool HnswBaseBuilt,
    bool MutationsApplied,
    bool MutationStatusCountsMatched,
    bool GenerationMovementMatchedCommittedMutations,
    bool FinalRunComparedToTruth,
    bool ReturnedResultsAreLiveAndNotTombstoned,
    bool AllowsApproximateRecallBelowOne,
    bool AllowsUnderfill,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);
