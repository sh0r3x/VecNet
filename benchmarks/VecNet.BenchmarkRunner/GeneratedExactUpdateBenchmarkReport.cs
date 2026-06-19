namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactUpdateBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactUpdateEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactUpdateWorkloadInfo Workload,
    GeneratedExactUpdateCountInfo Counts,
    GeneratedExactUpdateMutationInfo Mutations,
    GeneratedExactUpdateFilterInputInfo RawAllowlistInput,
    GeneratedExactUpdateFilterInputInfo CandidateSetInput,
    GeneratedExactUpdateCandidateSetInfo CandidateSet,
    GeneratedExactUpdateSearchesInfo Searches,
    GeneratedExactUpdateMeasurementInfo Measurement,
    GeneratedExactUpdateMetricsInfo Metrics,
    GeneratedExactUpdateValidationInfo Validation,
    GeneratedExactUpdateMemoryEstimateInfo MemoryEstimates,
    GeneratedExactUpdateEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactUpdateEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record GeneratedExactUpdateWorkloadInfo(
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    string Seed,
    string MutationOrder,
    string IdPolicy);

public sealed record GeneratedExactUpdateCountInfo(
    int PhysicalVectorCount,
    int LiveVectorCount,
    int BaseVectorCount,
    int DeltaVectorCount,
    int TombstoneCount,
    double TombstoneRatio,
    string TombstoneRatioDenominator,
    int DeletedOrReservedIdCount,
    string DeletedOrReservedIdSemantics,
    string VectorCountSemantics);

public sealed record GeneratedExactUpdateMutationInfo(
    int InsertedCount,
    int DeletedCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int CommittedMutationCount,
    long GenerationBeforeMutations,
    long GenerationAfterMutations,
    long GenerationDelta,
    bool GenerationDeltaMatchesCommittedMutations,
    GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

public sealed record GeneratedExactUpdateMutationStatusCountInfo(
    int Committed,
    int DuplicateId,
    int UnknownId,
    int AlreadyDeleted,
    int ReadOnly,
    int Unsupported);

public sealed record GeneratedExactUpdateFilterInputInfo(
    string Kind,
    string SelectivityTarget,
    double ActualLiveSelectivity,
    int KnownLiveIdCountPerQuery,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    int InputIdCountPerQuery,
    int TotalKnownLiveIdCount,
    int TotalDuplicateIdCount,
    int TotalUnknownIdCount,
    string InputOrder,
    string GenerationFormula,
    string DuplicatePolicy,
    string UnknownIdPolicy,
    string MutationVisibilityPolicy);

public sealed record GeneratedExactUpdateCandidateSetInfo(
    string ConstructionStatus,
    string ConstructionOperation,
    string ConstructionTimingScope,
    string ConstructionAllocationScope,
    bool ConstructedAfterMutations,
    bool ConstructedBeforeWarmupAndMeasuredSearch,
    int ConstructedSetCount,
    int CountPerQuery,
    int MinCount,
    int MaxCount,
    double MeanCount,
    int TotalCandidateCount,
    string Binding,
    string StaleCandidateSetPolicy,
    string PersistenceScope);

public sealed record GeneratedExactUpdateSearchesInfo(
    GeneratedExactUpdateOperationSearchInfo UnfilteredSearch,
    GeneratedExactUpdateOperationSearchInfo RawAllowlistSearch,
    GeneratedExactUpdateOperationSearchInfo CandidateSetSearch);

public sealed record GeneratedExactUpdateOperationSearchInfo(
    string Name,
    string TimedOperation,
    SearchInfo Search);

public sealed record GeneratedExactUpdateMeasurementInfo(
    GeneratedExactUpdateOperationMeasurementInfo UnfilteredSearch,
    GeneratedExactUpdateOperationMeasurementInfo RawAllowlistSearch,
    GeneratedExactUpdateOperationMeasurementInfo CandidateSetSearch,
    MeasurementStatusInfo MutationLatencyAndAllocation,
    MeasurementStatusInfo LiveViewSave,
    MeasurementStatusInfo ResidentProcessMemory,
    WarmupInfo Warmup,
    string SharedExcludedOperations);

public sealed record GeneratedExactUpdateOperationMeasurementInfo(
    LatencyMeasurementInfo Latency,
    MeasurementStatusInfo ManagedAllocations,
    RepeatedRunInfo RepeatedRuns,
    RunToRunNoiseInfo RunToRunNoise);

public sealed record GeneratedExactUpdateMetricsInfo(
    GeneratedExactUpdateOperationMetricsInfo UnfilteredSearch,
    GeneratedExactUpdateOperationMetricsInfo RawAllowlistSearch,
    GeneratedExactUpdateOperationMetricsInfo CandidateSetSearch);

public sealed record GeneratedExactUpdateOperationMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    GeneratedExactFilteredResultIntegrityInfo ResultIntegrity);

public sealed record GeneratedExactUpdateValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool LiveTruthGenerated,
    bool MutationStatusCountsMatched,
    bool GenerationMovementMatchedCommittedMutations,
    bool CandidateSetsConstructedAfterMutations,
    bool FinalRunUnfilteredComparedToTruth,
    bool FinalRunRawAllowlistComparedToTruth,
    bool FinalRunCandidateSetComparedToTruth,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactUpdateMemoryEstimateInfo(
    string Status,
    string Scope,
    long PhysicalIdPayloadLowerBoundBytes,
    long PhysicalVectorPayloadLowerBoundBytes,
    long LiveVectorPayloadLowerBoundBytes,
    long CandidateSetOrdinalPayloadLowerBoundBytes,
    MeasurementStatusInfo TombstoneDeletedReservationRetainedMemory,
    MeasurementStatusInfo ResidentProcessMemory,
    string NonGoals);

public sealed record GeneratedExactUpdateEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
