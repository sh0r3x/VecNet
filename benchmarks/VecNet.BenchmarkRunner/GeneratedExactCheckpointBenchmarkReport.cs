namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCheckpointBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactCheckpointEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactCheckpointWorkloadInfo Workload,
    GeneratedExactCheckpointCountInfo PreCheckpointCounts,
    GeneratedExactCheckpointResultInfo CheckpointResult,
    GeneratedExactCheckpointCountInfo PostCheckpointCounts,
    GeneratedExactCheckpointMutationInfo Mutations,
    GeneratedExactUpdateFilterInputInfo RawAllowlistInput,
    GeneratedExactUpdateFilterInputInfo CandidateSetInput,
    GeneratedExactCheckpointCandidateSetInfo CandidateSet,
    GeneratedExactCheckpointOperationsInfo Operations,
    GeneratedExactCheckpointMeasurementInfo Measurement,
    GeneratedExactCheckpointOutputsInfo Outputs,
    GeneratedExactCheckpointMetricsInfo Metrics,
    GeneratedExactCheckpointValidationInfo Validation,
    GeneratedExactCheckpointMemoryEstimateInfo MemoryEstimates,
    GeneratedExactCheckpointEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactCheckpointEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool PreviewReadinessEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string PreviewReadinessReason,
    string[] Limitations);

public sealed record GeneratedExactCheckpointWorkloadInfo(
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    string Seed,
    string CheckpointMode,
    string CheckpointTargetPolicy,
    bool LiveViewSaveComparisonMeasured,
    bool RawAllowlistBehaviorIncluded,
    bool CandidateSetBehaviorIncluded,
    bool NoChangesBehaviorIncluded,
    bool FailurePathBehaviorIncluded,
    string MutationOrder,
    string IdPolicy);

public sealed record GeneratedExactCheckpointCountInfo(
    int PhysicalVectorCount,
    int LiveVectorCount,
    int BaseVectorCount,
    int DeltaVectorCount,
    int VisibilityTombstoneCount,
    int DeletedReservedIdCount,
    double TombstoneRatio,
    string TombstoneRatioDenominator,
    long Generation,
    string DeletedReservedIdSemantics,
    string VectorCountSemantics);

public sealed record GeneratedExactCheckpointResultInfo(
    string Status,
    long Generation,
    int PhysicalVectorCount,
    int LiveVectorCount,
    int BaseVectorCount,
    int DeltaVectorCount,
    int TombstoneCount,
    int DeletedReservedIdCount,
    int FoldedDeltaVectorCount,
    int FoldedTombstoneCount);

public sealed record GeneratedExactCheckpointMutationInfo(
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

public sealed record GeneratedExactCheckpointCandidateSetInfo(
    string ConstructionStatus,
    string ConstructionOperation,
    string ConstructionTimingScope,
    bool PreCheckpointCandidateSetsConstructed,
    bool PreCheckpointCandidateSetsStaleAfterPublishedCheckpoint,
    bool PostCheckpointCandidateSetsConstructed,
    int ConstructedSetCount,
    int CountPerQuery,
    int MinCount,
    int MaxCount,
    double MeanCount,
    int TotalCandidateCount,
    string Binding,
    string PersistenceScope);

public sealed record GeneratedExactCheckpointOperationsInfo(
    GeneratedExactCheckpointOperationInfo Checkpoint,
    MeasurementStatusInfo LiveViewSave,
    MeasurementStatusInfo PostCheckpointUnfilteredSearch,
    MeasurementStatusInfo PostCheckpointRawAllowlistSearch,
    MeasurementStatusInfo PostCheckpointCandidateSetSearch,
    MeasurementStatusInfo NoChanges,
    MeasurementStatusInfo FailureCases,
    MeasurementStatusInfo ResidentProcessMemory);

public sealed record GeneratedExactCheckpointOperationInfo(
    string Name,
    string TimedOperation,
    GeneratedExactCheckpointOperationRunInfo[] Runs,
    GeneratedExactCheckpointOperationAggregateInfo Aggregate);

public sealed record GeneratedExactCheckpointOperationRunInfo(
    int RunNumber,
    double ElapsedMilliseconds,
    string Status,
    long GenerationBeforeCheckpoint,
    long GenerationAfterCheckpoint,
    string OutputDirectoryPolicy);

public sealed record GeneratedExactCheckpointOperationAggregateInfo(
    int RunCount,
    double MeanElapsedMilliseconds,
    double MinElapsedMilliseconds,
    double MaxElapsedMilliseconds);

public sealed record GeneratedExactCheckpointMeasurementInfo(
    GeneratedExactCheckpointOperationMeasurementInfo Checkpoint,
    MeasurementStatusInfo CheckpointManagedAllocations,
    MeasurementStatusInfo LiveViewSave,
    MeasurementStatusInfo PostCheckpointSearchTiming,
    MeasurementStatusInfo ResidentProcessMemory,
    WarmupInfo Warmup,
    string SharedExcludedOperations);

public sealed record GeneratedExactCheckpointOperationMeasurementInfo(
    LatencyMeasurementInfo Latency,
    RepeatedRunInfo RepeatedRuns,
    RunToRunMetricNoiseInfo RunToRunNoise);

public sealed record GeneratedExactCheckpointOutputsInfo(
    GeneratedExactCheckpointOutputInfo CheckpointOutput,
    MeasurementStatusInfo SaveOutput);

public sealed record GeneratedExactCheckpointOutputInfo(
    string Status,
    string DirectoryPathPolicy,
    string DirectoryPath,
    int FileCount,
    long TotalBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    int OutputVectorCount,
    double BytesPerLiveVector,
    string ValidationOpenStatus,
    string ScanTimingScope);

public sealed record GeneratedExactCheckpointMetricsInfo(
    GeneratedExactCheckpointOperationMetricsInfo PreCheckpointInMemorySearch,
    GeneratedExactCheckpointOperationMetricsInfo PostCheckpointInMemorySearch,
    GeneratedExactCheckpointOperationMetricsInfo ReopenedCheckpointOutputSearch,
    GeneratedExactCheckpointOperationMetricsInfo PostCheckpointRawAllowlistSearch,
    GeneratedExactCheckpointOperationMetricsInfo PostCheckpointCandidateSetSearch);

public sealed record GeneratedExactCheckpointOperationMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    GeneratedExactFilteredResultIntegrityInfo ResultIntegrity);

public sealed record GeneratedExactCheckpointValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool LiveTruthGenerated,
    bool PreCheckpointInMemoryComparedToTruth,
    bool CheckpointResultStatusPublished,
    bool CheckpointResultCountsMatched,
    bool PostCheckpointCountsMatched,
    bool GenerationAdvancedExactlyOnce,
    bool PostCheckpointInMemoryComparedToTruth,
    bool ReopenedCheckpointOutputComparedToTruth,
    bool RawAllowlistComparedToTruth,
    bool CandidateSetComparedToTruth,
    bool PreCheckpointCandidateSetsRejectedAsStale,
    bool DeletedReservedIdsRejectedAfterCheckpoint,
    bool OutputBytesScannedOutsideCheckpointDuration,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool PreviewReadinessEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactCheckpointMemoryEstimateInfo(
    string Status,
    string Scope,
    long PreCheckpointPhysicalIdPayloadLowerBoundBytes,
    long PreCheckpointPhysicalVectorPayloadLowerBoundBytes,
    long PreCheckpointLiveVectorPayloadLowerBoundBytes,
    long PostCheckpointCompactIdPayloadLowerBoundBytes,
    long PostCheckpointCompactVectorPayloadLowerBoundBytes,
    long CheckpointSnapshotPayloadLowerBoundBytes,
    long CandidateSetOrdinalPayloadLowerBoundBytes,
    MeasurementStatusInfo TombstoneDeletedReservationRetainedMemory,
    MeasurementStatusInfo RetainedHashSetCapacity,
    MeasurementStatusInfo ResidentProcessMemory,
    MeasurementStatusInfo GcHeap,
    MeasurementStatusInfo WorkingSet,
    MeasurementStatusInfo PrivateBytes,
    MeasurementStatusInfo PeakMemory,
    string NonGoals);

public sealed record GeneratedExactCheckpointEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool PreviewReadinessEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string PreviewReadinessReason);
