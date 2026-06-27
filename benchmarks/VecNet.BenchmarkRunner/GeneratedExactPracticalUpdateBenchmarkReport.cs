namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactPracticalUpdateBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactPracticalUpdateEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactPracticalUpdateWorkloadInfo Workload,
    GeneratedExactPracticalUpdateCountsInfo Counts,
    GeneratedExactPracticalUpdateMutationInfo Mutations,
    GeneratedExactPracticalUpdateGenerationInfo Generations,
    GeneratedExactUpdateFilterInputInfo RawAllowlistInput,
    GeneratedExactUpdateFilterInputInfo CandidateSetInput,
    GeneratedExactPracticalUpdateCandidateSetInfo CandidateSet,
    GeneratedExactPracticalUpdateOperationsInfo Operations,
    GeneratedExactPracticalUpdateMeasurementInfo Measurement,
    GeneratedExactPracticalUpdateOutputsInfo Outputs,
    GeneratedExactPracticalUpdateMetricsInfo Metrics,
    GeneratedExactPracticalUpdateValidationInfo Validation,
    GeneratedExactPracticalUpdateResourceInfo Resources,
    GeneratedExactPracticalUpdateEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactPracticalUpdateEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string PreviewReadinessReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record GeneratedExactPracticalUpdateWorkloadInfo(
    int InitialBaseVectorCount,
    int InsertAttemptCount,
    int DeleteAttemptCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    string Seed,
    string MutationOrder,
    string IdPolicy,
    string CheckpointMode);

public sealed record GeneratedExactPracticalUpdateCountsInfo(
    int InitialBaseCount,
    int PhysicalVectorCountAfterMutation,
    int FinalLiveCountBeforeCheckpoint,
    int LiveBaseCountBeforeCheckpoint,
    int LiveDeltaCountBeforeCheckpoint,
    int TombstoneCountBeforeCheckpoint,
    int DeletedReservedIdCountBeforeCheckpoint,
    double TombstoneRatio,
    string TombstoneRatioDenominator,
    double DeltaInsertRatio,
    string DeltaInsertRatioDenominator,
    int PhysicalVectorCountAfterCheckpoint,
    int FinalLiveCountAfterCheckpoint,
    int DeltaCountAfterCheckpoint,
    int TombstoneCountAfterCheckpoint);

public sealed record GeneratedExactPracticalUpdateMutationInfo(
    int InsertAttemptCount,
    int InsertSuccessCount,
    int DeleteAttemptCount,
    int DeleteSuccessCount,
    int DuplicateInsertAttempts,
    int DuplicateInsertFailures,
    int UnknownDeleteAttempts,
    int UnknownDeleteFailures,
    int RepeatedDeleteAttempts,
    int RepeatedDeleteFailures,
    int CommittedMutationCount,
    GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

public sealed record GeneratedExactPracticalUpdateGenerationInfo(
    long BeforeMutation,
    long AfterMutation,
    long BeforeCheckpoint,
    long AfterCheckpoint,
    long MutationDelta,
    long CheckpointDelta,
    bool MutationDeltaMatchesCommittedMutations,
    bool CheckpointAdvancedExactlyOnce);

public sealed record GeneratedExactPracticalUpdateCandidateSetInfo(
    string ConstructionOperation,
    string ConstructionTimingScope,
    bool StaleCandidateSetConstructedBeforeMutation,
    bool StaleCandidateSetRejectedAfterMutation,
    bool FreshCandidateSetConstructedAfterMutation,
    int FreshConstructedSetCount,
    int FreshCountPerQuery,
    int FreshTotalCandidateCount,
    string Binding,
    string StalePolicy);

public sealed record GeneratedExactPracticalUpdateOperationsInfo(
    GeneratedExactPracticalUpdateTimedOperationInfo Mutations,
    GeneratedExactPracticalUpdateTimedOperationInfo PostMutationExactSearch,
    GeneratedExactPracticalUpdateTimedOperationInfo Checkpoint,
    GeneratedExactPracticalUpdateTimedOperationInfo Open,
    MeasurementStatusInfo RawAllowlistValidationSearch,
    MeasurementStatusInfo FreshCandidateSetValidationSearch,
    MeasurementStatusInfo StaleCandidateSetRejectionValidation);

public sealed record GeneratedExactPracticalUpdateTimedOperationInfo(
    string Name,
    string TimedOperation,
    GeneratedExactPracticalUpdateOperationRunInfo[] Runs,
    GeneratedExactPracticalUpdateOperationAggregateInfo Aggregate);

public sealed record GeneratedExactPracticalUpdateOperationRunInfo(
    int RunNumber,
    double ElapsedMilliseconds,
    string Status,
    long GenerationBefore,
    long GenerationAfter,
    int OperationCount,
    string TimingScope);

public sealed record GeneratedExactPracticalUpdateOperationAggregateInfo(
    int RunCount,
    double MeanElapsedMilliseconds,
    double MinElapsedMilliseconds,
    double MaxElapsedMilliseconds);

public sealed record GeneratedExactPracticalUpdateMeasurementInfo(
    GeneratedExactPracticalUpdateOperationMeasurementInfo Mutations,
    GeneratedExactPracticalUpdateOperationMeasurementInfo PostMutationExactSearch,
    GeneratedExactPracticalUpdateOperationMeasurementInfo Checkpoint,
    GeneratedExactPracticalUpdateOperationMeasurementInfo Open,
    MeasurementStatusInfo MutationManagedAllocations,
    MeasurementStatusInfo SearchManagedAllocations,
    MeasurementStatusInfo CheckpointManagedAllocations,
    MeasurementStatusInfo OpenManagedAllocations,
    WarmupInfo Warmup,
    string SharedExcludedOperations);

public sealed record GeneratedExactPracticalUpdateOperationMeasurementInfo(
    LatencyMeasurementInfo Latency,
    RepeatedRunInfo RepeatedRuns,
    RunToRunMetricNoiseInfo RunToRunNoise);

public sealed record GeneratedExactPracticalUpdateOutputsInfo(
    string CheckpointStatus,
    string CheckpointDirectoryPath,
    int CheckpointFileCount,
    long CheckpointOutputBytes,
    long CheckpointManifestBytes,
    long CheckpointIdsBytes,
    long CheckpointVectorsBytes,
    int CheckpointOutputVectorCount,
    string OutputByteScanTimingScope);

public sealed record GeneratedExactPracticalUpdateMetricsInfo(
    GeneratedExactCheckpointOperationMetricsInfo PostMutationExactSearch,
    GeneratedExactCheckpointOperationMetricsInfo RawAllowlistSearch,
    GeneratedExactCheckpointOperationMetricsInfo FreshCandidateSetSearch,
    GeneratedExactCheckpointOperationMetricsInfo ReopenedOutputSearch);

public sealed record GeneratedExactPracticalUpdateValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool LiveTruthGenerated,
    bool MutationCountsMatched,
    bool GenerationBeforeAfterMutationReported,
    bool PostMutationExactSearchComparedToTruth,
    bool RawAllowlistVisibleAfterMutation,
    bool FreshCandidateSetVisibleAfterMutation,
    bool StaleCandidateSetRejectedAfterMutation,
    bool CheckpointPublished,
    bool ReopenedOutputParity,
    bool CheckpointOutputBytesScannedOutsideTiming,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactPracticalUpdateResourceInfo(
    MeasurementStatusInfo ActualResidentMemory,
    MeasurementStatusInfo ActualProcessMemory,
    MeasurementStatusInfo ActualGcMemory,
    MeasurementStatusInfo ActualPrivateMemory,
    MeasurementStatusInfo ActualPeakMemory,
    MeasurementStatusInfo PeakTemporaryDisk,
    string FinalCheckpointOutputBytesStatus,
    string NonGoals);

public sealed record GeneratedExactPracticalUpdateEligibilityInfo(
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string PreviewReadinessReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason);
