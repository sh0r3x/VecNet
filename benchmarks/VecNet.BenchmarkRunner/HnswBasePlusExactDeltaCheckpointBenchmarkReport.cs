namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaCheckpointBenchmarkReport(
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
    HnswBasePlusExactDeltaCheckpointWorkloadInfo Workload,
    HnswBasePlusExactDeltaCheckpointCountInfo PreCheckpointCounts,
    HnswBasePlusExactDeltaCheckpointMutationInfo Mutations,
    HnswBasePlusExactDeltaCheckpointRunsInfo CheckpointRuns,
    HnswBasePlusExactDeltaCheckpointOperationInfo Checkpoint,
    HnswBasePlusExactDeltaCheckpointResultInfo CheckpointResult,
    HnswBasePlusExactDeltaCheckpointCountInfo PostCheckpointCounts,
    HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo NoChangesProbe,
    HnswBasePlusExactDeltaCheckpointOutputInfo Output,
    HnswBasePlusExactDeltaCheckpointOpenedValidationInfo OpenedValidation,
    HnswBasePlusExactDeltaCheckpointSearchSectionsInfo Searches,
    HnswBasePlusExactDeltaCheckpointMeasurementInfo Measurement,
    HnswBasePlusExactDeltaCheckpointValidationInfo Validation,
    HnswEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswBasePlusExactDeltaCheckpointWorkloadInfo(
    int BaseVectorCount,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int QueryCount,
    int TopK,
    int RunCount,
    int WarmupQueryCount,
    string Seed,
    string CheckpointDirectoryPolicy,
    string MutationOrder,
    string IdPolicy);

public sealed record HnswBasePlusExactDeltaCheckpointCountInfo(
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

public sealed record HnswBasePlusExactDeltaCheckpointMutationInfo(
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

public sealed record HnswBasePlusExactDeltaCheckpointRunsInfo(
    int RunCount,
    int DetailedValidationRunNumber,
    string DetailedValidationPolicy,
    HnswBasePlusExactDeltaCheckpointRunInfo[] Runs,
    HnswBasePlusExactDeltaCheckpointRunAggregateInfo Aggregate);

public sealed record HnswBasePlusExactDeltaCheckpointRunInfo(
    int RunNumber,
    string CheckpointDirectory,
    string Status,
    double ElapsedMilliseconds,
    long ManagedAllocatedBytes,
    long GenerationBeforeCheckpoint,
    long GenerationAfterCheckpoint,
    bool GenerationAdvancedExactlyOnce,
    HnswBasePlusExactDeltaCheckpointPhaseSetInfo Phases);

public sealed record HnswBasePlusExactDeltaCheckpointRunAggregateInfo(
    int RunCount,
    double MeanElapsedMilliseconds,
    double MinElapsedMilliseconds,
    double MaxElapsedMilliseconds,
    double MeanManagedAllocatedBytes,
    long MinManagedAllocatedBytes,
    long MaxManagedAllocatedBytes,
    string AggregateSemantics);

public sealed record HnswBasePlusExactDeltaCheckpointOperationInfo(
    string Status,
    string TimedOperation,
    double ElapsedMilliseconds,
    long ManagedAllocatedBytes,
    long GenerationBeforeCheckpoint,
    long GenerationAfterCheckpoint,
    bool GenerationAdvancedExactlyOnce,
    HnswBasePlusExactDeltaCheckpointPhaseSetInfo Phases,
    string ExcludedOperations);

public sealed record HnswBasePlusExactDeltaCheckpointPhaseSetInfo(
    HnswBasePlusExactDeltaCheckpointPhaseInfo LiveSnapshot,
    HnswBasePlusExactDeltaCheckpointPhaseInfo RebuildBuild,
    HnswBasePlusExactDeltaCheckpointPhaseInfo Save,
    HnswBasePlusExactDeltaCheckpointPhaseInfo OpenValidation,
    HnswBasePlusExactDeltaCheckpointPhaseInfo Publication);

public sealed record HnswBasePlusExactDeltaCheckpointPhaseInfo(
    string Status,
    long ElapsedTicks,
    double ElapsedMilliseconds,
    long ManagedAllocatedBytes,
    string Source);

public sealed record HnswBasePlusExactDeltaCheckpointResultInfo(
    string Status,
    long Generation,
    int RebuiltBaseVectorCount,
    int LiveVectorCount,
    int BasePhysicalVectorCount,
    int BaseLiveVectorCount,
    int DeltaPhysicalVectorCount,
    int DeltaLiveVectorCount,
    int BaseTombstoneCount,
    int DeltaTombstoneCount,
    int TombstoneCount,
    int DeletedReservedIdCount,
    int FoldedDeltaVectorCount,
    int FoldedBaseTombstoneCount,
    int FoldedDeltaTombstoneCount);

public sealed record HnswBasePlusExactDeltaCheckpointNoChangesProbeInfo(
    string Status,
    long GenerationBeforeProbe,
    long GenerationAfterProbe,
    bool GenerationUnchanged,
    bool OutputDirectoryRemainedEmpty,
    HnswBasePlusExactDeltaCheckpointPhaseSetInfo Phases);

public sealed record HnswBasePlusExactDeltaCheckpointOutputInfo(
    string Status,
    string DirectoryPath,
    int FileCount,
    long TotalBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    long LevelsBytes,
    long GraphBytes,
    int OutputVectorCount,
    double BytesPerLiveVector,
    string ValidationOpenStatus,
    string ScanTimingScope);

public sealed record HnswBasePlusExactDeltaCheckpointOpenedValidationInfo(
    string Status,
    int ExpectedVectorCount,
    int OpenedVectorCount,
    int IdMismatchCount,
    int VectorMismatchCount,
    HnswBasePlusExactDeltaCheckpointParityInfo RebuiltCompositeOpenedSearchParity,
    string Policy);

public sealed record HnswBasePlusExactDeltaCheckpointSearchSectionsInfo(
    HnswBasePlusExactDeltaCheckpointSearchSectionInfo PreCheckpointComposite,
    HnswBasePlusExactDeltaCheckpointSearchSectionInfo PostCheckpointRebuiltComposite,
    HnswBasePlusExactDeltaCheckpointSearchSectionInfo OpenedReadOnlyHnsw);

public sealed record HnswBasePlusExactDeltaCheckpointSearchSectionInfo(
    string Name,
    string TimedOperation,
    SearchInfo Search,
    MeasurementInfo Measurement,
    HnswBasePlusExactDeltaCheckpointMetricsInfo Metrics,
    HnswBasePlusExactDeltaUnderfillInfo Underfill);

public sealed record HnswBasePlusExactDeltaCheckpointMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    HnswBasePlusExactDeltaReturnedResultIntegrityInfo ReturnedResultIntegrity,
    string RecallDefinition,
    string DistanceValidationScope);

public sealed record HnswBasePlusExactDeltaCheckpointParityInfo(
    int QueryCount,
    int WrittenCountMismatchCount,
    int IdMismatchCount,
    int OrderMismatchCount,
    int DistanceMismatchCount,
    bool AllResultsMatched,
    string Policy);

public sealed record HnswBasePlusExactDeltaCheckpointMeasurementInfo(
    LatencyMeasurementInfo CheckpointLatency,
    MeasurementStatusInfo CheckpointManagedAllocations,
    HnswBasePlusExactDeltaCheckpointPhaseSetInfo PhaseDiagnostics,
    MeasurementStatusInfo OutputBytes,
    MeasurementStatusInfo Memory,
    WarmupInfo Warmup,
    string SharedExcludedOperations);

public sealed record HnswBasePlusExactDeltaCheckpointValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool LiveTruthGenerated,
    bool PreCheckpointCompositeComparedToTruth,
    bool CheckpointResultStatusPublished,
    bool CheckpointResultCountsMatched,
    bool CheckpointGenerationAdvancedExactlyOnce,
    bool PhaseDiagnosticsMeasuredForPublishedCheckpoint,
    bool CheckpointRepeatedRunEvidencePresent,
    int DetailedValidationRunNumber,
    bool DetailedValidationUsesFinalRun,
    bool PostCheckpointCountsMatched,
    bool PostCheckpointRebuiltCompositeComparedToTruth,
    bool OpenedReadOnlyHnswIdVectorValidationPassed,
    bool OpenedReadOnlyHnswComparedToTruth,
    bool RebuiltCompositeOpenedHnswSearchParityPassed,
    bool ReturnedResultIntegrityPassedForAllSearches,
    bool NoChangesCheckpointProbePassed,
    bool DeletedReservedIdsRejectedAfterCheckpoint,
    bool OutputBytesScannedOutsideCheckpointDuration,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);
