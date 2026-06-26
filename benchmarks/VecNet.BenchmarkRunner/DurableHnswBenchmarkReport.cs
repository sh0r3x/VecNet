namespace VecNet.BenchmarkRunner;

public sealed record DurableHnswBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    DurableHnswEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    DurableHnswWorkloadInfo Workload,
    HnswConfigurationInfo Hnsw,
    DurableHnswOperationsInfo Operations,
    DurableHnswMeasurementInfo Measurement,
    DurableHnswOutputsInfo Outputs,
    DurableHnswMetricsInfo Metrics,
    DurableHnswValidationInfo Validation,
    DurableHnswMemoryEstimateInfo MemoryEstimates,
    DurableHnswEligibilityInfo Eligibility,
    string[] Notes);

public sealed record DurableHnswEvidenceInfo(
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

public sealed record DurableHnswWorkloadInfo(
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    int RunCount,
    int WarmupQueryCount,
    string InsertionOrder,
    string SaveOpenLifecycle,
    string DurableFileFamilyName);

public sealed record DurableHnswOperationsInfo(
    DurableHnswOperationInfo Build,
    DurableHnswOperationInfo Save,
    DurableHnswOperationInfo Open,
    DurableHnswOpenedSearchOperationInfo OpenedSearch,
    MeasurementStatusInfo SourceSearch,
    MeasurementStatusInfo ResidentProcessMemory);

public sealed record DurableHnswOperationInfo(
    string Name,
    string TimedOperation,
    DurableHnswOperationRunInfo[] Runs,
    DurableHnswOperationAggregateInfo Aggregate);

public sealed record DurableHnswOperationRunInfo(
    int RunNumber,
    double ElapsedMilliseconds,
    string Status,
    string SnapshotDirectory);

public sealed record DurableHnswOperationAggregateInfo(
    int RunCount,
    double MeanElapsedMilliseconds,
    double MinElapsedMilliseconds,
    double MaxElapsedMilliseconds);

public sealed record DurableHnswOpenedSearchOperationInfo(
    string Name,
    string TimedOperation,
    SearchRunInfo[] Runs,
    AggregateTimingInfo Aggregate);

public sealed record DurableHnswMeasurementInfo(
    DurableHnswOperationMeasurementInfo Build,
    DurableHnswOperationMeasurementInfo Save,
    DurableHnswOperationMeasurementInfo Open,
    DurableHnswSearchMeasurementInfo OpenedSearch,
    MeasurementStatusInfo SourceSearch,
    MeasurementStatusInfo ResidentProcessMemory,
    WarmupInfo Warmup,
    string SharedExcludedOperations);

public sealed record DurableHnswOperationMeasurementInfo(
    LatencyMeasurementInfo Latency,
    MeasurementStatusInfo ManagedAllocations,
    RepeatedRunInfo RepeatedRuns,
    RunToRunMetricNoiseInfo RunToRunNoise);

public sealed record DurableHnswSearchMeasurementInfo(
    LatencyMeasurementInfo Latency,
    MeasurementStatusInfo ManagedAllocations,
    RepeatedRunInfo RepeatedRuns,
    RunToRunNoiseInfo RunToRunNoise);

public sealed record DurableHnswOutputsInfo(
    DurableHnswSnapshotOutputInfo SnapshotOutput,
    MeasurementStatusInfo TemporaryDisk,
    MeasurementStatusInfo PeakDisk);

public sealed record DurableHnswSnapshotOutputInfo(
    string Status,
    string DirectoryPathPolicy,
    string DirectoryPath,
    int FileCount,
    long TotalBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    long LevelsBytes,
    long GraphBytes,
    int VectorCount,
    double BytesPerVector,
    string ValidationOpenStatus,
    string ScanTimingScope);

public sealed record DurableHnswMetricsInfo(
    DurableHnswOperationMetricsInfo SourceHnsw,
    DurableHnswOperationMetricsInfo OpenedHnsw,
    bool SourceAndOpenedRecallEqual,
    bool SourceAndOpenedOrderedAgreementEqual,
    bool SourceAndOpenedDistanceIntegrityEqual,
    string RecallEquivalenceReason);

public sealed record DurableHnswOperationMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    int MissingResultCount,
    int ExtraResultCount,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    HnswReturnedResultIntegrityInfo ReturnedResultIntegrity,
    string RecallDefinition,
    string DistanceValidationScope);

public sealed record DurableHnswParityInfo(
    int QueryCount,
    int WrittenCountMismatchCount,
    int IdMismatchCount,
    int OrderMismatchCount,
    int DistanceMismatchCount,
    bool AllResultsMatched,
    string Policy);

public sealed record DurableHnswReadOnlyMutationInfo(
    string Status,
    string ExceptionType,
    bool RejectedBeforeVectorValidation,
    string Operation,
    string Reason);

public sealed record DurableHnswValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool ExactTruthGenerated,
    bool SourceHnswBuilt,
    bool SourceHnswSaved,
    bool OpenedHnswOpened,
    bool OpenedIndexReadOnly,
    bool SourceHnswComparedToTruth,
    bool OpenedHnswComparedToTruth,
    bool ReturnedResultIntegrityPassedForSource,
    bool ReturnedResultIntegrityPassedForOpened,
    DurableHnswParityInfo SavedOpenedParity,
    DurableHnswReadOnlyMutationInfo OpenedReadOnlyMutation,
    bool OutputBytesScannedOutsideSaveOpenDuration,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record DurableHnswMemoryEstimateInfo(
    string Status,
    string Scope,
    long VectorPayloadBytes,
    long IdPayloadBytes,
    long LevelPayloadBytes,
    long GraphCountPayloadBytes,
    long GraphNeighborPayloadBytes,
    long SearchWorkspaceBytes,
    long DurableOutputBytes,
    MeasurementStatusInfo ResidentProcessMemory,
    MeasurementStatusInfo GcHeap,
    MeasurementStatusInfo WorkingSet,
    MeasurementStatusInfo PrivateBytes,
    MeasurementStatusInfo PeakMemory,
    string[] Exclusions);

public sealed record DurableHnswEligibilityInfo(
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
