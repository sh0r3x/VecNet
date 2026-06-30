namespace VecNet.BenchmarkRunner;

public sealed record HnswMemorySmokeReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    HnswMemorySmokeEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    ScenarioInfo Scenario,
    IndexInfo Index,
    HnswMemorySmokeWorkloadInfo Workload,
    HnswConfigurationInfo Hnsw,
    HnswMemorySmokeActualMemoryInfo ActualMemory,
    HnswMemorySmokePeakMemoryInfo PeakMemory,
    HnswMemorySmokeLayoutLowerBoundsInfo LayoutLowerBounds,
    HnswMemorySmokeStorageSizeInfo StorageSize,
    HnswMemorySmokeValidationInfo Validation,
    HnswMemorySmokeEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswMemorySmokeEvidenceInfo(
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

public sealed record HnswMemorySmokeWorkloadInfo(
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int WarmupQueries,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    int SampleIntervalMilliseconds,
    string InsertionOrder,
    string DurableLifecycle,
    string Boundary);

public sealed record HnswMemorySmokeActualMemoryInfo(
    string Status,
    string Scope,
    string MeasurementMethod,
    string ClaimBoundary,
    HnswMemorySampleInfo BaselineProcess,
    HnswMemorySampleInfo PostDatasetGeneration,
    HnswMemorySampleInfo PostSourceBuildRetained,
    HnswMemorySampleInfo PostSourceWarmSearchRetained,
    HnswMemorySampleInfo PostSaveRetained,
    HnswMemorySampleInfo PostOpenReadOnlyRetained,
    HnswMemorySampleInfo PostOpenedWarmSearchRetained,
    HnswMemorySampleInfo PostValidationRetained,
    HnswMemoryUnsupportedInfo Unsupported,
    string[] Limitations);

public sealed record HnswMemorySampleInfo(
    string Name,
    string Boundary,
    HnswMemoryMetricInfo ManagedHeapSizeBytes,
    HnswMemoryMetricInfo GcCommittedBytes,
    HnswMemoryMetricInfo GcFragmentedBytes,
    HnswMemoryMetricInfo ProcessPrivateBytes,
    HnswMemoryMetricInfo ProcessWorkingSetBytes,
    HnswMemoryMetricInfo ProcessPeakWorkingSetBytes);

public sealed record HnswMemoryMetricInfo(
    string Status,
    long? ValueBytes,
    long? DeltaFromBaselineBytes,
    string Unit,
    bool ContextOnly,
    string Reason);

public sealed record HnswMemoryUnsupportedInfo(
    MeasurementStatusInfo ObjectAccurateIdMapRetainedMemory,
    MeasurementStatusInfo ObjectAccurateGraphLayerObjectMemory,
    MeasurementStatusInfo ObjectHeadersArrayHeadersAlignmentAndSlack,
    MeasurementStatusInfo NeighborCandidateRetainedLayout,
    MeasurementStatusInfo IndexOnlyPrivateBytes,
    MeasurementStatusInfo OpenedOnlyRetainedMemory,
    MeasurementStatusInfo SaveManagedAllocations,
    MeasurementStatusInfo OpenManagedAllocations,
    MeasurementStatusInfo TrueProcessPeakMemory,
    MeasurementStatusInfo PeakTemporaryDisk);

public sealed record HnswMemorySmokePeakMemoryInfo(
    string Status,
    string Scope,
    string ClaimBoundary,
    HnswMemoryPeakOperationInfo Build,
    HnswMemoryPeakOperationInfo Save,
    HnswMemoryPeakOperationInfo Open,
    MeasurementStatusInfo SourceSearchWarmupPeakMemory,
    MeasurementStatusInfo OpenedSearchWarmupPeakMemory,
    MeasurementStatusInfo PeakTemporaryDiskBytes,
    string[] Limitations);

public sealed record HnswMemoryPeakOperationInfo(
    string Name,
    string Status,
    int SampleIntervalMilliseconds,
    int SampleCount,
    HnswMemorySampleInfo StartSample,
    HnswMemorySampleInfo EndSample,
    HnswMemoryMetricInfo PeakObservedManagedHeapSizeBytes,
    HnswMemoryMetricInfo PeakObservedGcCommittedBytes,
    HnswMemoryMetricInfo PeakObservedPrivateBytes,
    HnswMemoryMetricInfo PeakObservedWorkingSetBytes,
    string MissedShortPeakCaveat,
    string WholeProcessCaveat,
    string TimedScope,
    string ExcludedOperations);

public sealed record HnswMemorySmokeLayoutLowerBoundsInfo(
    string Status,
    string ClaimBoundary,
    long VectorPayloadLowerBoundBytes,
    long IdPayloadLowerBoundBytes,
    long LevelPayloadLowerBoundBytes,
    long GraphCountPayloadLowerBoundBytes,
    long GraphNeighborPayloadLowerBoundBytes,
    long GraphPayloadLowerBoundBytes,
    long IdMapEntryPayloadLowerBoundBytes,
    MeasurementStatusInfo SearchWorkspacePayloadLowerBoundBytes,
    MeasurementStatusInfo BuildScratchPayloadLowerBoundBytes,
    long SourceRetainedPayloadLowerBoundBytes,
    long OpenedRetainedPayloadLowerBoundBytes,
    HnswMemoryLayerLowerBoundInfo[] Layers,
    string Exclusions);

public sealed record HnswMemoryLayerLowerBoundInfo(
    int Layer,
    int Stride,
    long CountPayloadLowerBoundBytes,
    long NeighborPayloadLowerBoundBytes);

public sealed record HnswMemorySmokeStorageSizeInfo(
    string Status,
    string Boundary,
    string SnapshotDirectoryPathPolicy,
    string SnapshotDirectory,
    int FileCount,
    long TotalBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    long LevelsBytes,
    long GraphBytes,
    double BytesPerVector,
    string ScanTimingScope,
    MeasurementStatusInfo PeakObservedOutputDirectoryBytes,
    MeasurementStatusInfo PeakTemporaryDiskBytes);

public sealed record HnswMemorySmokeValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool SourceHnswBuilt,
    bool SourceWarmSearchExecuted,
    bool SourceHnswSaved,
    bool OpenedHnswOpened,
    bool OpenedIndexReadOnly,
    bool OpenedWarmSearchExecuted,
    bool SourceOpenedParityChecked,
    DurableHnswParityInfo SourceOpenedParity,
    HnswReturnedResultIntegrityInfo SourceReturnedResultIntegrity,
    HnswReturnedResultIntegrityInfo OpenedReturnedResultIntegrity,
    bool ActualPeakLowerBoundAndStorageSectionsSeparated,
    bool UnsupportedFieldsExplicitlyMarked,
    bool WorkingSetContextOnly,
    bool SampledPeakLabelsPresent,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record HnswMemorySmokeEligibilityInfo(
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
