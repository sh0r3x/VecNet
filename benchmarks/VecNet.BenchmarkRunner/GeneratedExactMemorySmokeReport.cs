namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactMemorySmokeReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactMemorySmokeEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactMemorySmokeWorkloadInfo Workload,
    GeneratedExactMemorySmokeActualMemoryInfo ActualMemory,
    GeneratedExactMemorySmokeLayoutLowerBoundsInfo LayoutLowerBounds,
    GeneratedExactMemorySmokeOutputsInfo Outputs,
    GeneratedExactMemorySmokeValidationInfo Validation,
    GeneratedExactMemorySmokeEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactMemorySmokeEvidenceInfo(
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

public sealed record GeneratedExactMemorySmokeWorkloadInfo(
    int BaseVectorCount,
    int PhysicalVectorCountAfterMutation,
    int LiveVectorCountAfterMutation,
    int LiveBaseVectorCountAfterMutation,
    int LiveDeltaVectorCountAfterMutation,
    int TombstoneCountAfterMutation,
    int DeletedReservedIdCountAfterMutation,
    double TombstoneRatio,
    double DeltaRatio,
    int QueryCount,
    int TopK,
    int WarmupQueries,
    string AllowlistKind,
    string CandidateSetKind,
    int RawAllowlistKnownIdCountPerQuery,
    int CandidateSetKnownIdCountPerQuery,
    int CandidateSetCount,
    int CandidateSetOrdinalCount,
    string MutationOrder,
    string Boundary);

public sealed record GeneratedExactMemorySmokeActualMemoryInfo(
    string Status,
    string Scope,
    string MeasurementMethod,
    string ClaimBoundary,
    GeneratedExactMemorySampleInfo BaselineProcess,
    GeneratedExactMemorySampleInfo PostDatasetGeneration,
    GeneratedExactMemorySampleInfo PostIndexBuildRetained,
    GeneratedExactMemorySampleInfo PostWarmSearchRetained,
    GeneratedExactMemorySampleInfo RawAllowlistWorkspaceRetained,
    GeneratedExactMemorySampleInfo CandidateSetRetained,
    GeneratedExactMemorySampleInfo PostMutationRetained,
    GeneratedExactMemorySampleInfo PostSaveRetained,
    GeneratedExactMemorySampleInfo PostOpenReadOnlyRetained,
    GeneratedExactMemorySampleInfo OpenedReadOnlyWarmSearchRetained,
    GeneratedExactMemorySampleInfo PostCheckpointRetained,
    GeneratedExactMemoryUnsupportedInfo Unsupported,
    string[] Limitations);

public sealed record GeneratedExactMemorySampleInfo(
    string Name,
    string Boundary,
    GeneratedExactMemoryMetricInfo ManagedHeapSizeBytes,
    GeneratedExactMemoryMetricInfo GcCommittedBytes,
    GeneratedExactMemoryMetricInfo GcFragmentedBytes,
    GeneratedExactMemoryMetricInfo ProcessPrivateBytes,
    GeneratedExactMemoryMetricInfo ProcessWorkingSetBytes,
    GeneratedExactMemoryMetricInfo ProcessPeakWorkingSetBytes,
    GeneratedExactMemoryMetricInfo PeakObservedPrivateBytes,
    GeneratedExactMemoryMetricInfo PeakObservedWorkingSetBytes);

public sealed record GeneratedExactMemoryMetricInfo(
    string Status,
    long? ValueBytes,
    long? DeltaFromBaselineBytes,
    string Unit,
    bool ContextOnly,
    string Reason);

public sealed record GeneratedExactMemoryUnsupportedInfo(
    MeasurementStatusInfo ObjectAccurateIdMapRetainedMemory,
    MeasurementStatusInfo ObjectAccurateTombstoneHashSetRetainedMemory,
    MeasurementStatusInfo ObjectAccurateDeletedReservationHashSetRetainedMemory,
    MeasurementStatusInfo IndexOnlyPrivateBytes,
    MeasurementStatusInfo OpenedOnlyRetainedMemory,
    MeasurementStatusInfo PeakTemporaryProcessMemory,
    MeasurementStatusInfo PeakTemporaryDisk);

public sealed record GeneratedExactMemorySmokeLayoutLowerBoundsInfo(
    string Status,
    string ClaimBoundary,
    long PhysicalIdPayloadLowerBoundBytes,
    long PhysicalVectorPayloadLowerBoundBytes,
    long LiveVectorPayloadLowerBoundBytes,
    long IdMapEntryPayloadLowerBoundBytes,
    long RawAllowlistWorkspacePayloadLowerBoundBytes,
    long CandidateSetOrdinalPayloadLowerBoundBytes,
    long CheckpointSnapshotPayloadLowerBoundBytes,
    long DurableIdPayloadBytes,
    long DurableVectorPayloadBytes,
    long DurablePayloadBytes,
    MeasurementStatusInfo TombstoneIdPayloadLowerBoundBytes,
    MeasurementStatusInfo DeletedReservedIdPayloadLowerBoundBytes,
    string Exclusions);

public sealed record GeneratedExactMemorySmokeOutputsInfo(
    GeneratedExactMemorySmokeOutputInfo SaveOutput,
    GeneratedExactMemorySmokeOutputInfo CheckpointOutput,
    MeasurementStatusInfo PeakObservedSaveOutputDirectoryBytes,
    MeasurementStatusInfo PeakObservedCheckpointOutputDirectoryBytes,
    MeasurementStatusInfo PeakTemporaryDiskBytes,
    MeasurementStatusInfo PeakObservedPrivateBytesDuringSave,
    MeasurementStatusInfo PeakObservedPrivateBytesDuringCheckpoint,
    string Boundary);

public sealed record GeneratedExactMemorySmokeOutputInfo(
    string Status,
    string DirectoryPath,
    int FileCount,
    long FinalOutputBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    int OutputVectorCount,
    long DurableIdPayloadBytes,
    long DurableVectorPayloadBytes,
    string ScanTimingScope);

public sealed record GeneratedExactMemorySmokeValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool BaseIndexBuilt,
    bool WarmSearchExecuted,
    bool RawAllowlistWorkspaceConstructed,
    bool CandidateSetsConstructed,
    bool MutationCountsMatched,
    bool SaveOutputWritten,
    bool OpenReadOnlyCompleted,
    bool CheckpointPublished,
    bool ActualAndEstimateSectionsSeparated,
    bool UnsupportedFieldsExplicitlyMarked,
    bool WorkingSetContextOnly,
    bool PublicClaimEligible,
    bool PreviewReadinessEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactMemorySmokeEligibilityInfo(
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
