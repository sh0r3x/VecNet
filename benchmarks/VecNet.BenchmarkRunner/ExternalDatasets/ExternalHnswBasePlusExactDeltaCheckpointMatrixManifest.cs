namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest(
    string SchemaName,
    string SchemaVersion,
    string TaskId,
    string ScenarioName,
    string ReportId,
    string PresetName,
    DateTimeOffset GeneratedAtUtc,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    string OutputDirectory,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPostureInfo EvidencePosture,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo CacheTruth,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixDesignInfo Design,
    int CaseCount,
    string ValidationStatus,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] Cases,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate Aggregate,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibility Eligibility,
    string[] Notes);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixPostureInfo(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceKind,
    string Scope);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthInfo(
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
    int RequiredQueryCount,
    int RequiredTruthDepth,
    int RequiredBaseVectorCount,
    string? ErrorMessage);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixDesignInfo(
    string DatasetId,
    int Dimension,
    string Metric,
    int QueryCount,
    int WarmupQueryCount,
    int CheckpointRunCount,
    int[] TopKValues,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo[] UpdateProfiles,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo HnswProfile,
    string MatrixSeed,
    string HnswMatrixSeedBase,
    string WorkloadSeedPolicy,
    string HnswSeedPolicy,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseSeedInfo[] ResolvedCaseSeeds,
    string RowSelectionPolicy,
    string ExactUpdatedTruthPolicy,
    string CheckpointOutputPolicy,
    string MeasurementBoundaryPolicy,
    string ScopePolicy);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo(
    string Name,
    int ImmutableBaseStartRow,
    int ImmutableBaseEndRowInclusive,
    int ImmutableBaseRowCount,
    int DeltaStartRow,
    int DeltaEndRowInclusive,
    int DeltaRowCount,
    int UnusedStartRow,
    int UnusedEndRowInclusive,
    int UnusedRowCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int ExpectedPhysicalCandidateVectorCount,
    int ExpectedLiveVectorCount,
    int ExpectedDeletedReservedIdCount,
    string Description);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseSeedInfo(
    int CaseNumber,
    string CaseId,
    string WorkloadSeed,
    string HnswSeed);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string UpdateProfileName,
    string HnswProfileName,
    string DatasetId,
    string Metric,
    int Dimension,
    int QueryCount,
    int TopK,
    int CheckpointRunCount,
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
    int UnusedStartRow,
    int UnusedEndRowInclusive,
    int UnusedCandidateRowCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    int ExpectedPhysicalCandidateVectorCount,
    int ExpectedLiveVectorCount,
    int ExpectedDeletedReservedIdCount,
    string? LinkedReportPath,
    string? LinkedCheckpointDirectoryPath,
    string[] CommandArguments,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary LinkedReportValidation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary RepeatedCheckpointRuns,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary PhaseDiagnostics,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary OutputSummary,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary PreCheckpointSourceCompositeSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary PostCheckpointRebuiltCompositeSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary OpenedReadOnlyHnswSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary OpenedValidation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary RebuiltOpenedParity,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary DeletedReservation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary NoChanges,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary CountSummary,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixMemorySummary Memory,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary RecursiveEligibility,
    string? ErrorCategory,
    string? ErrorMessage);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary(
    string Status,
    bool LinkedReportInspected,
    bool? SchemaMatched,
    bool? ScenarioMatched,
    bool? CaseParametersMatched,
    bool? RequiredCheckpointSectionsPresent,
    bool? PhaseDiagnosticsPresent,
    bool? OpenedValidationPresent,
    bool? RebuiltOpenedParityPassed,
    bool? DeletedReservationValidated,
    bool? EligibilityFalse);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
    string Status,
    int RequestedRunCount,
    int? CompletedRunCount,
    int? PublishedRunCount,
    int? NoChangesRunCount,
    int? FailedRunCount,
    int? DetailedValidationRunNumber,
    bool? DetailedValidationUsesFinalRun,
    double? MeanElapsedMilliseconds,
    double? MinElapsedMilliseconds,
    double? MaxElapsedMilliseconds,
    double? MeanManagedAllocatedBytes,
    long? MinManagedAllocatedBytes,
    long? MaxManagedAllocatedBytes);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary(
    string Status,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary LiveSnapshot,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary RebuildBuild,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary Save,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary OpenValidation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary Publication);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseSummary(
    string Status,
    int MeasuredCount,
    int NotExecutedCount,
    int MissingCount,
    double? TotalElapsedMilliseconds,
    long? TotalManagedAllocatedBytes);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputSummary(
    string Status,
    int? FileCount,
    long? TotalBytes,
    long? ManifestBytes,
    long? IdsBytes,
    long? VectorsBytes,
    long? LevelsBytes,
    long? GraphBytes,
    int? OutputVectorCount,
    double? BytesPerLiveVector,
    string? ValidationOpenStatus,
    string? ScanTimingScope);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixSearchSummary(
    string Status,
    double? RecallAtK,
    double? OrderedAgreement,
    string? DistanceToleranceStatus,
    int? DistanceMismatchCount,
    int? MissingResultCount,
    int? ExtraResultCount,
    string? ReturnedResultIntegrityStatus,
    int? CheckedResultCount,
    int? UnknownIdCount,
    int? TombstonedIdCount,
    int? IntegrityDistanceMismatchCount,
    int? UnderfilledQueryCount,
    int? UnderfilledSlotCount,
    double? MeanQps,
    double? MeanLatencyP95Milliseconds,
    double? MeanManagedAllocatedBytesPerQuery);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedValidationSummary(
    string Status,
    int? ExpectedVectorCount,
    int? OpenedVectorCount,
    int? IdMismatchCount,
    int? VectorMismatchCount);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixParitySummary(
    string Status,
    int? QueryCount,
    int? WrittenCountMismatchCount,
    int? IdMismatchCount,
    int? OrderMismatchCount,
    int? DistanceMismatchCount,
    bool? AllResultsMatched);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationSummary(
    string Status,
    bool? DeletedReservedIdsRejectedAfterCheckpoint,
    int ExpectedDeletedReservedIdCount,
    int? ActualDeletedReservedIdCount);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesSummary(
    string Status,
    bool? GenerationUnchanged,
    bool? OutputDirectoryRemainedEmpty,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary Phases);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixCountSummary(
    string Status,
    int ExpectedBasePhysicalVectorCount,
    int ExpectedDeltaPhysicalVectorCount,
    int ExpectedPhysicalCandidateVectorCount,
    int ExpectedLiveVectorCount,
    int ExpectedDeletedReservedIdCount,
    int? PreCheckpointLiveVectorCount,
    int? PreCheckpointTombstoneCount,
    int? PreCheckpointDeletedReservedIdCount,
    int? PostCheckpointBasePhysicalVectorCount,
    int? PostCheckpointLiveVectorCount,
    int? PostCheckpointTombstoneCount,
    int? PostCheckpointDeletedReservedIdCount,
    double? PreCheckpointTombstoneRatio,
    double? PreCheckpointDeltaInsertRatio);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixMemorySummary(
    string Status,
    string Unit,
    string Reason);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
    string Status,
    bool LinkedReportInspected,
    int NonFalseEligibilityFlagCount,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    bool AllEligibilityFlagsFalse);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount,
    int LinkedReportCount,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthAggregate CacheTruth,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunAggregate CheckpointRuns,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixPhaseDiagnosticsSummary PhaseDiagnostics,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputAggregate OutputBytes,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary PreCheckpointSourceCompositeSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary PostCheckpointRebuiltCompositeSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary OpenedReadOnlyHnswSearch,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedAggregate OpenedValidation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixParityAggregate RebuiltOpenedParity,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationAggregate DeletedReservation,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesAggregate NoChanges,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixMemorySummary Memory,
    ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary RecursiveEligibility);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixCacheTruthAggregate(
    string Status,
    bool Available,
    bool AllCasesBlockedBySharedReadiness,
    int BlockedCaseCount,
    string? ErrorMessage);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixRepeatedRunAggregate(
    string Status,
    int RequestedRunCountPerCase,
    int RecordedCaseCount,
    int RequestedRunCountTotal,
    int CompletedRunCount,
    int PublishedRunCount,
    int NoChangesRunCount,
    int FailedRunCount,
    int FinalRunDetailedValidationCaseCount,
    double? MeanElapsedMilliseconds,
    double? MaxElapsedMilliseconds,
    double? MeanManagedAllocatedBytes,
    long? MaxManagedAllocatedBytes);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixOutputAggregate(
    string Status,
    int RecordedCaseCount,
    int TotalFileCount,
    long TotalBytes,
    long ManifestBytes,
    long IdsBytes,
    long VectorsBytes,
    long LevelsBytes,
    long GraphBytes,
    double? MinBytesPerLiveVector,
    double? MaxBytesPerLiveVector,
    string ScanTimingScope);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary(
    string Status,
    int RecordedCaseCount,
    double? MinRecallAtK,
    double? MaxRecallAtK,
    double? MinOrderedAgreement,
    double? MaxOrderedAgreement,
    int ReturnedResultIntegrityNotPassedCaseCount,
    int DistanceToleranceNotPassedCaseCount,
    int TotalUnderfilledQueryCount,
    int TotalUnderfilledSlotCount,
    double? MaxMeanManagedAllocatedBytesPerQuery);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixOpenedAggregate(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int IdMismatchCount,
    int VectorMismatchCount);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixParityAggregate(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int IdMismatchCount,
    int OrderMismatchCount,
    int DistanceMismatchCount);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixDeletedReservationAggregate(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int ExpectedDeletedReservedIdCountTotal,
    int ActualDeletedReservedIdCountTotal);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixNoChangesAggregate(
    string Status,
    int RecordedCaseCount,
    int PassedCaseCount,
    int GenerationChangedCaseCount,
    int OutputDirectoryNotEmptyCaseCount);

public sealed record ExternalHnswBasePlusExactDeltaCheckpointMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string ComparisonPublicationReason,
    string RegressionGateReason);
