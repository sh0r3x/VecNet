namespace VecNet.BenchmarkRunner;

public sealed record HnswBasePlusExactDeltaCheckpointMatrixManifest(
    string SchemaName,
    string SchemaVersion,
    string TaskId,
    string ScenarioName,
    string PresetName,
    DateTimeOffset GeneratedAtUtc,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    string OutputDirectory,
    HnswBasePlusExactDeltaCheckpointMatrixDesignInfo Design,
    int CaseCount,
    string ValidationStatus,
    HnswBasePlusExactDeltaCheckpointMatrixCaseManifest[] Cases,
    HnswBasePlusExactDeltaCheckpointMatrixAggregate Aggregate,
    HnswBasePlusExactDeltaCheckpointMatrixEligibility Eligibility,
    string[] Notes);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixDesignInfo(
    string Metric,
    int[] Dimensions,
    int[] TopKValues,
    HnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo[] HnswProfiles,
    HnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo[] UpdateProfiles,
    string WorkloadPolicy,
    string PresetPolicy,
    string ScopePolicy);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixHnswProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixUpdateProfileInfo(
    string Name,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    string Description);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string HnswProfileName,
    string UpdateProfileName,
    string Metric,
    int Dimension,
    int BaseVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    int InsertedDeltaVectorCount,
    int DeletedBaseVectorCount,
    int DeletedDeltaVectorCount,
    int DuplicateInsertAttempts,
    int UnknownDeleteAttempts,
    int RepeatedDeleteAttempts,
    string LinkedReportPath,
    string LinkedCheckpointDirectoryPath,
    string[] CommandArguments,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    HnswBasePlusExactDeltaCheckpointMatrixValidationSummary ValidationSummary,
    HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary RepeatedCheckpointRuns,
    HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary CheckpointSummary,
    HnswBasePlusExactDeltaCheckpointMatrixSearchSummary PreCheckpointSearch,
    HnswBasePlusExactDeltaCheckpointMatrixSearchSummary PostCheckpointSearch,
    HnswBasePlusExactDeltaCheckpointMatrixSearchSummary OpenedReadOnlySearch,
    HnswBasePlusExactDeltaCheckpointMatrixCountSummary CountSummary,
    HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary RecursiveEligibility,
    string? ErrorMessage);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixValidationSummary(
    string Status,
    bool? CheckpointResultStatusPublished,
    bool? CheckpointResultCountsMatched,
    bool? CheckpointGenerationAdvancedExactlyOnce,
    bool? PhaseDiagnosticsMeasuredForPublishedCheckpoint,
    bool? CheckpointRepeatedRunEvidencePresent,
    int? DetailedValidationRunNumber,
    bool? DetailedValidationUsesFinalRun,
    bool? OpenedReadOnlyHnswIdVectorValidationPassed,
    bool? RebuiltCompositeOpenedHnswSearchParityPassed,
    bool? ReturnedResultIntegrityPassedForAllSearches,
    bool? NoChangesCheckpointProbePassed,
    bool? DeletedReservedIdsRejectedAfterCheckpoint,
    bool? OutputBytesScannedOutsideCheckpointDuration);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixRepeatedRunSummary(
    string Status,
    int? RunCount,
    int? DetailedValidationRunNumber,
    double? MeanElapsedMilliseconds,
    double? MinElapsedMilliseconds,
    double? MaxElapsedMilliseconds,
    double? MeanManagedAllocatedBytes,
    long? MinManagedAllocatedBytes,
    long? MaxManagedAllocatedBytes);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixCheckpointSummary(
    string Status,
    double? FinalRunElapsedMilliseconds,
    long? FinalRunManagedAllocatedBytes,
    long? GenerationBeforeCheckpoint,
    long? GenerationAfterCheckpoint,
    bool? GenerationAdvancedExactlyOnce,
    int? OutputFileCount,
    long? OutputTotalBytes,
    string? OutputScanTimingScope);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixSearchSummary(
    string Status,
    double? RecallAtK,
    double? OrderedAgreement,
    string? ReturnedResultIntegrityStatus,
    int? UnderfilledQueryCount,
    int? UnderfilledSlotCount,
    double? MeanQps,
    double? MeanLatencyP95Milliseconds,
    double? MeanManagedAllocatedBytesPerQuery);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixCountSummary(
    string Status,
    int BasePhysicalVectorCount,
    int PhysicalVectorCount,
    int ExpectedLiveVectorCount,
    int? PreCheckpointLiveVectorCount,
    int? PreCheckpointTombstoneCount,
    int? PostCheckpointBasePhysicalVectorCount,
    int? PostCheckpointLiveVectorCount,
    int? PostCheckpointTombstoneCount,
    int? DeletedReservedIdCount,
    double? PreCheckpointTombstoneRatio,
    double? PreCheckpointDeltaInsertRatio);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary(
    string Status,
    bool LinkedReportInspected,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    bool AllEligibilityFlagsFalse);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount,
    int LinkedReportCount,
    int TotalCheckpointRunCount,
    int ValidationPassedCaseCount,
    int RepeatedCheckpointRunEvidenceCaseCount,
    HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary PreCheckpointSearch,
    HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary PostCheckpointSearch,
    HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary OpenedReadOnlySearch,
    HnswBasePlusExactDeltaCheckpointMatrixAggregateCheckpointSummary Checkpoint,
    HnswBasePlusExactDeltaCheckpointMatrixEligibilitySummary RecursiveEligibility);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixAggregateSearchSummary(
    string Status,
    int RecordedCaseCount,
    double? MinRecallAtK,
    double? MaxRecallAtK,
    double? MinOrderedAgreement,
    double? MaxOrderedAgreement,
    int? TotalUnderfilledQueryCount,
    int? TotalUnderfilledSlotCount,
    double? MaxMeanManagedAllocatedBytesPerQuery);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixAggregateCheckpointSummary(
    string Status,
    int RecordedCaseCount,
    int PublishedCaseCount,
    double? MeanCheckpointElapsedMilliseconds,
    double? MaxCheckpointElapsedMilliseconds,
    double? MeanCheckpointManagedAllocatedBytes,
    long? MaxCheckpointManagedAllocatedBytes,
    long? TotalOutputBytes);

public sealed record HnswBasePlusExactDeltaCheckpointMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonArtifactEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonArtifactReason,
    string RegressionGateReason);
