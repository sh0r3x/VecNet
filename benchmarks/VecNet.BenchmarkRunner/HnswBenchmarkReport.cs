namespace VecNet.BenchmarkRunner;

public sealed record HnswBenchmarkReport(
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
    SearchInfo Search,
    MeasurementInfo Measurement,
    HnswMemoryEstimateInfo MemoryEstimate,
    HnswMetricsInfo Metrics,
    HnswValidationInfo Validation,
    HnswEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record HnswConfigurationInfo(
    int M,
    int MMax,
    int MMax0,
    int EfConstruction,
    int EfSearch,
    string RandomSeed,
    string InsertionOrder,
    string MetricScope);

public sealed record HnswBuildInfo(
    string Status,
    double ElapsedMilliseconds,
    MeasurementStatusInfo ManagedAllocations,
    int VectorCount,
    int Dimension,
    string IncludedOperations,
    string ExcludedOperations);

public sealed record HnswMemoryEstimateInfo(
    string Status,
    string EstimateKind,
    string Unit,
    long TotalEstimatedBytes,
    long VectorBytes,
    long IdBytes,
    long LevelBytes,
    long GraphAdjacencyBytes,
    long GraphCountBytes,
    long SearchWorkspaceBytes,
    int MaxLayer,
    int LayerCount,
    HnswLayerMemoryEstimateInfo[] Layers,
    string Reason,
    string[] Exclusions);

public sealed record HnswLayerMemoryEstimateInfo(
    int Layer,
    int Stride,
    long NeighborBytes,
    long CountBytes);

public sealed record HnswMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    HnswReturnedResultIntegrityInfo ReturnedResultIntegrity,
    string RecallDefinition,
    string DistanceValidationScope);

public sealed record HnswReturnedResultIntegrityInfo(
    string Status,
    int CheckedResultCount,
    int QueryCountMismatchCount,
    int ResultCountViolationCount,
    int NonFiniteDistanceCount,
    int DuplicateIdCount,
    int UnknownIdCount,
    int DistanceMismatchCount,
    string Policy,
    string Reason);

public sealed record HnswValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool TruthGenerated,
    bool FinalRunComparedToTruth,
    bool AllowsApproximateRecallBelowOne,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record HnswEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
