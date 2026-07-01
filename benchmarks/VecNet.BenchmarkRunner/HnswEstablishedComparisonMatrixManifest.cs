namespace VecNet.BenchmarkRunner;

public sealed record HnswEstablishedComparisonMatrixManifest(
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
    HnswEstablishedComparisonSourcePinningInfo SourcePinning,
    HnswEstablishedComparisonMatrixDesignInfo Design,
    int CaseCount,
    HnswEstablishedComparisonMatrixCaseManifest[] Cases,
    HnswEstablishedComparisonMatrixAggregate Aggregate,
    HnswEstablishedComparisonMatrixEligibility Eligibility,
    string[] Notes);

public sealed record HnswEstablishedComparisonMatrixDesignInfo(
    int[] RepresentativeGeneratedDimensions,
    int[] OptionalAdversarialTailDimensions,
    HnswEstablishedComparisonMatrixProfileInfo[] Profiles,
    int[] TopKValues,
    string Metric,
    string WorkloadPolicy,
    string TailDimensionPolicy,
    string PresetPolicy);

public sealed record HnswEstablishedComparisonMatrixProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record HnswEstablishedComparisonMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string ProfileName,
    string Metric,
    int Dimension,
    string DimensionRole,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string DataSeed,
    string HnswSeed,
    int M,
    int EfConstruction,
    int EfSearch,
    string LinkedReportPath,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record HnswEstablishedComparisonMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount);

public sealed record HnswEstablishedComparisonMatrixEligibility(
    string ClaimClass,
    string PrivacyClass,
    string EvidenceStatus,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonPublicationReason,
    string RegressionGateReason);
