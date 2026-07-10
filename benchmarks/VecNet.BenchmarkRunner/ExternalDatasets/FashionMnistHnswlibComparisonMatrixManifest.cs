namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistHnswlibComparisonMatrixManifest(
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
    FashionMnistHnswlibComparisonMatrixCacheTruthInfo CacheTruth,
    HnswEstablishedComparisonSourcePinningInfo SourcePinning,
    FashionMnistHnswlibComparisonMatrixDesignInfo Design,
    int CaseCount,
    FashionMnistHnswlibComparisonMatrixCaseManifest[] Cases,
    FashionMnistHnswlibComparisonMatrixAggregate Aggregate,
    FashionMnistHnswlibComparisonMatrixEligibility Eligibility,
    string[] Notes);

public sealed record FashionMnistHnswlibComparisonMatrixCacheTruthInfo(
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
    string? ErrorMessage);

public sealed record FashionMnistHnswlibComparisonMatrixDesignInfo(
    FashionMnistHnswlibComparisonMatrixProfileInfo[] Profiles,
    int[] TopKValues,
    string DatasetId,
    int Dimension,
    string Metric,
    string WorkloadPolicy,
    string PresetPolicy);

public sealed record FashionMnistHnswlibComparisonMatrixProfileInfo(
    string Name,
    int M,
    int EfConstruction,
    int EfSearch);

public sealed record FashionMnistHnswlibComparisonMatrixCaseManifest(
    int CaseNumber,
    string CaseId,
    string ProfileName,
    string DatasetId,
    string Metric,
    int Dimension,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    string Seed,
    int M,
    int EfConstruction,
    int EfSearch,
    string LinkedReportPath,
    string? LinkedReportId,
    string Status,
    string ValidationStatus,
    string? ErrorMessage);

public sealed record FashionMnistHnswlibComparisonMatrixAggregate(
    int PassedCaseCount,
    int FailedCaseCount,
    int SkippedCaseCount,
    int BlockedCaseCount);

public sealed record FashionMnistHnswlibComparisonMatrixEligibility(
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
