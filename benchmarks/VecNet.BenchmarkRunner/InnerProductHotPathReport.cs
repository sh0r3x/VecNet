namespace VecNet.BenchmarkRunner;

public sealed record InnerProductHotPathReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    InnerProductHotPathEvidenceInfo Evidence,
    InnerProductHotPathSourceInfo Source,
    InnerProductHotPathRunnerInfo Runner,
    EnvironmentInfo Environment,
    InnerProductHotPathOptionsInfo Options,
    InnerProductHotPathCaseInfo[] Cases,
    InnerProductHotPathValidationSummaryInfo Validation,
    string[] Notes);

public sealed record InnerProductHotPathEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string Reason,
    string[] Limitations);

public sealed record InnerProductHotPathSourceInfo(string? Commit, bool Dirty);

public sealed record InnerProductHotPathRunnerInfo(string Name, string Version, string CommandScenario);

public sealed record InnerProductHotPathOptionsInfo(
    string Metric,
    int[] Dimensions,
    string[] OperationShapes,
    int VectorCount,
    int QueryCount,
    int Runs,
    int WarmupIterations,
    int EfConstruction,
    int EfSearch,
    string Seed);

public sealed record InnerProductHotPathCaseInfo(
    string CaseId,
    string Metric,
    int Dimension,
    string DimensionClass,
    string OperationShape,
    InnerProductHotPathWorkloadInfo Workload,
    InnerProductHotPathImplementationMeasurementInfo CurrentScalar,
    InnerProductHotPathImplementationMeasurementInfo CandidateSharedDot,
    InnerProductHotPathCaseValidationInfo Validation);

public sealed record InnerProductHotPathWorkloadInfo(
    int VectorCount,
    int QueryCount,
    long DistanceCallCount,
    string CallShape,
    string IncludedOperations,
    string ExcludedOperations);

public sealed record InnerProductHotPathImplementationMeasurementInfo(
    string Implementation,
    string Primitive,
    long DistanceCallCount,
    double ElapsedMilliseconds,
    double DistanceCallsPerSecond,
    long ManagedAllocatedBytes,
    double ManagedAllocatedBytesPerDistanceCall,
    string ChecksumCategory,
    float Checksum);

public sealed record InnerProductHotPathCaseValidationInfo(
    string Status,
    int ComparedDistanceCount,
    int FiniteMatchCount,
    int PositiveInfinityMatchCount,
    int NegativeInfinityMatchCount,
    int NaNMatchCount,
    int CategoryMismatchCount,
    int FiniteDistanceMismatchCount,
    double MaxFiniteAbsoluteDelta,
    string[] DriftExamples);

public sealed record InnerProductHotPathValidationSummaryInfo(
    string Status,
    int CaseCount,
    int PassedCaseCount,
    int FailedCaseCount,
    int ComparedDistanceCount,
    int CategoryMismatchCount,
    int FiniteDistanceMismatchCount,
    int PositiveInfinityComparisons,
    int NegativeInfinityComparisons,
    int NaNComparisons,
    string Policy);
