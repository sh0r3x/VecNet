namespace VecNet.BenchmarkRunner;

public sealed record HnswEstablishedComparisonReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    HnswEstablishedComparisonEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    HnswEstablishedComparisonSourcePinningInfo SourcePinning,
    HnswEstablishedComparisonDesignInfo Design,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    HnswEstablishedComparisonMethodologyInfo Methodology,
    HnswEstablishedComparisonParametersInfo Parameters,
    HnswEstablishedComparisonImplementationResult VecNet,
    HnswEstablishedComparisonImplementationResult Hnswlib,
    HnswEstablishedComparisonValidationInfo Validation,
    HnswEstablishedComparisonEligibilityInfo Eligibility,
    string[] Notes);

public sealed record HnswEstablishedComparisonEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonPublicationReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record HnswEstablishedComparisonSourcePinningInfo(
    string ExternalImplementation,
    string PackageName,
    string PackageSource,
    string PackageVersion,
    string SourceDistributionSha256,
    string License,
    string LicensePosture,
    string NativeBoundary,
    string ProductDependencyPosture);

public sealed record HnswEstablishedComparisonDesignInfo(
    int[] RepresentativeGeneratedDimensions,
    int[] OptionalAdversarialTailDimensions,
    int CurrentDimension,
    string CurrentDimensionRole,
    string Metric,
    string WorkloadPolicy,
    string TailDimensionPolicy);

public sealed record HnswEstablishedComparisonMethodologyInfo(
    string IdenticalInputsPolicy,
    string TimingScope,
    string ExcludedOperations,
    string LatencyPercentileEstimator,
    string AggregateSemantics,
    string ThreadingPolicy,
    string PythonBoundary,
    string ResultValidationPolicy);

public sealed record HnswEstablishedComparisonParametersInfo(
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount,
    int TopK,
    int Runs,
    int WarmupQueries,
    int M,
    int EfConstruction,
    int HnswlibEfConstruction,
    int EfSearch,
    int HnswlibEf,
    string DataSeed,
    string HnswSeed,
    string InsertionOrder,
    int ThreadCount);

public sealed record HnswEstablishedComparisonImplementationResult(
    string Name,
    string ImplementationType,
    string Version,
    string RuntimeBoundary,
    string Identity,
    HnswEstablishedComparisonBuildInfo Build,
    HnswEstablishedComparisonSearchInfo Search,
    HnswEstablishedComparisonMetricsInfo Metrics,
    MeasurementStatusInfo Memory,
    MeasurementStatusInfo PersistedBytes);

public sealed record HnswEstablishedComparisonBuildInfo(
    string Status,
    double ElapsedMilliseconds,
    MeasurementStatusInfo ManagedAllocations,
    string IncludedOperations,
    string ExcludedOperations);

public sealed record HnswEstablishedComparisonSearchInfo(
    string Status,
    int MeasuredQueryCount,
    double ElapsedMilliseconds,
    double LatencyP50Milliseconds,
    double LatencyP95Milliseconds,
    double LatencyP99Milliseconds,
    double Qps,
    HnswEstablishedComparisonSearchRunInfo[] Runs,
    HnswEstablishedComparisonAggregateTimingInfo Aggregate,
    MeasurementStatusInfo ManagedAllocations);

public sealed record HnswEstablishedComparisonSearchRunInfo(
    int RunNumber,
    int MeasuredQueryCount,
    double ElapsedMilliseconds,
    double LatencyP50Milliseconds,
    double LatencyP95Milliseconds,
    double LatencyP99Milliseconds,
    double Qps,
    long? ManagedAllocatedBytes,
    double? ManagedAllocatedBytesPerQuery);

public sealed record HnswEstablishedComparisonAggregateTimingInfo(
    int RunCount,
    int MeasuredQueryCountPerRun,
    double MeanElapsedMilliseconds,
    double MinElapsedMilliseconds,
    double MaxElapsedMilliseconds,
    double MeanLatencyP50Milliseconds,
    double MeanLatencyP95Milliseconds,
    double MeanLatencyP99Milliseconds,
    double MeanQps,
    double MinQps,
    double MaxQps,
    double? MeanManagedAllocatedBytes,
    double? MeanManagedAllocatedBytesPerQuery);

public sealed record HnswEstablishedComparisonMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    HnswReturnedResultIntegrityInfo ReturnedResultIntegrity,
    string RecallDefinition,
    string OrderedAgreementScope);

public sealed record HnswEstablishedComparisonValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool TruthGenerated,
    bool IdenticalVectorsQueriesIdsAndParameters,
    bool VecNetComparedToTruth,
    bool HnswlibComparedToTruth,
    bool VecNetReturnedResultIntegrityPassed,
    bool HnswlibReturnedResultIntegrityPassed,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record HnswEstablishedComparisonEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ComparisonPublicationEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string ComparisonPublicationReason,
    string RegressionGateReason);
