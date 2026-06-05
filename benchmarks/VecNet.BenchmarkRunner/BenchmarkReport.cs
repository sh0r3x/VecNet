namespace VecNet.BenchmarkRunner;

public sealed record BenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ClaimClass,
    string PrivacyClass,
    EvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    SearchInfo Search,
    MeasurementInfo Measurement,
    MetricsInfo Metrics,
    BaselineInfo Baseline,
    ValidationInfo Validation,
    string[] Notes);

public sealed record EvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    string PublicClaimReason,
    string[] Limitations);

public sealed record RunnerInfo(string Name, string Version, string[] Arguments);

public sealed record CommandInfo(string Scenario, string[] Arguments);

public sealed record EnvironmentInfo(
    string OsDescription,
    string ProcessArchitecture,
    string FrameworkDescription,
    string RuntimeIdentifier,
    int ProcessorCount,
    bool ServerGc,
    int VectorFloatCount);

public sealed record DatasetInfo(
    string Kind,
    string SourceVerificationStatus,
    string Distribution,
    string Seed,
    string Metric,
    int Dimension,
    int VectorCount,
    int QueryCount);

public sealed record TruthInfo(string Kind, int Depth, string TiePolicy);

public sealed record ScenarioInfo(
    string Name,
    int TopK,
    int MeasuredQueryCount,
    int Concurrency,
    string ExcludedFromSearchTiming);

public sealed record IndexInfo(
    string Profile,
    string Type,
    string Metric,
    int Dimension,
    int VectorCount,
    string Configuration);

public sealed record SearchInfo(
    int MeasuredQueryCount,
    double ElapsedMilliseconds,
    double LatencyP50Milliseconds,
    double LatencyP95Milliseconds,
    double LatencyP99Milliseconds,
    double Qps,
    SearchRunInfo[] Runs,
    AggregateTimingInfo Aggregate);

public sealed record SearchRunInfo(
    int RunNumber,
    int MeasuredQueryCount,
    double ElapsedMilliseconds,
    double LatencyP50Milliseconds,
    double LatencyP95Milliseconds,
    double LatencyP99Milliseconds,
    double Qps,
    long ManagedAllocatedBytes,
    double ManagedAllocatedBytesPerQuery);

public sealed record AggregateTimingInfo(
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
    double MeanManagedAllocatedBytes,
    long MinManagedAllocatedBytes,
    long MaxManagedAllocatedBytes,
    double MeanManagedAllocatedBytesPerQuery,
    double MinManagedAllocatedBytesPerQuery,
    double MaxManagedAllocatedBytesPerQuery);

public sealed record MeasurementInfo(
    LatencyMeasurementInfo Latency,
    MeasurementStatusInfo ManagedAllocations,
    MeasurementStatusInfo Memory,
    RepeatedRunInfo RepeatedRuns,
    RunToRunNoiseInfo RunToRunNoise,
    WarmupInfo Warmup);

public sealed record LatencyMeasurementInfo(
    string Status,
    string Unit,
    string SampleScope,
    string TimedOperation,
    string ExcludedOperations,
    string PercentileEstimator,
    string AggregateSemantics,
    string RawSampleDisclosure);

public sealed record MeasurementStatusInfo(
    string Status,
    string Value,
    string Unit,
    string Reason);

public sealed record RepeatedRunInfo(
    string Status,
    int RunCount,
    bool VarianceMeasured,
    string Reason);

public sealed record RunToRunNoiseInfo(
    string Status,
    int RunCount,
    bool NoiseMeasured,
    string Scope,
    string Statistics,
    string Reason,
    string NonGoals,
    RunToRunMetricNoiseInfo ElapsedMilliseconds,
    RunToRunMetricNoiseInfo Qps,
    RunToRunMetricNoiseInfo LatencyP50Milliseconds,
    RunToRunMetricNoiseInfo LatencyP95Milliseconds,
    RunToRunMetricNoiseInfo LatencyP99Milliseconds,
    RunToRunMetricNoiseInfo ManagedAllocatedBytesPerQuery);

public sealed record RunToRunMetricNoiseInfo(
    string Status,
    string Unit,
    double? Mean,
    double? SampleStandardDeviation,
    double? CoefficientOfVariation,
    double? Min,
    double? Max,
    double? Spread,
    string Reason);

public sealed record WarmupInfo(
    string Status,
    int WarmupCount,
    string Reason);

public sealed record MetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount);

public sealed record BaselineInfo(
    string? BaselineReportId,
    string Suitability,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string Reason);

public sealed record ValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool TruthGenerated,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool ReportIsPrivateRaw);
