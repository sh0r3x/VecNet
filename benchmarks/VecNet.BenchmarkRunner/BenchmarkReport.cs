namespace VecNet.BenchmarkRunner;

public sealed record BenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ClaimClass,
    string PrivacyClass,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    SearchInfo Search,
    MetricsInfo Metrics,
    ValidationInfo Validation,
    string[] Notes);

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
    double Qps);

public sealed record MetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount);

public sealed record ValidationInfo(
    string Status,
    bool FiniteVectors,
    bool TruthGenerated,
    bool ReportIsPrivateRaw);
