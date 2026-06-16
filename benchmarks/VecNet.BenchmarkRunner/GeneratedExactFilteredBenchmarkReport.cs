namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactFilteredBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactFilteredEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactFilterInfo Filter,
    SearchInfo Search,
    MeasurementInfo Measurement,
    GeneratedExactFilteredMetricsInfo Metrics,
    GeneratedExactFilteredValidationInfo Validation,
    GeneratedExactFilteredEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactFilteredEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record GeneratedExactFilterInfo(
    string Kind,
    string SelectivityTarget,
    double ActualSelectivity,
    int VisibleCountPerQuery,
    int KnownIdCountPerQuery,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    int AllowlistLengthPerQuery,
    int MinVisibleCount,
    int MaxVisibleCount,
    double MeanVisibleCount,
    int TotalKnownIdCount,
    int TotalDuplicateIdCount,
    int TotalUnknownIdCount,
    string AllowlistOrder,
    string GenerationFormula,
    string DuplicatePolicy,
    string UnknownIdPolicy);

public sealed record GeneratedExactFilteredMetricsInfo(
    double RecallAtK,
    double OrderedAgreement,
    string DistanceToleranceStatus,
    int DistanceMismatchCount,
    int MissingResultCount,
    int ExtraResultCount,
    GeneratedExactFilteredResultIntegrityInfo FilteredResultIntegrity,
    string RecallDefinition,
    string DistanceValidationScope);

public sealed record GeneratedExactFilteredResultIntegrityInfo(
    string Status,
    int QueryCountMismatchCount,
    int CheckedResultCount,
    int MissingResultCount,
    int ExtraResultCount,
    int WrongIdCount,
    int OrderMismatchCount,
    int NonFiniteDistanceCount,
    int DistanceMismatchCount,
    string Policy,
    string Reason);

public sealed record GeneratedExactFilteredValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool TruthGenerated,
    bool FinalRunComparedToTruth,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactFilteredEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
