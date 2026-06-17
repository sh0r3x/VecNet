namespace VecNet.BenchmarkRunner;

public sealed record GeneratedExactCandidateSetBenchmarkReport(
    string SchemaName,
    string SchemaVersion,
    string ReportId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ScenarioName,
    string ClaimClass,
    string PrivacyClass,
    GeneratedExactCandidateSetEvidenceInfo Evidence,
    RepositoryInfo Repository,
    RunnerInfo Runner,
    CommandInfo Command,
    EnvironmentInfo Environment,
    DatasetInfo Dataset,
    TruthInfo Truth,
    ScenarioInfo Scenario,
    IndexInfo Index,
    GeneratedExactCandidateInputInfo CandidateInput,
    GeneratedExactCandidateSetInfo CandidateSet,
    SearchInfo Search,
    MeasurementInfo Measurement,
    GeneratedExactFilteredMetricsInfo Metrics,
    GeneratedExactCandidateSetValidationInfo Validation,
    GeneratedExactCandidateSetEligibilityInfo Eligibility,
    string[] Notes);

public sealed record GeneratedExactCandidateSetEvidenceInfo(
    string Status,
    string Scope,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason,
    string[] Limitations);

public sealed record GeneratedExactCandidateInputInfo(
    string Kind,
    string SelectivityTarget,
    double ActualSelectivity,
    int KnownIdCountPerQuery,
    int DuplicateIdCountPerQuery,
    int UnknownIdCountPerQuery,
    int InputIdCountPerQuery,
    int MinKnownVisibleCount,
    int MaxKnownVisibleCount,
    double MeanKnownVisibleCount,
    int TotalKnownIdCount,
    int TotalDuplicateIdCount,
    int TotalUnknownIdCount,
    string InputOrder,
    string GenerationFormula,
    string DuplicatePolicy,
    string UnknownIdPolicy,
    string ApplicationScope);

public sealed record GeneratedExactCandidateSetInfo(
    string ConstructionStatus,
    string ConstructionOperation,
    string ConstructionTimingScope,
    string ConstructionAllocationScope,
    bool ConstructedBeforeMeasuredSearch,
    int ConstructedSetCount,
    int CountPerQuery,
    int MinCount,
    int MaxCount,
    double MeanCount,
    int TotalCandidateCount,
    string Binding,
    string DuplicateHandling,
    string UnknownIdHandling,
    string PersistenceScope);

public sealed record GeneratedExactCandidateSetValidationInfo(
    string Status,
    string EvidenceStatus,
    bool FiniteVectors,
    bool TruthGenerated,
    bool CandidateSetsConstructed,
    bool FinalRunComparedToTruth,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    bool ReportIsPrivateRaw);

public sealed record GeneratedExactCandidateSetEligibilityInfo(
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string PublicClaimReason,
    string BaselineCandidateReason,
    string RegressionGateReason);
