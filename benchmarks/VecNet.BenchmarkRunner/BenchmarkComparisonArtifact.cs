namespace VecNet.BenchmarkRunner;

public sealed record BenchmarkComparisonArtifact(
    string SchemaName,
    string SchemaVersion,
    string ComparisonId,
    DateTimeOffset GeneratedAtUtc,
    string TaskId,
    string ClaimClass,
    string PrivacyClass,
    ComparisonEvidenceInfo Evidence,
    string ArtifactKind,
    ComparisonArtifactIdentity BaselineArtifact,
    ComparisonArtifactIdentity CurrentArtifact,
    ComparisonCompatibility Compatibility,
    MetricComparisonEntry[] Metrics,
    MatrixCaseComparisonEntry[] Cases,
    MatrixComparisonSummary? MatrixSummary,
    ComparisonWarningSummary Warnings,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string[] Notes);

public sealed record ComparisonEvidenceInfo(
    string Status,
    string Scope,
    string Policy,
    string Reason);

public sealed record ComparisonArtifactIdentity(
    string Kind,
    string Path,
    string? Id,
    string? SchemaName,
    string? SchemaVersion,
    DateTimeOffset? GeneratedAtUtc,
    RepositoryInfo? Repository,
    bool BaselineCandidateEligible,
    bool Dirty);

public sealed record ComparisonCompatibility(
    string Status,
    CompatibilityReason[] Reasons);

public sealed record CompatibilityReason(
    string Code,
    string Field,
    string? Expected,
    string? Actual,
    string Message);

public sealed record MetricComparisonEntry(
    string Name,
    string Unit,
    string Direction,
    double? BaselineValue,
    double? CurrentValue,
    double? AbsoluteDelta,
    double? Ratio,
    double? PercentChange,
    double? WorseFraction,
    double? ImprovementFraction,
    AvailableNoiseInfo AvailableNoise,
    string WarningLabel);

public sealed record AvailableNoiseInfo(
    string Status,
    double? Fraction,
    double? Percent,
    string Reason);

public sealed record MatrixCaseComparisonEntry(
    string CaseKey,
    string? BaselineReportId,
    string? CurrentReportId,
    ComparisonCompatibility Compatibility,
    MetricComparisonEntry[] Metrics,
    ComparisonWarningSummary Warnings);

public sealed record MatrixComparisonSummary(
    int ComparableCaseCount,
    int NotComparableCaseCount,
    int ImprovedCaseCount,
    int UnchangedCaseCount,
    int NoisyOrInconclusiveCaseCount,
    int CorrectnessWarningCount,
    int PerformanceWarningCount,
    int AllocationWarningCount);

public sealed record ComparisonWarningSummary(
    string Status,
    string[] Labels,
    int CorrectnessWarningCount,
    int PerformanceWarningCount,
    int AllocationWarningCount,
    int DirtyCurrentWarningCount,
    int NoiseDominatedCount,
    int NoiseUnavailableCount,
    int ImprovedCount,
    int ImprovedWithinNoiseCount,
    int UnchangedCount,
    int NotComparableCount);
