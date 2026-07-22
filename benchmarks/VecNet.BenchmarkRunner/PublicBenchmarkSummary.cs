using System.Text.Json;

namespace VecNet.BenchmarkRunner;

public sealed record PublicBenchmarkSummary(
    string SchemaName,
    string SchemaVersion,
    string SummaryId,
    DateTimeOffset GeneratedAtUtc,
    string ClaimClass,
    string ClaimText,
    PublicBenchmarkClaimScope Scope,
    bool PublicClaimEligible,
    string PublicClaimStatus,
    PublicBenchmarkSourceIdentity Source,
    PublicBenchmarkCommand[] Commands,
    PublicBenchmarkEnvironment Environment,
    PublicBenchmarkDatasetIdentity Dataset,
    PublicBenchmarkMeasurementCategories Measurements,
    PublicBenchmarkPrivacyStatus Privacy,
    string[] Limitations,
    PublicBenchmarkReviewStatus Review,
    ExactGeneratedPublicEvidenceValidationInfo? EvidenceValidation = null);

public sealed record PublicBenchmarkClaimScope(
    string ProductSurface,
    string Algorithm,
    string Metric,
    string Scenario,
    string EvidenceClass,
    string ClaimScope);

public sealed record PublicBenchmarkSourceIdentity(
    string PackageId,
    string PackageVersion,
    string GitCommit,
    string GitTag,
    string WorkingTreeStatus);

public sealed record PublicBenchmarkCommand(
    string Name,
    string[] Arguments,
    string WorkingDirectoryDisclosure);

public sealed record PublicBenchmarkEnvironment(
    string OperatingSystem,
    string Architecture,
    string Runtime,
    string Cpu,
    string Memory,
    string Storage,
    string Disclosure);

public sealed record PublicBenchmarkDatasetIdentity(
    string Kind,
    string Name,
    string Version,
    string Source,
    string License,
    string Checksum,
    string Seed,
    int VectorCount,
    int QueryCount,
    int Dimension,
    string Metric,
    string TruthMethod);

public sealed record PublicBenchmarkMeasurementCategories(
    PublicBenchmarkMeasurementCategory Latency,
    PublicBenchmarkMeasurementCategory Throughput,
    PublicBenchmarkMeasurementCategory Recall,
    PublicBenchmarkMeasurementCategory ManagedAllocation,
    PublicBenchmarkMeasurementCategory ProcessMemory,
    PublicBenchmarkMeasurementCategory GcHeap,
    PublicBenchmarkMeasurementCategory SampledPeaks,
    PublicBenchmarkMeasurementCategory LayoutEstimate,
    PublicBenchmarkMeasurementCategory PersistedBytes,
    PublicBenchmarkMeasurementCategory BuildTime,
    PublicBenchmarkMeasurementCategory OpenTime,
    PublicBenchmarkMeasurementCategory CheckpointTime);

public sealed record PublicBenchmarkMeasurementCategory(
    string Status,
    string Unit,
    string Semantics,
    string[] Values,
    string[] Limitations);

public sealed record PublicBenchmarkPrivacyStatus(
    string RedactionStatus,
    string PublicArtifactKind,
    string RawArtifactDisclosure,
    string[] RedactedFields,
    PublicBenchmarkSourceArtifact[] SourceArtifacts);

public sealed record PublicBenchmarkSourceArtifact(
    string ArtifactKind,
    string ArtifactSchemaName,
    string ArtifactIdDisclosure,
    bool PubliclyIncluded);

public sealed record PublicBenchmarkReviewStatus(
    string Status,
    string Reviewer,
    string ReviewedAtUtc,
    string Notes);

public sealed record PublicBenchmarkSummaryValidationResult(bool IsValid, string[] Errors);

public static class PublicBenchmarkSummaryGate
{
    public const string SchemaName = "VecNet.PublicBenchmarkSummary";
    public const string SchemaVersion = "0.1";
    public const string AcceptedPublicClaimStatus = "accepted-public-claim-candidate";

    private static readonly string[] RequiredMeasurementCategories =
    [
        "latency",
        "throughput",
        "recall",
        "managedAllocation",
        "processMemory",
        "gcHeap",
        "sampledPeaks",
        "layoutEstimate",
        "persistedBytes",
        "buildTime",
        "openTime",
        "checkpointTime"
    ];

    private static readonly string[] PrivateRawReportSchemaNames =
    [
        "VecNet.BenchmarkReport",
        "VecNet.ExactCandidateSetBenchmarkReport",
        "VecNet.ExactCheckpointBenchmarkReport",
        "VecNet.ExactFilteredBenchmarkReport",
        "VecNet.ExactMemorySmokeReport",
        "VecNet.ExactOpenedSearchBenchmarkReport",
        "VecNet.ExactPracticalUpdateBenchmarkReport",
        "VecNet.ExactUpdateBenchmarkReport",
        "VecNet.HnswBenchmarkReport",
        "VecNet.HnswMemorySmokeReport",
        "VecNet.HnswBasePlusExactDeltaBenchmarkReport",
        "VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport",
        "VecNet.HnswAllowlistFilteringBenchmarkReport",
        "VecNet.DurableHnswBenchmarkReport",
        "VecNet.ExternalBenchmarkReport",
        "VecNet.ExternalHnswBenchmarkReport",
        "VecNet.ExternalDurableHnswBenchmarkReport",
        "VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport",
        "VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport",
        "VecNet.ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport",
        "VecNet.ExternalHnswAllowlistFilteringBenchmarkReport"
    ];

    private static readonly string[] PrivateStringFragments =
    [
        "VecNet.BenchmarkRunner.Artifacts",
        "BenchmarkDotNet.Artifacts",
        ".nuget\\packages",
        ".nuget/packages",
        "NUGET_PACKAGES",
        "private package feed",
        "private-feed",
        "local package source",
        "pkgs.dev.azure.com",
        "artifactory",
        "myget",
        "nuget.local",
        "TestResults",
        "file://"
    ];

    public static PublicBenchmarkSummaryValidationResult Validate(PublicBenchmarkSummary? summary)
    {
        if (summary is null)
        {
            return new PublicBenchmarkSummaryValidationResult(false, ["summary must be present"]);
        }

        List<string> errors = [];

        RequireEqual(errors, "schemaName", SchemaName, summary.SchemaName);
        RequireEqual(errors, "schemaVersion", SchemaVersion, summary.SchemaVersion);
        RequireNonWhiteSpace(errors, "summaryId", summary.SummaryId);
        Require(errors, summary.GeneratedAtUtc != default, "generatedAtUtc must be present");
        RequireNonWhiteSpace(errors, "claimClass", summary.ClaimClass);
        RequireNonWhiteSpace(errors, "claimText", summary.ClaimText);
        Require(errors, summary.Scope is not null, "scope must be present");
        Require(errors, summary.PublicClaimEligible, "publicClaimEligible must be true");
        RequireEqual(errors, "publicClaimStatus", AcceptedPublicClaimStatus, summary.PublicClaimStatus);
        Require(errors, summary.Source is not null, "source must be present");
        Require(errors, summary.Commands is { Length: > 0 }, "commands must contain at least one command");
        Require(errors, summary.Environment is not null, "environment must be present");
        Require(errors, summary.Dataset is not null, "dataset must be present");
        Require(errors, summary.Measurements is not null, "measurements must be present");
        Require(errors, summary.Privacy is not null, "privacy must be present");
        Require(errors, summary.Limitations is { Length: > 0 }, "limitations must contain at least one limitation");
        Require(errors, summary.Review is not null, "review must be present");

        ValidateScope(summary.Scope, errors);
        ValidateSource(summary.Source, errors);
        ValidateCommands(summary.Commands, errors);
        ValidateEnvironment(summary.Environment, errors);
        ValidateDataset(summary.Dataset, errors);
        ValidateMeasurements(summary.Measurements, errors);
        ValidatePrivacy(summary.Privacy, errors);
        ValidateReview(summary.Review, errors);
        ValidateEvidenceValidation(summary, errors);
        ValidateRedaction(summary, errors);

        return new PublicBenchmarkSummaryValidationResult(errors.Count == 0, errors.ToArray());
    }

    private static void ValidateScope(PublicBenchmarkClaimScope? scope, List<string> errors)
    {
        if (scope is null)
        {
            return;
        }

        RequireNonWhiteSpace(errors, "scope.productSurface", scope.ProductSurface);
        RequireNonWhiteSpace(errors, "scope.algorithm", scope.Algorithm);
        RequireNonWhiteSpace(errors, "scope.metric", scope.Metric);
        RequireNonWhiteSpace(errors, "scope.scenario", scope.Scenario);
        RequireNonWhiteSpace(errors, "scope.evidenceClass", scope.EvidenceClass);
        RequireNonWhiteSpace(errors, "scope.claimScope", scope.ClaimScope);
        RequireEqual(errors, "scope.evidenceClass", "E4", scope.EvidenceClass);
    }

    private static void ValidateSource(PublicBenchmarkSourceIdentity? source, List<string> errors)
    {
        if (source is null)
        {
            return;
        }

        bool hasPackageIdentity =
            !string.IsNullOrWhiteSpace(source.PackageId) &&
            !string.IsNullOrWhiteSpace(source.PackageVersion);
        bool hasGitIdentity =
            !string.IsNullOrWhiteSpace(source.GitCommit) ||
            !string.IsNullOrWhiteSpace(source.GitTag);
        Require(errors, hasPackageIdentity || hasGitIdentity, "source must include package or git identity");
        RequireNonWhiteSpace(errors, "source.workingTreeStatus", source.WorkingTreeStatus);
        Require(
            errors,
            string.Equals(source.WorkingTreeStatus, "clean", StringComparison.Ordinal) ||
                string.Equals(source.WorkingTreeStatus, "published-package", StringComparison.Ordinal),
            "source.workingTreeStatus must be clean or published-package");
    }

    private static void ValidateCommands(PublicBenchmarkCommand[]? commands, List<string> errors)
    {
        if (commands is null)
        {
            return;
        }

        for (int i = 0; i < commands.Length; i++)
        {
            PublicBenchmarkCommand command = commands[i];
            RequireNonWhiteSpace(errors, $"commands[{i}].name", command.Name);
            Require(errors, command.Arguments is { Length: > 0 }, $"commands[{i}].arguments must contain at least one argument");
            RequireNonWhiteSpace(errors, $"commands[{i}].workingDirectoryDisclosure", command.WorkingDirectoryDisclosure);
        }
    }

    private static void ValidateEnvironment(PublicBenchmarkEnvironment? environment, List<string> errors)
    {
        if (environment is null)
        {
            return;
        }

        RequireNonWhiteSpace(errors, "environment.operatingSystem", environment.OperatingSystem);
        RequireNonWhiteSpace(errors, "environment.architecture", environment.Architecture);
        RequireNonWhiteSpace(errors, "environment.runtime", environment.Runtime);
        RequireNonWhiteSpace(errors, "environment.cpu", environment.Cpu);
        RequireNonWhiteSpace(errors, "environment.memory", environment.Memory);
        RequireNonWhiteSpace(errors, "environment.storage", environment.Storage);
        RequireNonWhiteSpace(errors, "environment.disclosure", environment.Disclosure);
    }

    private static void ValidateDataset(PublicBenchmarkDatasetIdentity? dataset, List<string> errors)
    {
        if (dataset is null)
        {
            return;
        }

        RequireNonWhiteSpace(errors, "dataset.kind", dataset.Kind);
        RequireNonWhiteSpace(errors, "dataset.name", dataset.Name);
        RequireNonWhiteSpace(errors, "dataset.version", dataset.Version);
        RequireNonWhiteSpace(errors, "dataset.source", dataset.Source);
        RequireNonWhiteSpace(errors, "dataset.license", dataset.License);
        RequireNonWhiteSpace(errors, "dataset.checksum", dataset.Checksum);
        RequireNonWhiteSpace(errors, "dataset.seed", dataset.Seed);
        Require(errors, dataset.VectorCount > 0, "dataset.vectorCount must be positive");
        Require(errors, dataset.QueryCount > 0, "dataset.queryCount must be positive");
        Require(errors, dataset.Dimension > 0, "dataset.dimension must be positive");
        RequireNonWhiteSpace(errors, "dataset.metric", dataset.Metric);
        RequireNonWhiteSpace(errors, "dataset.truthMethod", dataset.TruthMethod);
    }

    private static void ValidateMeasurements(PublicBenchmarkMeasurementCategories? measurements, List<string> errors)
    {
        if (measurements is null)
        {
            return;
        }

        ValidateMeasurementCategory("latency", measurements.Latency, errors);
        ValidateMeasurementCategory("throughput", measurements.Throughput, errors);
        ValidateMeasurementCategory("recall", measurements.Recall, errors);
        ValidateMeasurementCategory("managedAllocation", measurements.ManagedAllocation, errors);
        ValidateMeasurementCategory("processMemory", measurements.ProcessMemory, errors);
        ValidateMeasurementCategory("gcHeap", measurements.GcHeap, errors);
        ValidateMeasurementCategory("sampledPeaks", measurements.SampledPeaks, errors);
        ValidateMeasurementCategory("layoutEstimate", measurements.LayoutEstimate, errors);
        ValidateMeasurementCategory("persistedBytes", measurements.PersistedBytes, errors);
        ValidateMeasurementCategory("buildTime", measurements.BuildTime, errors);
        ValidateMeasurementCategory("openTime", measurements.OpenTime, errors);
        ValidateMeasurementCategory("checkpointTime", measurements.CheckpointTime, errors);
    }

    private static void ValidateMeasurementCategory(
        string name,
        PublicBenchmarkMeasurementCategory? category,
        List<string> errors)
    {
        if (category is null)
        {
            errors.Add($"measurements.{name} must be present");
            return;
        }

        RequireNonWhiteSpace(errors, $"measurements.{name}.status", category.Status);
        RequireNonWhiteSpace(errors, $"measurements.{name}.unit", category.Unit);
        RequireNonWhiteSpace(errors, $"measurements.{name}.semantics", category.Semantics);
        Require(errors, category.Values is { Length: > 0 }, $"measurements.{name}.values must contain at least one value or explicit absence");
        Require(errors, category.Limitations is not null, $"measurements.{name}.limitations must be present");
    }

    private static void ValidatePrivacy(PublicBenchmarkPrivacyStatus? privacy, List<string> errors)
    {
        if (privacy is null)
        {
            return;
        }

        RequireNonWhiteSpace(errors, "privacy.redactionStatus", privacy.RedactionStatus);
        RequireEqual(errors, "privacy.redactionStatus", "redacted", privacy.RedactionStatus);
        RequireNonWhiteSpace(errors, "privacy.publicArtifactKind", privacy.PublicArtifactKind);
        RequireEqual(errors, "privacy.publicArtifactKind", "redacted-public-summary", privacy.PublicArtifactKind);
        RequireNonWhiteSpace(errors, "privacy.rawArtifactDisclosure", privacy.RawArtifactDisclosure);
        Require(errors, privacy.RedactedFields is not null, "privacy.redactedFields must be present");
        Require(errors, privacy.SourceArtifacts is not null, "privacy.sourceArtifacts must be present");

        if (privacy.SourceArtifacts is null)
        {
            return;
        }

        for (int i = 0; i < privacy.SourceArtifacts.Length; i++)
        {
            PublicBenchmarkSourceArtifact artifact = privacy.SourceArtifacts[i];
            RequireNonWhiteSpace(errors, $"privacy.sourceArtifacts[{i}].artifactKind", artifact.ArtifactKind);
            RequireNonWhiteSpace(errors, $"privacy.sourceArtifacts[{i}].artifactSchemaName", artifact.ArtifactSchemaName);
            RequireNonWhiteSpace(errors, $"privacy.sourceArtifacts[{i}].artifactIdDisclosure", artifact.ArtifactIdDisclosure);

            if (artifact.PubliclyIncluded && IsPrivateRawArtifact(artifact.ArtifactKind, artifact.ArtifactSchemaName))
            {
                errors.Add($"privacy.sourceArtifacts[{i}] cannot publicly include private raw runner artifacts");
            }
        }
    }

    private static void ValidateReview(PublicBenchmarkReviewStatus? review, List<string> errors)
    {
        if (review is null)
        {
            return;
        }

        RequireNonWhiteSpace(errors, "review.status", review.Status);
        RequireNonWhiteSpace(errors, "review.reviewer", review.Reviewer);
        RequireNonWhiteSpace(errors, "review.reviewedAtUtc", review.ReviewedAtUtc);
        RequireNonWhiteSpace(errors, "review.notes", review.Notes);
    }

    private static void ValidateEvidenceValidation(PublicBenchmarkSummary summary, List<string> errors)
    {
        bool isExactGeneratedSummary =
            ContainsOrdinalIgnoreCase(summary.Scope?.Scenario ?? "", GeneratedExactSearchOptions.ScenarioName) ||
            (summary.Commands?.Any(command => string.Equals(command.Name, GeneratedExactSearchOptions.ScenarioName, StringComparison.Ordinal)) ?? false);

        if (!isExactGeneratedSummary)
        {
            return;
        }

        ExactGeneratedPublicEvidenceValidationInfo? validation = summary.EvidenceValidation;
        if (validation is null)
        {
            errors.Add("evidenceValidation must be present for exact-generated public summaries");
            return;
        }

        RequireEqual(errors, "evidenceValidation.policyName", ExactGeneratedPublicEvidencePolicy.PolicyName, validation.PolicyName);
        RequireEqual(errors, "evidenceValidation.policyVersion", ExactGeneratedPublicEvidencePolicy.PolicyVersion, validation.PolicyVersion);
        Require(errors, validation.Acceptable, "evidenceValidation.acceptable must be true");
        Require(
            errors,
            validation.Status == "passed-strict" ||
                validation.Status == "accepted-near-tie-order-only",
            "evidenceValidation.status must be passed-strict or accepted-near-tie-order-only");
        RequireNonWhiteSpace(errors, "evidenceValidation.classification", validation.Classification);
        Require(
            errors,
            validation.AcceptedRecallFloor == ExactGeneratedPublicEvidencePolicy.AcceptedRecallFloor,
            "evidenceValidation.acceptedRecallFloor must match the exact-generated public evidence policy");
        Require(
            errors,
            validation.RecallAtK >= ExactGeneratedPublicEvidencePolicy.AcceptedRecallFloor,
            "evidenceValidation.recallAtK must satisfy the accepted recall floor");
        RequireEqual(errors, "evidenceValidation.distanceToleranceStatus", "passed", validation.DistanceToleranceStatus);
        Require(errors, validation.DistanceMismatchCount == 0, "evidenceValidation.distanceMismatchCount must be zero");
        Require(errors, validation.MissingResultCount == 0, "evidenceValidation.missingResultCount must be zero");
        Require(errors, validation.DuplicateResultCount == 0, "evidenceValidation.duplicateResultCount must be zero");
        Require(errors, validation.WrongIdAwayFromNearTieCount == 0, "evidenceValidation.wrongIdAwayFromNearTieCount must be zero");
        RequireNonWhiteSpace(errors, "evidenceValidation.nearTieTolerancePolicy", validation.NearTieTolerancePolicy);
        RequireNonWhiteSpace(errors, "evidenceValidation.explanation", validation.Explanation);
        Require(errors, validation.Diagnostics is { Length: > 0 }, "evidenceValidation.diagnostics must be present");

        if (validation.Status == "passed-strict")
        {
            Require(errors, validation.RecallAtK == 1, "evidenceValidation strict status requires recallAtK of 1.0");
            Require(errors, validation.OrderedAgreement == 1, "evidenceValidation strict status requires orderedAgreement of 1.0");
        }
        else
        {
            Require(
                errors,
                validation.OrderMismatchCount > 0 || validation.BoundaryNearTieMismatchCount > 0,
                "evidenceValidation near-tie status requires order or boundary near-tie mismatches");
        }
    }

    private static void ValidateRedaction(PublicBenchmarkSummary summary, List<string> errors)
    {
        if (LooksLikePrivateRawReportId(summary.SummaryId))
        {
            errors.Add("summaryId must be a public summary identifier, not a private raw runner report identifier");
        }

        HashSet<string> seenMeasurementNames = [];
        foreach ((string path, string value) in EnumerateStringValues(summary))
        {
            if (ContainsLocalPathOrPrivateRoot(value))
            {
                errors.Add($"{path} contains a local path, private package feed or private artifact/cache root");
            }

            if (ContainsAmbiguousMemoryWording(value))
            {
                errors.Add($"{path} appears to collapse distinct memory measurement categories into one ambiguous memory value");
            }

            if (PathMatchesMeasurementCategory(path))
            {
                seenMeasurementNames.Add(path.Split('.')[1]);
            }
        }

        foreach (string requiredMeasurementCategory in RequiredMeasurementCategories)
        {
            if (!seenMeasurementNames.Contains(requiredMeasurementCategory))
            {
                errors.Add($"measurements.{requiredMeasurementCategory} must be serialized as a separate category");
            }
        }
    }

    private static bool IsPrivateRawArtifact(string artifactKind, string artifactSchemaName) =>
        ContainsOrdinalIgnoreCase(artifactKind, "private-raw") ||
        ContainsOrdinalIgnoreCase(artifactKind, "private raw") ||
        PrivateRawReportSchemaNames.Any(
            schemaName => string.Equals(schemaName, artifactSchemaName, StringComparison.Ordinal));

    private static bool LooksLikePrivateRawReportId(string value)
    {
        string[] privateRawPrefixes =
        [
            "exact-generated-",
            "exact-generated-filtered-",
            "generated-exact-",
            "hnsw-generated-",
            "hnsw-base-plus-exact-delta-",
            "hnsw-allowlist-filtering-",
            "fashion-mnist-",
            "durable-hnsw-"
        ];

        return privateRawPrefixes.Any(prefix => value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
    }

    private static bool ContainsLocalPathOrPrivateRoot(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        if (PrivateStringFragments.Any(fragment => ContainsOrdinalIgnoreCase(value, fragment)))
        {
            return true;
        }

        if (ContainsWindowsDriveAbsolutePath(value))
        {
            return true;
        }

        if (value.Contains(@"\\", StringComparison.Ordinal))
        {
            return true;
        }

        string trimmed = value.TrimStart();
        return ContainsUnixPrivateRootPath(trimmed);
    }

    private static bool ContainsWindowsDriveAbsolutePath(string value)
    {
        for (int i = 0; i <= value.Length - 3; i++)
        {
            if (char.IsLetter(value[i]) &&
                value[i + 1] == ':' &&
                (value[i + 2] == '\\' || value[i + 2] == '/'))
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsUnixPrivateRootPath(string value)
    {
        string[] privateUnixRoots =
        [
            "/home/",
            "/Users/",
            "/root/",
            "/tmp/",
            "/var/",
            "/mnt/",
            "/workspace/"
        ];

        return privateUnixRoots.Any(root => value.Contains(root, StringComparison.Ordinal));
    }

    private static bool ContainsAmbiguousMemoryWording(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        string lower = value.ToLowerInvariant();
        string[] directAmbiguousPhrases =
        [
            "single memory",
            "one memory",
            "overall memory",
            "total memory",
            "combined memory",
            "memory value",
            "memory number",
            "memory total",
            "memory covers"
        ];

        if (directAmbiguousPhrases.Any(phrase => lower.Contains(phrase, StringComparison.Ordinal)))
        {
            return true;
        }

        if (!lower.Contains("memory", StringComparison.Ordinal))
        {
            return false;
        }

        int distinctCategoryMentions = 0;
        string[] collapsedCategoryPhrases =
        [
            "layout",
            "process",
            "private bytes",
            "managed allocation",
            "allocation",
            "gc heap",
            "sampled peak",
            "persisted bytes",
            "persisted"
        ];

        foreach (string phrase in collapsedCategoryPhrases)
        {
            if (lower.Contains(phrase, StringComparison.Ordinal))
            {
                distinctCategoryMentions++;
            }
        }

        return distinctCategoryMentions >= 2;
    }

    private static bool PathMatchesMeasurementCategory(string path)
    {
        if (!path.StartsWith("measurements.", StringComparison.Ordinal))
        {
            return false;
        }

        string[] parts = path.Split('.');
        return parts.Length >= 3 && RequiredMeasurementCategories.Contains(parts[1], StringComparer.Ordinal);
    }

    private static IReadOnlyList<(string Path, string Value)> EnumerateStringValues(PublicBenchmarkSummary summary)
    {
        using JsonDocument document = JsonDocument.Parse(ReportWriter.Serialize(summary));
        List<(string Path, string Value)> values = [];
        CollectStringValues(document.RootElement, "", values);
        return values;
    }

    private static void CollectStringValues(
        JsonElement element,
        string path,
        List<(string Path, string Value)> values)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (JsonProperty property in element.EnumerateObject())
                {
                    string childPath = string.IsNullOrEmpty(path)
                        ? property.Name
                        : $"{path}.{property.Name}";
                    CollectStringValues(property.Value, childPath, values);
                }

                break;
            case JsonValueKind.Array:
                int index = 0;
                foreach (JsonElement child in element.EnumerateArray())
                {
                    CollectStringValues(child, $"{path}[{index}]", values);
                    index++;
                }

                break;
            case JsonValueKind.String:
                values.Add((path, element.GetString() ?? string.Empty));
                break;
        }
    }

    private static void RequireEqual(List<string> errors, string field, string expected, string? actual)
    {
        if (!string.Equals(expected, actual, StringComparison.Ordinal))
        {
            errors.Add($"{field} must be {expected}");
        }
    }

    private static void RequireNonWhiteSpace(List<string> errors, string field, string? value)
    {
        Require(errors, !string.IsNullOrWhiteSpace(value), $"{field} must be present");
    }

    private static void Require(List<string> errors, bool condition, string message)
    {
        if (!condition)
        {
            errors.Add(message);
        }
    }

    private static bool ContainsOrdinalIgnoreCase(string value, string fragment) =>
        value.Contains(fragment, StringComparison.OrdinalIgnoreCase);
}
