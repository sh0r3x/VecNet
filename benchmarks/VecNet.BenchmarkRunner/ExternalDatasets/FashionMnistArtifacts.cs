using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record LabelMetadata(
    int Count,
    byte MinValue,
    byte MaxValue,
    int[] Histogram,
    bool StoredInConvertedVectors,
    bool StoredInTruthArtifact);

public sealed record FashionMnistLabelMetadata(LabelMetadata Base, LabelMetadata Query);

public sealed record ExternalDatasetSource(
    string MaintainerUrl,
    string DownloadRoot,
    string OfficialReadmeUrl,
    string LicenseUrl,
    string AccessDate,
    string CitationDate,
    string VersionStatus);

public sealed record ExternalDatasetLicense(
    string Name,
    string Copyright,
    string AttributionRequirement,
    string RedistributionPosture);

public sealed record ExternalDatasetPrivacy(
    string PrivacyClass,
    string EvidenceClass,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible);

public sealed record ExternalDatasetShape(
    int BaseCount,
    int QueryCount,
    int Dimension,
    int ImageRows,
    int ImageColumns,
    string SourceDataType,
    string ConvertedDataType);

public sealed record ExternalDatasetMetric(
    string UpstreamName,
    string VecNetMetric,
    string RankingNote,
    string DistanceNote);

public sealed record ExternalRawFileManifestEntry(
    string FileName,
    string SourceUrl,
    string Role,
    int ExpectedCount,
    string OfficialMd5,
    string ComputedSha256,
    long ByteSize,
    string VerificationStatus,
    string RelativePath);

public sealed record ExternalConvertedMatrixEntry(
    string Role,
    string RelativePath,
    int RowCount,
    int Dimension,
    string Format,
    string SchemaVersion,
    string Sha256);

public sealed record ConversionManifestArtifact(
    string SchemaName,
    string SchemaVersion,
    string DatasetId,
    string AdmittingTaskId,
    string MatrixSchemaName,
    string MatrixSchemaVersion,
    string ByteOrder,
    string Normalization,
    string Determinism,
    string InvalidationRule,
    RawFileVerification[] InputRawFiles,
    ExternalConvertedMatrixEntry[] OutputMatrices,
    FashionMnistLabelMetadata Labels);

public sealed record ConversionManifestSummary(
    string ConverterIdentity,
    string ManifestRelativePath,
    string ManifestSha256,
    string OutputFormat,
    string NormalizationRule,
    string DeterminismRule,
    ExternalConvertedMatrixEntry[] OutputFiles);

public sealed record ExternalTruthNeighbor(ulong Id, float SquaredDistance);

public sealed record ExternalTruthQuery(int QueryOrdinal, ExternalTruthNeighbor[] Neighbors);

public sealed record ExternalExactTruthArtifact(
    string SchemaName,
    string SchemaVersion,
    string DatasetId,
    string TaskId,
    int BaseCount,
    int QuerySubsetCount,
    int Dimension,
    string Metric,
    int TruthDepth,
    string TiePolicy,
    string[] SourceRawSha256,
    string ConverterIdentity,
    ExternalTruthQuery[] Queries);

public sealed record TruthManifestSummary(
    string Kind,
    string RelativePath,
    string Sha256,
    int QuerySubsetCount,
    int TruthDepth,
    string TiePolicy);

public sealed record EvidenceManifestSummary(
    string RelativePath,
    string Sha256,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible);

public sealed record ExternalDatasetManifest(
    string SchemaName,
    string SchemaVersion,
    string DatasetId,
    string AdmittingTaskId,
    ExternalDatasetSource Source,
    ExternalDatasetLicense License,
    ExternalDatasetPrivacy Privacy,
    ExternalDatasetShape Shape,
    ExternalDatasetMetric Metric,
    ExternalRawFileManifestEntry[] RawFiles,
    FashionMnistLabelMetadata Labels,
    ConversionManifestSummary Conversion,
    TruthManifestSummary Truth,
    EvidenceManifestSummary Evidence,
    RepositoryInfo Repository,
    string[] Notes);

public sealed record ExternalExactValidationEvidence(
    string SchemaName,
    string SchemaVersion,
    string TaskId,
    string DatasetId,
    string[] SourceUrls,
    string[] RawSha256,
    string[] ConvertedVectorSha256,
    string TruthArtifactSha256,
    int QuerySubsetCount,
    int TruthDepth,
    string IndexUnderValidation,
    string Metric,
    string UpstreamMetric,
    ExternalValidationOutcome Validation,
    string DistanceTolerancePolicy,
    ExternalWorkflowTiming Timing,
    MeasurementStatusInfo ManagedAllocations,
    MeasurementStatusInfo Memory,
    bool PublicClaimEligible,
    bool BaselineCandidateEligible,
    bool RegressionGateEligible,
    string EvidenceClass,
    string PrivacyClass);

public sealed record ExternalValidationOutcome(
    string Status,
    double RecallAtK,
    double OrderedAgreement,
    int MissingResultCount,
    int ExtraResultCount,
    int DistanceMismatchCount);

public sealed record ExternalWorkflowTiming(
    string Status,
    double ValidationElapsedMilliseconds,
    string Reason);

public sealed record FashionMnistAdmissionResult(
    ExternalDatasetManifest Manifest,
    ExternalExactValidationEvidence Evidence,
    string ManifestPath,
    string EvidencePath,
    string TruthPath,
    string ConversionManifestPath);
