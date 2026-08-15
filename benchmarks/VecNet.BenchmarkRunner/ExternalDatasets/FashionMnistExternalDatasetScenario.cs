using System.Diagnostics;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalDatasetScenario
{
    private const string TaskId = "VEC-023";
    private const string ConverterIdentity = "VecNet.BenchmarkRunner.ExternalDatasets.FashionMnistExternalDatasetScenario/0.1";

    public static FashionMnistAdmissionResult Run(FashionMnistExternalDatasetOptions options, IReadOnlyList<string> commandArguments)
    {
        return Run(options, commandArguments, FashionMnistDatasetSpecification.Official);
    }

    internal static FashionMnistAdmissionResult Run(
        FashionMnistExternalDatasetOptions options,
        IReadOnlyList<string> commandArguments,
        FashionMnistDatasetSpecification spec)
    {
        spec = spec.WithMetricIdentity(options.Metric);
        ValidateOptions(options, spec);

        DatasetPaths paths = DatasetPaths.Create(options.CacheRoot, spec.DatasetId);
        Directory.CreateDirectory(paths.RawDirectory);
        Directory.CreateDirectory(paths.ConvertedDirectory);
        Directory.CreateDirectory(paths.TruthDirectory);
        Directory.CreateDirectory(paths.ManifestDirectory);
        Directory.CreateDirectory(paths.EvidenceDirectory);

        EnsureRawFiles(paths, spec, options.DownloadRawFiles);
        RawFileVerification[] rawFiles = VerifyRawFiles(paths, spec);

        IdxImageSet baseImages = IdxFileReader.ReadImages(Path.Combine(paths.RawDirectory, spec.TrainImages.FileName), spec.BaseCount, spec.ImageRows, spec.ImageColumns);
        IdxLabelSet baseLabels = IdxFileReader.ReadLabels(Path.Combine(paths.RawDirectory, spec.TrainLabels.FileName), spec.BaseCount);
        IdxImageSet queryImages = IdxFileReader.ReadImages(Path.Combine(paths.RawDirectory, spec.QueryImages.FileName), spec.QueryCount, spec.ImageRows, spec.ImageColumns);
        IdxLabelSet queryLabels = IdxFileReader.ReadLabels(Path.Combine(paths.RawDirectory, spec.QueryLabels.FileName), spec.QueryCount);

        float[] baseVectors = DenseFloat32Matrix.ConvertImages(baseImages);
        float[] queryVectors = DenseFloat32Matrix.ConvertImages(queryImages);
        if (options.Metric == VectorMetric.Cosine)
        {
            FashionMnistExactTruth.ValidateNonZeroRows(baseVectors, spec.BaseCount, spec.Dimension, "base");
            FashionMnistExactTruth.ValidateNonZeroRows(queryVectors, options.QueryCount, spec.Dimension, "query");
        }

        DenseFloat32Matrix.Write(paths.BaseMatrixPath, spec.BaseCount, spec.Dimension, baseVectors);
        DenseFloat32Matrix.Write(paths.QueryMatrixPath, spec.QueryCount, spec.Dimension, queryVectors);
        string baseMatrixSha256 = FileChecksum.ComputeSha256(paths.BaseMatrixPath);
        string queryMatrixSha256 = FileChecksum.ComputeSha256(paths.QueryMatrixPath);

        FashionMnistLabelMetadata labels = new(ToLabelMetadata(baseLabels), ToLabelMetadata(queryLabels));
        ExternalConvertedMatrixEntry[] convertedMatrices =
        [
            new("base", paths.RelativeBaseMatrixPath, spec.BaseCount, spec.Dimension, DenseFloat32Matrix.SchemaName, DenseFloat32Matrix.SchemaVersion, baseMatrixSha256),
            new("query", paths.RelativeQueryMatrixPath, spec.QueryCount, spec.Dimension, DenseFloat32Matrix.SchemaName, DenseFloat32Matrix.SchemaVersion, queryMatrixSha256)
        ];

        var conversionManifest = new ConversionManifestArtifact(
            "VecNetDenseFloat32MatrixConversion",
            "0.1",
            spec.DatasetId,
            GetTaskId(options.Metric),
            DenseFloat32Matrix.SchemaName,
            DenseFloat32Matrix.SchemaVersion,
            "little-endian",
            "none; uint8 pixels are converted to float32 values 0..255 without scaling",
            "same verified raw SHA-256 values, converter identity and conversion parameters produce byte-identical matrix files",
            "converted artifacts are invalid if any input raw SHA-256 or converter identity changes",
            rawFiles,
            convertedMatrices,
            labels);
        ReportWriter.WriteJson(conversionManifest, paths.ConversionManifestPath);
        string conversionManifestSha256 = FileChecksum.ComputeSha256(paths.ConversionManifestPath);

        Stopwatch validationStopwatch = Stopwatch.StartNew();
        TruthSet truth = FashionMnistExactTruth.Generate(
            baseVectors,
            spec.BaseCount,
            queryVectors,
            spec.QueryCount,
            spec.Dimension,
            options.QueryCount,
            options.TruthDepth,
            options.Metric);
        var truthArtifact = FashionMnistExactTruth.CreateArtifact(
            spec.DatasetId,
            truth,
            spec.BaseCount,
            options.QueryCount,
            spec.Dimension,
            rawFiles.Select(file => file.ComputedSha256).ToArray(),
            ConverterIdentity,
            options.Metric,
            GetTaskId(options.Metric));
        ReportWriter.WriteJson(truthArtifact, paths.TruthPath);
        string truthSha256 = FileChecksum.ComputeSha256(paths.TruthPath);

        SearchResult[][] actual = ValidateExactIndex(baseVectors, queryVectors, spec.BaseCount, options.QueryCount, spec.Dimension, options.TruthDepth, options.Metric);
        validationStopwatch.Stop();

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            actual,
            options.TruthDepth,
            spec.Dimension,
            options.Metric);
        int extraResultCount = CountExtraResults(truth, actual, options.TruthDepth);

        ExternalExactValidationEvidence evidence = CreateEvidence(
            spec,
            rawFiles,
            [baseMatrixSha256, queryMatrixSha256],
            truthSha256,
            options,
            comparison,
            extraResultCount,
            validationStopwatch.Elapsed.TotalMilliseconds);
        ReportWriter.WriteJson(evidence, paths.EvidencePath);
        string evidenceSha256 = FileChecksum.ComputeSha256(paths.EvidencePath);

        ExternalDatasetManifest manifest = CreateManifest(
            spec,
            rawFiles,
            labels,
            convertedMatrices,
            conversionManifestSha256,
            truthSha256,
            evidenceSha256,
            options,
            paths,
            RepositoryInfo.Create());
        ReportWriter.WriteJson(manifest, paths.ManifestPath);

        return new FashionMnistAdmissionResult(
            manifest,
            evidence,
            paths.ManifestPath,
            paths.EvidencePath,
            paths.TruthPath,
            paths.ConversionManifestPath);
    }

    private static void ValidateOptions(FashionMnistExternalDatasetOptions options, FashionMnistDatasetSpecification spec)
    {
        if (string.IsNullOrWhiteSpace(options.CacheRoot))
        {
            throw new ArgumentException("Cache root must not be empty.", nameof(options));
        }

        if (options.QueryCount <= 0 || options.QueryCount > spec.QueryCount)
        {
            throw new ArgumentException($"Query count must be in the range 1..{spec.QueryCount}.", nameof(options));
        }

        if (options.TruthDepth <= 0 || options.TruthDepth > spec.BaseCount)
        {
            throw new ArgumentException($"Truth depth must be in the range 1..{spec.BaseCount}.", nameof(options));
        }

        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine))
        {
            throw new ArgumentException("Fashion-MNIST external admission supports only SquaredEuclidean, InnerProduct and Cosine.", nameof(options));
        }
    }

    private static void EnsureRawFiles(DatasetPaths paths, FashionMnistDatasetSpecification spec, bool downloadRawFiles)
    {
        using var httpClient = downloadRawFiles ? new HttpClient() : null;
        foreach (FashionMnistRawFileSpec rawFile in spec.RawFiles)
        {
            string path = Path.Combine(paths.RawDirectory, rawFile.FileName);
            if (File.Exists(path))
            {
                continue;
            }

            if (!downloadRawFiles)
            {
                throw new FileNotFoundException(
                    $"Raw file '{rawFile.FileName}' is missing under '{paths.RawDirectory}'. Re-run with --download true to fetch only the official Fashion-MNIST raw files.",
                    path);
            }

            byte[] bytes = httpClient!.GetByteArrayAsync(rawFile.SourceUrl).GetAwaiter().GetResult();
            File.WriteAllBytes(path, bytes);
        }
    }

    private static RawFileVerification[] VerifyRawFiles(DatasetPaths paths, FashionMnistDatasetSpecification spec) =>
        spec.RawFiles
            .Select(rawFile => FileChecksum.VerifyRawFile(Path.Combine(paths.RawDirectory, rawFile.FileName), rawFile))
            .ToArray();

    private static LabelMetadata ToLabelMetadata(IdxLabelSet labels) =>
        new(labels.Count, labels.MinValue, labels.MaxValue, labels.Histogram, StoredInConvertedVectors: false, StoredInTruthArtifact: false);

    private static SearchResult[][] ValidateExactIndex(
        ReadOnlySpan<float> baseVectors,
        ReadOnlySpan<float> queryVectors,
        int baseCount,
        int queryCount,
        int dimension,
        int truthDepth,
        VectorMetric metric)
    {
        var index = new ExactFlatIndex(dimension, metric);
        for (int baseRow = 0; baseRow < baseCount; baseRow++)
        {
            index.Add((ulong)baseRow, baseVectors.Slice(baseRow * dimension, dimension));
        }

        var actual = new SearchResult[queryCount][];
        var buffer = new SearchResult[truthDepth];
        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            int written = index.Search(queryVectors.Slice(queryRow * dimension, dimension), buffer);
            var queryResults = new SearchResult[written];
            buffer.AsSpan(0, written).CopyTo(queryResults);
            actual[queryRow] = queryResults;
        }

        return actual;
    }

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static ExternalExactValidationEvidence CreateEvidence(
        FashionMnistDatasetSpecification spec,
        RawFileVerification[] rawFiles,
        string[] convertedSha256,
        string truthSha256,
        FashionMnistExternalDatasetOptions options,
        ResultComparison comparison,
        int extraResultCount,
        double validationElapsedMilliseconds)
    {
        string status = comparison.RecallAtK == 1 &&
            comparison.OrderedAgreement == 1 &&
            comparison.DistanceToleranceStatus == "passed" &&
            comparison.MissingResultCount == 0 &&
            extraResultCount == 0
                ? "passed"
                : "failed";

        return new ExternalExactValidationEvidence(
            "VecNet.ExternalExactValidation",
            "0.1",
            GetTaskId(options.Metric),
            spec.DatasetId,
            rawFiles.Select(file => file.SourceUrl).ToArray(),
            rawFiles.Select(file => file.ComputedSha256).ToArray(),
            convertedSha256,
            truthSha256,
            options.QueryCount,
            options.TruthDepth,
            "public ExactFlatIndex",
            options.Metric.ToString(),
            GetUpstreamMetricName(options.Metric),
            new ExternalValidationOutcome(
                status,
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                comparison.MissingResultCount,
                extraResultCount,
                comparison.DistanceMismatchCount),
            GetDistanceTolerancePolicy(options.Metric),
            new ExternalWorkflowTiming(
                "privateWorkflowDiagnostic",
                validationElapsedMilliseconds,
                "Validation elapsed time is private workflow diagnostics only, not benchmark latency or QPS evidence."),
            new MeasurementStatusInfo(
                "notMeasured",
                "absent",
                "bytes",
                "Managed allocation measurement is not part of VEC-023 private exact validation evidence."),
            new MeasurementStatusInfo(
                "notMeasured",
                "absent",
                "bytes",
                "Resident/process memory is explicitly not measured."),
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "local-evidence",
            "private-raw");
    }

    private static ExternalDatasetManifest CreateManifest(
        FashionMnistDatasetSpecification spec,
        RawFileVerification[] rawFiles,
        FashionMnistLabelMetadata labels,
        ExternalConvertedMatrixEntry[] convertedMatrices,
        string conversionManifestSha256,
        string truthSha256,
        string evidenceSha256,
        FashionMnistExternalDatasetOptions options,
        DatasetPaths paths,
        RepositoryInfo repository) =>
        new(
            "VecNet.ExternalDatasetManifest",
            "0.1",
            spec.DatasetId,
            GetTaskId(options.Metric),
            new ExternalDatasetSource(
                spec.MaintainerUrl,
                spec.DownloadRoot,
                spec.OfficialReadmeUrl,
                spec.LicenseUrl,
                spec.AccessDate,
                spec.CitationDate,
                "no release tag available; official raw file URLs plus source MD5 checksums are pinned"),
            new ExternalDatasetLicense(
                spec.LicenseName,
                spec.Copyright,
                "preserve MIT copyright and license notice in any later approved public summary or redistribution",
                "raw, converted, truth and evidence artifacts remain local ignored cache files in VEC-023"),
            new ExternalDatasetPrivacy(
                "private-raw",
                "local-evidence",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false),
            new ExternalDatasetShape(
                spec.BaseCount,
                spec.QueryCount,
                spec.Dimension,
                spec.ImageRows,
                spec.ImageColumns,
                "uint8-source",
                "float32"),
            new ExternalDatasetMetric(
                GetUpstreamMetricName(options.Metric),
                options.Metric.ToString(),
                GetMetricRankingNote(options.Metric),
                GetMetricDistanceNote(options.Metric)),
            rawFiles.Select(file => new ExternalRawFileManifestEntry(
                file.FileName,
                file.SourceUrl,
                file.Role,
                file.ExpectedCount,
                file.OfficialMd5,
                file.ComputedSha256,
                file.ByteSize,
                file.VerificationStatus,
                paths.RelativeRawPath(file.FileName))).ToArray(),
            labels,
            new ConversionManifestSummary(
                ConverterIdentity,
                paths.RelativeConversionManifestPath,
                conversionManifestSha256,
                DenseFloat32Matrix.SchemaName,
                "uint8 pixels converted to float32 values 0..255 without normalization",
                "deterministic row-major little-endian matrix bytes from verified raw SHA-256 inputs",
                convertedMatrices),
            new TruthManifestSummary(
                FashionMnistExactTruth.Kind(options.Metric),
                paths.RelativeTruthPath,
                truthSha256,
                options.QueryCount,
                options.TruthDepth,
                FashionMnistExactTruth.TiePolicy(options.Metric)),
            new EvidenceManifestSummary(
                paths.RelativeEvidencePath,
                evidenceSha256,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false),
            repository,
            [
                "Private Fashion-MNIST admission evidence only; not a public benchmark claim.",
                "Only the four official Fashion-MNIST raw IDX gzip files are supported by VEC-023.",
                "Labels are recorded as private metadata only and are absent from converted vector matrices and truth artifacts.",
                GetMetricCacheNote(options.Metric),
                "ANN-Benchmarks HDF5 import, ANN algorithms, public summaries, hard regression gates and resident/process memory comparison are out of scope."
            ]);

    private static string GetTaskId(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "VEC-239",
            VectorMetric.InnerProduct => "VEC-350",
            _ => TaskId
        };

    private static string GetUpstreamMetricName(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "cosine",
            VectorMetric.InnerProduct => "raw-inner-product",
            _ => "euclidean"
        };

    private static string GetMetricRankingNote(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "Cosine ranks by ascending canonical distance over VecNet-normalized vectors.",
            VectorMetric.InnerProduct => "Raw inner product ranks by ascending VecNet canonical negative-dot distance over unnormalized vectors.",
            _ => "Euclidean and squared Euclidean preserve nearest-neighbor order for non-negative distances."
        };

    private static string GetMetricDistanceNote(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "VecNet private evidence records canonical cosine distances: 1 - dot(normalizedQuery, normalizedBase).",
            VectorMetric.InnerProduct => "VecNet private evidence records canonical inner-product distances: -dot(rawQuery, rawBase).",
            _ => "VecNet private evidence records canonical squared distances."
        };

    private static string GetDistanceTolerancePolicy(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "ResultComparer non-squared-L2 tolerance is used for canonical cosine distance; ordering requires exact ID order agreement for the selected truth depth.",
            VectorMetric.InnerProduct => "ResultComparer non-squared-L2 tolerance is used for canonical negative-dot distance; ordering requires exact ID order agreement for the selected truth depth.",
            _ => "D-026 squared-L2 tolerance used by ResultComparer; ordering requires exact ID order agreement for the selected truth depth."
        };

    private static string GetMetricCacheNote(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.Cosine => "Cosine evidence validates all selected base/query rows are nonzero and stores unnormalized float32 pixel vectors.",
            VectorMetric.InnerProduct => "Raw inner-product evidence stores unnormalized float32 pixel vectors under a distinct Fashion-MNIST identity and computes canonical negative-dot truth.",
            _ => "Squared-L2 behavior and existing Fashion-MNIST euclidean identity are preserved."
        };

    private sealed record DatasetPaths(
        string CacheRoot,
        string DatasetId,
        string RawDirectory,
        string ConvertedDirectory,
        string TruthDirectory,
        string ManifestDirectory,
        string EvidenceDirectory,
        string BaseMatrixPath,
        string QueryMatrixPath,
        string ConversionManifestPath,
        string TruthPath,
        string ManifestPath,
        string EvidencePath)
    {
        public string RelativeBaseMatrixPath => Relative("converted", DatasetId, "base.f32le");
        public string RelativeQueryMatrixPath => Relative("converted", DatasetId, "query.f32le");
        public string RelativeConversionManifestPath => Relative("converted", DatasetId, "conversion-manifest.json");
        public string RelativeTruthPath => Relative("truth", DatasetId, "exact-truth.json");
        public string RelativeEvidencePath => Relative("evidence", DatasetId, "exact-validation.json");

        public static DatasetPaths Create(string cacheRoot, string datasetId)
        {
            string raw = Path.Combine(cacheRoot, "raw", FashionMnistDatasetSpecification.RawDatasetId);
            string converted = Path.Combine(cacheRoot, "converted", datasetId);
            string truth = Path.Combine(cacheRoot, "truth", datasetId);
            string manifests = Path.Combine(cacheRoot, "manifests", datasetId);
            string evidence = Path.Combine(cacheRoot, "evidence", datasetId);
            return new DatasetPaths(
                cacheRoot,
                datasetId,
                raw,
                converted,
                truth,
                manifests,
                evidence,
                Path.Combine(converted, "base.f32le"),
                Path.Combine(converted, "query.f32le"),
                Path.Combine(converted, "conversion-manifest.json"),
                Path.Combine(truth, "exact-truth.json"),
                Path.Combine(manifests, "dataset-manifest.json"),
                Path.Combine(evidence, "exact-validation.json"));
        }

        public string RelativeRawPath(string fileName) => Relative("raw", FashionMnistDatasetSpecification.RawDatasetId, fileName);

        private static string Relative(params string[] parts) => string.Join('/', parts);
    }
}
