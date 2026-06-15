using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec039FashionMnistExternalHnswIndependentTests
{
    [Theory]
    [InlineData("EXTERNAL-FASHION-MNIST-HNSW", "--CACHE-ROOT", "cache", "--OUTPUT", "out.json", "--METRIC", "SQUARED-EUCLIDEAN", "--M", "2", "--EF-CONSTRUCTION", "2", "--EF-SEARCH", "1", "--TOP-K", "1", "--RUNS", "5", "--WARMUP-QUERIES", "0", "--HNSW-SEED", "42")]
    [InlineData("external-fashion-mnist-hnsw", "--metric", "SquaredEuclidean", "--m", "64", "--ef-construction", "4096", "--ef-search", "4096", "--top-k", "4096")]
    public void ParseExternalFashionMnistHnsw_AcceptsCaseInsensitiveNamesAndBounds(params string[] args)
    {
        FashionMnistExternalHnswBenchmarkOptions options = CommandLine.ParseExternalFashionMnistHnsw(args);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.InRange(options.M, 2, 64);
        Assert.InRange(options.Runs, 1, 5);
        Assert.True(options.EfConstruction >= options.M);
        Assert.True(options.EfSearch >= options.TopK);
    }

    [Theory]
    [InlineData("external-fashion-mnist-hnsw", "--query-count")]
    [InlineData("external-fashion-mnist-hnsw", "bare-token")]
    [InlineData("external-fashion-mnist-hnsw", "--output", "report.json", "--cache-root")]
    [InlineData("external-fashion-mnist-hnsw", "--download-raw-files", "false")]
    [InlineData("external-fashion-mnist-hnsw", "--truth", "truth.json")]
    [InlineData("external-fashion-mnist-hnsw", "--manifest", "dataset-manifest.json")]
    [InlineData("external-fashion-mnist-hnsw", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw", "--baseline-report-id", "baseline")]
    public void ParseExternalFashionMnistHnsw_RejectsUnsupportedAndMalformedOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_IndependentSyntheticCache_PreservesCacheAndEmitsPrivateRepeatedRunJson()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "json-shape",
            baseCount: 40,
            queryCount: 5,
            truthDepth: 6,
            rows: 3,
            columns: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "external-hnsw-independent-report.json");
        Dictionary<string, string> before = SnapshotCacheFiles(cacheRoot);

        var options = new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            QueryCount: 3,
            TopK: 4,
            Runs: 2,
            WarmupQueries: 5,
            VectorMetric.SquaredEuclidean,
            M: 3,
            EfConstruction: 9,
            EfSearch: 4,
            HnswSeed: 0x3900UL);

        ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-hnsw", "--query-count", "3", "--top-k", "4", "--runs", "2", "--warmup-queries", "5"]);
        FashionMnistExternalHnswBenchmarkScenario.Write(report, outputPath);

        Dictionary<string, string> after = SnapshotCacheFiles(cacheRoot);
        Assert.Equal(before, after);
        Assert.True(File.Exists(outputPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-039", root.GetProperty("taskId").GetString());
        Assert.Equal("external-fashion-mnist-hnsw", root.GetProperty("scenarioName").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("external-hnsw-smoke", root.GetProperty("evidence").GetProperty("scope").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("dataset").GetProperty("datasetId").GetString());
        Assert.Equal("manifests/fashion-mnist-784-euclidean/dataset-manifest.json", root.GetProperty("dataset").GetProperty("admissionManifest").GetProperty("relativePath").GetString());
        Assert.Equal(admission.Manifest.Truth.Sha256, root.GetProperty("truth").GetProperty("sha256").GetString());
        Assert.Equal(3, root.GetProperty("workload").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(4, root.GetProperty("workload").GetProperty("topK").GetInt32());
        Assert.Equal("HnswIndex", root.GetProperty("index").GetProperty("type").GetString());
        Assert.Equal("0x0000000000003900", root.GetProperty("hnsw").GetProperty("randomSeed").GetString());
        Assert.Equal("measured", root.GetProperty("build").GetProperty("status").GetString());
        Assert.Equal("internal HnswIndex.Search(query, results, workspace)", root.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Contains("download", root.GetProperty("measurement").GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", root.GetProperty("measurement").GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.Equal(2, root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(5, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal(2, root.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal(3, root.GetProperty("search").GetProperty("aggregate").GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("allowsApproximateRecallBelowOne").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("estimated", root.GetProperty("memoryEstimate").GetProperty("status").GetString());

        string json = File.ReadAllText(outputPath);
        Assert.DoesNotContain("latencyTicks", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("comparisonArtifact", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("baselineCandidateEligibility", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("regressionThreshold", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_LowRecallExternalHnswStillPassesWhenReturnedResultsAreWellFormed()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "low-recall",
            baseCount: 96,
            queryCount: 8,
            truthDepth: 10,
            rows: 4,
            columns: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        ExternalHnswBenchmarkReport? belowPerfect = null;

        for (ulong seed = 0x3900; seed < 0x3940 && belowPerfect is null; seed++)
        {
            ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(
                new FashionMnistExternalHnswBenchmarkOptions(
                    cacheRoot,
                    Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "low-recall-" + seed.ToString("X", CultureInfo.InvariantCulture) + ".json"),
                    QueryCount: 8,
                    TopK: 10,
                    Runs: 1,
                    WarmupQueries: 0,
                    VectorMetric.SquaredEuclidean,
                    M: 2,
                    EfConstruction: 2,
                    EfSearch: 10,
                    HnswSeed: seed),
                ["external-fashion-mnist-hnsw"]);

            if (report.Validation.Status == "passed" &&
                report.Metrics.ReturnedResultIntegrity.Status == "passed" &&
                report.Metrics.RecallAtK < 1)
            {
                belowPerfect = report;
            }
        }

        Assert.NotNull(belowPerfect);
        Assert.Equal("passed", belowPerfect.Validation.Status);
        Assert.True(belowPerfect.Validation.AllowsApproximateRecallBelowOne);
        Assert.InRange(belowPerfect.Metrics.RecallAtK, 0, 0.999999);
        Assert.Equal("passed", belowPerfect.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.False(belowPerfect.Eligibility.PublicClaimEligible);
        Assert.False(belowPerfect.Eligibility.BaselineCandidateEligible);
        Assert.False(belowPerfect.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ValidateReturnedResults_QueryCountMismatchAndDistanceToleranceAreReportedIndependently()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "integrity-independent",
            baseCount: 7,
            queryCount: 3,
            truthDepth: 2,
            rows: 2,
            columns: 3);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadDatasetForIntegrity(admission);
        SearchResult matching = ResultFor(dataset, queryRow: 0, id: 0);
        SearchResult mismatching = matching with { Distance = matching.Distance + 1000f };
        SearchResult[][] actual =
        [
            [mismatching],
            [ResultFor(dataset, queryRow: 1, id: 1)]
        ];

        HnswReturnedResultIntegrityInfo integrity = FashionMnistExternalHnswBenchmarkScenario.ValidateReturnedResults(
            dataset,
            actual,
            expectedQueryCount: 3,
            topK: 2);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(2, integrity.CheckedResultCount);
        Assert.Equal(1, integrity.QueryCountMismatchCount);
        Assert.Equal(0, integrity.ResultCountViolationCount);
        Assert.Equal(0, integrity.NonFiniteDistanceCount);
        Assert.Equal(0, integrity.DuplicateIdCount);
        Assert.Equal(0, integrity.UnknownIdCount);
        Assert.Equal(1, integrity.DistanceMismatchCount);
    }

    [Fact]
    public void CompareGeneratedExact_RejectsExternalHnswWithoutMutatingInputs()
    {
        string directory = CreateArtifactDirectory("comparison");
        string exactPath = Path.Combine(directory, "exact.json");
        BenchmarkReport exact = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 4,
                VectorCount: 16,
                QueryCount: 3,
                TopK: 2,
                Seed: 0x5EED0390,
                exactPath,
                BaselineReportId: null,
                Runs: 3,
                WarmupQueries: 1),
            ["exact-generated"]);
        ReportWriter.Write(exact, exactPath);

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "comparison-external",
            baseCount: 16,
            queryCount: 3,
            truthDepth: 2,
            rows: 2,
            columns: 2);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string externalPath = Path.Combine(directory, "external-hnsw.json");
        ExternalHnswBenchmarkReport external = FashionMnistExternalHnswBenchmarkScenario.Run(
            new FashionMnistExternalHnswBenchmarkOptions(cacheRoot, externalPath, 3, 2, 1, 0, VectorMetric.SquaredEuclidean, 2, 2, 2, 0x390UL),
            ["external-fashion-mnist-hnsw"]);
        FashionMnistExternalHnswBenchmarkScenario.Write(external, externalPath);

        string exactBefore = File.ReadAllText(exactPath);
        string externalBefore = File.ReadAllText(externalPath);
        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(exactPath, externalPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact"]);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "artifactKindMismatch");
        Assert.Empty(comparison.Metrics);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal(exactBefore, File.ReadAllText(exactPath));
        Assert.Equal(externalBefore, File.ReadAllText(externalPath));
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(
        string prefix,
        int baseCount,
        int queryCount,
        int truthDepth,
        int rows,
        int columns)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows, columns);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadDatasetForIntegrity(FashionMnistAdmissionResult admission)
    {
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        ExternalConvertedMatrixEntry baseEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "base");
        ExternalConvertedMatrixEntry queryEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "query");
        float[] baseVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le"),
            (ulong)baseEntry.RowCount,
            (uint)baseEntry.Dimension);
        float[] queryVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "query.f32le"),
            (ulong)queryEntry.RowCount,
            (uint)queryEntry.Dimension);

        return new FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset(
            new FashionMnistExternalHnswBenchmarkScenario.DatasetPaths(cacheRoot, admission.Manifest.DatasetId, admission.ManifestPath),
            admission.Manifest,
            FileChecksum.ComputeSha256(admission.ManifestPath),
            ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(admission.TruthPath))!,
            admission.Manifest.Truth.Sha256,
            baseVectors,
            queryVectors,
            baseEntry.RowCount,
            queryEntry.RowCount,
            baseEntry.Dimension);
    }

    private static SearchResult ResultFor(FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset, int queryRow, ulong id) =>
        new(id, SquaredEuclidean(dataset.GetQueryVector(queryRow), dataset.GetBaseVector(checked((int)id))));

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static Dictionary<string, string> SnapshotCacheFiles(string cacheRoot) =>
        Directory.EnumerateFiles(cacheRoot, "*", SearchOption.AllDirectories)
            .Select(path => new
            {
                RelativePath = Path.GetRelativePath(cacheRoot, path).Replace('\\', '/'),
                Sha256 = FileChecksum.ComputeSha256(path)
            })
            .OrderBy(item => item.RelativePath, StringComparer.Ordinal)
            .ToDictionary(item => item.RelativePath, item => item.Sha256, StringComparer.Ordinal);

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 5)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 1)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 89)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 7)).ToArray());

        static FashionMnistRawFileSpec Spec(string path, string fileName, string role, int expectedCount) =>
            new(fileName, role, expectedCount, FileChecksum.ComputeMd5(path), downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            datasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: baseCount,
            QueryCount: queryCount,
            ImageRows: rows,
            ImageColumns: columns,
            Dimension: checked(rows * columns),
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", baseCount),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", baseCount),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", queryCount),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", queryCount));
    }

    private static byte[] CreatePixels(int count, int dimension, int offset)
    {
        var payload = new byte[checked(count * dimension)];
        for (int row = 0; row < count; row++)
        {
            for (int column = 0; column < dimension; column++)
            {
                payload[(row * dimension) + column] = (byte)((row * 23 + column * 19 + offset + ((row + column) % 7) * 11) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count, int offset)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)((i + offset) % 10);
        }

        return labels;
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2051);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2049);
        WriteInt32BigEndian(decoded, count);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream Gzip(byte[] decoded)
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            gzip.Write(decoded);
        }

        compressed.Position = 0;
        return compressed;
    }

    private static void WriteInt32BigEndian(Stream stream, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, value);
        stream.Write(buffer);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec039-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
