using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec127FashionMnistExternalHnswBasePlusExactDeltaIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveBoundaryValuesWithoutChangingDefaultScenario()
    {
        FashionMnistExternalHnswBasePlusExactDeltaOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-BASE-PLUS-EXACT-DELTA",
                    "--CACHE-ROOT", "cache",
                    "--OUTPUT", NewArtifactPath("boundary.json"),
                    "--QUERY-COUNT", "1",
                    "--TOP-K", "4096",
                    "--BASE-VECTORS", "4096",
                    "--INSERTIONS", "1",
                    "--DELETES", "0",
                    "--DELTA-DELETES", "0",
                    "--DUPLICATE-INSERTS", "0",
                    "--UNKNOWN-DELETES", "0",
                    "--REPEATED-DELETES", "0",
                    "--RUNS", "5",
                    "--WARMUP-QUERIES", "0",
                    "--METRIC", "SQUARED-EUCLIDEAN",
                    "--SEED", "0xFFFFFFFF",
                    "--M", "64",
                    "--EF-CONSTRUCTION", "4096",
                    "--EF-SEARCH", "4096",
                    "--HNSW-SEED", "0xFFFFFFFFFFFFFFFF"
                ]);

        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, "external-fashion-mnist-hnsw-base-plus-exact-delta");
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(4096, options.TopK);
        Assert.Equal(4097, options.LiveVectorCount);
        Assert.Equal(5, options.Runs);
        Assert.Equal(uint.MaxValue, options.Seed);
        Assert.Equal(64, options.M);
        Assert.Equal(4096, options.EfConstruction);
        Assert.Equal(4096, options.EfSearch);
        Assert.Equal(ulong.MaxValue, options.HnswSeed);
    }

    [Theory]
    [InlineData("--download", "false")]
    [InlineData("--download-style", "never")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--preset", "smoke")]
    [InlineData("--output-dir", "matrix")]
    [InlineData("--manifest", "manifest.json")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--checkpoint-directory", "checkpoint")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--work-directory", "work")]
    [InlineData("--vecnet-snapshot-directory", "snapshot")]
    [InlineData("--hnswlib-index", "hnswlib.bin")]
    [InlineData("--dimension", "784")]
    [InlineData("--vectors", "60000")]
    [InlineData("--queries", "50")]
    public void Parser_RejectsDownloadMatrixDurableFilterComparisonAndAliasOptions(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DefaultWorkload_IsTheVec126DeterministicSmokeShape()
    {
        FashionMnistExternalHnswBasePlusExactDeltaOptions defaults =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);

        Assert.Equal(50, defaults.QueryCount);
        Assert.Equal(100, defaults.TopK);
        Assert.Equal(58_000, defaults.BaseVectorCount);
        Assert.Equal(1_000, defaults.InsertedDeltaCount);
        Assert.Equal(59_000, defaults.PhysicalCandidateVectorCount);
        Assert.Equal(1_000, defaults.DeletedBaseCount);
        Assert.Equal(100, defaults.DeletedDeltaCount);
        Assert.Equal(57_900, defaults.LiveVectorCount);
        Assert.Equal(1, defaults.DuplicateInsertAttempts);
        Assert.Equal(1, defaults.UnknownDeleteAttempts);
        Assert.Equal(1, defaults.RepeatedDeleteAttempts);
        Assert.Equal(0x5EED2127u, defaults.Seed);
        Assert.Equal(16, defaults.M);
        Assert.Equal(128, defaults.EfConstruction);
        Assert.Equal(192, defaults.EfSearch);
        Assert.Equal(0x484E535700012700UL, defaults.HnswSeed);
        Assert.True(defaults.EfSearch >= defaults.TopK);
    }

    [Fact]
    public void SyntheticAdmittedCacheRun_DoesNotModifyCacheOrCreateDatasetSidecars()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("cache-readonly", baseCount: 30, queryCount: 5, truthDepth: 5);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        Dictionary<string, string> before = SnapshotCacheFiles(cacheRoot);
        string outputPath = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "external-composite", "report.json");

        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                new FashionMnistExternalHnswBasePlusExactDeltaOptions(
                    cacheRoot,
                    outputPath,
                    QueryCount: 4,
                    TopK: 4,
                    BaseVectorCount: 20,
                    InsertedDeltaCount: 5,
                    DeletedBaseCount: 3,
                    DeletedDeltaCount: 1,
                    DuplicateInsertAttempts: 1,
                    UnknownDeleteAttempts: 1,
                    RepeatedDeleteAttempts: 1,
                    Runs: 1,
                    WarmupQueries: 0,
                    VectorMetric.SquaredEuclidean,
                    Seed: 0x5EED7127,
                    M: 2,
                    EfConstruction: 8,
                    EfSearch: 4,
                    HnswSeed: 0x7127),
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);
        FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(report, outputPath);

        Assert.Equal(before, SnapshotCacheFiles(cacheRoot));
        Assert.True(File.Exists(outputPath));
        Assert.DoesNotContain(Path.GetRelativePath(cacheRoot, outputPath), before.Keys);
        Assert.Contains(report.Notes, note => note.Contains("does not download, convert or regenerate", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(admission.Manifest.Truth.Sha256, report.ExistingTruthGuard.Sha256);
        Assert.Equal("scalar-reference-external-live-hnsw-base-plus-exact-delta", report.UpdatedTruth.Kind);
    }

    [Fact]
    public void UpdatedTruthLiveIds_ArePostUpdateBasePlusDeltaMinusTombstones()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("live-truth", baseCount: 24, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            NewArtifactPath("live-truth-report.json"),
            QueryCount: 3,
            TopK: 4,
            BaseVectorCount: 12,
            InsertedDeltaCount: 5,
            DeletedBaseCount: 3,
            DeletedDeltaCount: 2,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED7128,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x7128);

        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);
        ulong[] liveIds = InvokeBuildLiveIds(options);
        TruthSet truth = InvokeGenerateLiveTruth(LoadDataset(admission, report), options, liveIds);

        Assert.Equal<ulong>([3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16], liveIds);
        Assert.Equal(liveIds.Length, report.UpdatedTruth.LiveVectorCount);
        Assert.Equal(liveIds.Length, report.Counts.LiveVectorCount);
        Assert.Equal(4, truth.Depth);
        Assert.All(truth.Results.SelectMany(row => row), item =>
        {
            Assert.Contains(item.Id, liveIds);
            Assert.DoesNotContain(item.Id, new ulong[] { 0, 1, 2, 12, 13 });
        });
        Assert.Contains("readiness guard", report.ExistingTruthGuard.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.False(report.UpdatedTruth.Persisted);
        Assert.Contains("post-update live view", report.UpdatedTruth.Source, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReturnedResultIntegrity_ReportsMalformedCasesIndependently()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("integrity-independent", baseCount: 16, queryCount: 4, truthDepth: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            NewArtifactPath("integrity-report.json"),
            QueryCount: 4,
            TopK: 2,
            BaseVectorCount: 10,
            InsertedDeltaCount: 3,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED7129,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x7129);
        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(options, [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadDataset(admission, report);
        SearchResult valid = ResultFor(dataset, queryRow: 0, id: 2);

        HnswBasePlusExactDeltaReturnedResultIntegrityInfo integrity =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.ValidateReturnedResults(
                dataset,
                [
                    [
                        valid,
                        valid,
                        new SearchResult(0, ResultFor(dataset, queryRow: 0, id: 0).Distance),
                        new SearchResult(13, 1),
                        new SearchResult(3, float.PositiveInfinity),
                        new SearchResult(4, ResultFor(dataset, queryRow: 0, id: 4).Distance + 10_000)
                    ],
                    [ResultFor(dataset, queryRow: 1, id: 5)]
                ],
                options,
                liveIds: [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(1, integrity.QueryCountMismatchCount);
        Assert.Equal(1, integrity.ResultCountViolationCount);
        Assert.Equal(1, integrity.DuplicateIdCount);
        Assert.Equal(1, integrity.UnknownIdCount);
        Assert.Equal(1, integrity.TombstonedIdCount);
        Assert.Equal(1, integrity.NonFiniteDistanceCount);
        Assert.True(integrity.DistanceMismatchCount >= 1);
        Assert.Contains("distance must match recomputed squared-L2", integrity.Policy, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SerializedReport_RecordsMeasurementBoundariesMemoryAndRecursiveNoClaimEligibility()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("serialization", baseCount: 28, queryCount: 5, truthDepth: 5);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = NewArtifactPath("serialization-report.json");
        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                new FashionMnistExternalHnswBasePlusExactDeltaOptions(
                    cacheRoot,
                    outputPath,
                    QueryCount: 3,
                    TopK: 3,
                    BaseVectorCount: 18,
                    InsertedDeltaCount: 4,
                    DeletedBaseCount: 2,
                    DeletedDeltaCount: 1,
                    DuplicateInsertAttempts: 1,
                    UnknownDeleteAttempts: 1,
                    RepeatedDeleteAttempts: 1,
                    Runs: 2,
                    WarmupQueries: 1,
                    VectorMetric.SquaredEuclidean,
                    Seed: 0x5EED7130,
                    M: 2,
                    EfConstruction: 8,
                    EfSearch: 3,
                    HnswSeed: 0x7130),
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);
        FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(report, outputPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");

        Assert.Equal("internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", measurement.GetProperty("latency").GetProperty("timedOperation").GetString());
        string excluded = measurement.GetProperty("latency").GetProperty("excludedOperations").GetString()!;
        Assert.Contains("cache checks", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("HNSW base build", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("update application", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("exact updated truth", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("warmup", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("report writing", excluded, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", measurement.GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Contains("caller-owned SearchResult[]", measurement.GetProperty("managedAllocations").GetProperty("reason").GetString(), StringComparison.Ordinal);
        Assert.Equal("notMeasured", measurement.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", measurement.GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal("measured", measurement.GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("executed", measurement.GetProperty("warmup").GetProperty("status").GetString());

        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "candidateEligibility", "regressionDecision", "publicClaimStatus", "checkpointDirectory", "snapshotDirectory", "hnswlibPython");
    }

    [Fact]
    public void ReportSchema_CanRepresentNonPerfectRecallAndUnderfillWithoutChangingValidationPosture()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("honest-metadata", baseCount: 26, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = NewArtifactPath("honest-metadata-report.json");
        ExternalHnswBasePlusExactDeltaBenchmarkReport baseReport =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                new FashionMnistExternalHnswBasePlusExactDeltaOptions(
                    cacheRoot,
                    outputPath,
                    QueryCount: 3,
                    TopK: 3,
                    BaseVectorCount: 16,
                    InsertedDeltaCount: 3,
                    DeletedBaseCount: 2,
                    DeletedDeltaCount: 1,
                    DuplicateInsertAttempts: 0,
                    UnknownDeleteAttempts: 0,
                    RepeatedDeleteAttempts: 0,
                    Runs: 1,
                    WarmupQueries: 0,
                    VectorMetric.SquaredEuclidean,
                    Seed: 0x5EED7131,
                    M: 2,
                    EfConstruction: 8,
                    EfSearch: 3,
                    HnswSeed: 0x7131),
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName]);
        ExternalHnswBasePlusExactDeltaBenchmarkReport honestNonPerfect = baseReport with
        {
            Metrics = baseReport.Metrics with
            {
                RecallAtK = 0.75,
                OrderedAgreement = 0.50,
                MissingResultCount = 2,
                ExtraResultCount = 1
            },
            Underfill = baseReport.Underfill with
            {
                TotalReturnedResults = 7,
                UnderfilledQueryCount = 2,
                UnderfilledSlotCount = 2
            },
            Validation = baseReport.Validation with
            {
                Status = "passed",
                AllowsApproximateRecallBelowOne = true,
                AllowsUnderfill = true
            }
        };
        FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(honestNonPerfect, outputPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;

        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("allowsApproximateRecallBelowOne").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("allowsUnderfill").GetBoolean());
        Assert.Equal(0.75, root.GetProperty("metrics").GetProperty("recallAtK").GetDouble(), precision: 6);
        Assert.Equal(0.50, root.GetProperty("metrics").GetProperty("orderedAgreement").GetDouble(), precision: 6);
        Assert.Equal(2, root.GetProperty("underfill").GetProperty("underfilledQueryCount").GetInt32());
        Assert.Equal(2, root.GetProperty("underfill").GetProperty("underfilledSlotCount").GetInt32());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 3, columns: 5);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false),
            ["external-fashion-mnist", "--download", "false"],
            spec);
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadDataset(
        FashionMnistAdmissionResult admission,
        ExternalHnswBasePlusExactDeltaBenchmarkReport report)
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
            report.Dataset.AdmissionManifest.Sha256,
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

    private static ulong[] InvokeBuildLiveIds(FashionMnistExternalHnswBasePlusExactDeltaOptions options)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaScenario).GetMethod(
            "BuildLiveIds",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<ulong[]>(method.Invoke(null, [options]));
    }

    private static TruthSet InvokeGenerateLiveTruth(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        ulong[] liveIds)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaScenario).GetMethod(
            "GenerateLiveTruth",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<TruthSet>(method.Invoke(null, [dataset, options, liveIds]));
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 17)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 3)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 61)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 5)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 13 + offset + (row % 5) * 7) % 251);
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

    private static string NewArtifactPath(string fileName)
    {
        string directory = CreateArtifactDirectory(Path.GetFileNameWithoutExtension(fileName));
        return Path.Combine(directory, fileName);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec127-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static void AssertNoBooleanPropertyTrueForNames(JsonElement element, params string[] propertyNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    propertyNames.Any(name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    Assert.Fail($"Property '{property.Name}' must not be true.");
                }

                AssertNoBooleanPropertyTrueForNames(property.Value, propertyNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoBooleanPropertyTrueForNames(item, propertyNames);
            }
        }
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                AssertNoPropertyNamed(property.Value, disallowedNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, disallowedNames);
            }
        }
    }
}
