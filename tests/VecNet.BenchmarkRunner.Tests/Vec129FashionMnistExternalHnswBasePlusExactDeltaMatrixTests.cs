using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec129FashionMnistExternalHnswBasePlusExactDeltaMatrixTests
{
    [Fact]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix_UsesPrivateSmokeDefaults()
    {
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(
                ["external-fashion-mnist-hnsw-base-plus-exact-delta-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(0x5EED2128u, options.Seed);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.EndsWith("fashion-mnist-external-hnsw-base-plus-exact-delta-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--preset", "large")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--preset", " ")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--runs", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--runs", "6")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--seed", "0xNOTHEX")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--duplicate-inserts", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--unknown-deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--repeated-deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--output-dir", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--manifest", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--top-k", "10")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--m", "16")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--ef-search", "192")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--base-vectors", "59000")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--insertions", "500")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--deletes", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--delta-deletes", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--hnswlib-python", "python")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", "--checkpoint-directory", "checkpoint")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--output-dir", "matrix")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix_RejectsInvalidOrOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
    }

    [Fact]
    public void ExpandCases_StandardPresetMatchesAcceptedEightCaseOrderAndSeeds()
    {
        string outputDirectory = NewArtifactDirectory("expand-standard");
        var options = new FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions(
            "standard",
            "VecNet.DatasetCache",
            QueryCount: 50,
            Runs: 1,
            WarmupQueries: 3,
            VectorMetric.Cosine,
            Seed: 0x5EED2128,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            outputDirectory,
            Path.Combine(outputDirectory, "manifest.json"));

        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase[] cases =
            FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExpandCases(options);

        Assert.Equal(8, cases.Length);
        Assert.Equal(
            [
                "case-001-10k-low-churn-wide-m16-ef192",
                "case-002-10k-low-churn-wide-m16-ef384",
                "case-003-10k-tombstone-heavy-wide-m16-ef192",
                "case-004-10k-tombstone-heavy-wide-m16-ef384",
                "case-005-100k-low-churn-wide-m16-ef192",
                "case-006-100k-low-churn-wide-m16-ef384",
                "case-007-100k-tombstone-heavy-wide-m16-ef192",
                "case-008-100k-tombstone-heavy-wide-m16-ef384"
            ],
            cases.Select(item => item.CaseId).ToArray());

        Assert.All(cases, matrixCase =>
        {
            Assert.Equal("VecNet.DatasetCache", matrixCase.Options.CacheRoot);
            Assert.Equal(50, matrixCase.Options.QueryCount);
            Assert.Equal(1, matrixCase.Options.Runs);
            Assert.Equal(3, matrixCase.Options.WarmupQueries);
            Assert.Equal(VectorMetric.Cosine, matrixCase.Options.Metric);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.Equal(1, matrixCase.Options.DuplicateInsertAttempts);
            Assert.Equal(1, matrixCase.Options.UnknownDeleteAttempts);
            Assert.Equal(1, matrixCase.Options.RepeatedDeleteAttempts);
            Assert.False(Path.IsPathRooted(matrixCase.RelativeReportPath));
            Assert.StartsWith(outputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        });

        Assert.Equal(0x5EED2128u, cases[0].Options.Seed);
        Assert.Equal(0x5EED212Fu, cases[^1].Options.Seed);
        Assert.Equal("0x484E535700012801", FormatHex(cases[0].Options.HnswSeed));
        Assert.Equal("0x484E535700012808", FormatHex(cases[^1].Options.HnswSeed));
        Assert.Equal(59_000, cases[0].Options.BaseVectorCount);
        Assert.Equal(500, cases[0].Options.InsertedDeltaCount);
        Assert.Equal(100, cases[0].Options.DeletedBaseCount);
        Assert.Equal(0, cases[0].Options.DeletedDeltaCount);
        Assert.Equal(59_400, cases[0].Options.LiveVectorCount);
        Assert.Equal(56_000, cases[2].Options.BaseVectorCount);
        Assert.Equal(2_000, cases[2].Options.InsertedDeltaCount);
        Assert.Equal(5_000, cases[2].Options.DeletedBaseCount);
        Assert.Equal(500, cases[2].Options.DeletedDeltaCount);
        Assert.Equal(52_500, cases[2].Options.LiveVectorCount);
    }

    [Fact]
    public void Run_WithMissingCacheBlocksEveryStandardCaseWithoutFakeLinkedReports()
    {
        string outputDirectory = NewArtifactDirectory("missing-cache");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        string[] arguments =
        [
            "external-fashion-mnist-hnsw-base-plus-exact-delta-matrix",
            "--preset", "standard",
            "--cache-root", Path.Combine(outputDirectory, "missing-cache"),
            "--query-count", "50",
            "--runs", "1",
            "--warmup-queries", "3",
            "--metric", "squared-euclidean",
            "--seed", "0x5EED2128",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(arguments);

        ExternalHnswBasePlusExactDeltaMatrixManifest manifest =
            FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.Run(options, arguments);
        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-129", manifest.TaskId);
        Assert.Equal("external-fashion-mnist-hnsw-base-plus-exact-delta-matrix", manifest.ScenarioName);
        Assert.Equal("standard", manifest.PresetName);
        Assert.Equal("unavailable", manifest.CacheTruth.Status);
        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.SkippedCaseCount);
        Assert.Equal(8, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal(0, manifest.Aggregate.ReturnedResultIntegrityNotPassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.DistanceToleranceNotPassedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.False(manifest.Eligibility.ComparisonPublicationEligible);
        Assert.Equal(0, manifest.Aggregate.Eligibility.LinkedReportNonFalseEligibilityCount);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("blocked", matrixCase.Status);
            Assert.Equal("blocked", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.False(Path.IsPathRooted(matrixCase.LinkedReportPath));
            Assert.False(File.Exists(Path.Combine(outputDirectory, matrixCase.LinkedReportPath)));
            Assert.Equal("notAvailable", matrixCase.RecallOrderSummary.Status);
            Assert.Equal("notAvailable", matrixCase.IntegritySummary.Status);
            Assert.Equal("notAvailable", matrixCase.UnderfillSummary.Status);
            Assert.Equal("notAvailable", matrixCase.AllocationSummary.Status);
            Assert.Equal("notAvailable", matrixCase.MutationSummary.Status);
            Assert.Equal("notAvailable", matrixCase.CountSummary.Status);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("cacheTruth").GetProperty("datasetId").GetString());
        Assert.Equal([10, 100], ToIntArray(root.GetProperty("design").GetProperty("topKValues")));
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible", "comparisonPublicationEligible");
        Assert.DoesNotContain("\"taskId\": \"VEC-127\"", File.ReadAllText(manifestPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LinkedVec127ReportSummariesAndAggregatesAreCopiedRecursivelyWithFalseEligibility()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("linked-summary", baseCount: 30, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputDirectory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix");
        Directory.CreateDirectory(outputDirectory);
        var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            Path.Combine(outputDirectory, "case-001-10k-low-churn-wide-m16-ef192.json"),
            QueryCount: 3,
            TopK: 3,
            BaseVectorCount: 18,
            InsertedDeltaCount: 4,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED2128,
            M: 2,
            EfConstruction: 8,
            EfSearch: 8,
            HnswSeed: 0x484E535700012801);
        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                caseOptions,
                FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.CreateCaseArguments(caseOptions));
        var matrixCase = new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase(
            "case-001-10k-low-churn-wide-m16-ef192",
            "low-churn",
            "wide-m16-ef192",
            "case-001-10k-low-churn-wide-m16-ef192.json",
            new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExternalDeltaUpdateProfile(
                "low-churn",
                BaseRowCount: 18,
                DeltaRowCount: 4,
                DeletedBaseCount: 2,
                DeletedDeltaCount: 1,
                ExpectedLiveCount: 19,
                "test profile"),
            new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExternalDeltaHnswProfile(
                "wide-m16-ef192",
                M: 2,
                EfConstruction: 8,
                EfSearch: 8),
            caseOptions);

        ExternalHnswBasePlusExactDeltaMatrixCaseManifest manifestCase =
            InvokeCreateCaseManifest(matrixCase, report);
        ExternalHnswBasePlusExactDeltaMatrixAggregate aggregate =
            InvokeCreateAggregate([manifestCase], passed: 1, failed: 0, blocked: 0);

        Assert.Equal("passed", manifestCase.Status);
        Assert.Equal(report.ReportId, manifestCase.LinkedReportId);
        Assert.Equal("recorded", manifestCase.RecallOrderSummary.Status);
        Assert.Equal(report.Metrics.RecallAtK, manifestCase.RecallOrderSummary.RecallAtK);
        Assert.Equal(report.Metrics.OrderedAgreement, manifestCase.RecallOrderSummary.OrderedAgreement);
        Assert.Equal("passed", manifestCase.IntegritySummary.Status);
        Assert.Equal(report.Metrics.ReturnedResultIntegrity.CheckedResultCount, manifestCase.IntegritySummary.CheckedResultCount);
        Assert.Equal("recorded", manifestCase.UnderfillSummary.Status);
        Assert.Equal(report.Underfill.UnderfilledSlotCount, manifestCase.UnderfillSummary.UnderfilledSlotCount);
        Assert.Equal("recorded", manifestCase.AllocationSummary.Status);
        Assert.Equal(report.Search.Aggregate.MeanManagedAllocatedBytesPerQuery, manifestCase.AllocationSummary.MeanManagedAllocatedBytesPerSearchCall);
        Assert.Equal("notMeasured", manifestCase.AllocationSummary.MemoryStatus);
        Assert.Equal("recorded", manifestCase.MutationSummary.Status);
        Assert.True(manifestCase.MutationSummary.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(report.Mutations.StatusCounts.Committed, manifestCase.MutationSummary.StatusCommitted);
        Assert.Equal("recorded", manifestCase.CountSummary.Status);
        Assert.Equal(report.Counts.LiveVectorCount, manifestCase.CountSummary.LiveVectorCount);
        Assert.False(manifestCase.EligibilitySummary.PublicClaimEligible);
        Assert.False(manifestCase.EligibilitySummary.BaselineCandidateEligible);
        Assert.False(manifestCase.EligibilitySummary.RegressionGateEligible);
        Assert.False(manifestCase.EligibilitySummary.ValidationPublicClaimEligible);

        Assert.Equal(1, aggregate.PassedCaseCount);
        Assert.Equal(0, aggregate.ReturnedResultIntegrityNotPassedCaseCount);
        Assert.Equal(0, aggregate.DistanceToleranceNotPassedCaseCount);
        Assert.Equal(report.Metrics.RecallAtK, aggregate.Recall.MinimumRecallAtK);
        Assert.Equal(report.Metrics.RecallAtK, aggregate.Recall.MaximumRecallAtK);
        Assert.Equal(report.Metrics.OrderedAgreement, aggregate.Order.MinimumOrderedAgreement);
        Assert.Equal(report.Underfill.UnderfilledSlotCount, aggregate.Underfill.TotalUnderfilledSlotCount);
        Assert.Equal(report.Search.Aggregate.MeanManagedAllocatedBytesPerQuery, aggregate.Allocation.MaximumMeanManagedAllocatedBytesPerSearchCall);
        Assert.Equal(report.Mutations.CommittedMutationCount, aggregate.Mutations.TotalCommittedMutationCount);
        Assert.Equal(report.Counts.LiveVectorCount, aggregate.Counts.MinimumLiveVectorCount);
        Assert.Equal(0, aggregate.Eligibility.LinkedReportNonFalseEligibilityCount);
        Assert.False(aggregate.Eligibility.ManifestPublicClaimEligible);
    }

    private static ExternalHnswBasePlusExactDeltaMatrixCaseManifest InvokeCreateCaseManifest(
        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase matrixCase,
        ExternalHnswBasePlusExactDeltaBenchmarkReport report)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario).GetMethod(
            "CreateCaseManifest",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaMatrixCaseManifest>(
            method.Invoke(
                null,
                [
                    1,
                    matrixCase,
                    FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.CreateCaseArguments(matrixCase.Options),
                    report,
                    "passed",
                    "passed",
                    null
                ]));
    }

    private static ExternalHnswBasePlusExactDeltaMatrixAggregate InvokeCreateAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario).GetMethod(
            "CreateAggregate",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaMatrixAggregate>(
            method.Invoke(null, [cases, passed, failed, blocked]));
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = NewArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 3, columns: 5);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false),
            ["external-fashion-mnist", "--download", "false"],
            spec);
    }

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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 71)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 113)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 31 + column * 17 + offset) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)(i % 10);
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

    private static int[] ToIntArray(JsonElement array) =>
        array.EnumerateArray().Select(value => value.GetInt32()).ToArray();

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec129-{prefix}-{Guid.NewGuid():N}"));
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
}
