using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec140FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixTests
{
    [Fact]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix_UsesPrivateSmokeDefaults()
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.EndsWith("fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", "large")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--preset", " ")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--output-dir", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--manifest", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--query-count", "50")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--runs", "2")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--warmup-queries", "3")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--metric", "InnerProduct")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--seed", "0x5EED2139")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--top-k", "10")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--base-vectors", "59000")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--insertions", "500")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--deletes", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--delta-deletes", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--m", "16")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--ef-construction", "128")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--ef-search", "192")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--hnsw-seed", "0x484E535700013901")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint-matrix", "--checkpoint-directory", "checkpoint")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint", "--output-dir", "matrix")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix_RejectsInvalidOrOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
    }

    [Fact]
    public void ExpandCases_SmokeAndStandardPresetsMatchAcceptedOrderProfilesAndSeeds()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        var smokeOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions(
            "smoke",
            "VecNet.DatasetCache",
            outputDirectory,
            Path.Combine(outputDirectory, "manifest.json"),
            VectorMetric.Cosine);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] smokeCases =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(smokeOptions);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase[] standardCases =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(smokeOptions with { PresetName = "standard" });

        Assert.Single(smokeCases);
        Assert.Equal(4, standardCases.Length);
        Assert.Equal(
            [
                "case-001-10k-low-churn-wide-m16-ef192",
                "case-002-10k-tombstone-heavy-wide-m16-ef192",
                "case-003-100k-low-churn-wide-m16-ef192",
                "case-004-100k-tombstone-heavy-wide-m16-ef192"
            ],
            standardCases.Select(item => item.CaseId).ToArray());
        Assert.Equal(standardCases[0].CaseId, smokeCases[0].CaseId);

        Assert.All(standardCases, matrixCase =>
        {
            Assert.Equal("VecNet.DatasetCache", matrixCase.Options.CacheRoot);
            Assert.Equal(50, matrixCase.Options.QueryCount);
            Assert.Equal(2, matrixCase.Options.Runs);
            Assert.Equal(3, matrixCase.Options.WarmupQueries);
            Assert.Equal(VectorMetric.Cosine, matrixCase.Options.Metric);
            Assert.Equal(16, matrixCase.Options.M);
            Assert.Equal(128, matrixCase.Options.EfConstruction);
            Assert.Equal(192, matrixCase.Options.EfSearch);
            Assert.Equal(1, matrixCase.Options.DuplicateInsertAttempts);
            Assert.Equal(1, matrixCase.Options.UnknownDeleteAttempts);
            Assert.Equal(1, matrixCase.Options.RepeatedDeleteAttempts);
            Assert.False(Path.IsPathRooted(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathRooted(matrixCase.RelativeCheckpointDirectoryPath));
            Assert.EndsWith("checkpoint-report.json", matrixCase.RelativeReportPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(matrixCase.CaseId, matrixCase.Options.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);
        });

        Assert.Equal(10, standardCases[0].Options.TopK);
        Assert.Equal(59_000, standardCases[0].Options.BaseVectorCount);
        Assert.Equal(500, standardCases[0].Options.InsertedDeltaCount);
        Assert.Equal(100, standardCases[0].Options.DeletedBaseCount);
        Assert.Equal(0, standardCases[0].Options.DeletedDeltaCount);
        Assert.Equal(59_500, standardCases[0].Options.PhysicalCandidateVectorCount);
        Assert.Equal(59_400, standardCases[0].Options.LiveVectorCount);

        Assert.Equal(56_000, standardCases[1].Options.BaseVectorCount);
        Assert.Equal(2_000, standardCases[1].Options.InsertedDeltaCount);
        Assert.Equal(5_000, standardCases[1].Options.DeletedBaseCount);
        Assert.Equal(500, standardCases[1].Options.DeletedDeltaCount);
        Assert.Equal(58_000, standardCases[1].Options.PhysicalCandidateVectorCount);
        Assert.Equal(52_500, standardCases[1].Options.LiveVectorCount);

        Assert.Equal(0x5EED2139u, standardCases[0].Options.Seed);
        Assert.Equal(0x5EED213Cu, standardCases[^1].Options.Seed);
        Assert.Equal("0x484E535700013901", FormatHex(standardCases[0].Options.HnswSeed));
        Assert.Equal("0x484E535700013904", FormatHex(standardCases[^1].Options.HnswSeed));
    }

    [Fact]
    public void Run_WithMissingCacheBlocksEveryStandardCaseWithoutFakeLinkedReports()
    {
        string outputDirectory = NewArtifactDirectory("missing-cache");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        string[] arguments =
        [
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName,
            "--preset", "standard",
            "--cache-root", Path.Combine(outputDirectory, "missing-cache"),
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(arguments);

        ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest manifest =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.Run(options, arguments);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-140", manifest.TaskId);
        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, manifest.ScenarioName);
        Assert.Equal("standard", manifest.PresetName);
        Assert.Equal("failed", manifest.ValidationStatus);
        Assert.Equal("unavailable", manifest.CacheTruth.Status);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(4, manifest.Aggregate.BlockedCaseCount);
        Assert.True(manifest.Aggregate.CacheTruth.AllCasesBlockedBySharedReadiness);
        Assert.Equal(0, manifest.Aggregate.LinkedReportCount);
        Assert.Equal("notMeasured", manifest.Aggregate.Memory.Status);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("blocked", matrixCase.Status);
            Assert.Equal("blocked", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportPath);
            Assert.Null(matrixCase.LinkedCheckpointDirectoryPath);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.Equal("notAvailable", matrixCase.LinkedReportValidation.Status);
            Assert.Equal("notAvailable", matrixCase.RepeatedCheckpointRuns.Status);
            Assert.Equal("notAvailable", matrixCase.PhaseDiagnostics.Status);
            Assert.Equal("notAvailable", matrixCase.OutputSummary.Status);
            Assert.Equal("notAvailable", matrixCase.PreCheckpointSourceCompositeSearch.Status);
            Assert.Equal("notAvailable", matrixCase.OpenedValidation.Status);
            Assert.Equal("notMeasured", matrixCase.Memory.Status);
            Assert.Equal("cacheTruthReadiness", matrixCase.ErrorCategory);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal([10, 100], ToIntArray(root.GetProperty("design").GetProperty("topKValues")));
        Assert.Equal("notMeasured", root.GetProperty("aggregate").GetProperty("memory").GetProperty("status").GetString());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "comparisonPublicationEligible", "regressionGateEligible");
        Assert.DoesNotContain("\"taskId\": \"VEC-138\"", File.ReadAllText(manifestPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LinkedVec138ReportSummariesAreCopiedWithCheckpointAggregatesAndFalseEligibility()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("linked-summary", baseCount: 48, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputDirectory = Path.Combine(cacheRoot, "matrix");
        Directory.CreateDirectory(outputDirectory);
        var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            Path.Combine(outputDirectory, "case-001-10k-low-churn-wide-m16-ef192", "checkpoint-report.json"),
            Path.Combine(outputDirectory, "case-001-10k-low-churn-wide-m16-ef192", "checkpoint-output"),
            QueryCount: 5,
            TopK: 6,
            BaseVectorCount: 36,
            InsertedDeltaCount: 8,
            DeletedBaseCount: 5,
            DeletedDeltaCount: 3,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 2,
            WarmupQueries: 2,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED2139,
            M: 4,
            EfConstruction: 16,
            EfSearch: 10,
            HnswSeed: 0x484E535700013901);
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                caseOptions,
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.CreateCaseArguments(caseOptions));
        var matrixCase = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase(
            "case-001-10k-low-churn-wide-m16-ef192",
            "low-churn",
            "wide-m16-ef192",
            "case-001-10k-low-churn-wide-m16-ef192/checkpoint-report.json",
            "case-001-10k-low-churn-wide-m16-ef192/checkpoint-output",
            new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExternalCheckpointUpdateProfile(
                "low-churn",
                BaseRowCount: 36,
                DeltaRowCount: 8,
                DeletedBaseCount: 5,
                DeletedDeltaCount: 3,
                ExpectedPhysicalCandidateCount: 44,
                ExpectedLiveCount: 36,
                ExpectedDeletedReservedIdCount: 8,
                "test profile"),
            caseOptions);

        ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest manifestCase =
            InvokeCreateCaseManifest(matrixCase, report);

        Assert.Equal("passed", manifestCase.Status);
        Assert.Equal(report.ReportId, manifestCase.LinkedReportId);
        Assert.Equal("passed", manifestCase.LinkedReportValidation.Status);
        Assert.Equal("recorded", manifestCase.RepeatedCheckpointRuns.Status);
        Assert.Equal(2, manifestCase.RepeatedCheckpointRuns.CompletedRunCount);
        Assert.Equal(2, manifestCase.RepeatedCheckpointRuns.PublishedRunCount);
        Assert.Equal(2, manifestCase.RepeatedCheckpointRuns.DetailedValidationRunNumber);
        Assert.Equal("recorded", manifestCase.PhaseDiagnostics.Status);
        Assert.Equal(2, manifestCase.PhaseDiagnostics.RebuildBuild.MeasuredCount);
        Assert.Equal("recorded", manifestCase.OutputSummary.Status);
        Assert.True(manifestCase.OutputSummary.TotalBytes > 0);
        Assert.Equal("outsideCheckpointDuration", manifestCase.OutputSummary.ScanTimingScope);

        Assert.Equal("recorded", manifestCase.PreCheckpointSourceCompositeSearch.Status);
        Assert.Equal(report.Searches.PreCheckpointSourceComposite.Metrics.RecallAtK, manifestCase.PreCheckpointSourceCompositeSearch.RecallAtK);
        Assert.Equal("passed", manifestCase.PreCheckpointSourceCompositeSearch.ReturnedResultIntegrityStatus);
        Assert.Equal("recorded", manifestCase.PostCheckpointRebuiltCompositeSearch.Status);
        Assert.Equal("recorded", manifestCase.OpenedReadOnlyHnswSearch.Status);
        Assert.Equal("passed", manifestCase.OpenedValidation.Status);
        Assert.True(manifestCase.RebuiltOpenedParity.AllResultsMatched);
        Assert.Equal("passed", manifestCase.DeletedReservation.Status);
        Assert.Equal(8, manifestCase.DeletedReservation.ActualDeletedReservedIdCount);
        Assert.Equal("passed", manifestCase.NoChanges.Status);
        Assert.Equal("recorded", manifestCase.CountSummary.Status);
        Assert.Equal(36, manifestCase.CountSummary.PreCheckpointLiveVectorCount);
        Assert.Equal(0, manifestCase.CountSummary.PostCheckpointTombstoneCount);
        Assert.Equal("notMeasured", manifestCase.Memory.Status);
        Assert.True(manifestCase.RecursiveEligibility.AllEligibilityFlagsFalse);
        Assert.Equal(0, manifestCase.RecursiveEligibility.NonFalseEligibilityFlagCount);
    }

    private static ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest InvokeCreateCaseManifest(
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.MatrixCase matrixCase,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario).GetMethod(
            "CreateCaseManifest",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        var linkedValidation = new ExternalHnswBasePlusExactDeltaCheckpointMatrixLinkedReportValidationSummary(
            "passed",
            LinkedReportInspected: true,
            SchemaMatched: true,
            ScenarioMatched: true,
            CaseParametersMatched: true,
            RequiredCheckpointSectionsPresent: true,
            PhaseDiagnosticsPresent: true,
            OpenedValidationPresent: true,
            RebuiltOpenedParityPassed: true,
            DeletedReservationValidated: true,
            EligibilityFalse: true);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaCheckpointMatrixCaseManifest>(
            method.Invoke(
                null,
                [
                    1,
                    matrixCase,
                    FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.CreateCaseArguments(matrixCase.Options),
                    report,
                    "passed",
                    "passed",
                    null,
                    null,
                    true,
                    linkedValidation
                ]));
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 89)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 131)).ToArray());
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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 37 + offset) % 251);
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
            string.Create(CultureInfo.InvariantCulture, $"vec140-{prefix}-{Guid.NewGuid():N}"));
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
