using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec151FashionMnistExternalHnswAllowlistFilteringTests
{
    [Fact]
    public void ParseExternalFashionMnistHnswAllowlistFiltering_UsesAcceptedPrivateDefaults()
    {
        FashionMnistExternalHnswAllowlistFilteringOptions options =
            CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.EndsWith("fashion-mnist-external-hnsw-allowlist-filtered.json", options.OutputPath);
        Assert.EndsWith(Path.Combine("vec-151-smoke", "opened-output"), options.OpenedIndexDirectory);
        Assert.EndsWith(Path.Combine("vec-151-smoke", "checkpoint-output"), options.CheckpointDirectory);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(58_000, options.BaseVectorCount);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(59_000, options.PhysicalCandidateVectorCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(100, options.DeletedDeltaCount);
        Assert.Equal(57_900, options.LiveVectorCount);
        Assert.Equal("fallback-boundary", options.FilterProfile);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(0x5EED2151u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(128, options.EfConstruction);
        Assert.Equal(192, options.EfSearch);
        Assert.Equal(0x484E535700015100UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--snapshot-directory", "snapshot")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--hnswlib-python", "python")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--metric", "Cosine")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--filter", "all")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--base-vectors", "0")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--insertions", "0")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--deletes", "11", "--base-vectors", "10")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--delta-deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--delta-deletes", "2", "--insertions", "1")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--top-k", "10", "--base-vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--filter", "fallback-boundary", "--base-vectors", "16", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "32")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--filter", "broad", "--base-vectors", "16", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "32")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--m", "1")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--output", "")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--opened-index-directory", "")]
    [InlineData("external-fashion-mnist-hnsw-allowlist-filtered", "--checkpoint-directory", "")]
    public void ParseExternalFashionMnistHnswAllowlistFiltering_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_FallbackBoundarySyntheticCache_EmitsAllSectionsAndExactParity()
    {
        ExternalHnswAllowlistFilteringBenchmarkReport report = RunReport("fallback-boundary");

        Assert.Equal("VecNet.ExternalHnswAllowlistFilteringBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-151", report.TaskId);
        Assert.Equal(FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Contains("readiness guard", report.ExistingTruthGuard.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("scalar-reference-external-filtered-live-hnsw-allowlist", report.FilteredTruth.Kind);
        Assert.False(report.FilteredTruth.Persisted);
        Assert.Equal(5, report.FilteredTruth.QueryCount);
        Assert.Equal(5, report.FilteredTruth.TruthDepth);
        Assert.Equal(36, report.FilteredTruth.LiveVectorCount);
        Assert.Equal(5, report.FilteredTruth.Queries.Length);
        Assert.All(report.FilteredTruth.Queries, query => Assert.InRange(query.Neighbors.Length, 1, 5));

        Assert.Equal("fallback-boundary", report.Allowlist.Profile);
        Assert.Equal(8, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.Equal(4, report.Allowlist.LiveBaseAllowedCountPerQuery);
        Assert.Equal(4, report.Allowlist.LiveDeltaAllowedCountPerQuery);
        Assert.Equal(1, report.Allowlist.DuplicateInputIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.UnknownIdCountPerQuery);
        Assert.Equal(2, report.Allowlist.TombstonedInputIdCountPerQuery);
        Assert.Equal(5, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(0, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("exactFallback", report.Branches.ExpectedBranch);

        Assert.Equal(36, report.PreCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(31, report.PreCheckpointCounts.BaseLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(5, report.PreCheckpointCounts.DeltaLiveVectorCount);
        Assert.Equal(36, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(36, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal("notMeasured", report.Memory.Status);

        AssertFallbackSection(report.Searches.ImmutableHnsw);
        AssertFallbackSection(report.Searches.OpenedHnsw);
        AssertFallbackSection(report.Searches.SourceComposite);
        AssertFallbackSection(report.Searches.RebuiltComposite);
        AssertFallbackSection(report.Searches.CheckpointOpenedHnsw);
        Assert.Equal("measured", report.Searches.SourceComposite.ExactFilteredDeltaScan.Status);
        Assert.Equal(4, report.Searches.SourceComposite.ExactFilteredDeltaScan.AllowedLiveDeltaCountPerQuery);
        Assert.Equal("measuredZeroAfterCheckpoint", report.Searches.RebuiltComposite.ExactFilteredDeltaScan.Status);
        Assert.True(report.Parity.ImmutableOpenedHnsw.AllResultsMatched);
        Assert.True(report.Parity.RebuiltCompositeCheckpointOpenedHnsw.AllResultsMatched);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.CacheAndTruthReadinessPassed);
        Assert.True(report.Validation.ExactFilteredTruthGeneratedFromLiveView);
        Assert.True(report.Validation.ExactFallbackParityPassedForAllSearches);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);
        Assert.True(report.Validation.TombstoneSuppressionPassed);
        Assert.True(report.Validation.MemoryNotMeasured);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void Run_BroadSyntheticCache_RecordsEmissionIntegrityUnderfillAndFalseEligibility()
    {
        ExternalHnswAllowlistFilteringBenchmarkReport report = RunReport("broad");

        Assert.Equal("broad", report.Allowlist.Profile);
        Assert.Equal(9, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.Equal(0, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(5, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("broadEmission", report.Branches.ExpectedBranch);

        AssertBroadSection(report.Searches.ImmutableHnsw);
        AssertBroadSection(report.Searches.OpenedHnsw);
        AssertBroadSection(report.Searches.SourceComposite);
        AssertBroadSection(report.Searches.RebuiltComposite);
        AssertBroadSection(report.Searches.CheckpointOpenedHnsw);
        Assert.Equal("measured", report.Searches.SourceComposite.ExactFilteredDeltaScan.Status);
        Assert.True(report.Validation.BroadEmissionIntegrityPassedForAllSearches);
        Assert.True(report.Validation.BranchConsistencyPassed);
        Assert.False(report.Validation.PublicClaimEligible);

        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswAllowlistFilteringBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("broad", root.GetProperty("allowlist").GetProperty("profile").GetString());
        Assert.Equal("notMeasured", root.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal(5, root.GetProperty("filteredTruth").GetProperty("queries").GetArrayLength());
        Assert.Equal("immutableHnsw", root.GetProperty("searches").GetProperty("immutableHnsw").GetProperty("name").GetString());
        Assert.Equal("openedHnsw", root.GetProperty("searches").GetProperty("openedHnsw").GetProperty("name").GetString());
        Assert.Equal("sourceComposite", root.GetProperty("searches").GetProperty("sourceComposite").GetProperty("name").GetString());
        Assert.Equal("rebuiltComposite", root.GetProperty("searches").GetProperty("rebuiltComposite").GetProperty("name").GetString());
        Assert.Equal("checkpointOpenedHnsw", root.GetProperty("searches").GetProperty("checkpointOpenedHnsw").GetProperty("name").GetString());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "downloadRawFiles", "truthRefresh", "preset", "manifest", "outputDir", "hnswlibPython");
    }

    [Fact]
    public void Run_MissingCacheFailsClosedWithoutWritingOutputs()
    {
        string cacheRoot = CreateArtifactDirectory("missing-cache");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string openedDirectory = Path.Combine(cacheRoot, "opened");
        string checkpointDirectory = Path.Combine(cacheRoot, "checkpoint");

        var options = new FashionMnistExternalHnswAllowlistFilteringOptions(
            cacheRoot,
            outputPath,
            openedDirectory,
            checkpointDirectory,
            QueryCount: 1,
            TopK: 1,
            BaseVectorCount: 4,
            InsertedDeltaCount: 1,
            DeletedBaseCount: 1,
            DeletedDeltaCount: 0,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            FilterProfile: "fallback-boundary",
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1510,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x1510);

        Assert.Throws<FileNotFoundException>(() =>
            FashionMnistExternalHnswAllowlistFilteringScenario.Run(
                options,
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName]));
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(openedDirectory));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    private static ExternalHnswAllowlistFilteringBenchmarkReport RunReport(string filter)
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission($"allowlist-{filter}", baseCount: 48, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = CreateArtifactDirectory($"report-{filter}");
        string[] arguments =
        [
            FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName,
            "--cache-root", cacheRoot,
            "--output", Path.Combine(directory, "report.json"),
            "--opened-index-directory", Path.Combine(directory, "opened"),
            "--checkpoint-directory", Path.Combine(directory, "checkpoint"),
            "--query-count", "5",
            "--top-k", "5",
            "--base-vectors", "36",
            "--insertions", "8",
            "--deletes", "5",
            "--delta-deletes", "3",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "2",
            "--filter", filter,
            "--runs", "1",
            "--warmup-queries", "2",
            "--metric", "squared-euclidean",
            "--seed", "0x5EED1511",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000001511"
        ];

        FashionMnistExternalHnswAllowlistFilteringOptions options =
            CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(arguments);
        ExternalHnswAllowlistFilteringBenchmarkReport report =
            FashionMnistExternalHnswAllowlistFilteringScenario.Run(options, arguments);
        FashionMnistExternalHnswAllowlistFilteringScenario.Write(report, options.OutputPath);
        Assert.True(File.Exists(options.OutputPath));
        return report;
    }

    private static void AssertFallbackSection(HnswAllowlistSearchSectionInfo section)
    {
        Assert.Equal("passed", section.ExactFallbackValidation.Status);
        Assert.Equal(0, section.ExactFallbackValidation.CountMismatchCount);
        Assert.Equal(0, section.ExactFallbackValidation.IdOrOrderMismatchCount);
        Assert.Equal(0, section.ExactFallbackValidation.DistanceMismatchCount);
        Assert.Equal("notApplicable", section.BroadEmissionValidation.Status);
        Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
        Assert.Equal("passed", section.TombstoneSuppression.Status);
        Assert.Equal("measured", section.Measurement.Latency.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.Equal(5, section.Underfill.QueryCount);
        Assert.Equal(5, section.Underfill.RequestedResultCountPerQuery);
    }

    private static void AssertBroadSection(HnswAllowlistSearchSectionInfo section)
    {
        Assert.Equal("notApplicable", section.ExactFallbackValidation.Status);
        Assert.Equal("passed", section.BroadEmissionValidation.Status);
        Assert.InRange(section.BroadEmissionValidation.RecallAtK, 0, 1);
        Assert.InRange(section.BroadEmissionValidation.OrderedAgreement, 0, 1);
        Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
        Assert.Equal("passed", section.TombstoneSuppression.Status);
        Assert.Equal("measured", section.Measurement.Latency.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.InRange(section.Underfill.UnderfilledQueryCount, 0, 5);
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 41)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 2)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 73)).ToArray());
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
                payload[(row * dimension) + column] = (byte)((row * 23 + column * 19 + offset + (row % 7) * 5) % 251);
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
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec151-{prefix}-{Guid.NewGuid():N}"));
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
