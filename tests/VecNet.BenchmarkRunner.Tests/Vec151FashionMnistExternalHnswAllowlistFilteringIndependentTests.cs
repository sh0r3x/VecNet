using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec151FashionMnistExternalHnswAllowlistFilteringIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveBoundaryValuesAndKeepsPrivateSmokeDefaults()
    {
        string root = NewArtifactDirectory("parser-boundary");

        FashionMnistExternalHnswAllowlistFilteringOptions options =
            CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-ALLOWLIST-FILTERED",
                    "--CACHE-ROOT", "cache-root",
                    "--OUTPUT", Path.Combine(root, "report.json"),
                    "--OPENED-INDEX-DIRECTORY", Path.Combine(root, "opened-output"),
                    "--CHECKPOINT-DIRECTORY", Path.Combine(root, "checkpoint-output"),
                    "--QUERY-COUNT", "1",
                    "--TOP-K", "4096",
                    "--BASE-VECTORS", "4096",
                    "--INSERTIONS", "1",
                    "--DELETES", "0",
                    "--DELTA-DELETES", "0",
                    "--DUPLICATE-INSERTS", "0",
                    "--UNKNOWN-DELETES", "0",
                    "--REPEATED-DELETES", "0",
                    "--FILTER", "FALLBACK-BOUNDARY",
                    "--RUNS", "5",
                    "--WARMUP-QUERIES", "0",
                    "--METRIC", "COSINE",
                    "--SEED", "0xFFFFFFFF",
                    "--M", "64",
                    "--EF-CONSTRUCTION", "4096",
                    "--EF-SEARCH", "4096",
                    "--HNSW-SEED", "0xFFFFFFFFFFFFFFFF"
                ]);

        Assert.Equal("cache-root", options.CacheRoot);
        Assert.Equal("fallback-boundary", options.FilterProfile);
        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
        Assert.Equal(4_097, options.PhysicalCandidateVectorCount);
        Assert.Equal(4_097, options.LiveVectorCount);
        Assert.Equal(4_096, options.TopK);
        Assert.Equal(5, options.Runs);
        Assert.Equal(uint.MaxValue, options.Seed);
        Assert.Equal(64, options.M);
        Assert.Equal(4_096, options.EfConstruction);
        Assert.Equal(4_096, options.EfSearch);
        Assert.Equal(ulong.MaxValue, options.HnswSeed);
    }

    [Theory]
    [InlineData("--download", "false")]
    [InlineData("--download-raw-files", "false")]
    [InlineData("--truth-refresh", "true")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--matrix", "standard")]
    [InlineData("--preset", "smoke")]
    [InlineData("--output-dir", "matrix")]
    [InlineData("--manifest", "manifest.json")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--comparison-report", "comparison.json")]
    [InlineData("--comparison-output", "comparison.json")]
    [InlineData("--comparison-artifact", "true")]
    [InlineData("--regression-gate", "true")]
    [InlineData("--public-claim", "true")]
    [InlineData("--public-claim-eligible", "true")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--faiss-index", "index.faiss")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--checkpoint-memory", "true")]
    [InlineData("--actual-memory", "true")]
    [InlineData("--dimension", "784")]
    [InlineData("--vectors", "60000")]
    [InlineData("--queries", "50")]
    public void Parser_RejectsDownloadTruthRefreshMatrixComparisonPublicClaimAndAliasSurfaces(
        string option,
        string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(option, exception.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--runs", "6")]
    [InlineData("--warmup-queries", "-1")]
    [InlineData("--metric", "InnerProduct")]
    [InlineData("--filter", "all")]
    [InlineData("--filter", "very-selective")]
    [InlineData("--top-k", "5", "--ef-search", "4")]
    [InlineData("--filter", "fallback-boundary", "--base-vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "16")]
    [InlineData("--filter", "broad", "--base-vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "9")]
    [InlineData("--m", "1")]
    [InlineData("--ef-construction", "1", "--m", "2")]
    [InlineData("--ef-search", "4097")]
    [InlineData("--insertions", "2", "--delta-deletes", "2")]
    public void Parser_RejectsMalformedBoundsUnsupportedProfilesAndBranchImpossibleWorkloads(params string[] options)
    {
        string[] args = [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, .. options];

        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(args));
    }

    [Fact]
    public void CorruptTruthGuard_FailsClosedBeforeReportOpenedOrCheckpointOutputs()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("corrupt-truth", baseCount: 18, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputRoot = NewArtifactDirectory("corrupt-truth-output");
        string reportPath = Path.Combine(outputRoot, "report.json");
        string openedDirectory = Path.Combine(outputRoot, "opened");
        string checkpointDirectory = Path.Combine(outputRoot, "checkpoint");
        File.WriteAllText(admission.TruthPath, """{"schemaName":"corrupt"}""");

        var options = new FashionMnistExternalHnswAllowlistFilteringOptions(
            cacheRoot,
            reportPath,
            openedDirectory,
            checkpointDirectory,
            QueryCount: 2,
            TopK: 2,
            BaseVectorCount: 12,
            InsertedDeltaCount: 3,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 1,
            FilterProfile: "fallback-boundary",
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED151D,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x151D);

        Assert.ThrowsAny<Exception>(() =>
            FashionMnistExternalHnswAllowlistFilteringScenario.Run(
                options,
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName]));

        Assert.False(File.Exists(reportPath));
        Assert.False(Directory.Exists(openedDirectory));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    [Fact]
    public void ProgramRun_FallbackBoundaryPersistsLiveViewTruthAllSectionsAndPrivatePosture()
    {
        JsonElement root = RunProgramReport(
            "fallback-boundary",
            baseCount: 32,
            queryCount: 5,
            truthDepth: 5,
            baseVectors: 20,
            insertions: 6,
            deletes: 3,
            deltaDeletes: 2,
            topK: 4,
            efSearch: 5,
            seed: "0x5EED2511");

        AssertCommonReportShape(root, "fallback-boundary", expectedQueryCount: 3, expectedTopK: 4);
        Assert.Equal("exactFallback", GetString(root.GetProperty("branches"), "expectedBranch"));
        Assert.Equal(3, root.GetProperty("branches").GetProperty("exactFallbackQueryCount").GetInt32());
        Assert.Equal(0, root.GetProperty("branches").GetProperty("broadEmissionQueryCount").GetInt32());
        Assert.Equal(5, root.GetProperty("allowlist").GetProperty("knownLiveAllowedCountPerQuery").GetInt32());
        Assert.Equal(3, root.GetProperty("allowlist").GetProperty("liveBaseAllowedCountPerQuery").GetInt32());
        Assert.Equal(2, root.GetProperty("allowlist").GetProperty("liveDeltaAllowedCountPerQuery").GetInt32());
        Assert.Equal(1, root.GetProperty("allowlist").GetProperty("duplicateInputIdCountPerQuery").GetInt32());
        Assert.Equal(1, root.GetProperty("allowlist").GetProperty("unknownIdCountPerQuery").GetInt32());
        Assert.Equal(2, root.GetProperty("allowlist").GetProperty("tombstonedInputIdCountPerQuery").GetInt32());

        ulong[][] expectedAllowedWindows =
        [
            [3, 4, 5, 22, 23],
            [4, 5, 6, 23, 24],
            [5, 6, 7, 24, 25]
        ];
        AssertFilteredTruthUsesOnlyExpectedLiveAllowlistWindows(root, expectedAllowedWindows);

        foreach (JsonElement section in EnumerateSearchSections(root))
        {
            Assert.Equal("passed", GetString(section.GetProperty("exactFallbackValidation"), "status"));
            Assert.Equal(0, section.GetProperty("exactFallbackValidation").GetProperty("countMismatchCount").GetInt32());
            Assert.Equal(0, section.GetProperty("exactFallbackValidation").GetProperty("idOrOrderMismatchCount").GetInt32());
            Assert.Equal(0, section.GetProperty("exactFallbackValidation").GetProperty("distanceMismatchCount").GetInt32());
            Assert.Equal("notApplicable", GetString(section.GetProperty("broadEmissionValidation"), "status"));
            AssertReturnedIntegrityAndMeasurement(section);
        }

        JsonElement searches = root.GetProperty("searches");
        JsonElement sourceDeltaScan = searches.GetProperty("sourceComposite").GetProperty("exactFilteredDeltaScan");
        Assert.Equal("measured", GetString(sourceDeltaScan, "status"));
        Assert.Equal(2, sourceDeltaScan.GetProperty("allowedLiveDeltaCountPerQuery").GetInt32());
        Assert.Equal(6, sourceDeltaScan.GetProperty("totalAllowedLiveDeltaCount").GetInt32());
        Assert.Equal("measuredZeroAfterCheckpoint", GetString(searches.GetProperty("rebuiltComposite").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("notApplicable", GetString(searches.GetProperty("checkpointOpenedHnsw").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.True(root.GetProperty("validation").GetProperty("exactFallbackParityPassedForAllSearches").GetBoolean());
        Assert.True(root.GetProperty("parity").GetProperty("immutableOpenedHnsw").GetProperty("allResultsMatched").GetBoolean());
        Assert.True(root.GetProperty("parity").GetProperty("rebuiltCompositeCheckpointOpenedHnsw").GetProperty("allResultsMatched").GetBoolean());
    }

    [Fact]
    public void ProgramRun_BroadEmissionRecordsIntegrityRecallUnderfillAndDeltaScanHonestly()
    {
        JsonElement root = RunProgramReport(
            "broad",
            baseCount: 36,
            queryCount: 5,
            truthDepth: 5,
            baseVectors: 22,
            insertions: 7,
            deletes: 2,
            deltaDeletes: 2,
            topK: 4,
            efSearch: 5,
            seed: "0x5EED2512");

        AssertCommonReportShape(root, "broad", expectedQueryCount: 3, expectedTopK: 4);
        Assert.Equal("broadEmission", GetString(root.GetProperty("branches"), "expectedBranch"));
        Assert.Equal(0, root.GetProperty("branches").GetProperty("exactFallbackQueryCount").GetInt32());
        Assert.Equal(3, root.GetProperty("branches").GetProperty("broadEmissionQueryCount").GetInt32());
        Assert.Equal(6, root.GetProperty("allowlist").GetProperty("knownLiveAllowedCountPerQuery").GetInt32());
        Assert.Equal(3, root.GetProperty("allowlist").GetProperty("liveBaseAllowedCountPerQuery").GetInt32());
        Assert.Equal(3, root.GetProperty("allowlist").GetProperty("liveDeltaAllowedCountPerQuery").GetInt32());
        Assert.True(root.GetProperty("validation").GetProperty("broadEmissionIntegrityPassedForAllSearches").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("returnedResultIntegrityPassedForAllSearches").GetBoolean());

        foreach (JsonElement section in EnumerateSearchSections(root))
        {
            Assert.Equal("notApplicable", GetString(section.GetProperty("exactFallbackValidation"), "status"));
            JsonElement broad = section.GetProperty("broadEmissionValidation");
            Assert.Equal("passed", GetString(broad, "status"));
            Assert.InRange(broad.GetProperty("recallAtK").GetDouble(), 0, 1);
            Assert.InRange(broad.GetProperty("orderedAgreement").GetDouble(), 0, 1);
            Assert.True(broad.GetProperty("missingResultCount").GetInt32() >= 0);
            Assert.Equal(0, broad.GetProperty("distanceMismatchCount").GetInt32());

            JsonElement underfill = section.GetProperty("underfill");
            int returned = underfill.GetProperty("totalReturnedResults").GetInt32();
            int exactAvailable = underfill.GetProperty("totalExactTruthAvailableResults").GetInt32();
            int requested = underfill.GetProperty("totalRequestedResultSlots").GetInt32();
            Assert.InRange(returned, 0, requested);
            Assert.Equal(requested, exactAvailable);
            Assert.Equal(requested - returned, underfill.GetProperty("underfilledSlotCount").GetInt32());
            AssertReturnedIntegrityAndMeasurement(section);
        }

        JsonElement sourceDeltaScan = root.GetProperty("searches").GetProperty("sourceComposite").GetProperty("exactFilteredDeltaScan");
        Assert.Equal("measured", GetString(sourceDeltaScan, "status"));
        Assert.Equal(5, sourceDeltaScan.GetProperty("liveDeltaScannedCountPerQuery").GetInt32());
        Assert.Equal(3, sourceDeltaScan.GetProperty("allowedLiveDeltaCountPerQuery").GetInt32());
        Assert.Equal(9, sourceDeltaScan.GetProperty("totalAllowedLiveDeltaCount").GetInt32());
    }

    private static JsonElement RunProgramReport(
        string profile,
        int baseCount,
        int queryCount,
        int truthDepth,
        int baseVectors,
        int insertions,
        int deletes,
        int deltaDeletes,
        int topK,
        int efSearch,
        string seed)
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            $"program-{profile}",
            baseCount,
            queryCount,
            truthDepth);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputRoot = NewArtifactDirectory($"program-{profile}-output");
        string reportPath = Path.Combine(outputRoot, "fashion-mnist-external-hnsw-allowlist-filtered.json");
        string openedDirectory = Path.Combine(outputRoot, "opened-output");
        string checkpointDirectory = Path.Combine(outputRoot, "checkpoint-output");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName,
                "--cache-root", cacheRoot,
                "--output", reportPath,
                "--opened-index-directory", openedDirectory,
                "--checkpoint-directory", checkpointDirectory,
                "--query-count", "3",
                "--top-k", topK.ToString(CultureInfo.InvariantCulture),
                "--base-vectors", baseVectors.ToString(CultureInfo.InvariantCulture),
                "--insertions", insertions.ToString(CultureInfo.InvariantCulture),
                "--deletes", deletes.ToString(CultureInfo.InvariantCulture),
                "--delta-deletes", deltaDeletes.ToString(CultureInfo.InvariantCulture),
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--filter", profile,
                "--runs", "2",
                "--warmup-queries", "1",
                "--metric", "squared-euclidean",
                "--seed", seed,
                "--m", "2",
                "--ef-construction", "8",
                "--ef-search", efSearch.ToString(CultureInfo.InvariantCulture),
                "--hnsw-seed", "0x0000000000002511"
            ]);

        Assert.Equal(0, exitCode);
        Assert.True(File.Exists(reportPath));
        Assert.True(Directory.Exists(openedDirectory));
        Assert.True(Directory.Exists(checkpointDirectory));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(reportPath));
        return document.RootElement.Clone();
    }

    private static void AssertCommonReportShape(JsonElement root, string profile, int expectedQueryCount, int expectedTopK)
    {
        Assert.Equal("VecNet.ExternalHnswAllowlistFilteringBenchmarkReport", GetString(root, "schemaName"));
        Assert.Equal("0.1", GetString(root, "schemaVersion"));
        Assert.Equal("VEC-151", GetString(root, "taskId"));
        Assert.Equal(FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, GetString(root, "scenarioName"));
        Assert.Equal("private-raw", GetString(root, "privacyClass"));
        Assert.Equal("local-evidence", GetString(root, "claimClass"));
        Assert.Equal("smoke", GetString(root.GetProperty("evidence"), "status"));
        Assert.Equal("fashion-mnist-784-euclidean", GetString(root.GetProperty("dataset"), "datasetId"));
        Assert.Equal(profile, GetString(root.GetProperty("allowlist"), "profile"));
        Assert.Equal(expectedQueryCount, root.GetProperty("workload").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(expectedTopK, root.GetProperty("workload").GetProperty("topK").GetInt32());
        JsonElement truth = root.GetProperty("filteredTruth");
        Assert.False(truth.GetProperty("persisted").GetBoolean());
        Assert.Contains("post-update live view", GetString(truth, "generationPolicy"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("truth artifact", GetString(truth, "existingTruthUsage"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not filtered truth", GetString(truth, "existingTruthUsage"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(expectedQueryCount, truth.GetProperty("queryCount").GetInt32());
        Assert.Equal(expectedTopK, truth.GetProperty("truthDepth").GetInt32());
        Assert.Equal(expectedQueryCount, truth.GetProperty("queries").GetArrayLength());

        JsonElement searches = root.GetProperty("searches");
        Assert.Equal("immutableHnsw", GetString(searches.GetProperty("immutableHnsw"), "name"));
        Assert.Equal("openedHnsw", GetString(searches.GetProperty("openedHnsw"), "name"));
        Assert.Equal("sourceComposite", GetString(searches.GetProperty("sourceComposite"), "name"));
        Assert.Equal("rebuiltComposite", GetString(searches.GetProperty("rebuiltComposite"), "name"));
        Assert.Equal("checkpointOpenedHnsw", GetString(searches.GetProperty("checkpointOpenedHnsw"), "name"));

        Assert.Equal("notMeasured", GetString(root.GetProperty("memory"), "status"));
        Assert.Equal("absent", GetString(root.GetProperty("memory"), "value"));
        Assert.Equal("passed", GetString(root.GetProperty("validation"), "status"));
        Assert.True(root.GetProperty("validation").GetProperty("memoryNotMeasured").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "downloadRawFiles", "truthRefresh", "preset", "outputDir", "hnswlibPython", "faissIndex");
    }

    private static void AssertFilteredTruthUsesOnlyExpectedLiveAllowlistWindows(JsonElement root, ulong[][] expectedAllowedWindows)
    {
        JsonElement queries = root.GetProperty("filteredTruth").GetProperty("queries");
        Assert.Equal(expectedAllowedWindows.Length, queries.GetArrayLength());
        for (int queryIndex = 0; queryIndex < queries.GetArrayLength(); queryIndex++)
        {
            HashSet<ulong> allowed = expectedAllowedWindows[queryIndex].ToHashSet();
            foreach (JsonElement neighbor in queries[queryIndex].GetProperty("neighbors").EnumerateArray())
            {
                ulong id = neighbor.GetProperty("id").GetUInt64();
                Assert.Contains(id, allowed);
                Assert.DoesNotContain(id, new ulong[] { 0, 1, 2, 20, 21 });
                Assert.True(float.IsFinite(neighbor.GetProperty("squaredDistance").GetSingle()));
            }
        }
    }

    private static IEnumerable<JsonElement> EnumerateSearchSections(JsonElement root)
    {
        JsonElement searches = root.GetProperty("searches");
        yield return searches.GetProperty("immutableHnsw");
        yield return searches.GetProperty("openedHnsw");
        yield return searches.GetProperty("sourceComposite");
        yield return searches.GetProperty("rebuiltComposite");
        yield return searches.GetProperty("checkpointOpenedHnsw");
    }

    private static void AssertReturnedIntegrityAndMeasurement(JsonElement section)
    {
        JsonElement integrity = section.GetProperty("returnedResultIntegrity");
        Assert.Equal("passed", GetString(integrity, "status"));
        Assert.Equal(0, integrity.GetProperty("unknownIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("tombstonedIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("notAllowedIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("duplicateIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("nonFiniteDistanceCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("distanceMismatchCount").GetInt32());

        JsonElement tombstone = section.GetProperty("tombstoneSuppression");
        Assert.Equal("passed", GetString(tombstone, "status"));
        Assert.Equal(1, tombstone.GetProperty("baseTombstoneInputCountPerQuery").GetInt32());
        Assert.Equal(1, tombstone.GetProperty("deltaTombstoneInputCountPerQuery").GetInt32());
        Assert.Equal(0, tombstone.GetProperty("returnedBaseTombstoneCount").GetInt32());
        Assert.Equal(0, tombstone.GetProperty("returnedDeltaTombstoneCount").GetInt32());

        JsonElement measurement = section.GetProperty("measurement");
        Assert.Equal("measured", GetString(measurement.GetProperty("latency"), "status"));
        Assert.Equal("perMeasuredSearchCall", GetString(measurement.GetProperty("latency"), "sampleScope"));
        Assert.Equal(GetString(section, "timedOperation"), GetString(measurement.GetProperty("latency"), "timedOperation"));
        Assert.Contains("allowlist generation", GetString(measurement.GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("truth", GetString(measurement.GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", GetString(measurement.GetProperty("managedAllocations"), "status"));
        Assert.Equal("bytesPerSearchCall", GetString(measurement.GetProperty("managedAllocations"), "unit"));
        Assert.Equal("notMeasured", GetString(measurement.GetProperty("memory"), "status"));
        Assert.Equal("measured", GetString(measurement.GetProperty("runToRunNoise"), "status"));
        Assert.Equal("executed", GetString(measurement.GetProperty("warmup"), "status"));
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = NewArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 4, columns: 4);
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 53)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 3)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 89)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 6)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 31 + offset + (row % 5) * 7) % 251);
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

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec151-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static string GetString(JsonElement element, string propertyName) =>
        element.GetProperty(propertyName).GetString() ?? string.Empty;

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
