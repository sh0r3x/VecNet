using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec138FashionMnistExternalHnswBasePlusExactDeltaCheckpointIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveBoundaryValuesAndKeepsAcceptedCheckpointShape()
    {
        string root = NewArtifactDirectory("parser-boundary");
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-BASE-PLUS-EXACT-DELTA-CHECKPOINT",
                    "--CACHE-ROOT", "cache-root",
                    "--OUTPUT", Path.Combine(root, "report.json"),
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
                    "--RUNS", "5",
                    "--WARMUP-QUERIES", "0",
                    "--METRIC", "SQUARED-EUCLIDEAN",
                    "--SEED", "0xFFFFFFFF",
                    "--M", "64",
                    "--EF-CONSTRUCTION", "4096",
                    "--EF-SEARCH", "4096",
                    "--HNSW-SEED", "0xFFFFFFFFFFFFFFFF"
                ]);

        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "external-fashion-mnist-hnsw-base-plus-exact-delta-checkpoint");
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
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
    [InlineData("--preset", "smoke")]
    [InlineData("--output-dir", "matrix")]
    [InlineData("--manifest", "manifest.json")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--checkpoint", "true")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--work-directory", "work")]
    [InlineData("--vecnet-snapshot-directory", "snapshot")]
    [InlineData("--hnswlib-index", "index.bin")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "report-id")]
    [InlineData("--public-claim", "true")]
    [InlineData("--public-claim-eligible", "true")]
    [InlineData("--comparison-artifact", "true")]
    [InlineData("--comparison-publication", "true")]
    [InlineData("--regression-gate", "true")]
    [InlineData("--actual-memory", "true")]
    [InlineData("--peak-memory", "true")]
    [InlineData("--concurrency", "2")]
    [InlineData("--dimension", "784")]
    [InlineData("--vectors", "60000")]
    [InlineData("--queries", "50")]
    public void Parser_RejectsRefreshMatrixDurableFilterComparisonMemoryConcurrencyAndAliasOptions(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(option, exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void InvalidTruthGuard_FailsClosedBeforeWritingReportOrCheckpointOutput()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("invalid-truth", baseCount: 16, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string reportPath = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "checkpoint", "report.json");
        string checkpointDirectory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "checkpoint-output");
        File.WriteAllText(admission.TruthPath, """{"schemaName":"corrupt"}""");

        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            reportPath,
            checkpointDirectory,
            QueryCount: 2,
            TopK: 2,
            BaseVectorCount: 10,
            InsertedDeltaCount: 2,
            DeletedBaseCount: 1,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED8138,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x8138);

        Assert.ThrowsAny<Exception>(() =>
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName]));
        Assert.False(File.Exists(reportPath));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    [Fact]
    public void ProgramRun_RepresentsFinalRunCheckpointValidationAndPrivatePostureInJson()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("program-report", baseCount: 44, queryCount: 6, truthDepth: 6);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputRoot = NewArtifactDirectory("program-output");
        string reportPath = Path.Combine(outputRoot, "fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint.json");
        string checkpointRoot = Path.Combine(outputRoot, "checkpoint-output");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
                "--cache-root", cacheRoot,
                "--output", reportPath,
                "--checkpoint-directory", checkpointRoot,
                "--query-count", "4",
                "--top-k", "4",
                "--base-vectors", "30",
                "--insertions", "6",
                "--deletes", "4",
                "--delta-deletes", "2",
                "--duplicate-inserts", "1",
                "--unknown-deletes", "1",
                "--repeated-deletes", "1",
                "--runs", "2",
                "--warmup-queries", "1",
                "--metric", "squared-euclidean",
                "--seed", "0x5EED8139",
                "--m", "2",
                "--ef-construction", "8",
                "--ef-search", "4",
                "--hnsw-seed", "0x8139"
            ]);

        Assert.Equal(0, exitCode);
        Assert.True(File.Exists(reportPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(reportPath));
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport", GetString(root, "schemaName"));
        Assert.Equal("0.1", GetString(root, "schemaVersion"));
        Assert.Equal("VEC-138", GetString(root, "taskId"));
        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, GetString(root, "scenarioName"));
        Assert.Equal("private-raw", GetString(root, "privacyClass"));
        Assert.Equal("local-evidence", GetString(root, "claimClass"));
        Assert.Equal("fashion-mnist-784-euclidean", GetString(root.GetProperty("dataset"), "datasetId"));
        Assert.Contains("readiness guard", GetString(root.GetProperty("existingTruthGuard"), "distanceSemantics"), StringComparison.OrdinalIgnoreCase);
        Assert.False(root.GetProperty("updatedTruth").GetProperty("persisted").GetBoolean());
        Assert.Contains("post-update live view", GetString(root.GetProperty("updatedTruth"), "source"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(30, root.GetProperty("workload").GetProperty("immutableBaseRowCount").GetInt32());
        Assert.Equal(6, root.GetProperty("workload").GetProperty("deltaRowCount").GetInt32());
        Assert.Equal(30, root.GetProperty("updatedTruth").GetProperty("liveVectorCount").GetInt32());

        AssertCheckpointRunsUseFreshDirectoriesAndFinalRun(root, checkpointRoot);
        AssertOpenedValidationAndParity(root);
        AssertSearchSections(root);
        AssertPrivateEligibilityAndMemory(root);
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "downloadRawFiles", "truthRefresh", "preset", "manifest", "outputDir", "snapshotDirectory", "hnswlibPython");
    }

    [Fact]
    public void LiveTruthAndIntegrityHelpers_ExposeTombstoneUnderfillAndDistanceIntegritySemantics()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("helpers", baseCount: 22, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            NewArtifactPath("helpers-report.json"),
            NewArtifactDirectory("helpers-checkpoint"),
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
            Seed: 0x5EED813A,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x813A);

        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName]);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadDataset(admission, report);
        ulong[] liveIds = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.BuildLiveIds(options);
        TruthSet truth = FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.GenerateLiveTruth(dataset, options, liveIds);

        Assert.Equal<ulong>([3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16], liveIds);
        Assert.Equal(12, report.UpdatedTruth.LiveVectorCount);
        Assert.Equal(12, report.PreCheckpointCounts.LiveVectorCount);
        Assert.Equal(12, report.PostCheckpointCounts.LiveVectorCount);
        Assert.All(truth.Results.SelectMany(row => row), item =>
        {
            Assert.Contains(item.Id, liveIds);
            Assert.DoesNotContain(item.Id, new ulong[] { 0, 1, 2, 12, 13 });
        });

        SearchResult valid = ResultFor(dataset, queryRow: 0, id: 3);
        HnswBasePlusExactDeltaReturnedResultIntegrityInfo malformed =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.ValidateReturnedResults(
                dataset,
                [
                    [
                        valid,
                        valid,
                        new SearchResult(0, ResultFor(dataset, queryRow: 0, id: 0).Distance),
                        new SearchResult(21, 1),
                        new SearchResult(4, float.NaN),
                        new SearchResult(5, ResultFor(dataset, queryRow: 0, id: 5).Distance + 10_000)
                    ],
                    [ResultFor(dataset, queryRow: 1, id: 6)]
                ],
                options,
                liveIds);

        Assert.Equal("failed", malformed.Status);
        Assert.Equal(1, malformed.QueryCountMismatchCount);
        Assert.Equal(1, malformed.ResultCountViolationCount);
        Assert.Equal(1, malformed.DuplicateIdCount);
        Assert.Equal(1, malformed.UnknownIdCount);
        Assert.Equal(1, malformed.TombstonedIdCount);
        Assert.Equal(1, malformed.NonFiniteDistanceCount);
        Assert.True(malformed.DistanceMismatchCount >= 1);
        Assert.Contains("tombstoned IDs must not be returned", malformed.Policy, StringComparison.OrdinalIgnoreCase);
        Assert.All(
            new[]
            {
                report.Searches.PreCheckpointSourceComposite,
                report.Searches.PostCheckpointRebuiltComposite,
                report.Searches.OpenedReadOnlyHnsw
            },
            section =>
            {
                Assert.Equal(3, section.Underfill.QueryCount);
                Assert.Equal(4, section.Underfill.RequestedResultCountPerQuery);
                Assert.Equal(12, section.Underfill.TotalRequestedResultSlots);
                Assert.Equal("passed", section.Metrics.ReturnedResultIntegrity.Status);
                Assert.Contains("Underfill is recorded", section.Underfill.Policy, StringComparison.OrdinalIgnoreCase);
            });
    }

    private static void AssertCheckpointRunsUseFreshDirectoriesAndFinalRun(JsonElement root, string checkpointRoot)
    {
        JsonElement checkpointRuns = root.GetProperty("checkpointRuns");
        JsonElement runs = checkpointRuns.GetProperty("runs");
        Assert.Equal(2, checkpointRuns.GetProperty("runCount").GetInt32());
        Assert.Equal(2, checkpointRuns.GetProperty("detailedValidationRunNumber").GetInt32());
        Assert.Equal(2, runs.GetArrayLength());
        Assert.Contains("final checkpoint run", GetString(checkpointRuns, "detailedValidationPolicy"), StringComparison.OrdinalIgnoreCase);

        var directories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string fullCheckpointRoot = Path.GetFullPath(checkpointRoot);
        int expectedRunNumber = 1;
        foreach (JsonElement run in runs.EnumerateArray())
        {
            string directory = GetString(run, "checkpointDirectory");
            Assert.Equal(expectedRunNumber, run.GetProperty("runNumber").GetInt32());
            Assert.Equal("Published", GetString(run, "status"));
            Assert.True(run.GetProperty("generationAdvancedExactlyOnce").GetBoolean());
            Assert.StartsWith(fullCheckpointRoot, directory, StringComparison.OrdinalIgnoreCase);
            Assert.EndsWith(string.Create(CultureInfo.InvariantCulture, $"checkpoint-run-{expectedRunNumber:000}"), directory, StringComparison.OrdinalIgnoreCase);
            Assert.True(directories.Add(directory));
            Assert.True(Directory.Exists(directory));
            Assert.True(File.Exists(Path.Combine(directory, "hnsw.manifest.json")));
            AssertMeasuredPhaseSet(run.GetProperty("phases"));
            expectedRunNumber++;
        }

        JsonElement finalRun = runs[1];
        JsonElement checkpoint = root.GetProperty("checkpoint");
        Assert.Equal(GetString(finalRun, "checkpointDirectory"), GetString(root.GetProperty("output"), "directoryPath"));
        Assert.Equal(finalRun.GetProperty("generationBeforeCheckpoint").GetInt64(), checkpoint.GetProperty("generationBeforeCheckpoint").GetInt64());
        Assert.Equal(finalRun.GetProperty("generationAfterCheckpoint").GetInt64(), checkpoint.GetProperty("generationAfterCheckpoint").GetInt64());
        Assert.Equal(finalRun.GetProperty("managedAllocatedBytes").GetInt64(), checkpoint.GetProperty("managedAllocatedBytes").GetInt64());
        AssertMeasuredPhaseSet(checkpoint.GetProperty("phases"));

        JsonElement noChanges = root.GetProperty("noChangesProbe");
        Assert.Equal("passed", GetString(noChanges, "status"));
        AssertNotExecutedPhaseSet(noChanges.GetProperty("phases"));

        JsonElement validation = root.GetProperty("validation");
        Assert.True(validation.GetProperty("checkpointRepeatedRunEvidencePresent").GetBoolean());
        Assert.True(validation.GetProperty("detailedValidationUsesFinalRun").GetBoolean());
        Assert.True(validation.GetProperty("phaseDiagnosticsMeasuredForPublishedCheckpoint").GetBoolean());
        Assert.True(validation.GetProperty("outputBytesScannedOutsideCheckpointDuration").GetBoolean());
    }

    private static void AssertOpenedValidationAndParity(JsonElement root)
    {
        JsonElement opened = root.GetProperty("openedValidation");
        Assert.Equal("passed", GetString(opened, "status"));
        Assert.Equal(30, opened.GetProperty("expectedVectorCount").GetInt32());
        Assert.Equal(30, opened.GetProperty("openedVectorCount").GetInt32());
        Assert.Equal(0, opened.GetProperty("idMismatchCount").GetInt32());
        Assert.Equal(0, opened.GetProperty("vectorMismatchCount").GetInt32());
        Assert.Contains("vector payloads matching", GetString(opened, "policy"), StringComparison.OrdinalIgnoreCase);

        JsonElement parity = opened.GetProperty("rebuiltCompositeOpenedSearchParity");
        Assert.True(parity.GetProperty("allResultsMatched").GetBoolean());
        Assert.Equal(0, parity.GetProperty("writtenCountMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("idMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("orderMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("distanceMismatchCount").GetInt32());
        Assert.Contains("must return the same count, IDs, order and distances", GetString(parity, "policy"), StringComparison.OrdinalIgnoreCase);

        JsonElement validation = root.GetProperty("validation");
        Assert.True(validation.GetProperty("openedReadOnlyHnswIdVectorValidationPassed").GetBoolean());
        Assert.True(validation.GetProperty("rebuiltCompositeOpenedHnswSearchParityPassed").GetBoolean());
        Assert.True(validation.GetProperty("deletedReservedIdsRejectedAfterCheckpoint").GetBoolean());
        Assert.True(validation.GetProperty("returnedResultIntegrityPassedForAllSearches").GetBoolean());
    }

    private static void AssertSearchSections(JsonElement root)
    {
        JsonElement searches = root.GetProperty("searches");
        AssertSearchSection(searches.GetProperty("preCheckpointSourceComposite"), "preCheckpointSourceComposite");
        AssertSearchSection(searches.GetProperty("postCheckpointRebuiltComposite"), "postCheckpointRebuiltComposite");
        AssertSearchSection(searches.GetProperty("openedReadOnlyHnsw"), "openedReadOnlyHnsw");

        Assert.NotEqual(
            GetString(searches.GetProperty("preCheckpointSourceComposite"), "timedOperation"),
            GetString(searches.GetProperty("postCheckpointRebuiltComposite"), "timedOperation"));
        Assert.NotEqual(
            GetString(searches.GetProperty("postCheckpointRebuiltComposite"), "timedOperation"),
            GetString(searches.GetProperty("openedReadOnlyHnsw"), "timedOperation"));
    }

    private static void AssertSearchSection(JsonElement section, string name)
    {
        Assert.Equal(name, GetString(section, "name"));
        JsonElement search = section.GetProperty("search");
        Assert.Equal(4, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(2, search.GetProperty("runs").GetArrayLength());
        Assert.Equal(2, search.GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal("measured", GetString(section.GetProperty("measurement").GetProperty("latency"), "status"));
        Assert.Equal("perMeasuredSearchCall", GetString(section.GetProperty("measurement").GetProperty("latency"), "sampleScope"));
        Assert.Contains("checkpoint call", GetString(section.GetProperty("measurement").GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", GetString(section.GetProperty("measurement").GetProperty("managedAllocations"), "status"));
        Assert.Equal("notMeasured", GetString(section.GetProperty("measurement").GetProperty("memory"), "status"));
        Assert.Equal("measured", GetString(section.GetProperty("measurement").GetProperty("runToRunNoise"), "status"));
        Assert.Equal("passed", GetString(section.GetProperty("metrics").GetProperty("returnedResultIntegrity"), "status"));
        Assert.Equal("passed", GetString(section.GetProperty("metrics"), "distanceToleranceStatus"));
        Assert.Equal(4, section.GetProperty("underfill").GetProperty("queryCount").GetInt32());
        Assert.Equal(4, section.GetProperty("underfill").GetProperty("requestedResultCountPerQuery").GetInt32());
        Assert.Equal(16, section.GetProperty("underfill").GetProperty("totalRequestedResultSlots").GetInt32());
        Assert.Equal(
            section.GetProperty("underfill").GetProperty("totalRequestedResultSlots").GetInt32() - section.GetProperty("underfill").GetProperty("totalReturnedResults").GetInt32(),
            section.GetProperty("underfill").GetProperty("underfilledSlotCount").GetInt32());
    }

    private static void AssertPrivateEligibilityAndMemory(JsonElement root)
    {
        JsonElement measurement = root.GetProperty("measurement");
        Assert.Equal("measured", GetString(measurement.GetProperty("checkpointLatency"), "status"));
        Assert.Contains("VEC-133", GetString(measurement.GetProperty("checkpointLatency"), "percentileEstimator"), StringComparison.Ordinal);
        Assert.Equal("measured", GetString(measurement.GetProperty("checkpointManagedAllocations"), "status"));
        Assert.Equal("notMeasured", GetString(measurement.GetProperty("memory"), "status"));
        Assert.Equal("absent", GetString(measurement.GetProperty("memory"), "value"));
        Assert.Contains("not measured", GetString(measurement.GetProperty("memory"), "reason"), StringComparison.OrdinalIgnoreCase);

        JsonElement output = root.GetProperty("output");
        Assert.Equal("recorded", GetString(output, "status"));
        Assert.Equal("passed", GetString(output, "validationOpenStatus"));
        Assert.Equal("outsideCheckpointDuration", GetString(output, "scanTimingScope"));
        Assert.True(output.GetProperty("totalBytes").GetInt64() > 0);
        Assert.True(output.GetProperty("bytesPerLiveVector").GetDouble() > 0);

        Assert.Equal("smoke", GetString(root.GetProperty("evidence"), "status"));
        Assert.Equal("private-raw", GetString(root, "privacyClass"));
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    private static void AssertMeasuredPhaseSet(JsonElement phases)
    {
        AssertMeasuredPhase(phases.GetProperty("liveSnapshot"));
        AssertMeasuredPhase(phases.GetProperty("rebuildBuild"));
        AssertMeasuredPhase(phases.GetProperty("save"));
        AssertMeasuredPhase(phases.GetProperty("openValidation"));
        AssertMeasuredPhase(phases.GetProperty("publication"));
    }

    private static void AssertMeasuredPhase(JsonElement phase)
    {
        Assert.Equal("Measured", GetString(phase, "status"));
        Assert.True(phase.GetProperty("elapsedTicks").GetInt64() >= 0);
        Assert.True(phase.GetProperty("elapsedMilliseconds").GetDouble() >= 0);
        Assert.True(phase.GetProperty("managedAllocatedBytes").GetInt64() >= 0);
        Assert.Contains("VEC-133", GetString(phase, "source"), StringComparison.Ordinal);
    }

    private static void AssertNotExecutedPhaseSet(JsonElement phases)
    {
        AssertNotExecutedPhase(phases.GetProperty("liveSnapshot"));
        AssertNotExecutedPhase(phases.GetProperty("rebuildBuild"));
        AssertNotExecutedPhase(phases.GetProperty("save"));
        AssertNotExecutedPhase(phases.GetProperty("openValidation"));
        AssertNotExecutedPhase(phases.GetProperty("publication"));
    }

    private static void AssertNotExecutedPhase(JsonElement phase)
    {
        Assert.Equal("NotExecuted", GetString(phase, "status"));
        Assert.Equal(0, phase.GetProperty("elapsedTicks").GetInt64());
        Assert.Equal(0, phase.GetProperty("elapsedMilliseconds").GetDouble());
        Assert.Equal(0, phase.GetProperty("managedAllocatedBytes").GetInt64());
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

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadDataset(
        FashionMnistAdmissionResult admission,
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport report)
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

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 31)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 1)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 67)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 4)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 17 + column * 23 + offset + (row % 3) * 11) % 251);
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
        string directory = NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName));
        return Path.Combine(directory, fileName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec138-independent-{prefix}-{Guid.NewGuid():N}"));
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
