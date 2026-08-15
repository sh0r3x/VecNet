using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec142FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeIndependentTests
{
    [Fact]
    public void Parser_DefaultPathsStayPrivateIgnoredArtifactPathsAndAcceptCaseInsensitiveAllowedOptions()
    {
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions defaults =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName]);

        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", defaults.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", defaults.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("vec-142-memory-smoke", defaults.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("checkpoint-output", defaults.CheckpointDirectory, StringComparison.OrdinalIgnoreCase);

        string root = NewArtifactDirectory("parser");
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions parsed =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-BASE-PLUS-EXACT-DELTA-CHECKPOINT-MEMORY-SMOKE",
                    "--CACHE-ROOT", "CacheRoot",
                    "--OUTPUT", Path.Combine(root, "report.json"),
                    "--CHECKPOINT-DIRECTORY", Path.Combine(root, "checkpoint-output"),
                    "--SAMPLE-INTERVAL-MS", "1000",
                    "--METRIC", "COSINE"
                ]);

        Assert.Equal("CacheRoot", parsed.CacheRoot);
        Assert.Equal(Path.Combine(root, "report.json"), parsed.OutputPath);
        Assert.Equal(Path.Combine(root, "checkpoint-output"), parsed.CheckpointDirectory);
        Assert.Equal(1000, parsed.SampleIntervalMilliseconds);
        Assert.Equal(1, FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.CheckpointRunCount);
        Assert.Equal(50, parsed.QueryCount);
        Assert.Equal(100, parsed.TopK);
        Assert.Equal(58_000, parsed.BaseVectorCount);
        Assert.Equal(1_000, parsed.InsertedDeltaCount);
        Assert.Equal(57_900, parsed.LiveVectorCount);
        Assert.Equal(1_100, parsed.DeletedReservedIdCount);
        Assert.Equal(3, parsed.WarmupQueries);
        Assert.Equal(VectorMetric.Cosine, parsed.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(parsed.Metric));
        Assert.Equal(16, parsed.M);
        Assert.Equal(128, parsed.EfConstruction);
        Assert.Equal(192, parsed.EfSearch);
    }

    [Theory]
    [InlineData("--dimension", "784")]
    [InlineData("--query-count", "1")]
    [InlineData("--queries", "1")]
    [InlineData("--top-k", "1")]
    [InlineData("--vectors", "60000")]
    [InlineData("--base-vectors", "58000")]
    [InlineData("--insertions", "1000")]
    [InlineData("--deletes", "1000")]
    [InlineData("--delta-deletes", "100")]
    [InlineData("--duplicate-inserts", "1")]
    [InlineData("--unknown-deletes", "1")]
    [InlineData("--repeated-deletes", "1")]
    [InlineData("--runs", "2")]
    [InlineData("--checkpoint-runs", "2")]
    [InlineData("--warmup-queries", "3")]
    [InlineData("--seed", "0x5EED2141")]
    [InlineData("--m", "16")]
    [InlineData("--ef-construction", "128")]
    [InlineData("--ef-search", "192")]
    [InlineData("--hnsw-seed", "0x484E535700014100")]
    [InlineData("--preset", "smoke")]
    [InlineData("--manifest", "manifest.json")]
    [InlineData("--output-dir", "matrix")]
    [InlineData("--download", "false")]
    [InlineData("--truth-refresh", "true")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--comparison", "true")]
    [InlineData("--comparison-artifact", "true")]
    [InlineData("--regression-gate", "true")]
    [InlineData("--public-claim", "true")]
    [InlineData("--preview-readiness", "true")]
    [InlineData("--package", "true")]
    [InlineData("--platform", "linux-x64")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--faiss-index", "index.bin")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--concurrency", "2")]
    [InlineData("--storage-size", "true")]
    public void Parser_RejectsOutOfScopeOptionFamilies(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(option, exception.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--sample-interval-ms", "not-a-number")]
    [InlineData("--sample-interval-ms", "-1")]
    [InlineData("--sample-interval-ms", "0")]
    [InlineData("--sample-interval-ms", "1001")]
    [InlineData("--cache-root", "")]
    [InlineData("--output", "")]
    [InlineData("--checkpoint-directory", "")]
    public void Parser_RejectsMalformedAllowedOptions(string option, string value)
    {
        Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, option, value]));
    }

    [Fact]
    public void InvalidTruthGuard_FailsClosedBeforeReportOrCheckpointOutput()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("invalid-truth", baseCount: 24, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string reportPath = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "memory-smoke", "report.json");
        string checkpointDirectory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "memory-smoke", "checkpoint-output");
        File.WriteAllText(admission.TruthPath, """{"schemaName":"corrupt","schemaVersion":"0.1"}""");

        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions(
            cacheRoot,
            reportPath,
            checkpointDirectory,
            SampleIntervalMilliseconds: 1,
            QueryCount: 2,
            TopK: 2,
            BaseVectorCount: 14,
            InsertedDeltaCount: 3,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1421,
            M: 2,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x1421);

        Assert.ThrowsAny<Exception>(() =>
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Run(
                options,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName]));
        Assert.False(File.Exists(reportPath));
        Assert.False(Directory.Exists(checkpointDirectory));
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_JsonSeparatesMemoryPeaksLowerBoundsStorageAndEligibility()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("json-report", baseCount: 52, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputRoot = NewArtifactDirectory("json-output");
        string reportPath = Path.Combine(outputRoot, "fashion-mnist-external-hnsw-base-plus-exact-delta-checkpoint-memory-smoke.json");
        string checkpointRoot = Path.Combine(outputRoot, "checkpoint-output");
        var options = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions(
            cacheRoot,
            reportPath,
            checkpointRoot,
            SampleIntervalMilliseconds: 1,
            QueryCount: 4,
            TopK: 5,
            BaseVectorCount: 38,
            InsertedDeltaCount: 7,
            DeletedBaseCount: 4,
            DeletedDeltaCount: 2,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            WarmupQueries: 1,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1422,
            M: 2,
            EfConstruction: 8,
            EfSearch: 5,
            HnswSeed: 0x1422);

        ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport report =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Run(
                options,
                [
                    FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName,
                    "--cache-root", cacheRoot,
                    "--output", reportPath,
                    "--checkpoint-directory", checkpointRoot,
                    "--sample-interval-ms", "1"
                ]);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeScenario.Write(report, reportPath);

        Assert.True(File.Exists(reportPath));
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(reportPath));
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaCheckpointMemorySmokeReport", GetString(root, "schemaName"));
        Assert.Equal("0.1", GetString(root, "schemaVersion"));
        Assert.Equal("VEC-142", GetString(root, "taskId"));
        Assert.Equal("private-raw", GetString(root, "privacyClass"));
        Assert.Equal("local-evidence", GetString(root, "claimClass"));
        Assert.Equal(FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, GetString(root.GetProperty("command"), "scenario"));

        JsonElement workload = root.GetProperty("workload");
        Assert.Equal(1, workload.GetProperty("checkpointRunCount").GetInt32());
        Assert.Equal(1, workload.GetProperty("sampleIntervalMilliseconds").GetInt32());
        Assert.Equal(38, workload.GetProperty("immutableBaseRowCount").GetInt32());
        Assert.Equal(7, workload.GetProperty("deltaRowCount").GetInt32());
        Assert.Equal(39, workload.GetProperty("expectedLiveVectorCount").GetInt32());
        Assert.Equal(6, workload.GetProperty("expectedDeletedReservedIdCount").GetInt32());
        Assert.Contains("deleted IDs remain reserved", GetString(workload, "externalIdPolicy"), StringComparison.OrdinalIgnoreCase);

        Assert.Equal(39, root.GetProperty("updatedTruth").GetProperty("liveVectorCount").GetInt32());
        Assert.False(root.GetProperty("updatedTruth").GetProperty("persisted").GetBoolean());
        Assert.Contains("post-update live view", GetString(root.GetProperty("updatedTruth"), "source"), StringComparison.OrdinalIgnoreCase);

        AssertMeasuredPhases(root.GetProperty("measuredPhases"));
        AssertActualMemorySchema(root.GetProperty("actualMemory"));
        AssertPeakMemorySchema(root.GetProperty("peakMemory"), sampleIntervalMilliseconds: 1);
        AssertLayoutLowerBounds(root.GetProperty("layoutLowerBounds"), options);
        AssertStorageOutput(root.GetProperty("storageOutput"), root.GetProperty("checkpointOutput"));
        AssertValidationBooleans(root.GetProperty("validation"));

        Assert.Equal("passed", GetString(root.GetProperty("openedValidation"), "status"));
        Assert.True(root.GetProperty("openedValidation").GetProperty("rebuiltCompositeOpenedSearchParity").GetProperty("allResultsMatched").GetBoolean());
        Assert.Equal("passed", GetString(root.GetProperty("noChangesProbe"), "status"));
        Assert.Equal("Published", GetString(root.GetProperty("checkpoint"), "status"));
        Assert.Equal("Published", GetString(root.GetProperty("checkpointResult"), "status"));

        AssertNoBooleanPropertyTrueForNames(
            root,
            "publicClaimEligible",
            "previewReadinessEligible",
            "baselineCandidateEligible",
            "comparisonArtifactEligible",
            "comparisonPublicationEligible",
            "regressionGateEligible");
        AssertNoPropertyNamed(root, "candidateEligibility", "regressionDecision", "publicClaimStatus", "baselineComparison", "downloadRawFiles", "truthRefresh");
    }

    private static void AssertMeasuredPhases(JsonElement phases)
    {
        foreach (string phaseName in new[]
        {
            "cacheTruthLoad",
            "immutableHnswBaseBuild",
            "compositeCreationAndExactDeltaTombstoneMutation",
            "exactUpdatedTruthGeneration",
            "preCheckpointSourceCompositeSearch",
            "checkpointPublication",
            "openedReadOnlyHnswOpen",
            "postCheckpointRebuiltCompositeSearch",
            "openedReadOnlyHnswSearch",
            "finalValidation"
        })
        {
            JsonElement phase = phases.GetProperty(char.ToLowerInvariant(phaseName[0]) + phaseName[1..]);
            Assert.Equal(phaseName, GetString(phase, "name"));
            Assert.Equal("measured", GetString(phase, "status"));
            Assert.True(phase.GetProperty("elapsedMilliseconds").GetDouble() >= 0);
            Assert.True(phase.GetProperty("managedAllocatedBytes").GetInt64() >= 0);
            Assert.Contains("sampled", GetString(phase, "memorySampling"), StringComparison.OrdinalIgnoreCase);
        }

        JsonElement diagnostics = phases.GetProperty("checkpointPhaseDiagnostics");
        foreach (string diagnosticName in new[] { "liveSnapshot", "rebuildBuild", "save", "openValidation", "publication" })
        {
            JsonElement diagnostic = diagnostics.GetProperty(diagnosticName);
            Assert.Equal("Measured", GetString(diagnostic, "status"));
            Assert.Contains("VEC-133", GetString(diagnostic, "source"), StringComparison.Ordinal);
        }
    }

    private static void AssertActualMemorySchema(JsonElement actualMemory)
    {
        Assert.Equal("measured", GetString(actualMemory, "status"));
        Assert.Equal("wholeProcessBoundarySamples", GetString(actualMemory, "scope"));
        Assert.Contains("whole-process", GetString(actualMemory, "claimBoundary"), StringComparison.OrdinalIgnoreCase);

        foreach (string sampleName in new[]
        {
            "baselineProcess",
            "postCacheTruthLoad",
            "postImmutableHnswBaseBuild",
            "postCompositeMutation",
            "postExactUpdatedTruthGeneration",
            "postPreCheckpointSearch",
            "postCheckpointPublication",
            "postOpenedReadOnlyHnswOpen",
            "postPostCheckpointRebuiltCompositeSearch",
            "postOpenedReadOnlyHnswSearch",
            "postFinalValidation"
        })
        {
            AssertMemorySample(actualMemory.GetProperty(sampleName));
        }

        JsonElement unsupported = actualMemory.GetProperty("unsupported");
        AssertStatus(unsupported.GetProperty("objectAccurateIdMapRetainedMemory"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("objectAccurateGraphLayerObjectMemory"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("objectAccurateTombstoneSetMemory"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("objectAccurateDeletedReservationSetMemory"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("objectHeadersArrayHeadersAlignmentAndSlack"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("neighborCandidateRetainedLayout"), "notAvailable", "absent");
        AssertStatus(unsupported.GetProperty("indexOnlyPrivateBytes"), "notMeasured", "absent");
        AssertStatus(unsupported.GetProperty("sourceCompositeOnlyRetainedMemory"), "notMeasured", "absent");
        AssertStatus(unsupported.GetProperty("openedOnlyRetainedMemory"), "notMeasured", "absent");
        AssertStatus(unsupported.GetProperty("trueProcessPeakMemory"), "notMeasured", "absent");
        AssertStatus(unsupported.GetProperty("peakTemporaryDisk"), "notMeasured", "absent");
    }

    private static void AssertMemorySample(JsonElement sample)
    {
        Assert.NotEmpty(GetString(sample, "name"));
        Assert.NotEmpty(GetString(sample, "boundary"));
        foreach (string metricName in new[]
        {
            "managedHeapSizeBytes",
            "gcCommittedBytes",
            "gcFragmentedBytes",
            "processPrivateBytes",
            "processWorkingSetBytes",
            "processPeakWorkingSetBytes"
        })
        {
            JsonElement metric = sample.GetProperty(metricName);
            Assert.Equal("measured", GetString(metric, "status"));
            Assert.Equal("bytes", GetString(metric, "unit"));
            Assert.True(metric.GetProperty("valueBytes").GetInt64() >= 0);
            Assert.NotEmpty(GetString(metric, "reason"));
        }

        Assert.False(sample.GetProperty("processPrivateBytes").GetProperty("contextOnly").GetBoolean());
        Assert.True(sample.GetProperty("processWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
        Assert.True(sample.GetProperty("processPeakWorkingSetBytes").GetProperty("contextOnly").GetBoolean());
    }

    private static void AssertPeakMemorySchema(JsonElement peakMemory, int sampleIntervalMilliseconds)
    {
        Assert.Equal("sampled", GetString(peakMemory, "status"));
        Assert.Equal("observedSampledWholeProcessPeaks", GetString(peakMemory, "scope"));
        Assert.Contains("true maximum", GetString(peakMemory, "claimBoundary"), StringComparison.OrdinalIgnoreCase);

        foreach (string operationName in new[]
        {
            "cacheTruthLoad",
            "immutableHnswBaseBuild",
            "compositeCreationAndMutation",
            "exactUpdatedTruthGeneration",
            "preCheckpointSourceCompositeSearch",
            "checkpointPublication",
            "openedReadOnlyHnswOpen",
            "postCheckpointRebuiltCompositeSearch",
            "openedReadOnlyHnswSearch",
            "finalValidation"
        })
        {
            JsonElement operation = peakMemory.GetProperty(operationName);
            Assert.Equal("sampled", GetString(operation, "status"));
            Assert.Equal(sampleIntervalMilliseconds, operation.GetProperty("sampleIntervalMilliseconds").GetInt32());
            Assert.True(operation.GetProperty("sampleCount").GetInt32() >= 2);
            AssertMemorySample(operation.GetProperty("startSample"));
            AssertMemorySample(operation.GetProperty("endSample"));
            AssertPeakMetric(operation.GetProperty("peakObservedManagedHeapSizeBytes"), contextOnly: false);
            AssertPeakMetric(operation.GetProperty("peakObservedGcCommittedBytes"), contextOnly: false);
            AssertPeakMetric(operation.GetProperty("peakObservedPrivateBytes"), contextOnly: false);
            AssertPeakMetric(operation.GetProperty("peakObservedWorkingSetBytes"), contextOnly: true);
            Assert.Contains("miss", GetString(operation, "missedShortPeakCaveat"), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("whole-process", GetString(operation, "wholeProcessCaveat"), StringComparison.OrdinalIgnoreCase);
            Assert.NotEmpty(GetString(operation, "timedScope"));
            Assert.NotEmpty(GetString(operation, "excludedOperations"));
        }

        AssertStatus(peakMemory.GetProperty("peakTemporaryDiskBytes"), "notMeasured", "absent");
        AssertStatus(peakMemory.GetProperty("trueProcessPeakMemory"), "notMeasured", "absent");
    }

    private static void AssertPeakMetric(JsonElement metric, bool contextOnly)
    {
        Assert.Equal("sampled", GetString(metric, "status"));
        Assert.Equal("bytes", GetString(metric, "unit"));
        Assert.True(metric.GetProperty("valueBytes").GetInt64() >= 0);
        Assert.Equal(contextOnly, metric.GetProperty("contextOnly").GetBoolean());
        Assert.Contains("observed sampled peak", GetString(metric, "reason"), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertLayoutLowerBounds(
        JsonElement lowerBounds,
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions options)
    {
        Assert.Equal("estimatedLowerBound", GetString(lowerBounds, "status"));
        Assert.Contains("payload-only", GetString(lowerBounds, "claimBoundary"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(15, lowerBounds.GetProperty("dimension").GetInt32());
        Assert.Equal(options.BaseVectorCount, lowerBounds.GetProperty("sourceBasePhysicalVectorCount").GetInt32());
        Assert.Equal(options.InsertedDeltaCount, lowerBounds.GetProperty("sourceDeltaPhysicalVectorCount").GetInt32());
        Assert.Equal(options.LiveVectorCount, lowerBounds.GetProperty("sourceLiveVectorCount").GetInt32());
        Assert.Equal(options.LiveVectorCount, lowerBounds.GetProperty("rebuiltOpenedVectorCount").GetInt32());
        Assert.Equal((long)options.BaseVectorCount * 15L * sizeof(float), lowerBounds.GetProperty("sourceBaseVectorPayloadLowerBoundBytes").GetInt64());
        Assert.Equal((long)options.InsertedDeltaCount * 15L * sizeof(float), lowerBounds.GetProperty("sourceDeltaVectorPayloadLowerBoundBytes").GetInt64());
        Assert.Equal((long)options.DeletedBaseCount * sizeof(ulong), lowerBounds.GetProperty("baseTombstoneIdPayloadLowerBoundBytes").GetInt64());
        Assert.Equal((long)options.DeletedDeltaCount * sizeof(ulong), lowerBounds.GetProperty("deltaTombstoneIdPayloadLowerBoundBytes").GetInt64());
        Assert.Equal((long)options.DeletedReservedIdCount * sizeof(ulong), lowerBounds.GetProperty("deletedReservedIdPayloadLowerBoundBytes").GetInt64());
        AssertStatus(lowerBounds.GetProperty("compositeSearchWorkspacePayloadLowerBoundBytes"), "estimatedLowerBound", null);
        AssertStatus(lowerBounds.GetProperty("openedSearchWorkspacePayloadLowerBoundBytes"), "estimatedLowerBound", null);
        Assert.True(lowerBounds.GetProperty("sourceBaseLayers").GetArrayLength() > 0);
        Assert.True(lowerBounds.GetProperty("rebuiltOpenedLayers").GetArrayLength() > 0);
        Assert.Contains("sourceCompositeLowerBound", GetString(lowerBounds, "formula"), StringComparison.Ordinal);
        Assert.Contains("rebuiltOpenedLowerBound", GetString(lowerBounds, "formula"), StringComparison.Ordinal);
        Assert.Contains("Excludes", GetString(lowerBounds, "exclusions"), StringComparison.Ordinal);
        Assert.Contains("Fashion-MNIST input arrays", GetString(lowerBounds, "exclusions"), StringComparison.Ordinal);
    }

    private static void AssertStorageOutput(JsonElement storageOutput, JsonElement checkpointOutput)
    {
        Assert.Equal("fileFacts", GetString(storageOutput, "status"));
        Assert.Contains("not memory", GetString(storageOutput, "memoryBoundary"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("outsideCheckpointDuration", GetString(storageOutput, "scanTimingScope"));
        Assert.Equal(GetString(checkpointOutput, "directoryPath"), GetString(storageOutput, "checkpointDirectory"));
        Assert.Equal(checkpointOutput.GetProperty("totalBytes").GetInt64(), storageOutput.GetProperty("totalBytes").GetInt64());
        Assert.Equal(checkpointOutput.GetProperty("manifestBytes").GetInt64(), storageOutput.GetProperty("manifestBytes").GetInt64());
        Assert.Equal(checkpointOutput.GetProperty("idsBytes").GetInt64(), storageOutput.GetProperty("idsBytes").GetInt64());
        Assert.Equal(checkpointOutput.GetProperty("vectorsBytes").GetInt64(), storageOutput.GetProperty("vectorsBytes").GetInt64());
        Assert.Equal(checkpointOutput.GetProperty("levelsBytes").GetInt64(), storageOutput.GetProperty("levelsBytes").GetInt64());
        Assert.Equal(checkpointOutput.GetProperty("graphBytes").GetInt64(), storageOutput.GetProperty("graphBytes").GetInt64());
        Assert.True(storageOutput.GetProperty("totalBytes").GetInt64() > 0);
        AssertStatus(storageOutput.GetProperty("peakObservedOutputDirectoryBytes"), "notMeasured", "absent");
        AssertStatus(storageOutput.GetProperty("peakTemporaryDiskBytes"), "notMeasured", "absent");
    }

    private static void AssertValidationBooleans(JsonElement validation)
    {
        Assert.Equal("passed", GetString(validation, "status"));
        foreach (string propertyName in new[]
        {
            "cacheAndTruthReadinessPassed",
            "existingTruthGuardLoaded",
            "updatedTruthGeneratedFromLiveView",
            "preCheckpointSourceCompositeComparedToTruth",
            "checkpointResultStatusPublished",
            "checkpointResultCountsMatched",
            "checkpointGenerationAdvancedExactlyOnce",
            "phaseDiagnosticsMeasuredForPublishedCheckpoint",
            "checkpointRunCountIsOne",
            "postCheckpointCountsMatched",
            "postCheckpointRebuiltCompositeComparedToTruth",
            "openedReadOnlyHnswOpened",
            "openedReadOnlyHnswIdVectorValidationPassed",
            "openedReadOnlyHnswComparedToTruth",
            "rebuiltCompositeOpenedHnswSearchParityPassed",
            "returnedResultIntegrityPassedForAllSearches",
            "noChangesCheckpointProbePassed",
            "deletedReservedIdsRejectedAfterCheckpoint",
            "actualPeakLowerBoundAndStorageSectionsSeparated",
            "outputBytesAreSeparateFileFacts",
            "unsupportedFieldsExplicitlyMarked",
            "workingSetContextOnly",
            "sampledPeakLabelsPresent",
            "outputBytesScannedOutsideCheckpointDuration",
            "reportIsPrivateRaw"
        })
        {
            Assert.True(validation.GetProperty(propertyName).GetBoolean(), propertyName);
        }

        Assert.False(validation.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(validation.GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(validation.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(validation.GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(validation.GetProperty("regressionGateEligible").GetBoolean());
    }

    private static void AssertStatus(JsonElement element, string expectedStatus, string? expectedValue)
    {
        Assert.Equal(expectedStatus, GetString(element, "status"));
        if (expectedValue is not null)
        {
            Assert.Equal(expectedValue, GetString(element, "value"));
        }

        Assert.NotEmpty(GetString(element, "reason"));
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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 13 + offset + (row % 5) * 17) % 251);
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
            string.Create(CultureInfo.InvariantCulture, $"vec142-independent-{prefix}-{Guid.NewGuid():N}"));
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

    private static void AssertNoPropertyNamed(JsonElement element, params string[] propertyNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(propertyNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                AssertNoPropertyNamed(property.Value, propertyNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, propertyNames);
            }
        }
    }
}
