using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec134HnswBasePlusExactDeltaCheckpointIndependentTests
{
    [Fact]
    public void ParserDefaultsMatchAcceptedVec132SmokeShape()
    {
        HnswBasePlusExactDeltaCheckpointOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaCheckpoint(
                [HnswBasePlusExactDeltaCheckpointOptions.ScenarioName]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(1_024, options.BaseVectorCount);
        Assert.Equal(128, options.InsertedDeltaCount);
        Assert.Equal(128, options.DeletedBaseCount);
        Assert.Equal(16, options.DeletedDeltaCount);
        Assert.Equal(16, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED2132u, options.Seed);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(128, options.EfSearch);
        Assert.Equal(0x484E535700013200UL, options.HnswSeed);
        Assert.Equal(1_152, options.PhysicalVectorCount);
        Assert.Equal(1_008, options.LiveVectorCount);
        Assert.False(Path.IsPathFullyQualified(options.OutputPath));
        Assert.False(Path.IsPathFullyQualified(options.CheckpointDirectory));
    }

    [Fact]
    public void ProgramCommandWritesRepeatedCheckpointReportWithCoherentFinalRunEvidence()
    {
        string root = NewArtifactDirectory("program-runs");
        string outputPath = Path.Combine(root, "checkpoint-report.json");
        string checkpointRoot = Path.Combine(root, "checkpoint-output");
        string[] args =
        [
            HnswBasePlusExactDeltaCheckpointOptions.ScenarioName,
            "--dimension", "5",
            "--vectors", "14",
            "--queries", "3",
            "--top-k", "2",
            "--insertions", "4",
            "--deletes", "3",
            "--delta-deletes", "1",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "2",
            "--runs", "3",
            "--warmup-queries", "1",
            "--seed", "0x5EED134B",
            "--m", "2",
            "--ef-construction", "4",
            "--ef-search", "4",
            "--hnsw-seed", "0x134B",
            "--output", outputPath,
            "--checkpoint-directory", checkpointRoot
        ];

        int exitCode = BenchmarkRunnerProgram.Run(args);

        Assert.Equal(0, exitCode);
        Assert.True(File.Exists(outputPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement rootElement = document.RootElement;

        Assert.Equal("VecNet.HnswBasePlusExactDeltaCheckpointBenchmarkReport", GetString(rootElement, "schemaName"));
        Assert.Equal("0.1", GetString(rootElement, "schemaVersion"));
        Assert.Equal("VEC-134", GetString(rootElement, "taskId"));
        Assert.Equal(HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, GetString(rootElement, "scenarioName"));
        Assert.Equal(HnswBasePlusExactDeltaCheckpointOptions.ScenarioName, GetString(rootElement.GetProperty("command"), "scenario"));
        Assert.Equal("private-raw", GetString(rootElement, "privacyClass"));
        Assert.Equal("local-evidence", GetString(rootElement, "claimClass"));

        JsonElement workload = rootElement.GetProperty("workload");
        Assert.Equal(14, workload.GetProperty("baseVectorCount").GetInt32());
        Assert.Equal(4, workload.GetProperty("insertedDeltaVectorCount").GetInt32());
        Assert.Equal(3, workload.GetProperty("deletedBaseVectorCount").GetInt32());
        Assert.Equal(1, workload.GetProperty("deletedDeltaVectorCount").GetInt32());
        Assert.Equal(3, workload.GetProperty("runCount").GetInt32());
        Assert.Equal(1, workload.GetProperty("warmupQueryCount").GetInt32());
        Assert.Equal("0x5EED134B", GetString(workload, "seed"));
        Assert.Contains("fresh ignored subdirectory", GetString(workload, "checkpointDirectoryPolicy"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("deleted IDs remain reserved", GetString(workload, "idPolicy"), StringComparison.OrdinalIgnoreCase);

        JsonElement hnsw = rootElement.GetProperty("hnsw");
        Assert.Equal(2, hnsw.GetProperty("m").GetInt32());
        Assert.Equal(4, hnsw.GetProperty("efConstruction").GetInt32());
        Assert.Equal(4, hnsw.GetProperty("efSearch").GetInt32());
        Assert.Equal("0x000000000000134B", GetString(hnsw, "randomSeed"));

        JsonElement preCounts = rootElement.GetProperty("preCheckpointCounts");
        JsonElement postCounts = rootElement.GetProperty("postCheckpointCounts");
        Assert.Equal(14, preCounts.GetProperty("basePhysicalVectorCount").GetInt32());
        Assert.Equal(11, preCounts.GetProperty("baseLiveVectorCount").GetInt32());
        Assert.Equal(4, preCounts.GetProperty("deltaPhysicalVectorCount").GetInt32());
        Assert.Equal(3, preCounts.GetProperty("deltaLiveVectorCount").GetInt32());
        Assert.Equal(4, preCounts.GetProperty("tombstoneCount").GetInt32());
        Assert.Equal(14, preCounts.GetProperty("liveVectorCount").GetInt32());
        Assert.Equal(4, preCounts.GetProperty("deletedReservedIdCount").GetInt32());
        Assert.Equal(0, postCounts.GetProperty("deltaPhysicalVectorCount").GetInt32());
        Assert.Equal(0, postCounts.GetProperty("tombstoneCount").GetInt32());
        Assert.Equal(preCounts.GetProperty("liveVectorCount").GetInt32(), postCounts.GetProperty("liveVectorCount").GetInt32());
        Assert.Equal(preCounts.GetProperty("deletedReservedIdCount").GetInt32(), postCounts.GetProperty("deletedReservedIdCount").GetInt32());

        AssertRepeatedCheckpointRuns(rootElement, checkpointRoot);
        AssertCheckpointDetailedValidationUsesFinalRun(rootElement);
        AssertNoChangesProbe(rootElement.GetProperty("noChangesProbe"));
        AssertOutputAndOpenedValidation(rootElement);
        AssertSearchSectionsRemainSeparated(rootElement);
        AssertPrivateSmokePosture(rootElement);
        AssertNoBooleanPropertyTrueForNames(
            rootElement,
            "publicClaimEligible",
            "baselineCandidateEligible",
            "comparisonArtifactEligible",
            "regressionGateEligible");
    }

    private static void AssertRepeatedCheckpointRuns(JsonElement rootElement, string checkpointRoot)
    {
        JsonElement checkpointRuns = rootElement.GetProperty("checkpointRuns");
        JsonElement runs = checkpointRuns.GetProperty("runs");
        Assert.Equal(3, checkpointRuns.GetProperty("runCount").GetInt32());
        Assert.Equal(3, checkpointRuns.GetProperty("detailedValidationRunNumber").GetInt32());
        Assert.Equal(3, runs.GetArrayLength());
        Assert.Contains("final checkpoint run", GetString(checkpointRuns, "detailedValidationPolicy"), StringComparison.OrdinalIgnoreCase);

        var elapsed = new List<double>();
        var allocations = new List<long>();
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
            Assert.True(File.Exists(Path.Combine(directory, "hnsw.ids.u64")));
            Assert.True(File.Exists(Path.Combine(directory, "hnsw.vectors.f32")));
            Assert.True(File.Exists(Path.Combine(directory, "hnsw.levels.i32")));
            Assert.True(File.Exists(Path.Combine(directory, "hnsw.graph.bin")));
            AssertMeasuredPhaseSet(run.GetProperty("phases"));

            elapsed.Add(run.GetProperty("elapsedMilliseconds").GetDouble());
            allocations.Add(run.GetProperty("managedAllocatedBytes").GetInt64());
            expectedRunNumber++;
        }

        JsonElement aggregate = checkpointRuns.GetProperty("aggregate");
        Assert.Equal(3, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(elapsed.Average(), aggregate.GetProperty("meanElapsedMilliseconds").GetDouble(), precision: 8);
        Assert.Equal(elapsed.Min(), aggregate.GetProperty("minElapsedMilliseconds").GetDouble(), precision: 8);
        Assert.Equal(elapsed.Max(), aggregate.GetProperty("maxElapsedMilliseconds").GetDouble(), precision: 8);
        Assert.Equal(allocations.Average(), aggregate.GetProperty("meanManagedAllocatedBytes").GetDouble(), precision: 8);
        Assert.Equal(allocations.Min(), aggregate.GetProperty("minManagedAllocatedBytes").GetInt64());
        Assert.Equal(allocations.Max(), aggregate.GetProperty("maxManagedAllocatedBytes").GetInt64());
        Assert.Contains("independently rebuilt equivalent checkpoint attempts", GetString(aggregate, "aggregateSemantics"), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertCheckpointDetailedValidationUsesFinalRun(JsonElement rootElement)
    {
        JsonElement checkpointRuns = rootElement.GetProperty("checkpointRuns");
        JsonElement finalRun = checkpointRuns.GetProperty("runs")[2];
        JsonElement checkpoint = rootElement.GetProperty("checkpoint");
        JsonElement validation = rootElement.GetProperty("validation");
        JsonElement measurement = rootElement.GetProperty("measurement");

        Assert.Equal("Published", GetString(checkpoint, "status"));
        Assert.Equal(GetString(finalRun, "status"), GetString(checkpoint, "status"));
        Assert.Equal(finalRun.GetProperty("generationBeforeCheckpoint").GetInt64(), checkpoint.GetProperty("generationBeforeCheckpoint").GetInt64());
        Assert.Equal(finalRun.GetProperty("generationAfterCheckpoint").GetInt64(), checkpoint.GetProperty("generationAfterCheckpoint").GetInt64());
        Assert.Equal(finalRun.GetProperty("managedAllocatedBytes").GetInt64(), checkpoint.GetProperty("managedAllocatedBytes").GetInt64());
        Assert.Equal(finalRun.GetProperty("elapsedMilliseconds").GetDouble(), checkpoint.GetProperty("elapsedMilliseconds").GetDouble(), precision: 8);
        AssertMeasuredPhaseSet(checkpoint.GetProperty("phases"));
        AssertMeasuredPhaseSet(measurement.GetProperty("phaseDiagnostics"));
        Assert.Equal("measured", GetString(measurement.GetProperty("checkpointLatency"), "status"));
        Assert.Equal("perCheckpointCall", GetString(measurement.GetProperty("checkpointLatency"), "sampleScope"));
        Assert.Contains("VEC-133", GetString(measurement.GetProperty("checkpointLatency"), "percentileEstimator"), StringComparison.Ordinal);
        Assert.Equal("measured", GetString(measurement.GetProperty("checkpointManagedAllocations"), "status"));
        Assert.Equal("bytesPerCheckpointCall", GetString(measurement.GetProperty("checkpointManagedAllocations"), "unit"));
        Assert.Equal(
            checkpointRuns.GetProperty("aggregate").GetProperty("meanManagedAllocatedBytes").GetDouble().ToString(CultureInfo.InvariantCulture),
            GetString(measurement.GetProperty("checkpointManagedAllocations"), "value"));

        Assert.Equal("passed", GetString(validation, "status"));
        Assert.True(validation.GetProperty("checkpointRepeatedRunEvidencePresent").GetBoolean());
        Assert.Equal(3, validation.GetProperty("detailedValidationRunNumber").GetInt32());
        Assert.True(validation.GetProperty("detailedValidationUsesFinalRun").GetBoolean());
        Assert.True(validation.GetProperty("phaseDiagnosticsMeasuredForPublishedCheckpoint").GetBoolean());
        Assert.True(validation.GetProperty("outputBytesScannedOutsideCheckpointDuration").GetBoolean());
    }

    private static void AssertNoChangesProbe(JsonElement noChangesProbe)
    {
        Assert.Equal("passed", GetString(noChangesProbe, "status"));
        Assert.True(noChangesProbe.GetProperty("generationUnchanged").GetBoolean());
        Assert.True(noChangesProbe.GetProperty("outputDirectoryRemainedEmpty").GetBoolean());
        AssertNotExecutedPhaseSet(noChangesProbe.GetProperty("phases"));
    }

    private static void AssertOutputAndOpenedValidation(JsonElement rootElement)
    {
        JsonElement output = rootElement.GetProperty("output");
        JsonElement finalRun = rootElement.GetProperty("checkpointRuns").GetProperty("runs")[2];
        Assert.Equal(GetString(finalRun, "checkpointDirectory"), GetString(output, "directoryPath"));
        Assert.Equal("recorded", GetString(output, "status"));
        Assert.Equal("passed", GetString(output, "validationOpenStatus"));
        Assert.Equal("outsideCheckpointDuration", GetString(output, "scanTimingScope"));
        Assert.True(output.GetProperty("totalBytes").GetInt64() > 0);
        Assert.True(output.GetProperty("bytesPerLiveVector").GetDouble() > 0);

        JsonElement opened = rootElement.GetProperty("openedValidation");
        Assert.Equal("passed", GetString(opened, "status"));
        Assert.Equal(0, opened.GetProperty("idMismatchCount").GetInt32());
        Assert.Equal(0, opened.GetProperty("vectorMismatchCount").GetInt32());
        Assert.Equal(opened.GetProperty("expectedVectorCount").GetInt32(), opened.GetProperty("openedVectorCount").GetInt32());

        JsonElement parity = opened.GetProperty("rebuiltCompositeOpenedSearchParity");
        Assert.True(parity.GetProperty("allResultsMatched").GetBoolean());
        Assert.Equal(0, parity.GetProperty("writtenCountMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("idMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("orderMismatchCount").GetInt32());
        Assert.Equal(0, parity.GetProperty("distanceMismatchCount").GetInt32());
    }

    private static void AssertSearchSectionsRemainSeparated(JsonElement rootElement)
    {
        JsonElement searches = rootElement.GetProperty("searches");
        JsonElement pre = searches.GetProperty("preCheckpointComposite");
        JsonElement post = searches.GetProperty("postCheckpointRebuiltComposite");
        JsonElement opened = searches.GetProperty("openedReadOnlyHnsw");

        AssertSearchSection(pre, "preCheckpointComposite", "pre-checkpoint");
        AssertSearchSection(post, "postCheckpointRebuiltComposite", "post-checkpoint");
        AssertSearchSection(opened, "openedReadOnlyHnsw", "opened read-only");

        Assert.NotEqual(GetString(pre, "timedOperation"), GetString(post, "timedOperation"));
        Assert.NotEqual(GetString(post, "timedOperation"), GetString(opened, "timedOperation"));
        Assert.DoesNotContain("CheckpointWithDiagnostics", GetString(pre, "timedOperation"), StringComparison.Ordinal);
        Assert.DoesNotContain("CheckpointWithDiagnostics", GetString(post, "timedOperation"), StringComparison.Ordinal);
        Assert.DoesNotContain("CheckpointWithDiagnostics", GetString(opened, "timedOperation"), StringComparison.Ordinal);

        JsonElement validation = rootElement.GetProperty("validation");
        Assert.True(validation.GetProperty("preCheckpointCompositeComparedToTruth").GetBoolean());
        Assert.True(validation.GetProperty("postCheckpointRebuiltCompositeComparedToTruth").GetBoolean());
        Assert.True(validation.GetProperty("openedReadOnlyHnswComparedToTruth").GetBoolean());
        Assert.True(validation.GetProperty("rebuiltCompositeOpenedHnswSearchParityPassed").GetBoolean());
        Assert.True(validation.GetProperty("returnedResultIntegrityPassedForAllSearches").GetBoolean());
        Assert.True(validation.GetProperty("noChangesCheckpointProbePassed").GetBoolean());
        Assert.True(validation.GetProperty("deletedReservedIdsRejectedAfterCheckpoint").GetBoolean());
    }

    private static void AssertSearchSection(JsonElement section, string name, string operationFragment)
    {
        Assert.Equal(name, GetString(section, "name"));
        Assert.Contains(operationFragment, GetString(section, "timedOperation"), StringComparison.OrdinalIgnoreCase);

        JsonElement search = section.GetProperty("search");
        JsonElement aggregate = search.GetProperty("aggregate");
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(3, search.GetProperty("runs").GetArrayLength());
        Assert.Equal(3, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(3, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());

        JsonElement measurement = section.GetProperty("measurement");
        Assert.Equal("measured", GetString(measurement.GetProperty("latency"), "status"));
        Assert.Equal("perMeasuredSearchCall", GetString(measurement.GetProperty("latency"), "sampleScope"));
        Assert.Equal(GetString(section, "timedOperation"), GetString(measurement.GetProperty("latency"), "timedOperation"));
        Assert.Contains("checkpoint call", GetString(measurement.GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", GetString(measurement.GetProperty("managedAllocations"), "status"));
        Assert.Equal("bytesPerSearchCall", GetString(measurement.GetProperty("managedAllocations"), "unit"));
        Assert.Equal("notMeasured", GetString(measurement.GetProperty("memory"), "status"));
        Assert.Equal("measured", GetString(measurement.GetProperty("repeatedRuns"), "status"));
        Assert.True(measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("measured", GetString(measurement.GetProperty("runToRunNoise"), "status"));
        Assert.Equal("executed", GetString(measurement.GetProperty("warmup"), "status"));

        JsonElement metrics = section.GetProperty("metrics");
        Assert.InRange(metrics.GetProperty("recallAtK").GetDouble(), 0, 1);
        Assert.InRange(metrics.GetProperty("orderedAgreement").GetDouble(), 0, 1);
        Assert.Equal("passed", GetString(metrics, "distanceToleranceStatus"));
        Assert.Equal("passed", GetString(metrics.GetProperty("returnedResultIntegrity"), "status"));

        JsonElement underfill = section.GetProperty("underfill");
        Assert.Equal(3, underfill.GetProperty("queryCount").GetInt32());
        Assert.Equal(2, underfill.GetProperty("requestedResultCountPerQuery").GetInt32());
        Assert.Equal(6, underfill.GetProperty("totalRequestedResultSlots").GetInt32());
        Assert.Equal(
            underfill.GetProperty("totalRequestedResultSlots").GetInt32() - underfill.GetProperty("totalReturnedResults").GetInt32(),
            underfill.GetProperty("underfilledSlotCount").GetInt32());
    }

    private static void AssertPrivateSmokePosture(JsonElement rootElement)
    {
        Assert.Equal("smoke", GetString(rootElement.GetProperty("evidence"), "status"));
        Assert.Equal("private-raw", GetString(rootElement, "privacyClass"));
        Assert.Equal("notMeasured", GetString(rootElement.GetProperty("measurement").GetProperty("memory"), "status"));
        Assert.Equal("absent", GetString(rootElement.GetProperty("measurement").GetProperty("memory"), "value"));
        Assert.False(rootElement.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("validation").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        Assert.True(rootElement.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
        Assert.False(rootElement.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(rootElement.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
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

    private static string GetString(JsonElement element, string propertyName) =>
        element.GetProperty(propertyName).GetString() ?? string.Empty;

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"test-agent-vec134-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

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
