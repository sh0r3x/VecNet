using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec061GeneratedExactUpdateTests
{
    [Fact]
    public void ParseGeneratedExactUpdate_UsesPrivateSmokeDefaults()
    {
        GeneratedExactUpdateOptions options = CommandLine.ParseGeneratedExactUpdate(["generated-exact-update"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.BaseVectorCount);
        Assert.Equal(11_000, options.PhysicalVectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal("broad", options.AllowlistKind);
        Assert.Equal("selective", options.CandidateSetKind);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2061u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-update", "--dimension")]
    [InlineData("generated-exact-update", "dimension", "8")]
    [InlineData("generated-exact-update", "--metric", "Unknown")]
    [InlineData("generated-exact-update", "--dimension", "0")]
    [InlineData("generated-exact-update", "--vectors", "0")]
    [InlineData("generated-exact-update", "--queries", "0")]
    [InlineData("generated-exact-update", "--top-k", "11", "--vectors", "10", "--insertions", "1", "--deletes", "1")]
    [InlineData("generated-exact-update", "--runs", "0")]
    [InlineData("generated-exact-update", "--runs", "6")]
    [InlineData("generated-exact-update", "--warmup-queries", "-1")]
    [InlineData("generated-exact-update", "--insertions", "0")]
    [InlineData("generated-exact-update", "--deletes", "0")]
    [InlineData("generated-exact-update", "--deletes", "11", "--vectors", "10")]
    [InlineData("generated-exact-update", "--duplicate-inserts", "-1")]
    [InlineData("generated-exact-update", "--unknown-deletes", "-1")]
    [InlineData("generated-exact-update", "--repeated-deletes", "-1")]
    [InlineData("generated-exact-update", "--allowlist", "unknown")]
    [InlineData("generated-exact-update", "--candidate-set", "unknown")]
    [InlineData("generated-exact-update", "--allowlist", "very-selective", "--top-k", "1")]
    [InlineData("generated-exact-update", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-update", "--unknown-ids", "-1")]
    [InlineData("generated-exact-update", "--filter", "broad")]
    [InlineData("generated-exact-update", "--preset", "smoke")]
    [InlineData("generated-exact-update", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-update", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-exact-update", "--m", "8")]
    [InlineData("generated-exact-update", "--output", "")]
    public void ParseGeneratedExactUpdate_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedExactUpdateReportWithCountsMutationsAndMeasuredModes()
    {
        string outputPath = NewArtifactPath("exact-update-report.json");
        string[] arguments =
        [
            "generated-exact-update",
            "--metric", "SquaredEuclidean",
            "--dimension", "13",
            "--vectors", "50",
            "--queries", "6",
            "--top-k", "7",
            "--insertions", "8",
            "--deletes", "5",
            "--duplicate-inserts", "3",
            "--unknown-deletes", "4",
            "--repeated-deletes", "2",
            "--allowlist", "broad",
            "--candidate-set", "selective",
            "--duplicate-ids", "2",
            "--unknown-ids", "3",
            "--runs", "3",
            "--warmup-queries", "4",
            "--seed", "0x5EED061A",
            "--output", outputPath
        ];
        GeneratedExactUpdateOptions options = CommandLine.ParseGeneratedExactUpdate(arguments);

        GeneratedExactUpdateBenchmarkReport report = GeneratedExactUpdateScenario.Run(options, arguments);
        GeneratedExactUpdateScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExactUpdateBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-061", report.TaskId);
        Assert.Equal("generated-exact-update", report.ScenarioName);
        Assert.Equal("generated-exact-update", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-exact-update-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
        Assert.Contains("baseline-candidate policy", report.Eligibility.BaselineCandidateReason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("regression-gate policy", report.Eligibility.RegressionGateReason, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(58, report.Counts.PhysicalVectorCount);
        Assert.Equal(53, report.Counts.LiveVectorCount);
        Assert.Equal(50, report.Counts.BaseVectorCount);
        Assert.Equal(8, report.Counts.DeltaVectorCount);
        Assert.Equal(5, report.Counts.TombstoneCount);
        Assert.Equal(5.0 / 58.0, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCount", report.Counts.TombstoneRatioDenominator);
        Assert.Equal(5, report.Counts.DeletedOrReservedIdCount);
        Assert.Contains("reserved", report.Counts.DeletedOrReservedIdSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("physical stored-row count", report.Counts.VectorCountSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(8, report.Mutations.InsertedCount);
        Assert.Equal(5, report.Mutations.DeletedCount);
        Assert.Equal(3, report.Mutations.DuplicateInsertAttempts);
        Assert.Equal(4, report.Mutations.UnknownDeleteAttempts);
        Assert.Equal(2, report.Mutations.RepeatedDeleteAttempts);
        Assert.Equal(13, report.Mutations.CommittedMutationCount);
        Assert.Equal(50, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(63, report.Mutations.GenerationAfterMutations);
        Assert.Equal(13, report.Mutations.GenerationDelta);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(13, report.Mutations.StatusCounts.Committed);
        Assert.Equal(3, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(4, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);
        Assert.Equal(0, report.Mutations.StatusCounts.ReadOnly);
        Assert.Equal(0, report.Mutations.StatusCounts.Unsupported);

        Assert.Equal("broad", report.RawAllowlistInput.Kind);
        Assert.Equal(27, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Equal(2, report.RawAllowlistInput.DuplicateIdCountPerQuery);
        Assert.Equal(3, report.RawAllowlistInput.UnknownIdCountPerQuery);
        Assert.Contains("post-mutation live view", report.RawAllowlistInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("selective", report.CandidateSetInput.Kind);
        Assert.Equal(6, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.Equal("constructedAfterMutationsOutsideMeasuredSearch", report.CandidateSet.ConstructionStatus);
        Assert.True(report.CandidateSet.ConstructedAfterMutations);
        Assert.True(report.CandidateSet.ConstructedBeforeWarmupAndMeasuredSearch);
        Assert.Equal(6, report.CandidateSet.ConstructedSetCount);
        Assert.Equal(6, report.CandidateSet.CountPerQuery);
        Assert.Equal(36, report.CandidateSet.TotalCandidateCount);

        AssertMeasuredOperation(report.Searches.UnfilteredSearch, "unfilteredSearch", "Search(query, results)");
        AssertMeasuredOperation(report.Searches.RawAllowlistSearch, "rawAllowlistSearch", "Search(query, allowedIds, results, workspace)");
        AssertMeasuredOperation(report.Searches.CandidateSetSearch, "candidateSetSearch", "Search(query, candidateSet, results)");
        Assert.Equal("measured", report.Measurement.UnfilteredSearch.Latency.Status);
        Assert.Equal("bytesPerSearchCall", report.Measurement.UnfilteredSearch.ManagedAllocations.Unit);
        Assert.Equal("measured", report.Measurement.RawAllowlistSearch.Latency.Status);
        Assert.Equal("measured", report.Measurement.CandidateSetSearch.Latency.Status);
        Assert.Equal("notMeasured", report.Measurement.MutationLatencyAndAllocation.Status);
        Assert.Contains("not measured", report.Measurement.MutationLatencyAndAllocation.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Measurement.LiveViewSave.Status);
        Assert.Contains("deferred", report.Measurement.LiveViewSave.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Contains("candidate-set construction", report.Measurement.SharedExcludedOperations, StringComparison.OrdinalIgnoreCase);

        AssertPassedMetrics(report.Metrics.UnfilteredSearch);
        AssertPassedMetrics(report.Metrics.RawAllowlistSearch);
        AssertPassedMetrics(report.Metrics.CandidateSetSearch);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LiveTruthGenerated);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.True(report.Validation.CandidateSetsConstructedAfterMutations);
        Assert.True(report.Validation.FinalRunUnfilteredComparedToTruth);
        Assert.True(report.Validation.FinalRunRawAllowlistComparedToTruth);
        Assert.True(report.Validation.FinalRunCandidateSetComparedToTruth);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);

        Assert.Equal("estimatedPayloadLowerBounds", report.MemoryEstimates.Status);
        Assert.Equal(58L * sizeof(ulong), report.MemoryEstimates.PhysicalIdPayloadLowerBoundBytes);
        Assert.Equal(58L * 13L * sizeof(float), report.MemoryEstimates.PhysicalVectorPayloadLowerBoundBytes);
        Assert.Equal(53L * 13L * sizeof(float), report.MemoryEstimates.LiveVectorPayloadLowerBoundBytes);
        Assert.Equal(36L * sizeof(int), report.MemoryEstimates.CandidateSetOrdinalPayloadLowerBoundBytes);
        Assert.Equal("notAvailable", report.MemoryEstimates.TombstoneDeletedReservationRetainedMemory.Status);
        Assert.Equal("notMeasured", report.MemoryEstimates.ResidentProcessMemory.Status);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactUpdateBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("generated-exact-update", root.GetProperty("scenarioName").GetString());
        Assert.Equal(58, root.GetProperty("counts").GetProperty("physicalVectorCount").GetInt32());
        Assert.Equal(53, root.GetProperty("counts").GetProperty("liveVectorCount").GetInt32());
        Assert.Equal(5, root.GetProperty("counts").GetProperty("tombstoneCount").GetInt32());
        Assert.Equal(8, root.GetProperty("mutations").GetProperty("insertedCount").GetInt32());
        Assert.Equal(13, root.GetProperty("mutations").GetProperty("committedMutationCount").GetInt32());
        Assert.Equal("public ExactFlatIndex.Search(query, results)", root.GetProperty("searches").GetProperty("unfilteredSearch").GetProperty("timedOperation").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", root.GetProperty("searches").GetProperty("rawAllowlistSearch").GetProperty("timedOperation").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", root.GetProperty("searches").GetProperty("candidateSetSearch").GetProperty("timedOperation").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("mutationLatencyAndAllocation").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("liveViewSave").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoPropertyNamed(root, "baseline", "candidateEligibility", "comparisonResult", "regressionDecision", "checkpoint", "rebuild", "vectorData", "sql", "hnswUpdate");
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Run_ValidatesAllMeasuredModesAcrossExactMetrics(VectorMetric metric)
    {
        GeneratedExactUpdateBenchmarkReport report = GeneratedExactUpdateScenario.Run(
            new GeneratedExactUpdateOptions(
                metric,
                Dimension: 9,
                BaseVectorCount: 32,
                QueryCount: 4,
                TopK: 5,
                Seed: 0x5EED6120,
                InsertedDeltaCount: 6,
                DeletedBaseCount: 4,
                DuplicateInsertAttempts: 2,
                UnknownDeleteAttempts: 2,
                RepeatedDeleteAttempts: 2,
                AllowlistKind: "all",
                CandidateSetKind: "very-selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("metric.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-update"]);

        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal("passed", report.Validation.Status);
        AssertPassedMetrics(report.Metrics.UnfilteredSearch);
        AssertPassedMetrics(report.Metrics.RawAllowlistSearch);
        AssertPassedMetrics(report.Metrics.CandidateSetSearch);
        Assert.Equal(38, report.Counts.PhysicalVectorCount);
        Assert.Equal(34, report.Counts.LiveVectorCount);
        Assert.Equal(6, report.Counts.DeltaVectorCount);
        Assert.Equal(4, report.Counts.TombstoneCount);
        Assert.Equal(10, report.Mutations.GenerationDelta);
        Assert.Equal("very-selective", report.CandidateSetInput.Kind);
        Assert.Equal(4, report.CandidateSet.CountPerQuery);
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndUpdateModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--insertions", "2"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--deletes", "2"]));
        Assert.Equal("generated-exact-update", GeneratedExactUpdateOptions.ScenarioName);
    }

    private static void AssertMeasuredOperation(
        GeneratedExactUpdateOperationSearchInfo operation,
        string expectedName,
        string expectedTimedOperationPart)
    {
        Assert.Equal(expectedName, operation.Name);
        Assert.Contains(expectedTimedOperationPart, operation.TimedOperation, StringComparison.Ordinal);
        Assert.Equal(6, operation.Search.MeasuredQueryCount);
        Assert.Equal(3, operation.Search.Runs.Length);
        Assert.Equal(3, operation.Search.Aggregate.RunCount);
        Assert.Equal(6, operation.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.True(operation.Search.ElapsedMilliseconds >= 0);
        Assert.True(operation.Search.Qps > 0 || double.IsPositiveInfinity(operation.Search.Qps));
    }

    private static void AssertPassedMetrics(GeneratedExactUpdateOperationMetricsInfo metrics)
    {
        Assert.Equal(1.0, metrics.RecallAtK);
        Assert.Equal(1.0, metrics.OrderedAgreement);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
        Assert.Equal("passed", metrics.ResultIntegrity.Status);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec061-" + Path.GetFileNameWithoutExtension(fileName) + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, string.Create(CultureInfo.InvariantCulture, $"Unexpected field '{property.Name}' was present."));
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
