using System.Globalization;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec061GeneratedExactUpdateIndependentTests
{
    [Fact]
    public void MinimalAllowedMutationWorkload_AllowsZeroFailureAttemptsAndEmptyPostMutationFilters()
    {
        GeneratedExactUpdateBenchmarkReport report = GeneratedExactUpdateScenario.Run(
            new GeneratedExactUpdateOptions(
                VectorMetric.InnerProduct,
                Dimension: 3,
                BaseVectorCount: 2,
                QueryCount: 2,
                TopK: 2,
                Seed: 0x5EED_6101,
                InsertedDeltaCount: 1,
                DeletedBaseCount: 1,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                AllowlistKind: "empty",
                CandidateSetKind: "empty",
                DuplicateIdsPerQuery: 3,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("minimal-empty.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-update", "--allowlist", "empty", "--candidate-set", "empty"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(3, report.Counts.PhysicalVectorCount);
        Assert.Equal(2, report.Counts.LiveVectorCount);
        Assert.Equal(2, report.Counts.BaseVectorCount);
        Assert.Equal(1, report.Counts.DeltaVectorCount);
        Assert.Equal(1, report.Counts.TombstoneCount);
        Assert.Equal(1.0 / 3.0, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal("physicalVectorCount", report.Counts.TombstoneRatioDenominator);
        Assert.Equal(1, report.Counts.DeletedOrReservedIdCount);
        Assert.Contains("reserved", report.Counts.DeletedOrReservedIdSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(1, report.Mutations.InsertedCount);
        Assert.Equal(1, report.Mutations.DeletedCount);
        Assert.Equal(0, report.Mutations.DuplicateInsertAttempts);
        Assert.Equal(0, report.Mutations.UnknownDeleteAttempts);
        Assert.Equal(0, report.Mutations.RepeatedDeleteAttempts);
        Assert.Equal(2, report.Mutations.CommittedMutationCount);
        Assert.Equal(2, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(4, report.Mutations.GenerationAfterMutations);
        Assert.Equal(2, report.Mutations.GenerationDelta);
        Assert.Equal(2, report.Mutations.StatusCounts.Committed);
        Assert.Equal(0, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(0, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(0, report.Mutations.StatusCounts.AlreadyDeleted);

        AssertEmptyFilterInput(report.RawAllowlistInput);
        AssertEmptyFilterInput(report.CandidateSetInput);
        Assert.Equal(2, report.CandidateSet.ConstructedSetCount);
        Assert.Equal(0, report.CandidateSet.CountPerQuery);
        Assert.Equal(0, report.CandidateSet.MinCount);
        Assert.Equal(0, report.CandidateSet.MaxCount);
        Assert.Equal(0, report.CandidateSet.MeanCount);
        Assert.Equal(0, report.CandidateSet.TotalCandidateCount);
        Assert.True(report.CandidateSet.ConstructedAfterMutations);
        Assert.True(report.CandidateSet.ConstructedBeforeWarmupAndMeasuredSearch);

        AssertPassed(report.Metrics.UnfilteredSearch);
        AssertPassed(report.Metrics.RawAllowlistSearch);
        AssertPassed(report.Metrics.CandidateSetSearch);
        Assert.Equal(4, report.Metrics.UnfilteredSearch.ResultIntegrity.CheckedResultCount);
        Assert.Equal(0, report.Metrics.RawAllowlistSearch.ResultIntegrity.CheckedResultCount);
        Assert.Equal(0, report.Metrics.CandidateSetSearch.ResultIntegrity.CheckedResultCount);
        Assert.Equal("notMeasured", report.Measurement.MutationLatencyAndAllocation.Status);
        Assert.Equal("notMeasured", report.Measurement.LiveViewSave.Status);
        Assert.Equal("notMeasured", report.Measurement.ResidentProcessMemory.Status);
        Assert.Equal("singleRun", report.Measurement.UnfilteredSearch.RepeatedRuns.Status);
        Assert.Equal("notMeasured", report.Measurement.UnfilteredSearch.RunToRunNoise.Status);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
    }

    [Fact]
    public void AllFiltersAreBuiltFromPostMutationLiveIdsRatherThanBaseOrPhysicalIds()
    {
        const int baseCount = 6;
        const int insertions = 4;
        const int deletes = 3;
        const int queryCount = 3;
        const int liveCount = baseCount + insertions - deletes;
        const int physicalCount = baseCount + insertions;

        GeneratedExactUpdateBenchmarkReport report = GeneratedExactUpdateScenario.Run(
            new GeneratedExactUpdateOptions(
                VectorMetric.Cosine,
                Dimension: 7,
                BaseVectorCount: baseCount,
                QueryCount: queryCount,
                TopK: liveCount,
                Seed: 0x5EED_6102,
                InsertedDeltaCount: insertions,
                DeletedBaseCount: deletes,
                DuplicateInsertAttempts: 5,
                UnknownDeleteAttempts: 4,
                RepeatedDeleteAttempts: 2,
                AllowlistKind: "all",
                CandidateSetKind: "all",
                DuplicateIdsPerQuery: 2,
                UnknownIdsPerQuery: 1,
                OutputPath: NewArtifactPath("all-live.json"),
                Runs: 2,
                WarmupQueries: 1),
            ["generated-exact-update", "--allowlist", "all", "--candidate-set", "all"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(physicalCount, report.Counts.PhysicalVectorCount);
        Assert.Equal(liveCount, report.Counts.LiveVectorCount);
        Assert.Equal(insertions, report.Counts.DeltaVectorCount);
        Assert.Equal(deletes, report.Counts.TombstoneCount);
        Assert.Equal((double)deletes / physicalCount, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal(deletes, report.Counts.DeletedOrReservedIdCount);

        Assert.Equal(liveCount, report.RawAllowlistInput.KnownLiveIdCountPerQuery);
        Assert.Equal(liveCount, report.CandidateSetInput.KnownLiveIdCountPerQuery);
        Assert.Equal(queryCount * liveCount, report.RawAllowlistInput.TotalKnownLiveIdCount);
        Assert.Equal(queryCount * liveCount, report.CandidateSetInput.TotalKnownLiveIdCount);
        Assert.Equal(1.0, report.RawAllowlistInput.ActualLiveSelectivity);
        Assert.Equal(1.0, report.CandidateSetInput.ActualLiveSelectivity);
        Assert.Equal(liveCount + 3, report.RawAllowlistInput.InputIdCountPerQuery);
        Assert.Equal(liveCount + 3, report.CandidateSetInput.InputIdCountPerQuery);
        Assert.Contains("post-mutation live ID", report.CandidateSetInput.GenerationFormula, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("tombstoned IDs are excluded", report.CandidateSetInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("committed delta IDs are eligible", report.CandidateSetInput.MutationVisibilityPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(liveCount, report.CandidateSet.CountPerQuery);
        Assert.Equal(liveCount, report.CandidateSet.MinCount);
        Assert.Equal(liveCount, report.CandidateSet.MaxCount);
        Assert.Equal(liveCount, report.CandidateSet.MeanCount);
        Assert.Equal(queryCount * liveCount, report.CandidateSet.TotalCandidateCount);
        Assert.Contains("generation-bound", report.CandidateSet.Binding, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("freshly built after the mutation workload", report.CandidateSet.StaleCandidateSetPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(insertions + deletes, report.Mutations.CommittedMutationCount);
        Assert.Equal(baseCount, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(baseCount + insertions + deletes, report.Mutations.GenerationAfterMutations);
        Assert.Equal(5, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(4, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.True(report.Validation.CandidateSetsConstructedAfterMutations);
        Assert.True(report.Validation.FinalRunUnfilteredComparedToTruth);
        Assert.True(report.Validation.FinalRunRawAllowlistComparedToTruth);
        Assert.True(report.Validation.FinalRunCandidateSetComparedToTruth);

        Assert.Equal("measured", report.Measurement.UnfilteredSearch.RunToRunNoise.Status);
        Assert.Equal("measured", report.Measurement.RawAllowlistSearch.RunToRunNoise.Status);
        Assert.Equal("measured", report.Measurement.CandidateSetSearch.RunToRunNoise.Status);
        Assert.Equal(queryCount * liveCount * sizeof(int), report.MemoryEstimates.CandidateSetOrdinalPayloadLowerBoundBytes);
        Assert.Equal("notAvailable", report.MemoryEstimates.TombstoneDeletedReservationRetainedMemory.Status);
    }

    [Fact]
    public void ParserAcceptsUpdateAliasesAtBoundsAndRejectsUnrelatedOptionFamilies()
    {
        GeneratedExactUpdateOptions parsed = CommandLine.ParseGeneratedExactUpdate(
            [
                "GENERATED-EXACT-UPDATE",
                "--METRIC", "cosine",
                "--DIMENSION", "1",
                "--VECTORS", "3",
                "--QUERIES", "1",
                "--TOP-K", "2",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "0",
                "--SEED", "0xFFFFFFFF",
                "--INSERTIONS", "1",
                "--DELETES", "2",
                "--DUPLICATE-INSERTS", "0",
                "--UNKNOWN-DELETES", "0",
                "--REPEATED-DELETES", "0",
                "--ALLOWLIST", "verySelective",
                "--CANDIDATE-SET", "verySelective",
                "--DUPLICATE-IDS", "0",
                "--UNKNOWN-IDS", "0",
                "--OUTPUT", "VecNet.BenchmarkRunner.Artifacts/vec061-independent-parse.json"
            ]);

        Assert.Equal(VectorMetric.Cosine, parsed.Metric);
        Assert.Equal(1, parsed.Dimension);
        Assert.Equal(3, parsed.BaseVectorCount);
        Assert.Equal(4, parsed.PhysicalVectorCount);
        Assert.Equal(2, parsed.TopK);
        Assert.Equal(1, parsed.InsertedDeltaCount);
        Assert.Equal(2, parsed.DeletedBaseCount);
        Assert.Equal("very-selective", parsed.AllowlistKind);
        Assert.Equal("very-selective", parsed.CandidateSetKind);
        Assert.Equal(5, parsed.Runs);
        Assert.Equal(uint.MaxValue, parsed.Seed);

        string[][] updateRejects =
        [
            ["generated-exact-update", "--seed", "-1"],
            ["generated-exact-update", "--seed", "0x100000000"],
            ["generated-exact-update", "--dimension", "1.5"],
            ["generated-exact-update", "--allowlist", " "],
            ["generated-exact-update", "--candidate-set", " "],
            ["generated-exact-update", "--baseline", "baseline.json"],
            ["generated-exact-update", "--current", "current.json"],
            ["generated-exact-update", "--query-count", "3"],
            ["generated-exact-update", "--truth-depth", "10"],
            ["generated-exact-update", "--download", "false"],
            ["generated-exact-update", "--output-dir", "matrix"],
            ["generated-exact-update", "--manifest", "manifest.json"],
            ["generated-exact-update", "--ef-construction", "64"],
            ["generated-exact-update", "--ef-search", "50"],
            ["generated-exact-update", "--hnsw-seed", "0x1234"]
        ];

        foreach (string[] args in updateRejects)
        {
            ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(args));
            Assert.NotEmpty(exception.Message);
        }

        Assert.Throws<ArgumentException>(() => CommandLine.Parse(["exact-generated", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--duplicate-inserts", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--allowlist", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSetMatrix(["generated-exact-candidate-set-matrix", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--allowlist", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--duplicate-inserts", "1"]));
    }

    private static void AssertEmptyFilterInput(GeneratedExactUpdateFilterInputInfo input)
    {
        Assert.Equal("empty", input.Kind);
        Assert.Equal(0, input.KnownLiveIdCountPerQuery);
        Assert.Equal(3, input.DuplicateIdCountPerQuery);
        Assert.Equal(2, input.UnknownIdCountPerQuery);
        Assert.Equal(5, input.InputIdCountPerQuery);
        Assert.Equal(0, input.TotalKnownLiveIdCount);
        Assert.Equal(6, input.TotalDuplicateIdCount);
        Assert.Equal(4, input.TotalUnknownIdCount);
        Assert.Equal(0, input.ActualLiveSelectivity);
        Assert.Contains("duplicate unknown IDs", input.DuplicatePolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ignored", input.UnknownIdPolicy, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertPassed(GeneratedExactUpdateOperationMetricsInfo metrics)
    {
        Assert.Equal(1.0, metrics.RecallAtK);
        Assert.Equal(1.0, metrics.OrderedAgreement);
        Assert.Equal("passed", metrics.DistanceToleranceStatus);
        Assert.Equal("passed", metrics.ResultIntegrity.Status);
        Assert.Equal(0, metrics.DistanceMismatchCount);
        Assert.Equal(0, metrics.MissingResultCount);
        Assert.Equal(0, metrics.ExtraResultCount);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec061-independent-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
