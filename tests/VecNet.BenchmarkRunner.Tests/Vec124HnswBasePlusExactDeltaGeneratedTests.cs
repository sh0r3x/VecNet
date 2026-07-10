using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec124HnswBasePlusExactDeltaGeneratedTests
{
    [Fact]
    public void ParseHnswBasePlusExactDeltaGenerated_UsesPrivateSmokeDefaults()
    {
        HnswBasePlusExactDeltaGeneratedOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaGenerated(["generated-hnsw-base-plus-exact-delta"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.BaseVectorCount);
        Assert.Equal(11_000, options.PhysicalVectorCount);
        Assert.Equal(10_000, options.LiveVectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(0, options.DeletedDeltaCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(16, options.M);
        Assert.Equal(200, options.EfConstruction);
        Assert.Equal(50, options.EfSearch);
        Assert.Equal(0x564543_034UL, options.HnswSeed);
        Assert.Equal(0x5EED2124u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--metric", "Cosine")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--metric", "InnerProduct")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--dimension", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--vectors", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--queries", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--top-k", "10", "--vectors", "4", "--insertions", "1", "--deletes", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--runs", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--runs", "6")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--insertions", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--deletes", "6", "--vectors", "5")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--delta-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--delta-deletes", "2", "--insertions", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--duplicate-inserts", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--unknown-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--repeated-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--top-k", "10", "--ef-search", "9")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--m", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--m", "65")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--m", "8", "--ef-construction", "7")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--ef-construction", "4097")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--ef-search", "4097")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--allowlist", "broad")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--candidate-set", "selective")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--preset", "smoke")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--output-dir", "matrix")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--manifest", "manifest.json")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--snapshot-directory", "snapshot")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--checkpoint-directory", "checkpoint")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--output", "")]
    public void ParseHnswBasePlusExactDeltaGenerated_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaGenerated(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_ProducesPrivateGeneratedCompositeReportWithUpdateRecallAndUnderfillMetadata()
    {
        string outputPath = NewArtifactPath("updated-hnsw.json");
        string[] arguments =
        [
            "generated-hnsw-base-plus-exact-delta",
            "--metric", "SquaredEuclidean",
            "--dimension", "11",
            "--vectors", "40",
            "--queries", "5",
            "--top-k", "6",
            "--insertions", "7",
            "--deletes", "4",
            "--delta-deletes", "2",
            "--duplicate-inserts", "3",
            "--unknown-deletes", "4",
            "--repeated-deletes", "3",
            "--runs", "3",
            "--warmup-queries", "2",
            "--seed", "0x5EED124A",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "10",
            "--hnsw-seed", "0x000000000000124A",
            "--output", outputPath
        ];
        HnswBasePlusExactDeltaGeneratedOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaGenerated(arguments);

        HnswBasePlusExactDeltaBenchmarkReport report =
            HnswBasePlusExactDeltaGeneratedScenario.Run(options, arguments);
        HnswBasePlusExactDeltaGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.HnswBasePlusExactDeltaBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-124", report.TaskId);
        Assert.Equal("generated-hnsw-base-plus-exact-delta", report.ScenarioName);
        Assert.Equal("generated-hnsw-base-plus-exact-delta", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-hnsw-base-plus-exact-delta-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Dataset.Metric);
        Assert.Equal(47, report.Dataset.VectorCount);
        Assert.Equal("scalar-reference-generated-live-hnsw-base-plus-exact-delta", report.Truth.Kind);
        Assert.Equal(6, report.Truth.Depth);
        Assert.Equal("HnswBasePlusExactDeltaIndex", report.Index.Type);
        Assert.Contains("internal", report.Index.Configuration, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(10, report.Hnsw.EfSearch);
        Assert.Equal("0x000000000000124A", report.Hnsw.RandomSeed);
        Assert.Equal("measured", report.Build.Status);
        Assert.True(report.Build.ElapsedMilliseconds >= 0);
        Assert.Equal("measured", report.Build.ManagedAllocations.Status);
        Assert.True(long.Parse(report.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);

        Assert.Equal(40, report.Workload.BaseVectorCount);
        Assert.Equal(7, report.Workload.InsertedDeltaVectorCount);
        Assert.Equal(4, report.Workload.DeletedBaseVectorCount);
        Assert.Equal(2, report.Workload.DeletedDeltaVectorCount);
        Assert.Contains("delta tombstone", report.Workload.MutationOrder, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("deleted IDs remain reserved", report.Workload.IdPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(40, report.Counts.BasePhysicalVectorCount);
        Assert.Equal(36, report.Counts.BaseLiveVectorCount);
        Assert.Equal(7, report.Counts.DeltaPhysicalVectorCount);
        Assert.Equal(5, report.Counts.DeltaLiveVectorCount);
        Assert.Equal(4, report.Counts.BaseTombstoneCount);
        Assert.Equal(2, report.Counts.DeltaTombstoneCount);
        Assert.Equal(6, report.Counts.TombstoneCount);
        Assert.Equal(41, report.Counts.LiveVectorCount);
        Assert.Equal(6, report.Counts.DeletedReservedIdCount);
        Assert.Equal(13, report.Counts.Generation);
        Assert.Equal(6.0 / 47.0, report.Counts.TombstoneRatio, precision: 12);
        Assert.Equal(7.0 / 40.0, report.Counts.DeltaInsertRatio, precision: 12);
        Assert.Contains("Base physical rows remain", report.Counts.CountSemantics, StringComparison.Ordinal);

        Assert.Equal(7, report.Mutations.InsertedCount);
        Assert.Equal(4, report.Mutations.DeletedBaseCount);
        Assert.Equal(2, report.Mutations.DeletedDeltaCount);
        Assert.Equal(3, report.Mutations.DuplicateInsertAttempts);
        Assert.Equal(4, report.Mutations.UnknownDeleteAttempts);
        Assert.Equal(3, report.Mutations.RepeatedDeleteAttempts);
        Assert.Equal(13, report.Mutations.CommittedMutationCount);
        Assert.Equal(0, report.Mutations.GenerationBeforeMutations);
        Assert.Equal(13, report.Mutations.GenerationAfterMutations);
        Assert.Equal(13, report.Mutations.GenerationDelta);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(13, report.Mutations.StatusCounts.Committed);
        Assert.Equal(3, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(4, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(3, report.Mutations.StatusCounts.AlreadyDeleted);
        Assert.Equal(0, report.Mutations.StatusCounts.ReadOnly);
        Assert.Equal(0, report.Mutations.StatusCounts.Unsupported);

        Assert.Equal(5, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("update application", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerSearchCall", report.Measurement.ManagedAllocations.Unit);
        Assert.Contains("caller-owned SearchResult[] and HnswBasePlusExactDeltaSearchWorkspace", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);

        Assert.InRange(report.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Contains("exact updated top-k", report.Metrics.RecallDefinition, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no tombstoned ID", report.Metrics.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(5, report.Underfill.QueryCount);
        Assert.Equal(6, report.Underfill.RequestedResultCountPerQuery);
        Assert.Equal(30, report.Underfill.TotalRequestedResultSlots);
        Assert.InRange(report.Underfill.TotalReturnedResults, 0, 30);
        Assert.InRange(report.Underfill.UnderfilledQueryCount, 0, 5);
        Assert.Equal(30 - report.Underfill.TotalReturnedResults, report.Underfill.UnderfilledSlotCount);
        Assert.Contains("Underfill is recorded", report.Underfill.Policy, StringComparison.Ordinal);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.LiveTruthGenerated);
        Assert.True(report.Validation.HnswBaseBuilt);
        Assert.True(report.Validation.MutationsApplied);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.True(report.Validation.FinalRunComparedToTruth);
        Assert.True(report.Validation.ReturnedResultsAreLiveAndNotTombstoned);
        Assert.True(report.Validation.AllowsApproximateRecallBelowOne);
        Assert.True(report.Validation.AllowsUnderfill);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswBasePlusExactDeltaBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("generated-hnsw-base-plus-exact-delta", root.GetProperty("scenarioName").GetString());
        Assert.Equal(40, root.GetProperty("counts").GetProperty("basePhysicalVectorCount").GetInt32());
        Assert.Equal(5, root.GetProperty("counts").GetProperty("deltaLiveVectorCount").GetInt32());
        Assert.Equal(6, root.GetProperty("counts").GetProperty("tombstoneCount").GetInt32());
        Assert.Equal(13, root.GetProperty("counts").GetProperty("generation").GetInt64());
        Assert.Equal(13, root.GetProperty("mutations").GetProperty("statusCounts").GetProperty("committed").GetInt32());
        Assert.Equal(3, root.GetProperty("mutations").GetProperty("statusCounts").GetProperty("alreadyDeleted").GetInt32());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal(0, root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("tombstonedIdCount").GetInt32());
        Assert.Equal(30, root.GetProperty("underfill").GetProperty("totalRequestedResultSlots").GetInt32());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoPropertyNamed(root, "preset", "manifest", "cacheRoot", "snapshotDirectory", "checkpointDirectory", "candidateEligibility", "regressionDecision", "publicClaimStatus");
    }

    [Fact]
    public void Run_ReportsUnderfillWithoutFailingValidationWhenBaseOverfetchIsExhaustedByTombstones()
    {
        HnswBasePlusExactDeltaBenchmarkReport report = HnswBasePlusExactDeltaGeneratedScenario.Run(
            new HnswBasePlusExactDeltaGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 4,
                BaseVectorCount: 8,
                QueryCount: 3,
                TopK: 6,
                Seed: 0x5EED124B,
                InsertedDeltaCount: 1,
                DeletedBaseCount: 3,
                DeletedDeltaCount: 0,
                DuplicateInsertAttempts: 0,
                UnknownDeleteAttempts: 0,
                RepeatedDeleteAttempts: 0,
                OutputPath: NewArtifactPath("underfill.json"),
                Runs: 1,
                WarmupQueries: 0,
                M: 2,
                EfConstruction: 2,
                EfSearch: 6,
                HnswSeed: 0x124B),
            ["generated-hnsw-base-plus-exact-delta"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(6, report.Counts.LiveVectorCount);
        Assert.Equal(18, report.Underfill.TotalRequestedResultSlots);
        Assert.InRange(report.Underfill.TotalReturnedResults, 0, 18);
        Assert.Equal(18 - report.Underfill.TotalReturnedResults, report.Underfill.UnderfilledSlotCount);
        Assert.True(report.Validation.AllowsUnderfill);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ValidateReturnedResults_FailsUnknownTombstonedDuplicateAndDistanceMismatches()
    {
        GeneratedDataset dataset = GeneratedDatasetFactory.Create(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 5,
                VectorCount: 6,
                QueryCount: 2,
                TopK: 1,
                Seed: 0x5EED124C,
                OutputPath: NewArtifactPath("dataset.json"),
                BaselineReportId: null));
        SearchResult live = ResultFor(dataset, queryRow: 0, id: 3);

        HnswBasePlusExactDeltaReturnedResultIntegrityInfo integrity =
            HnswBasePlusExactDeltaGeneratedScenario.ValidateReturnedResults(
                dataset,
                [
                    [
                        live,
                        live,
                        new SearchResult(1, ResultFor(dataset, queryRow: 0, id: 1).Distance),
                        new SearchResult(99, 1),
                        new SearchResult(4, float.NaN)
                    ],
                    [
                        new SearchResult(5, ResultFor(dataset, queryRow: 1, id: 5).Distance + 1)
                    ]
                ],
                topK: 2,
                liveIds: [3, 4, 5]);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(6, integrity.CheckedResultCount);
        Assert.Equal(1, integrity.ResultCountViolationCount);
        Assert.Equal(1, integrity.NonFiniteDistanceCount);
        Assert.Equal(1, integrity.DuplicateIdCount);
        Assert.Equal(1, integrity.UnknownIdCount);
        Assert.Equal(1, integrity.TombstonedIdCount);
        Assert.Equal(2, integrity.DistanceMismatchCount);
        Assert.Contains("tombstone", integrity.Policy, StringComparison.OrdinalIgnoreCase);
    }

    private static SearchResult ResultFor(GeneratedDataset dataset, int queryRow, ulong id) =>
        new(id, SquaredEuclidean(dataset.GetQuery(queryRow), dataset.GetVector(checked((int)id))));

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

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec124-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
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
