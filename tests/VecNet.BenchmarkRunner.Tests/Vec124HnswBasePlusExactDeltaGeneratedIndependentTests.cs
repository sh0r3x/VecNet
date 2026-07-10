using System.Globalization;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec124HnswBasePlusExactDeltaGeneratedIndependentTests
{
    [Fact]
    public void Parser_AcceptsCaseInsensitiveScenarioOptionsAndMetricButRejectsUnknownOption()
    {
        HnswBasePlusExactDeltaGeneratedOptions options = CommandLine.ParseHnswBasePlusExactDeltaGenerated(
            [
                "GENERATED-HNSW-BASE-PLUS-EXACT-DELTA",
                "--MeTrIc", "squaredeuclidean",
                "--DIMENSION", "7",
                "--Vectors", "12",
                "--Queries", "2",
                "--TOP-K", "3",
                "--Insertions", "4",
                "--Deletes", "2",
                "--Delta-Deletes", "1",
                "--Duplicate-Inserts", "0",
                "--Unknown-Deletes", "0",
                "--Repeated-Deletes", "0",
                "--Runs", "2",
                "--Warmup-Queries", "1",
                "--Seed", "0x5EED1240",
                "--M", "3",
                "--EF-Construction", "9",
                "--EF-Search", "5",
                "--HNSW-Seed", "0x1240",
                "--Output", NewArtifactPath("case-insensitive.json")
            ]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(7, options.Dimension);
        Assert.Equal(12, options.BaseVectorCount);
        Assert.Equal(4, options.InsertedDeltaCount);
        Assert.Equal(2, options.DeletedBaseCount);
        Assert.Equal(1, options.DeletedDeltaCount);
        Assert.Equal(2, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(3, options.M);
        Assert.Equal(9, options.EfConstruction);
        Assert.Equal(5, options.EfSearch);
        Assert.Equal(0x1240UL, options.HnswSeed);

        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaGenerated(
                ["generated-hnsw-base-plus-exact-delta", "--not-a-real-option", "1"]));
        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_MinimalWorkloadWithBaseAndDeltaDeletesReportsCoherentCountsAndNoUnderfill()
    {
        HnswBasePlusExactDeltaBenchmarkReport report = HnswBasePlusExactDeltaGeneratedScenario.Run(
            SmallOptions(
                baseVectorCount: 5,
                insertedDeltaCount: 2,
                deletedBaseCount: 1,
                deletedDeltaCount: 1,
                queryCount: 3,
                topK: 1,
                runs: 1,
                warmupQueries: 0,
                efSearch: 2),
            ["generated-hnsw-base-plus-exact-delta"]);

        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(5, report.Counts.BasePhysicalVectorCount);
        Assert.Equal(4, report.Counts.BaseLiveVectorCount);
        Assert.Equal(2, report.Counts.DeltaPhysicalVectorCount);
        Assert.Equal(1, report.Counts.DeltaLiveVectorCount);
        Assert.Equal(1, report.Counts.BaseTombstoneCount);
        Assert.Equal(1, report.Counts.DeltaTombstoneCount);
        Assert.Equal(2, report.Counts.TombstoneCount);
        Assert.Equal(5, report.Counts.LiveVectorCount);
        Assert.Equal(2, report.Counts.DeletedReservedIdCount);
        Assert.Equal(4, report.Counts.Generation);
        Assert.Equal(4, report.Mutations.StatusCounts.Committed);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.Equal(report.Underfill.TotalRequestedResultSlots, report.Underfill.TotalReturnedResults);
        Assert.Equal(0, report.Underfill.UnderfilledQueryCount);
        Assert.Equal(0, report.Underfill.UnderfilledSlotCount);
    }

    [Fact]
    public void FailedValidationPostureSerializesWhenMutationCountsOrGenerationMovementDoNotMatch()
    {
        HnswBasePlusExactDeltaBenchmarkReport report = HnswBasePlusExactDeltaGeneratedScenario.Run(
            SmallOptions(runs: 1, warmupQueries: 0),
            ["generated-hnsw-base-plus-exact-delta"]);

        HnswBasePlusExactDeltaBenchmarkReport statusMismatch = report with
        {
            Validation = report.Validation with
            {
                Status = "failed",
                MutationStatusCountsMatched = false
            }
        };
        HnswBasePlusExactDeltaBenchmarkReport generationMismatch = report with
        {
            Validation = report.Validation with
            {
                Status = "failed",
                GenerationMovementMatchedCommittedMutations = false
            },
            Mutations = report.Mutations with
            {
                GenerationDeltaMatchesCommittedMutations = false
            }
        };

        using JsonDocument statusDocument = SerializeToJson(statusMismatch);
        using JsonDocument generationDocument = SerializeToJson(generationMismatch);

        JsonElement statusValidation = statusDocument.RootElement.GetProperty("validation");
        Assert.Equal("failed", statusValidation.GetProperty("status").GetString());
        Assert.False(statusValidation.GetProperty("mutationStatusCountsMatched").GetBoolean());
        Assert.True(statusValidation.GetProperty("generationMovementMatchedCommittedMutations").GetBoolean());

        JsonElement generationValidation = generationDocument.RootElement.GetProperty("validation");
        Assert.Equal("failed", generationValidation.GetProperty("status").GetString());
        Assert.True(generationValidation.GetProperty("mutationStatusCountsMatched").GetBoolean());
        Assert.False(generationValidation.GetProperty("generationMovementMatchedCommittedMutations").GetBoolean());
        Assert.False(generationDocument.RootElement
            .GetProperty("mutations")
            .GetProperty("generationDeltaMatchesCommittedMutations")
            .GetBoolean());
    }

    [Fact]
    public void ValidateReturnedResults_FailsQueryCountMismatchAndResultCountViolations()
    {
        GeneratedDataset dataset = GeneratedDatasetFactory.Create(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 4,
                VectorCount: 5,
                QueryCount: 3,
                TopK: 1,
                Seed: 0x5EED1241,
                OutputPath: NewArtifactPath("integrity-dataset.json"),
                BaselineReportId: null));

        HnswBasePlusExactDeltaReturnedResultIntegrityInfo queryMismatch =
            HnswBasePlusExactDeltaGeneratedScenario.ValidateReturnedResults(
                dataset,
                [
                    [ResultFor(dataset, queryRow: 0, id: 0)],
                    [ResultFor(dataset, queryRow: 1, id: 1)]
                ],
                topK: 1,
                liveIds: [0, 1, 2, 3, 4]);

        Assert.Equal("failed", queryMismatch.Status);
        Assert.Equal(1, queryMismatch.QueryCountMismatchCount);
        Assert.Equal(0, queryMismatch.ResultCountViolationCount);
        Assert.Equal(2, queryMismatch.CheckedResultCount);

        HnswBasePlusExactDeltaReturnedResultIntegrityInfo resultCountViolation =
            HnswBasePlusExactDeltaGeneratedScenario.ValidateReturnedResults(
                dataset,
                [
                    [ResultFor(dataset, queryRow: 0, id: 0), ResultFor(dataset, queryRow: 0, id: 1)],
                    [ResultFor(dataset, queryRow: 1, id: 2)],
                    [ResultFor(dataset, queryRow: 2, id: 3)]
                ],
                topK: 1,
                liveIds: [0, 1, 2, 3, 4]);

        Assert.Equal("failed", resultCountViolation.Status);
        Assert.Equal(0, resultCountViolation.QueryCountMismatchCount);
        Assert.Equal(1, resultCountViolation.ResultCountViolationCount);
        Assert.Equal(0, resultCountViolation.UnknownIdCount);
        Assert.Equal(0, resultCountViolation.TombstonedIdCount);
        Assert.Equal(0, resultCountViolation.DuplicateIdCount);
    }

    [Fact]
    public void UnderfillMetadataCountsFullAndPartiallyUnderfilledCapturedResults()
    {
        HnswBasePlusExactDeltaGeneratedOptions options = SmallOptions(queryCount: 3, topK: 2);

        HnswBasePlusExactDeltaUnderfillInfo full = InvokeCreateUnderfill(
            options,
            [
                [new SearchResult(1, 0.1f), new SearchResult(2, 0.2f)],
                [new SearchResult(3, 0.3f), new SearchResult(4, 0.4f)],
                [new SearchResult(5, 0.5f), new SearchResult(6, 0.6f)]
            ]);

        Assert.Equal(3, full.QueryCount);
        Assert.Equal(2, full.RequestedResultCountPerQuery);
        Assert.Equal(6, full.TotalRequestedResultSlots);
        Assert.Equal(6, full.TotalReturnedResults);
        Assert.Equal(0, full.UnderfilledQueryCount);
        Assert.Equal(0, full.UnderfilledSlotCount);

        HnswBasePlusExactDeltaUnderfillInfo partial = InvokeCreateUnderfill(
            options,
            [
                [new SearchResult(1, 0.1f), new SearchResult(2, 0.2f)],
                [new SearchResult(3, 0.3f)],
                []
            ]);

        Assert.Equal(6, partial.TotalRequestedResultSlots);
        Assert.Equal(3, partial.TotalReturnedResults);
        Assert.Equal(2, partial.UnderfilledQueryCount);
        Assert.Equal(3, partial.UnderfilledSlotCount);
        Assert.Contains("fewer than requested top-k", partial.Policy, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WarmupAndRepeatedRunMetadataPreserveRunnerSemantics()
    {
        HnswBasePlusExactDeltaBenchmarkReport singleRun = HnswBasePlusExactDeltaGeneratedScenario.Run(
            SmallOptions(queryCount: 2, topK: 1, runs: 1, warmupQueries: 0),
            ["generated-hnsw-base-plus-exact-delta"]);
        HnswBasePlusExactDeltaBenchmarkReport repeated = HnswBasePlusExactDeltaGeneratedScenario.Run(
            SmallOptions(queryCount: 2, topK: 1, runs: 2, warmupQueries: 5),
            ["generated-hnsw-base-plus-exact-delta"]);

        Assert.Equal("singleRun", singleRun.Measurement.RepeatedRuns.Status);
        Assert.False(singleRun.Measurement.RepeatedRuns.VarianceMeasured);
        Assert.Equal("notMeasured", singleRun.Measurement.RunToRunNoise.Status);
        Assert.False(singleRun.Measurement.RunToRunNoise.NoiseMeasured);
        Assert.Equal("absent", singleRun.Measurement.Warmup.Status);
        Assert.Equal(0, singleRun.Measurement.Warmup.WarmupCount);

        Assert.Equal("measured", repeated.Measurement.RepeatedRuns.Status);
        Assert.True(repeated.Measurement.RepeatedRuns.VarianceMeasured);
        Assert.Equal(2, repeated.Measurement.RepeatedRuns.RunCount);
        Assert.Equal("measured", repeated.Measurement.RunToRunNoise.Status);
        Assert.True(repeated.Measurement.RunToRunNoise.NoiseMeasured);
        Assert.Equal(2, repeated.Search.Runs.Length);
        Assert.All(repeated.Search.Runs, run => Assert.Equal(2, run.MeasuredQueryCount));
        Assert.Equal("executed", repeated.Measurement.Warmup.Status);
        Assert.Equal(5, repeated.Measurement.Warmup.WarmupCount);
        Assert.Equal(2, repeated.Search.Aggregate.MeasuredQueryCountPerRun);
    }

    [Fact]
    public void SerializedJsonPreservesPrivateNoClaimPostureAndNoEligibilityFieldsAreTrue()
    {
        string outputPath = NewArtifactPath("private-posture.json");
        HnswBasePlusExactDeltaBenchmarkReport report = HnswBasePlusExactDeltaGeneratedScenario.Run(
            SmallOptions(outputPath: outputPath),
            ["generated-hnsw-base-plus-exact-delta", "--output", outputPath]);

        HnswBasePlusExactDeltaGeneratedScenario.Write(report, outputPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;

        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "candidateEligibility", "regressionDecision", "publicClaimStatus");
    }

    private static HnswBasePlusExactDeltaGeneratedOptions SmallOptions(
        int baseVectorCount = 8,
        int insertedDeltaCount = 2,
        int deletedBaseCount = 1,
        int deletedDeltaCount = 0,
        int queryCount = 2,
        int topK = 2,
        int runs = 1,
        int warmupQueries = 0,
        int efSearch = 4,
        string? outputPath = null) =>
        new(
            VectorMetric.SquaredEuclidean,
            Dimension: 4,
            BaseVectorCount: baseVectorCount,
            QueryCount: queryCount,
            TopK: topK,
            Seed: 0x5EED1242,
            InsertedDeltaCount: insertedDeltaCount,
            DeletedBaseCount: deletedBaseCount,
            DeletedDeltaCount: deletedDeltaCount,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputPath: outputPath ?? NewArtifactPath("small-run.json"),
            Runs: runs,
            WarmupQueries: warmupQueries,
            M: 2,
            EfConstruction: 4,
            EfSearch: efSearch,
            HnswSeed: 0x1242);

    private static HnswBasePlusExactDeltaUnderfillInfo InvokeCreateUnderfill(
        HnswBasePlusExactDeltaGeneratedOptions options,
        SearchResult[][] actual)
    {
        MethodInfo? method = typeof(HnswBasePlusExactDeltaGeneratedScenario).GetMethod(
            "CreateUnderfill",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        object? result = method.Invoke(null, [options, actual]);
        return Assert.IsType<HnswBasePlusExactDeltaUnderfillInfo>(result);
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

    private static JsonDocument SerializeToJson(HnswBasePlusExactDeltaBenchmarkReport report)
    {
        string outputPath = NewArtifactPath("synthetic-validation.json");
        HnswBasePlusExactDeltaGeneratedScenario.Write(report, outputPath);
        return JsonDocument.Parse(File.ReadAllText(outputPath));
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(
                CultureInfo.InvariantCulture,
                $"vec124-independent-{Path.GetFileNameWithoutExtension(fileName)}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
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
