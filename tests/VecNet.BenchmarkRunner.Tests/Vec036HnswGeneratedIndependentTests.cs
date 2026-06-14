using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec036HnswGeneratedIndependentTests
{
    [Fact]
    public void ParseHnswGenerated_AcceptsInclusiveHnswParameterBounds()
    {
        HnswGeneratedOptions minimums = CommandLine.ParseHnswGenerated(
            [
                "hnsw-generated",
                "--metric", "SquaredEuclidean",
                "--dimension", "1",
                "--vectors", "1",
                "--queries", "1",
                "--top-k", "1",
                "--runs", "1",
                "--m", "2",
                "--ef-construction", "2",
                "--ef-search", "1",
                "--hnsw-seed", "0"
            ]);
        HnswGeneratedOptions maximums = CommandLine.ParseHnswGenerated(
            [
                "hnsw-generated",
                "--metric", "SquaredEuclidean",
                "--dimension", "3",
                "--vectors", "5",
                "--queries", "2",
                "--top-k", "5",
                "--runs", "5",
                "--m", "64",
                "--ef-construction", "4096",
                "--ef-search", "4096",
                "--hnsw-seed", "0xFFFFFFFFFFFFFFFF"
            ]);

        Assert.Equal(2, minimums.M);
        Assert.Equal(2, minimums.EfConstruction);
        Assert.Equal(1, minimums.EfSearch);
        Assert.Equal(0UL, minimums.HnswSeed);
        Assert.Equal(64, maximums.M);
        Assert.Equal(4096, maximums.EfConstruction);
        Assert.Equal(4096, maximums.EfSearch);
        Assert.Equal(ulong.MaxValue, maximums.HnswSeed);
    }

    [Theory]
    [InlineData("hnsw-generated", "--baseline-report-id", "baseline")]
    [InlineData("hnsw-generated", "--preset", "smoke")]
    [InlineData("hnsw-generated", "--output-dir", "VecNet.BenchmarkRunner.Artifacts/matrix")]
    [InlineData("hnsw-generated", "--manifest", "manifest.json")]
    [InlineData("hnsw-generated", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnsw-generated", "--download", "false")]
    [InlineData("hnsw-generated", "--query-count", "3")]
    [InlineData("hnsw-generated", "--truth-depth", "10")]
    public void ParseHnswGenerated_RejectsExactMatrixBaselineAndExternalDatasetOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(args));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_BelowPerfectRecallStillPassesWithReturnedResultIntegrityInJson()
    {
        HnswBenchmarkReport? selected = null;
        for (uint seed = 0x5EED3660; seed < 0x5EED36C0 && selected is null; seed++)
        {
            HnswBenchmarkReport report = HnswGeneratedScenario.Run(
                new HnswGeneratedOptions(
                    VectorMetric.SquaredEuclidean,
                    Dimension: 18,
                    VectorCount: 72,
                    QueryCount: 6,
                    TopK: 8,
                    Seed: seed,
                    OutputPath: NewArtifactPath("below-perfect.json"),
                    Runs: 1,
                    WarmupQueries: 0,
                    M: 2,
                    EfConstruction: 2,
                    EfSearch: 8,
                    HnswSeed: 0x3600UL),
                ["hnsw-generated"]);

            if (report.Metrics.RecallAtK < 1)
            {
                selected = report;
            }
        }

        Assert.NotNull(selected);
        string json = ReportWriter.Serialize(selected);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement metrics = root.GetProperty("metrics");
        JsonElement integrity = metrics.GetProperty("returnedResultIntegrity");

        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("allowsApproximateRecallBelowOne").GetBoolean());
        Assert.InRange(metrics.GetProperty("recallAtK").GetDouble(), 0, 0.999999);
        Assert.Equal("passed", metrics.GetProperty("distanceToleranceStatus").GetString());
        Assert.Equal("passed", integrity.GetProperty("status").GetString());
        Assert.True(integrity.GetProperty("checkedResultCount").GetInt32() > 0);
        Assert.Equal(0, integrity.GetProperty("duplicateIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("unknownIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("nonFiniteDistanceCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("distanceMismatchCount").GetInt32());
        Assert.Contains("exact top-k recall/order are recorded, not required", metrics.GetProperty("distanceValidationScope").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateReturnedResults_EnforcesToleranceBoundaryAndSurplusQueryMismatch()
    {
        GeneratedDataset dataset = CreateDataset(dimension: 9, vectorCount: 6, queryCount: 2, seed: 0x5EED3604);
        float expected = SquaredEuclidean(dataset.GetQuery(0), dataset.GetVector(1));
        float tolerance = SquaredEuclideanTolerance(dataset.Dimension, expected);

        HnswReturnedResultIntegrityInfo withinTolerance = HnswGeneratedScenario.ValidateReturnedResults(
            dataset,
            [
                [new SearchResult(1, expected + (tolerance * 0.5f))],
                [ResultFor(dataset, queryRow: 1, id: 2)]
            ],
            topK: 1);
        HnswReturnedResultIntegrityInfo beyondToleranceWithExtraQuery = HnswGeneratedScenario.ValidateReturnedResults(
            dataset,
            [
                [new SearchResult(1, expected + (tolerance * 4))],
                [ResultFor(dataset, queryRow: 1, id: 2)],
                [new SearchResult(0, 0)]
            ],
            topK: 1);

        Assert.Equal("passed", withinTolerance.Status);
        Assert.Equal(0, withinTolerance.DistanceMismatchCount);
        Assert.Equal("failed", beyondToleranceWithExtraQuery.Status);
        Assert.Equal(1, beyondToleranceWithExtraQuery.QueryCountMismatchCount);
        Assert.Equal(1, beyondToleranceWithExtraQuery.DistanceMismatchCount);
        Assert.Equal(2, beyondToleranceWithExtraQuery.CheckedResultCount);

        string json = ReportWriter.Serialize(beyondToleranceWithExtraQuery);
        using JsonDocument document = JsonDocument.Parse(json);
        Assert.Equal("failed", document.RootElement.GetProperty("status").GetString());
        Assert.Equal(1, document.RootElement.GetProperty("queryCountMismatchCount").GetInt32());
        Assert.Equal(1, document.RootElement.GetProperty("distanceMismatchCount").GetInt32());
    }

    [Fact]
    public void Run_RepeatedRunsAndWarmupKeepMeasuredCountsAndAllocationAggregatesScopedToQueries()
    {
        HnswBenchmarkReport report = HnswGeneratedScenario.Run(
            new HnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 10,
                VectorCount: 40,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED3605,
                OutputPath: NewArtifactPath("runs-warmup.json"),
                Runs: 5,
                WarmupQueries: 17,
                M: 4,
                EfConstruction: 12,
                EfSearch: 6,
                HnswSeed: 0x3605UL),
            ["hnsw-generated", "--runs", "5", "--warmup-queries", "17"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement search = root.GetProperty("search");
        JsonElement aggregate = search.GetProperty("aggregate");
        JsonElement runs = search.GetProperty("runs");
        JsonElement measurement = root.GetProperty("measurement");

        Assert.Equal(3, root.GetProperty("scenario").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(5, runs.GetArrayLength());
        Assert.Equal(5, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(3, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal("measured", measurement.GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.Equal("measured", measurement.GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("executed", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(17, measurement.GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Contains("excluded", measurement.GetProperty("warmup").GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("HNSW build", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("warmup", measurement.GetProperty("managedAllocations").GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"measuredQueryCount\":20", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"measuredQueryCountPerRun\":20", json, StringComparison.Ordinal);

        double[] allocationsPerQuery = runs.EnumerateArray()
            .Select(run => run.GetProperty("managedAllocatedBytesPerQuery").GetDouble())
            .ToArray();
        Assert.Equal(allocationsPerQuery.Average(), aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble(), precision: 12);
        Assert.Equal(aggregate.GetProperty("meanManagedAllocatedBytesPerQuery").GetDouble(), double.Parse(
            measurement.GetProperty("managedAllocations").GetProperty("value").GetString()!,
            CultureInfo.InvariantCulture), precision: 12);
    }

    [Fact]
    public void HnswReportSchema_IsSeparateFromGeneratedExactAndUnsupportedByGeneratedExactComparison()
    {
        string directory = NewArtifactDirectory("schema-separation");
        string hnswPath = Path.Combine(directory, "hnsw.json");
        string exactPath = Path.Combine(directory, "exact.json");
        string comparisonPath = Path.Combine(directory, "comparison.json");

        HnswBenchmarkReport hnsw = HnswGeneratedScenario.Run(
            new HnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 2,
                TopK: 3,
                Seed: 0x5EED3606,
                OutputPath: hnswPath,
                Runs: 1,
                WarmupQueries: 0,
                M: 4,
                EfConstruction: 12,
                EfSearch: 4,
                HnswSeed: 0x3606UL),
            ["hnsw-generated"]);
        BenchmarkReport exact = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 2,
                TopK: 3,
                Seed: 0x5EED3606,
                OutputPath: exactPath,
                BaselineReportId: null),
            ["exact-generated"]);
        HnswGeneratedScenario.Write(hnsw, hnswPath);
        ReportWriter.Write(exact, exactPath);

        using JsonDocument hnswDocument = JsonDocument.Parse(File.ReadAllText(hnswPath));
        using JsonDocument exactDocument = JsonDocument.Parse(File.ReadAllText(exactPath));
        JsonElement hnswRoot = hnswDocument.RootElement;
        JsonElement exactRoot = exactDocument.RootElement;

        Assert.Equal("VecNet.HnswBenchmarkReport", hnswRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.BenchmarkReport", exactRoot.GetProperty("schemaName").GetString());
        Assert.True(hnswRoot.TryGetProperty("hnsw", out _));
        Assert.True(hnswRoot.TryGetProperty("memoryEstimate", out _));
        Assert.True(hnswRoot.GetProperty("metrics").TryGetProperty("returnedResultIntegrity", out _));
        Assert.False(hnswRoot.TryGetProperty("baseline", out _));
        Assert.False(exactRoot.TryGetProperty("hnsw", out _));
        Assert.False(exactRoot.TryGetProperty("memoryEstimate", out _));
        Assert.False(exactRoot.GetProperty("metrics").TryGetProperty("returnedResultIntegrity", out _));
        Assert.True(exactRoot.TryGetProperty("baseline", out _));

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(hnswPath, hnswPath, comparisonPath),
            ["compare-generated-exact"]);

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Metrics);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema");
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Fact]
    public void Run_DoesNotEmitMatrixExternalBaselineRegressionOrPublicClaimBehavior()
    {
        HnswBenchmarkReport report = HnswGeneratedScenario.Run(
            new HnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 12,
                VectorCount: 48,
                QueryCount: 4,
                TopK: 5,
                Seed: 0x5EED3607,
                OutputPath: NewArtifactPath("posture.json"),
                Runs: 3,
                WarmupQueries: 2,
                M: 4,
                EfConstruction: 16,
                EfSearch: 8,
                HnswSeed: 0x3607UL),
            ["hnsw-generated"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("validation").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("generated-no-external-source", root.GetProperty("dataset").GetProperty("sourceVerificationStatus").GetString());
        Assert.Equal("estimated", root.GetProperty("memoryEstimate").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        AssertNoPropertyNamed(
            root,
            "baseline",
            "baselineReportId",
            "candidateEligibility",
            "preset",
            "presetName",
            "caseCount",
            "cases",
            "manifest",
            "cacheRoot",
            "download",
            "truthDepth",
            "comparisonResult",
            "regressionPassed",
            "regressionDecision",
            "regressionThreshold",
            "publicClaimPassed",
            "publicClaimStatus",
            "residentMemoryBytes",
            "processMemoryBytes",
            "workingSetBytes");
        Assert.DoesNotContain("\"publicClaimEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", json, StringComparison.OrdinalIgnoreCase);
    }

    private static GeneratedDataset CreateDataset(int dimension, int vectorCount, int queryCount, uint seed) =>
        GeneratedDatasetFactory.Create(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                dimension,
                vectorCount,
                queryCount,
                TopK: 1,
                Seed: seed,
                OutputPath: NewArtifactPath("dataset.json"),
                BaselineReportId: null));

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

    private static float SquaredEuclideanTolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec036-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string NewArtifactPath(string fileName) =>
        Path.Combine(NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName)), fileName);

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
