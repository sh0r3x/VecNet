using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec046GeneratedExactFilteredIndependentTests
{
    [Fact]
    public void ParseGeneratedExactFiltered_AcceptsCaseInsensitiveNamesAliasAndBounds()
    {
        GeneratedExactFilteredOptions options = CommandLine.ParseGeneratedExactFiltered(
            [
                "EXACT-GENERATED-FILTERED",
                "--METRIC", "cosine",
                "--DIMENSION", "1",
                "--VECTORS", "5",
                "--QUERIES", "2",
                "--TOP-K", "5",
                "--FILTER", "verySelective",
                "--DUPLICATE-IDS", "2",
                "--UNKNOWN-IDS", "3",
                "--RUNS", "5",
                "--WARMUP-QUERIES", "0",
                "--SEED", "0xFFFFFFFF",
                "--OUTPUT", "VecNet.BenchmarkRunner.Artifacts/vec046-independent-parse.json"
            ]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal(1, options.Dimension);
        Assert.Equal(5, options.VectorCount);
        Assert.Equal(2, options.QueryCount);
        Assert.Equal(5, options.TopK);
        Assert.Equal("very-selective", options.FilterKind);
        Assert.Equal(2, options.DuplicateIdsPerQuery);
        Assert.Equal(3, options.UnknownIdsPerQuery);
        Assert.Equal(5, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(uint.MaxValue, options.Seed);
    }

    [Theory]
    [InlineData("exact-generated-filtered", "--seed", "-1")]
    [InlineData("exact-generated-filtered", "--seed", "0x100000000")]
    [InlineData("exact-generated-filtered", "--dimension", "1.5")]
    [InlineData("exact-generated-filtered", "--filter", " ")]
    [InlineData("exact-generated-filtered", "--baseline-report-id", "baseline")]
    [InlineData("exact-generated-filtered", "--current", "current.json")]
    [InlineData("exact-generated-filtered", "--manifest", "manifest.json")]
    [InlineData("exact-generated-filtered", "--output-dir", "matrix")]
    [InlineData("exact-generated-filtered", "--download", "false")]
    [InlineData("exact-generated-filtered", "--query-count", "3")]
    [InlineData("exact-generated-filtered", "--truth-depth", "10")]
    [InlineData("exact-generated-filtered", "--m", "8")]
    [InlineData("exact-generated-filtered", "--ef-search", "50")]
    [InlineData("exact-generated-filtered", "--hnsw-seed", "0x46")]
    public void ParseGeneratedExactFiltered_RejectsMalformedValuesAndOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, "all", 24, 24, 1.0)]
    [InlineData(VectorMetric.InnerProduct, "broad", 12, 12, 0.5)]
    [InlineData(VectorMetric.Cosine, "selective", 3, 3, 0.125)]
    [InlineData(VectorMetric.SquaredEuclidean, "very-selective", 4, 4, 1.0 / 6.0)]
    [InlineData(VectorMetric.InnerProduct, "empty", 0, 0, 0.0)]
    public void Run_FilterKindsAcrossMetricsCarrySchemaEligibilityAndSelectivityMetadata(
        VectorMetric metric,
        string filterKind,
        int expectedVisible,
        int expectedKnown,
        double expectedSelectivity)
    {
        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(
            new GeneratedExactFilteredOptions(
                metric,
                Dimension: 9,
                VectorCount: 24,
                QueryCount: 4,
                TopK: 5,
                Seed: 0x5EED4610,
                FilterKind: filterKind,
                DuplicateIdsPerQuery: 2,
                UnknownIdsPerQuery: 3,
                OutputPath: NewArtifactPath("filter-kind.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["exact-generated-filtered", "--filter", filterKind]);

        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-046", report.TaskId);
        Assert.Equal("exact-generated-filtered", report.ScenarioName);
        Assert.Equal(metric.ToString(), report.Dataset.Metric);
        Assert.Equal(metric.ToString(), report.Index.Metric);
        Assert.Equal(filterKind, report.Filter.Kind);
        Assert.Equal(expectedVisible, report.Filter.VisibleCountPerQuery);
        Assert.Equal(expectedKnown, report.Filter.KnownIdCountPerQuery);
        Assert.Equal(expectedVisible, report.Filter.MinVisibleCount);
        Assert.Equal(expectedVisible, report.Filter.MaxVisibleCount);
        Assert.Equal(expectedVisible, report.Filter.MeanVisibleCount);
        Assert.Equal(expectedKnown + 5, report.Filter.AllowlistLengthPerQuery);
        Assert.Equal(expectedKnown * 4, report.Filter.TotalKnownIdCount);
        Assert.Equal(8, report.Filter.TotalDuplicateIdCount);
        Assert.Equal(12, report.Filter.TotalUnknownIdCount);
        Assert.Equal(expectedSelectivity, report.Filter.ActualSelectivity, precision: 12);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal(4 * Math.Min(5, expectedVisible), report.Metrics.FilteredResultIntegrity.CheckedResultCount);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("generated-no-external-source", root.GetProperty("dataset").GetProperty("sourceVerificationStatus").GetString());
        AssertNoPropertyNamed(
            root,
            "baselineReportId",
            "candidateEligibility",
            "comparisonResult",
            "regressionDecision",
            "regressionThreshold",
            "publicClaimStatus",
            "preset",
            "cases",
            "manifest",
            "cacheRoot",
            "download",
            "truthDepth",
            "hnsw",
            "memoryEstimate",
            "residentMemoryBytes",
            "processMemoryBytes",
            "workingSetBytes");
    }

    [Fact]
    public void Run_DuplicateAndUnknownHeavyEmptyFilterKeepsVisibleSetEmpty()
    {
        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(
            new GeneratedExactFilteredOptions(
                VectorMetric.Cosine,
                Dimension: 6,
                VectorCount: 11,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED4611,
                FilterKind: "empty",
                DuplicateIdsPerQuery: 5,
                UnknownIdsPerQuery: 7,
                OutputPath: NewArtifactPath("empty-filter.json"),
                Runs: 1,
                WarmupQueries: 2),
            ["exact-generated-filtered", "--filter", "empty"]);

        Assert.Equal(0, report.Filter.VisibleCountPerQuery);
        Assert.Equal(0, report.Filter.KnownIdCountPerQuery);
        Assert.Equal(5, report.Filter.DuplicateIdCountPerQuery);
        Assert.Equal(7, report.Filter.UnknownIdCountPerQuery);
        Assert.Equal(12, report.Filter.AllowlistLengthPerQuery);
        Assert.Equal(0, report.Filter.TotalKnownIdCount);
        Assert.Equal(15, report.Filter.TotalDuplicateIdCount);
        Assert.Equal(21, report.Filter.TotalUnknownIdCount);
        Assert.Contains("duplicate unknown IDs", report.Filter.DuplicatePolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ignored", report.Filter.UnknownIdPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("passed", report.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.CheckedResultCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.MissingResultCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.ExtraResultCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.UnresolvedWrongIdCount);
        Assert.Equal(0, report.Metrics.FilteredResultIntegrity.UnresolvedOrderMismatchCount);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
    }

    [Fact]
    public void Run_RepeatedRunsWarmupAndAllocationMetadataStayScopedToMeasuredQueries()
    {
        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(
            new GeneratedExactFilteredOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 10,
                VectorCount: 40,
                QueryCount: 3,
                TopK: 6,
                Seed: 0x5EED4612,
                FilterKind: "selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 2,
                OutputPath: NewArtifactPath("runs-warmup.json"),
                Runs: 5,
                WarmupQueries: 17),
            ["exact-generated-filtered", "--runs", "5", "--warmup-queries", "17"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");
        JsonElement search = root.GetProperty("search");
        JsonElement aggregate = search.GetProperty("aggregate");
        JsonElement runs = search.GetProperty("runs");

        Assert.Equal(3, root.GetProperty("scenario").GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(3, search.GetProperty("measuredQueryCount").GetInt32());
        Assert.Equal(5, runs.GetArrayLength());
        Assert.Equal(5, aggregate.GetProperty("runCount").GetInt32());
        Assert.Equal(3, aggregate.GetProperty("measuredQueryCountPerRun").GetInt32());
        Assert.Equal("measured", measurement.GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", measurement.GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Contains("allowlist generation", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("filter workspace construction", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("result capture", measurement.GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", measurement.GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", measurement.GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Contains("workspace", measurement.GetProperty("managedAllocations").GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("notMeasured", measurement.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", measurement.GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal("measured", measurement.GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.True(measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("measured", measurement.GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("executed", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(17, measurement.GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Contains("excluded", measurement.GetProperty("warmup").GetProperty("reason").GetString(), StringComparison.OrdinalIgnoreCase);
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
    public void ValidateFilteredResults_ReportsQueryCountMissingIdOrderDistanceAndNonFiniteFailures()
    {
        var truth = new TruthSet(
            [
                [
                    new TruthItem(1, 1),
                    new TruthItem(2, 2),
                    new TruthItem(3, 3)
                ],
                [
                    new TruthItem(4, 4)
                ],
                [
                    new TruthItem(5, 5)
                ]
            ],
            depth: 3);
        SearchResult[][] actual =
        [
            [
                new SearchResult(1, 1),
                new SearchResult(99, float.PositiveInfinity)
            ],
            []
        ];

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            actual,
            topK: 3,
            dimension: 8,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(1, comparison.Integrity.QueryCountMismatchCount);
        Assert.Equal(2, comparison.Integrity.CheckedResultCount);
        Assert.Equal(2, comparison.Integrity.MissingResultCount);
        Assert.Equal(0, comparison.Integrity.ExtraResultCount);
        Assert.Equal(1, comparison.Integrity.WrongIdCount);
        Assert.Equal(1, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.NonFiniteDistanceCount);
        Assert.Equal(1, comparison.Integrity.DistanceMismatchCount);
        Assert.Equal("notApplicable", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
        Assert.Equal(0.25, comparison.RecallAtK);
        Assert.Equal(0.25, comparison.OrderedAgreement);
    }

    [Fact]
    public void ValidateFilteredResults_UsesSquaredL2ToleranceBoundary()
    {
        const int dimension = 128;
        const float expectedDistance = 2048f;
        float tolerance = SquaredEuclideanTolerance(dimension, expectedDistance);
        var truth = new TruthSet([[new TruthItem(7, expectedDistance)]], depth: 1);

        GeneratedExactFilteredResultComparison withinTolerance = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(7, expectedDistance + (tolerance * 0.5f))]],
            topK: 1,
            dimension,
            VectorMetric.SquaredEuclidean);
        GeneratedExactFilteredResultComparison beyondTolerance = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(7, expectedDistance + (tolerance * 4f))]],
            topK: 1,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("passed", withinTolerance.Integrity.Status);
        Assert.Equal(0, withinTolerance.Integrity.DistanceMismatchCount);
        Assert.Equal("failed", beyondTolerance.Integrity.Status);
        Assert.Equal(1, beyondTolerance.Integrity.DistanceMismatchCount);
        Assert.Equal(1.0, beyondTolerance.RecallAtK);
        Assert.Equal(1.0, beyondTolerance.OrderedAgreement);
    }

    [Fact]
    public void ValidateFilteredResults_AcceptsSquaredL2NearTieOrderPermutationOnlyWhenSetAndDistancesMatch()
    {
        const int dimension = 386;
        const float leftDistance = 1000f;
        float rightDistance = leftDistance + SquaredEuclideanTolerance(dimension, leftDistance);
        var truth = new TruthSet(
            [[new TruthItem(1, leftDistance), new TruthItem(2, rightDistance)]],
            depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, rightDistance), new SearchResult(1, leftDistance)]],
            topK: 2,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("passed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(2, comparison.Integrity.WrongIdCount);
        Assert.Equal(2, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(2, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(0, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.DistanceMismatchCount);
        Assert.Equal("acceptedNearTie", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("accepted D-026 near-tie/order-equivalence case", comparison.Integrity.Classification);
    }

    [Fact]
    public void ValidateFilteredResults_RejectsWrongIdEvenWhenDistanceIsNearExpected()
    {
        const int dimension = 386;
        const float distance = 1000f;
        var truth = new TruthSet([[new TruthItem(1, distance), new TruthItem(2, distance)]], depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(1, distance), new SearchResult(999, distance)]],
            topK: 2,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(0.5, comparison.RecallAtK);
        Assert.Equal(0.5, comparison.OrderedAgreement);
        Assert.Equal(1, comparison.Integrity.WrongIdCount);
        Assert.Equal(1, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal("notApplicable", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
    }

    [Fact]
    public void ValidateFilteredResults_RejectsSafelySeparatedSquaredL2OrderPermutation()
    {
        var truth = new TruthSet([[new TruthItem(1, 1f), new TruthItem(2, 10f)]], depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, 10f), new SearchResult(1, 1f)]],
            topK: 2,
            dimension: 386,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(2, comparison.Integrity.WrongIdCount);
        Assert.Equal(2, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(2, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(2, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal("notApplicable", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
    }

    [Fact]
    public void ValidateFilteredResults_RejectsNearTiePermutationWithWrongReturnedDistance()
    {
        const int dimension = 386;
        const float leftDistance = 1000f;
        float rightDistance = leftDistance + SquaredEuclideanTolerance(dimension, leftDistance);
        var truth = new TruthSet([[new TruthItem(1, leftDistance), new TruthItem(2, rightDistance)]], depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, rightDistance + (SquaredEuclideanTolerance(dimension, rightDistance) * 4f)), new SearchResult(1, leftDistance)]],
            topK: 2,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(1, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.DistanceMismatchCount);
        Assert.Equal("unresolved", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
    }

    [Fact]
    public void CompareGeneratedExact_RejectsMixedExactAndFilteredReportsWithoutMutatingInputs()
    {
        string directory = NewArtifactDirectory("comparison");
        string exactPath = Path.Combine(directory, "exact.json");
        string filteredPath = Path.Combine(directory, "filtered.json");
        string comparisonPath = Path.Combine(directory, "comparison.json");
        BenchmarkReport exact = GeneratedExactSearchScenario.Run(
            new GeneratedExactSearchOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED4613,
                exactPath,
                BaselineReportId: null,
                Runs: 3,
                WarmupQueries: 1),
            ["exact-generated"]);
        GeneratedExactFilteredBenchmarkReport filtered = GeneratedExactFilteredScenario.Run(
            new GeneratedExactFilteredOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 8,
                VectorCount: 24,
                QueryCount: 3,
                TopK: 4,
                Seed: 0x5EED4613,
                FilterKind: "broad",
                DuplicateIdsPerQuery: 0,
                UnknownIdsPerQuery: 0,
                OutputPath: filteredPath,
                Runs: 3,
                WarmupQueries: 1),
            ["exact-generated-filtered"]);
        ReportWriter.Write(exact, exactPath);
        GeneratedExactFilteredScenario.Write(filtered, filteredPath);
        string exactBefore = File.ReadAllText(exactPath);
        string filteredBefore = File.ReadAllText(filteredPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(exactPath, filteredPath, comparisonPath),
            ["compare-generated-exact"]);

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "artifactKindMismatch");
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        Assert.Empty(comparison.Metrics);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal(exactBefore, File.ReadAllText(exactPath));
        Assert.Equal(filteredBefore, File.ReadAllText(filteredPath));
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
            "vec046-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
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
