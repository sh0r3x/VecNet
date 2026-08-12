using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec149HnswAllowlistFilteringTests
{
    [Fact]
    public void ParseHnswAllowlistFiltering_UsesPrivateSmokeDefaults()
    {
        HnswAllowlistFilteringOptions options =
            CommandLine.ParseHnswAllowlistFiltering(["generated-hnsw-allowlist-filtered"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(32, options.Dimension);
        Assert.Equal(512, options.BaseVectorCount);
        Assert.Equal(576, options.PhysicalVectorCount);
        Assert.Equal(536, options.LiveVectorCount);
        Assert.Equal(8, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(64, options.InsertedDeltaCount);
        Assert.Equal(32, options.DeletedBaseCount);
        Assert.Equal(8, options.DeletedDeltaCount);
        Assert.Equal("fallback-boundary", options.FilterProfile);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(64, options.EfSearch);
        Assert.Equal(0x5EED2148u, options.Seed);
        Assert.Equal(0x484E535700014800UL, options.HnswSeed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OpenedIndexDirectory);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.CheckpointDirectory);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-allowlist-filtered", "--filter", "selective")]
    [InlineData("generated-hnsw-allowlist-filtered", "--dimension", "0")]
    [InlineData("generated-hnsw-allowlist-filtered", "--vectors", "0")]
    [InlineData("generated-hnsw-allowlist-filtered", "--queries", "0")]
    [InlineData("generated-hnsw-allowlist-filtered", "--top-k", "9", "--ef-search", "8")]
    [InlineData("generated-hnsw-allowlist-filtered", "--runs", "6")]
    [InlineData("generated-hnsw-allowlist-filtered", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-allowlist-filtered", "--insertions", "0")]
    [InlineData("generated-hnsw-allowlist-filtered", "--deletes", "9", "--vectors", "8")]
    [InlineData("generated-hnsw-allowlist-filtered", "--delta-deletes", "9", "--insertions", "8")]
    [InlineData("generated-hnsw-allowlist-filtered", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("generated-hnsw-allowlist-filtered", "--filter", "fallback-boundary", "--vectors", "16", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "32")]
    [InlineData("generated-hnsw-allowlist-filtered", "--filter", "broad", "--vectors", "16", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0", "--ef-search", "32")]
    [InlineData("generated-hnsw-allowlist-filtered", "--m", "1")]
    [InlineData("generated-hnsw-allowlist-filtered", "--ef-construction", "4097")]
    [InlineData("generated-hnsw-allowlist-filtered", "--output", "")]
    [InlineData("generated-hnsw-allowlist-filtered", "--opened-index-directory", "")]
    [InlineData("generated-hnsw-allowlist-filtered", "--checkpoint-directory", "")]
    [InlineData("generated-hnsw-allowlist-filtered", "--preset", "smoke")]
    [InlineData("generated-hnsw-allowlist-filtered", "--output-dir", "matrix")]
    [InlineData("generated-hnsw-allowlist-filtered", "--manifest", "manifest.json")]
    [InlineData("generated-hnsw-allowlist-filtered", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-hnsw-allowlist-filtered", "--sample-interval-ms", "10")]
    public void ParseHnswAllowlistFiltering_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswAllowlistFiltering(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    [InlineData("InnerProduct")]
    [InlineData("innerproduct")]
    public void ParseHnswAllowlistFiltering_AcceptsGeneratedHnswMetrics(string metric)
    {
        HnswAllowlistFilteringOptions options =
            CommandLine.ParseHnswAllowlistFiltering(
                [HnswAllowlistFilteringOptions.ScenarioName, "--metric", metric]);

        Assert.True(options.Metric is VectorMetric.Cosine or VectorMetric.InnerProduct);
    }

    [Fact]
    public void Run_FallbackBoundaryReportCoversAllSectionsAndExactParity()
    {
        HnswAllowlistFilteringBenchmarkReport report = RunReport("fallback-boundary", efSearch: 8);

        Assert.Equal("VecNet.HnswAllowlistFilteringBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-149", report.TaskId);
        Assert.Equal("generated-hnsw-allowlist-filtered", report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fallback-boundary", report.Allowlist.Profile);
        Assert.Equal(8, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.Equal(1, report.Allowlist.DuplicateInputIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.UnknownIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.TombstonedInputIdCountPerQuery);
        Assert.Equal(4, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(0, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("exactFallback", report.Branches.ExpectedBranch);
        Assert.Equal("passed", report.Branches.BranchConsistencyStatus);

        Assert.Equal(40, report.PreCheckpointCounts.BasePhysicalVectorCount);
        Assert.Equal(36, report.PreCheckpointCounts.BaseLiveVectorCount);
        Assert.Equal(8, report.PreCheckpointCounts.DeltaPhysicalVectorCount);
        Assert.Equal(6, report.PreCheckpointCounts.DeltaLiveVectorCount);
        Assert.Equal(6, report.PreCheckpointCounts.TombstoneCount);
        Assert.Equal(42, report.PostCheckpointCounts.LiveVectorCount);
        Assert.Equal("Published", report.CheckpointResult.Status);
        Assert.Equal(42, report.CheckpointResult.LiveVectorCount);
        Assert.Equal("notMeasured", report.Memory.Status);

        AssertFallbackSection(report.Searches.ImmutableHnsw);
        AssertFallbackSection(report.Searches.OpenedHnsw);
        AssertFallbackSection(report.Searches.SourceComposite);
        AssertFallbackSection(report.Searches.RebuiltComposite);
        AssertFallbackSection(report.Searches.CheckpointOpenedHnsw);
        Assert.Equal("measured", report.Searches.SourceComposite.ExactFilteredDeltaScan.Status);
        Assert.Equal("measuredZeroAfterCheckpoint", report.Searches.RebuiltComposite.ExactFilteredDeltaScan.Status);
        Assert.True(report.Parity.ImmutableOpenedHnsw.AllResultsMatched);
        Assert.True(report.Parity.RebuiltCompositeCheckpointOpenedHnsw.AllResultsMatched);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.ExactFallbackParityPassedForAllSearches);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);
        Assert.True(report.Validation.TombstoneSuppressionPassed);
        Assert.True(report.Validation.MemoryNotMeasured);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonArtifactEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void Run_BroadReportRecordsEmissionRecallUnderfillIntegrityAndFalseEligibility()
    {
        HnswAllowlistFilteringBenchmarkReport report = RunReport("broad", efSearch: 8);

        Assert.Equal("broad", report.Allowlist.Profile);
        Assert.Equal(0, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(4, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("broadEmission", report.Branches.ExpectedBranch);

        AssertBroadSection(report.Searches.ImmutableHnsw);
        AssertBroadSection(report.Searches.OpenedHnsw);
        AssertBroadSection(report.Searches.SourceComposite);
        AssertBroadSection(report.Searches.RebuiltComposite);
        AssertBroadSection(report.Searches.CheckpointOpenedHnsw);
        Assert.Equal("measured", report.Searches.SourceComposite.ExactFilteredDeltaScan.Status);
        Assert.True(report.Validation.BroadEmissionIntegrityPassedForAllSearches);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);
        Assert.True(report.Validation.BranchConsistencyPassed);
        Assert.False(report.Validation.PublicClaimEligible);

        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswAllowlistFilteringBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("broad", root.GetProperty("allowlist").GetProperty("profile").GetString());
        Assert.Equal("notMeasured", root.GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("immutableHnsw", root.GetProperty("searches").GetProperty("immutableHnsw").GetProperty("name").GetString());
        Assert.Equal("openedHnsw", root.GetProperty("searches").GetProperty("openedHnsw").GetProperty("name").GetString());
        Assert.Equal("sourceComposite", root.GetProperty("searches").GetProperty("sourceComposite").GetProperty("name").GetString());
        Assert.Equal("rebuiltComposite", root.GetProperty("searches").GetProperty("rebuiltComposite").GetProperty("name").GetString());
        Assert.Equal("checkpointOpenedHnsw", root.GetProperty("searches").GetProperty("checkpointOpenedHnsw").GetProperty("name").GetString());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "comparisonArtifactEligible", "regressionGateEligible");
        AssertNoPropertyNamed(root, "cacheRoot", "preset", "manifest", "outputDir", "fashionMnist", "hnswlib");
    }

    private static HnswAllowlistFilteringBenchmarkReport RunReport(string filter, int efSearch)
    {
        string directory = NewArtifactDirectory(filter);
        string[] arguments =
        [
            "generated-hnsw-allowlist-filtered",
            "--metric", "SquaredEuclidean",
            "--dimension", "9",
            "--vectors", "40",
            "--queries", "4",
            "--top-k", "5",
            "--insertions", "8",
            "--deletes", "4",
            "--delta-deletes", "2",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "1",
            "--filter", filter,
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED1490",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", efSearch.ToString(CultureInfo.InvariantCulture),
            "--hnsw-seed", "0x0000000000001490",
            "--output", Path.Combine(directory, "report.json"),
            "--opened-index-directory", Path.Combine(directory, "opened"),
            "--checkpoint-directory", Path.Combine(directory, "checkpoint")
        ];

        HnswAllowlistFilteringOptions options = CommandLine.ParseHnswAllowlistFiltering(arguments);
        HnswAllowlistFilteringBenchmarkReport report = HnswAllowlistFilteringScenario.Run(options, arguments);
        HnswAllowlistFilteringScenario.Write(report, options.OutputPath);
        Assert.True(File.Exists(options.OutputPath));
        return report;
    }

    private static void AssertFallbackSection(HnswAllowlistSearchSectionInfo section)
    {
        Assert.Equal("passed", section.ExactFallbackValidation.Status);
        Assert.Equal(0, section.ExactFallbackValidation.CountMismatchCount);
        Assert.Equal(0, section.ExactFallbackValidation.IdOrOrderMismatchCount);
        Assert.Equal(0, section.ExactFallbackValidation.DistanceMismatchCount);
        Assert.Equal("notApplicable", section.BroadEmissionValidation.Status);
        Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
        Assert.Equal("passed", section.TombstoneSuppression.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.Equal(4, section.Underfill.QueryCount);
        Assert.Equal(5, section.Underfill.RequestedResultCountPerQuery);
    }

    private static void AssertBroadSection(HnswAllowlistSearchSectionInfo section)
    {
        Assert.Equal("notApplicable", section.ExactFallbackValidation.Status);
        Assert.Equal("passed", section.BroadEmissionValidation.Status);
        Assert.InRange(section.BroadEmissionValidation.RecallAtK, 0, 1);
        Assert.InRange(section.BroadEmissionValidation.OrderedAgreement, 0, 1);
        Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
        Assert.Equal(0, section.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
        Assert.Equal("passed", section.TombstoneSuppression.Status);
        Assert.Equal("measured", section.Measurement.Latency.Status);
        Assert.Equal("measured", section.Measurement.ManagedAllocations.Status);
        Assert.Equal("notMeasured", section.Measurement.Memory.Status);
        Assert.InRange(section.Underfill.UnderfilledQueryCount, 0, 4);
    }

    private static string NewArtifactDirectory(string name)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec149-{name}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertNoBooleanPropertyTrueForNames(JsonElement element, params string[] names)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    names.Any(name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    throw new Xunit.Sdk.XunitException($"Property {property.Name} was unexpectedly true.");
                }

                AssertNoBooleanPropertyTrueForNames(property.Value, names);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoBooleanPropertyTrueForNames(item, names);
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
