using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec149HnswAllowlistFilteringIndependentTests
{
    [Fact]
    public void ParserDefaultsAndAcceptedProfilesStayBoundedToPrivateGeneratedSmoke()
    {
        HnswAllowlistFilteringOptions defaults =
            CommandLine.ParseHnswAllowlistFiltering([HnswAllowlistFilteringOptions.ScenarioName]);

        Assert.Equal(VectorMetric.SquaredEuclidean, defaults.Metric);
        Assert.Equal("fallback-boundary", defaults.FilterProfile);
        Assert.Equal(32, defaults.Dimension);
        Assert.Equal(512, defaults.BaseVectorCount);
        Assert.Equal(64, defaults.InsertedDeltaCount);
        Assert.Equal(32, defaults.DeletedBaseCount);
        Assert.Equal(8, defaults.DeletedDeltaCount);
        Assert.Equal(8, defaults.QueryCount);
        Assert.Equal(10, defaults.TopK);
        Assert.Equal(1, defaults.Runs);
        Assert.Equal(1, defaults.WarmupQueries);
        Assert.Equal(64, defaults.EfSearch);
        Assert.Equal(536, defaults.LiveVectorCount);
        Assert.False(Path.IsPathFullyQualified(defaults.OutputPath));
        Assert.False(Path.IsPathFullyQualified(defaults.OpenedIndexDirectory));
        Assert.False(Path.IsPathFullyQualified(defaults.CheckpointDirectory));

        foreach (string profile in new[] { "empty", "very-selective", "fallback-boundary", "broad", "all" })
        {
            HnswAllowlistFilteringOptions options = CommandLine.ParseHnswAllowlistFiltering(
                [HnswAllowlistFilteringOptions.ScenarioName, "--filter", profile]);

            Assert.Equal(profile, options.FilterProfile);
            Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        }
    }

    [Theory]
    [InlineData("--metric", "InnerProduct")]
    [InlineData("--metric", "Cosine")]
    [InlineData("--filter", "selective")]
    [InlineData("--filter", "low-churn")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "all")]
    [InlineData("--preset", "standard")]
    [InlineData("--matrix-seed", "0x5EED2148")]
    [InlineData("--cache-root", "VecNet.DatasetCache")]
    [InlineData("--dataset", "fashion-mnist")]
    [InlineData("--baseline-report-id", "baseline")]
    [InlineData("--comparison-report", "comparison.json")]
    [InlineData("--comparison-output", "comparison.json")]
    [InlineData("--sample-interval-ms", "10")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--checkpoint-memory", "true")]
    public void ParserRejectsUnsupportedMetricProfilesAndOutOfScopeSurfaces(params string[] option)
    {
        string[] args = [HnswAllowlistFilteringOptions.ScenarioName, .. option];

        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswAllowlistFiltering(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("--top-k", "9", "--ef-search", "8")]
    [InlineData("--filter", "fallback-boundary", "--vectors", "10", "--insertions", "2", "--deletes", "4", "--delta-deletes", "2", "--ef-search", "16")]
    [InlineData("--filter", "broad", "--vectors", "10", "--insertions", "2", "--deletes", "4", "--delta-deletes", "2", "--ef-search", "16")]
    [InlineData("--filter", "all", "--vectors", "10", "--insertions", "2", "--deletes", "4", "--delta-deletes", "2", "--ef-search", "16")]
    public void ParserRejectsBranchImpossibleWorkloads(params string[] option)
    {
        string[] args = [HnswAllowlistFilteringOptions.ScenarioName, .. option];

        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswAllowlistFiltering(args));
    }

    [Fact]
    public void EmptyProfileSerializesExactFallbackZeroLiveAllowlistAndTombstoneAccounting()
    {
        HnswAllowlistFilteringBenchmarkReport report = RunSmallReport("empty");
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        AssertSchema(root, "empty");
        Assert.Equal(0, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.Equal(0, report.Allowlist.DuplicateInputIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.UnknownIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.TombstonedInputIdCountPerQuery);
        Assert.Equal(report.Workload.QueryCount, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(0, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("exactFallback", report.Branches.ExpectedBranch);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.ExactLiveViewTruthGenerated);
        Assert.True(report.Validation.ExactFallbackParityPassedForAllSearches);
        Assert.True(report.Validation.TombstoneSuppressionPassed);

        foreach (JsonElement section in EnumerateSearchSections(root))
        {
            Assert.Equal("passed", GetString(section.GetProperty("exactFallbackValidation"), "status"));
            Assert.Equal("notApplicable", GetString(section.GetProperty("broadEmissionValidation"), "status"));
            Assert.Equal("passed", GetString(section.GetProperty("returnedResultIntegrity"), "status"));
            Assert.Equal(0, section.GetProperty("returnedResultIntegrity").GetProperty("checkedResultCount").GetInt32());
            Assert.Equal(0, section.GetProperty("underfill").GetProperty("totalReturnedResults").GetInt32());
            Assert.Equal(0, section.GetProperty("underfill").GetProperty("totalExactTruthAvailableResults").GetInt32());
            Assert.Equal("passed", GetString(section.GetProperty("tombstoneSuppression"), "status"));
            Assert.Equal(1, section.GetProperty("tombstoneSuppression").GetProperty("baseTombstoneInputCountPerQuery").GetInt32());
            Assert.Equal(0, section.GetProperty("tombstoneSuppression").GetProperty("returnedBaseTombstoneCount").GetInt32());
            Assert.Equal(0, section.GetProperty("tombstoneSuppression").GetProperty("returnedDeltaTombstoneCount").GetInt32());
            AssertSearchCallMeasurement(section);
        }

        AssertCompositeDeltaScanShape(root);
        AssertPrivateNoClaimPosture(root);
    }

    [Fact]
    public void VerySelectiveFallbackMaintainsExactTruthParityAcrossAllSections()
    {
        HnswAllowlistFilteringBenchmarkReport report = RunSmallReport("very-selective");

        Assert.Equal("very-selective", report.Allowlist.Profile);
        Assert.Equal(report.Workload.TopK - 1, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.Equal(1, report.Allowlist.DuplicateInputIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.UnknownIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.TombstonedInputIdCountPerQuery);
        Assert.Equal(report.Workload.QueryCount, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(0, report.Branches.BroadEmissionQueryCount);

        foreach (HnswAllowlistSearchSectionInfo section in GetTypedSections(report))
        {
            Assert.Equal("passed", section.ExactFallbackValidation.Status);
            Assert.Equal(report.Workload.QueryCount, section.ExactFallbackValidation.QueryCount);
            Assert.Equal(0, section.ExactFallbackValidation.CountMismatchCount);
            Assert.Equal(0, section.ExactFallbackValidation.IdOrOrderMismatchCount);
            Assert.Equal(0, section.ExactFallbackValidation.DistanceMismatchCount);
            Assert.Equal("notApplicable", section.BroadEmissionValidation.Status);
            Assert.Equal("passed", section.ReturnedResultIntegrity.Status);
            Assert.Equal(0, section.ReturnedResultIntegrity.UnknownIdCount);
            Assert.Equal(0, section.ReturnedResultIntegrity.TombstonedIdCount);
            Assert.Equal(0, section.ReturnedResultIntegrity.NotAllowedIdCount);
            Assert.Equal(report.Workload.QueryCount * report.Allowlist.KnownLiveAllowedCountPerQuery, section.Underfill.TotalReturnedResults);
            Assert.Equal(section.Underfill.TotalReturnedResults, section.Underfill.TotalExactTruthAvailableResults);
            Assert.Equal(0, section.Underfill.UnderfilledSlotCount);
        }

        Assert.True(report.Parity.ImmutableOpenedHnsw.AllResultsMatched);
        Assert.True(report.Parity.RebuiltCompositeCheckpointOpenedHnsw.AllResultsMatched);
        Assert.True(report.Validation.ExactFallbackParityPassedForAllSearches);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonArtifactEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void AllProfileBroadEmissionRecordsIntegrityRecallUnderfillAndDeltaScanHonestly()
    {
        HnswAllowlistFilteringBenchmarkReport report = RunSmallReport("all");
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        AssertSchema(root, "all");
        Assert.Equal(report.Workload.BaseVectorCount + report.Workload.InsertedDeltaVectorCount - report.Workload.DeletedBaseVectorCount - report.Workload.DeletedDeltaVectorCount, report.Allowlist.KnownLiveAllowedCountPerQuery);
        Assert.True(report.Allowlist.KnownLiveAllowedCountPerQuery > report.Hnsw.EfSearch);
        Assert.Equal(1, report.Allowlist.DuplicateInputIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.UnknownIdCountPerQuery);
        Assert.Equal(1, report.Allowlist.TombstonedInputIdCountPerQuery);
        Assert.Equal(0, report.Branches.ExactFallbackQueryCount);
        Assert.Equal(report.Workload.QueryCount, report.Branches.BroadEmissionQueryCount);
        Assert.Equal("broadEmission", report.Branches.ExpectedBranch);
        Assert.True(report.Validation.BroadEmissionIntegrityPassedForAllSearches);
        Assert.True(report.Validation.ReturnedResultIntegrityPassedForAllSearches);

        foreach (JsonElement section in EnumerateSearchSections(root))
        {
            Assert.Equal("notApplicable", GetString(section.GetProperty("exactFallbackValidation"), "status"));
            JsonElement broad = section.GetProperty("broadEmissionValidation");
            Assert.Equal("passed", GetString(broad, "status"));
            Assert.InRange(broad.GetProperty("recallAtK").GetDouble(), 0, 1);
            Assert.InRange(broad.GetProperty("orderedAgreement").GetDouble(), 0, 1);
            Assert.True(broad.GetProperty("missingResultCount").GetInt32() >= 0);
            Assert.Equal(0, broad.GetProperty("distanceMismatchCount").GetInt32());

            JsonElement integrity = section.GetProperty("returnedResultIntegrity");
            Assert.Equal("passed", GetString(integrity, "status"));
            Assert.Equal(0, integrity.GetProperty("unknownIdCount").GetInt32());
            Assert.Equal(0, integrity.GetProperty("tombstonedIdCount").GetInt32());
            Assert.Equal(0, integrity.GetProperty("notAllowedIdCount").GetInt32());
            Assert.Equal(0, integrity.GetProperty("duplicateIdCount").GetInt32());

            JsonElement underfill = section.GetProperty("underfill");
            int returned = underfill.GetProperty("totalReturnedResults").GetInt32();
            int exactAvailable = underfill.GetProperty("totalExactTruthAvailableResults").GetInt32();
            int requested = underfill.GetProperty("totalRequestedResultSlots").GetInt32();
            Assert.InRange(returned, 0, requested);
            Assert.Equal(requested, exactAvailable);
            Assert.Equal(requested - returned, underfill.GetProperty("underfilledSlotCount").GetInt32());
            AssertSearchCallMeasurement(section);
        }

        JsonElement searches = root.GetProperty("searches");
        Assert.Equal(report.Workload.QueryCount * (report.Workload.InsertedDeltaVectorCount - report.Workload.DeletedDeltaVectorCount),
            searches.GetProperty("sourceComposite").GetProperty("exactFilteredDeltaScan").GetProperty("totalLiveDeltaScannedCount").GetInt32());
        Assert.True(searches.GetProperty("sourceComposite").GetProperty("exactFilteredDeltaScan").GetProperty("totalAllowedLiveDeltaCount").GetInt32() > 0);
        Assert.Equal("measuredZeroAfterCheckpoint", GetString(searches.GetProperty("rebuiltComposite").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("notApplicable", GetString(searches.GetProperty("checkpointOpenedHnsw").GetProperty("exactFilteredDeltaScan"), "status"));
        AssertPrivateNoClaimPosture(root);
    }

    private static HnswAllowlistFilteringBenchmarkReport RunSmallReport(string profile)
    {
        string directory = NewArtifactDirectory(profile);
        string[] args =
        [
            HnswAllowlistFilteringOptions.ScenarioName,
            "--dimension", "7",
            "--vectors", "24",
            "--queries", "3",
            "--top-k", "4",
            "--insertions", "6",
            "--deletes", "3",
            "--delta-deletes", "1",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "1",
            "--repeated-deletes", "1",
            "--filter", profile,
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED1491",
            "--m", "2",
            "--ef-construction", "8",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000001491",
            "--output", Path.Combine(directory, "report.json"),
            "--opened-index-directory", Path.Combine(directory, "opened"),
            "--checkpoint-directory", Path.Combine(directory, "checkpoint")
        ];

        HnswAllowlistFilteringOptions options = CommandLine.ParseHnswAllowlistFiltering(args);
        HnswAllowlistFilteringBenchmarkReport report = HnswAllowlistFilteringScenario.Run(options, args);
        HnswAllowlistFilteringScenario.Write(report, options.OutputPath);

        Assert.True(File.Exists(options.OutputPath));
        Assert.Equal("passed", report.Validation.Status);
        return report;
    }

    private static IEnumerable<HnswAllowlistSearchSectionInfo> GetTypedSections(HnswAllowlistFilteringBenchmarkReport report)
    {
        yield return report.Searches.ImmutableHnsw;
        yield return report.Searches.OpenedHnsw;
        yield return report.Searches.SourceComposite;
        yield return report.Searches.RebuiltComposite;
        yield return report.Searches.CheckpointOpenedHnsw;
    }

    private static IEnumerable<JsonElement> EnumerateSearchSections(JsonElement root)
    {
        JsonElement searches = root.GetProperty("searches");
        yield return searches.GetProperty("immutableHnsw");
        yield return searches.GetProperty("openedHnsw");
        yield return searches.GetProperty("sourceComposite");
        yield return searches.GetProperty("rebuiltComposite");
        yield return searches.GetProperty("checkpointOpenedHnsw");
    }

    private static void AssertSchema(JsonElement root, string profile)
    {
        Assert.Equal("VecNet.HnswAllowlistFilteringBenchmarkReport", GetString(root, "schemaName"));
        Assert.Equal("0.1", GetString(root, "schemaVersion"));
        Assert.Equal("VEC-149", GetString(root, "taskId"));
        Assert.Equal(HnswAllowlistFilteringOptions.ScenarioName, GetString(root, "scenarioName"));
        Assert.Equal("private-raw", GetString(root, "privacyClass"));
        Assert.Equal("local-evidence", GetString(root, "claimClass"));
        Assert.Equal(profile, GetString(root.GetProperty("allowlist"), "profile"));
        Assert.Equal(HnswAllowlistFilteringOptions.ScenarioName, GetString(root.GetProperty("command"), "scenario"));
    }

    private static void AssertCompositeDeltaScanShape(JsonElement root)
    {
        JsonElement searches = root.GetProperty("searches");
        Assert.Equal("measured", GetString(searches.GetProperty("sourceComposite").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("measuredZeroAfterCheckpoint", GetString(searches.GetProperty("rebuiltComposite").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("notApplicable", GetString(searches.GetProperty("immutableHnsw").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("notApplicable", GetString(searches.GetProperty("openedHnsw").GetProperty("exactFilteredDeltaScan"), "status"));
        Assert.Equal("notApplicable", GetString(searches.GetProperty("checkpointOpenedHnsw").GetProperty("exactFilteredDeltaScan"), "status"));
    }

    private static void AssertSearchCallMeasurement(JsonElement section)
    {
        JsonElement measurement = section.GetProperty("measurement");
        Assert.Equal("measured", GetString(measurement.GetProperty("latency"), "status"));
        Assert.Equal("perMeasuredSearchCall", GetString(measurement.GetProperty("latency"), "sampleScope"));
        Assert.Equal(GetString(section, "timedOperation"), GetString(measurement.GetProperty("latency"), "timedOperation"));
        Assert.Contains("allowlist generation", GetString(measurement.GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("checkpoint/rebuild", GetString(measurement.GetProperty("latency"), "excludedOperations"), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", GetString(measurement.GetProperty("managedAllocations"), "status"));
        Assert.Equal("bytesPerSearchCall", GetString(measurement.GetProperty("managedAllocations"), "unit"));
        Assert.Equal("notMeasured", GetString(measurement.GetProperty("memory"), "status"));
        Assert.Equal("singleRun", GetString(measurement.GetProperty("repeatedRuns"), "status"));
        Assert.Equal("absent", GetString(measurement.GetProperty("warmup"), "status"));
    }

    private static void AssertPrivateNoClaimPosture(JsonElement root)
    {
        Assert.Equal("notMeasured", GetString(root.GetProperty("memory"), "status"));
        Assert.Equal("absent", GetString(root.GetProperty("memory"), "value"));
        Assert.Equal("passed", GetString(root.GetProperty("validation"), "status"));
        Assert.True(root.GetProperty("validation").GetProperty("memoryNotMeasured").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(
            root,
            "publicClaimEligible",
            "baselineCandidateEligible",
            "comparisonArtifactEligible",
            "regressionGateEligible");
        AssertNoPropertyNamed(root, "manifest", "outputDir", "fashionMnist", "hnswlib");
    }

    private static string NewArtifactDirectory(string profile)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"test-agent-vec149-{profile}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

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
