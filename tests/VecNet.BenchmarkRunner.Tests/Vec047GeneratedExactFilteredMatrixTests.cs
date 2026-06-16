using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec047GeneratedExactFilteredMatrixTests
{
    [Fact]
    public void ParseGeneratedExactFilteredMatrix_UsesBoundedSmokeDefaults()
    {
        GeneratedExactFilteredMatrixOptions options = CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(128, options.VectorCount);
        Assert.Equal(4, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.Equal(0x5EED2047u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.Equal(Path.Combine(options.OutputDirectory, "exact-filtered-matrix-manifest.json"), options.ManifestPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("exact-generated-filtered-matrix", "--preset", "large")]
    [InlineData("exact-generated-filtered-matrix", "--preset", "standard", "--vectors", "99")]
    [InlineData("exact-generated-filtered-matrix", "--vectors", "9")]
    [InlineData("exact-generated-filtered-matrix", "--queries", "0")]
    [InlineData("exact-generated-filtered-matrix", "--runs", "0")]
    [InlineData("exact-generated-filtered-matrix", "--runs", "6")]
    [InlineData("exact-generated-filtered-matrix", "--warmup-queries", "-1")]
    [InlineData("exact-generated-filtered-matrix", "--duplicate-ids", "-1")]
    [InlineData("exact-generated-filtered-matrix", "--unknown-ids", "-1")]
    [InlineData("exact-generated-filtered-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("exact-generated-filtered-matrix", "--dimension", "32")]
    [InlineData("exact-generated-filtered-matrix", "--top-k", "10")]
    [InlineData("exact-generated-filtered-matrix", "--filter", "broad")]
    [InlineData("exact-generated-filtered-matrix", "--output", "case.json")]
    [InlineData("exact-generated-filtered-matrix", "--baseline-report-id", "baseline")]
    [InlineData("exact-generated-filtered-matrix", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("exact-generated-filtered-matrix", "--download", "false")]
    [InlineData("exact-generated-filtered-matrix", "--output-dir")]
    [InlineData("exact-generated-filtered-matrix", "--output-dir", "--manifest")]
    [InlineData("exact-generated-filtered-matrix", "--output-dir", "")]
    [InlineData("exact-generated-filtered-matrix", "--manifest", "")]
    public void ParseGeneratedExactFilteredMatrix_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseGeneratedExactFilteredMatrix_AcceptsStandardPresetAndCallerOptions()
    {
        GeneratedExactFilteredMatrixOptions options = CommandLine.ParseGeneratedExactFilteredMatrix(
            [
                "exact-generated-filtered-matrix",
                "--preset", "STANDARD",
                "--vectors", "100",
                "--queries", "3",
                "--runs", "2",
                "--warmup-queries", "1",
                "--seed", "0x5EED0470",
                "--duplicate-ids", "2",
                "--unknown-ids", "3",
                "--output-dir", "VecNet.BenchmarkRunner.Artifacts/exact-filtered-standard",
                "--manifest", "VecNet.BenchmarkRunner.Artifacts/exact-filtered-standard/manifest.json"
            ]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(100, options.VectorCount);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(2, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED0470u, options.Seed);
        Assert.Equal(2, options.DuplicateIdsPerQuery);
        Assert.Equal(3, options.UnknownIdsPerQuery);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/exact-filtered-standard", options.OutputDirectory);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/exact-filtered-standard/manifest.json", options.ManifestPath);
    }

    [Fact]
    public void ExpandCases_SmokePresetCoversBoundedMetricsDimensionsTopKAndFilters()
    {
        string outputDirectory = NewArtifactDirectory("expand-smoke");
        var options = new GeneratedExactFilteredMatrixOptions(
            "smoke",
            VectorCount: 32,
            QueryCount: 2,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0471,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 2,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixScenario.GeneratedExactFilteredMatrixCase[] cases =
            GeneratedExactFilteredMatrixScenario.ExpandCases(options);

        Assert.Equal(8, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct],
            cases.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal([10], cases.Select(item => item.Options.TopK).Distinct().ToArray());
        Assert.Equal(["broad", "selective"], cases.Select(item => item.Options.FilterKind).Distinct().ToArray());
        Assert.Equal(
            Enumerable.Range(0, cases.Length).Select(offset => unchecked(0x5EED0471u + (uint)offset)).ToArray(),
            cases.Select(item => item.Options.Seed).ToArray());
        Assert.All(cases, item =>
        {
            Assert.Equal(32, item.Options.VectorCount);
            Assert.Equal(2, item.Options.QueryCount);
            Assert.Equal(3, item.Options.Runs);
            Assert.Equal(1, item.Options.WarmupQueries);
            Assert.Equal(1, item.Options.DuplicateIdsPerQuery);
            Assert.Equal(2, item.Options.UnknownIdsPerQuery);
            Assert.StartsWith(outputDirectory, item.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(item.Options.FilterKind, item.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void ExpandCases_StandardPresetIsBroaderAndStillBounded()
    {
        string outputDirectory = NewArtifactDirectory("expand-standard");
        var options = new GeneratedExactFilteredMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0xFFFF_FFF0,
            DuplicateIdsPerQuery: 2,
            UnknownIdsPerQuery: 3,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixScenario.GeneratedExactFilteredMatrixCase[] cases =
            GeneratedExactFilteredMatrixScenario.ExpandCases(options);

        Assert.Equal(120, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine],
            cases.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386, 768], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal([10, 100], cases.Select(item => item.Options.TopK).Distinct().ToArray());
        Assert.Equal(["all", "broad", "selective", "very-selective", "empty"], cases.Select(item => item.Options.FilterKind).Distinct().ToArray());
        Assert.Contains(cases, item => item.Options.Metric == VectorMetric.Cosine && item.Options.Dimension == 768 && item.Options.TopK == 100 && item.Options.FilterKind == "empty");
        Assert.Equal(0xFFFF_FFF0u, cases[0].Options.Seed);
        Assert.Equal(103u, cases[^1].Options.Seed);
        Assert.EndsWith("case-120-cosine-768d-100k-empty.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases, item =>
        {
            Assert.Equal(100, item.Options.VectorCount);
            Assert.Equal(1, item.Options.QueryCount);
            Assert.Equal(2, item.Options.Runs);
            Assert.Equal(1, item.Options.WarmupQueries);
            Assert.True(item.Options.TopK <= item.Options.VectorCount);
            Assert.True(item.Options.FilterKind != "very-selective" || item.Options.TopK > 1);
            Assert.Equal(2, item.Options.DuplicateIdsPerQuery);
            Assert.Equal(3, item.Options.UnknownIdsPerQuery);
        });
    }

    [Fact]
    public void CreateCaseArguments_RoundTripThroughGeneratedExactFilteredParser()
    {
        GeneratedExactFilteredOptions caseOptions = GeneratedExactFilteredMatrixScenario.ExpandCases(
            new GeneratedExactFilteredMatrixOptions(
                "smoke",
                VectorCount: 32,
                QueryCount: 2,
                Runs: 2,
                WarmupQueries: 1,
                Seed: 0x5EED0472,
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputDirectory: NewArtifactDirectory("case-args"),
                ManifestPath: "manifest.json"))[3].Options;

        string[] arguments = GeneratedExactFilteredMatrixScenario.CreateCaseArguments(caseOptions);
        GeneratedExactFilteredOptions parsed = CommandLine.ParseGeneratedExactFiltered(arguments);

        Assert.Equal("exact-generated-filtered", arguments[0]);
        Assert.Equal(caseOptions, parsed);
        Assert.Contains("--filter", arguments);
        Assert.Contains("--duplicate-ids", arguments);
        Assert.Contains("--unknown-ids", arguments);
        Assert.Contains("--output", arguments);
    }

    [Fact]
    public void Run_WritesPerCaseFilteredReportsAndDistinctPrivateManifest()
    {
        string outputDirectory = NewArtifactDirectory("run-smoke");
        string manifestPath = Path.Combine(outputDirectory, "exact-filtered-matrix-manifest.json");
        var options = new GeneratedExactFilteredMatrixOptions(
            "smoke",
            VectorCount: 32,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 1,
            Seed: 0x5EED0473,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 2,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "exact-generated-filtered-matrix",
            "--vectors", "32",
            "--queries", "2",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED0473",
            "--duplicate-ids", "1",
            "--unknown-ids", "2",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(options, arguments);
        GeneratedExactFilteredMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.ExactFilteredBenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-047", manifest.TaskId);
        Assert.Equal("exact-generated-filtered-matrix", manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(8, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal("local-evidence", manifest.Eligibility.ClaimClass);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("smoke", manifest.Eligibility.EvidenceStatus);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Equal(arguments, manifest.Runner.Arguments);
        Assert.True(File.Exists(manifestPath));

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.ReportId);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);
            Assert.Equal(32, matrixCase.VectorCount);
            Assert.Equal(2, matrixCase.QueryCount);
            Assert.Equal(10, matrixCase.TopK);
            Assert.Equal(1, matrixCase.Runs);
            Assert.Equal(1, matrixCase.WarmupQueries);
            Assert.Equal(1, matrixCase.DuplicateIdCountPerQuery);
            Assert.Equal(2, matrixCase.UnknownIdCountPerQuery);
            Assert.Null(matrixCase.ErrorMessage);
        });

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal("VecNet.ExactFilteredBenchmarkMatrixManifest", manifestRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-047", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal("exact-generated-filtered-matrix", manifestRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("smoke", manifestRoot.GetProperty("presetName").GetString());
        Assert.Equal(8, manifestRoot.GetProperty("caseCount").GetInt32());
        AssertFalseMatrixEligibility(manifestRoot);
        AssertNoComparisonBaselineOrExternalFields(manifestRoot);

        GeneratedExactFilteredMatrixCaseManifest sampledCase = Assert.Single(
            manifest.Cases,
            item => item.Metric == "InnerProduct" && item.Dimension == 128 && item.FilterKind == "selective");
        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(sampledCase.ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-046", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("exact-generated-filtered", reportRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("exact-generated-filtered", reportRoot.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal(sampledCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
        Assert.Equal(sampledCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
        Assert.Equal(sampledCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
        Assert.Equal(sampledCase.TopK, reportRoot.GetProperty("truth").GetProperty("depth").GetInt32());
        Assert.Equal(sampledCase.FilterKind, reportRoot.GetProperty("filter").GetProperty("kind").GetString());
        Assert.Equal(sampledCase.DuplicateIdCountPerQuery, reportRoot.GetProperty("filter").GetProperty("duplicateIdCountPerQuery").GetInt32());
        Assert.Equal(sampledCase.UnknownIdCountPerQuery, reportRoot.GetProperty("filter").GetProperty("unknownIdCountPerQuery").GetInt32());
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", reportRoot.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
        Assert.Contains("allowlist generation", reportRoot.GetProperty("measurement").GetProperty("latency").GetProperty("excludedOperations").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity").GetProperty("status").GetString());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoComparisonBaselineOrExternalFields(reportRoot);
    }

    [Fact]
    public void Run_RecordsFailedCasesWithoutAbortingEntireMatrix()
    {
        string outputDirectory = NewArtifactDirectory("failed-case");
        var options = new GeneratedExactFilteredMatrixOptions(
            "smoke",
            VectorCount: 5,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0474,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(options, ["exact-generated-filtered-matrix"]);

        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(8, manifest.Aggregate.FailedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.NotNull(matrixCase.ErrorMessage);
            Assert.Contains("top-k", matrixCase.ErrorMessage!, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(matrixCase.ReportPath));
        });
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseMatrix(["exact-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseComparison(["compare-generated-exact", "--baseline", "a.json", "--current", "b.json"]);
        _ = CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--download", "false"]);
        _ = CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--cache-root", "VecNet.DatasetCache"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseMatrix(["exact-generated-matrix", "--filter", "broad"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--duplicate-ids", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--preset", "smoke"]));
        Assert.Equal("exact-generated-filtered-matrix", GeneratedExactFilteredMatrixOptions.ScenarioName);
        Assert.Equal("exact-generated-filtered", GeneratedExactFilteredOptions.ScenarioName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec047-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertFalseMatrixEligibility(JsonElement root)
    {
        JsonElement eligibility = root.GetProperty("eligibility");
        Assert.Equal("local-evidence", eligibility.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", eligibility.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", eligibility.GetProperty("evidenceStatus").GetString());
        Assert.False(eligibility.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(eligibility.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(eligibility.GetProperty("regressionGateEligible").GetBoolean());
    }

    private static void AssertNoComparisonBaselineOrExternalFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baseline",
            "baselineReportId",
            "candidateEligibility",
            "baselineReportPath",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "latencyDeltaPercent",
            "qpsRatio",
            "allocationDeltaBytes",
            "allocationRatio",
            "regressionPassed",
            "regressionDecision",
            "regressionThreshold",
            "threshold",
            "delta",
            "ratio",
            "publicClaimPassed",
            "publicClaimStatus",
            "cacheRoot",
            "download",
            "truthDepth",
            "residentMemoryBytes",
            "processMemoryBytes",
            "workingSetBytes",
            "hnsw",
            "efSearch",
            "efConstruction",
            "retainedIdMap",
            "precompiledFilter");
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
