using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec037HnswGeneratedMatrixTests
{
    [Fact]
    public void ParseHnswGeneratedMatrix_UsesBoundedSmokeDefaults()
    {
        HnswGeneratedMatrixOptions options = CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(128, options.VectorCount);
        Assert.Equal(4, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2037u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.Equal(Path.Combine(options.OutputDirectory, "hnsw-matrix-manifest.json"), options.ManifestPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnsw-generated-matrix", "--preset", "large")]
    [InlineData("hnsw-generated-matrix", "--preset", "standard", "--vectors", "49")]
    [InlineData("hnsw-generated-matrix", "--vectors", "9")]
    [InlineData("hnsw-generated-matrix", "--queries", "0")]
    [InlineData("hnsw-generated-matrix", "--runs", "0")]
    [InlineData("hnsw-generated-matrix", "--runs", "6")]
    [InlineData("hnsw-generated-matrix", "--warmup-queries", "-1")]
    [InlineData("hnsw-generated-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("hnsw-generated-matrix", "--dimension", "32")]
    [InlineData("hnsw-generated-matrix", "--top-k", "10")]
    [InlineData("hnsw-generated-matrix", "--m", "4")]
    [InlineData("hnsw-generated-matrix", "--ef-construction", "16")]
    [InlineData("hnsw-generated-matrix", "--ef-search", "10")]
    [InlineData("hnsw-generated-matrix", "--hnsw-seed", "0x37")]
    [InlineData("hnsw-generated-matrix", "--baseline-report-id", "baseline")]
    [InlineData("hnsw-generated-matrix", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnsw-generated-matrix", "--download", "false")]
    [InlineData("hnsw-generated-matrix", "--unknown-option", "1")]
    [InlineData("hnsw-generated-matrix", "--output-dir")]
    [InlineData("hnsw-generated-matrix", "--output-dir", "--manifest")]
    [InlineData("hnsw-generated-matrix", "--output-dir", "")]
    [InlineData("hnsw-generated-matrix", "--manifest", "")]
    public void ParseHnswGeneratedMatrix_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseHnswGeneratedMatrix_AcceptsExplicitStandardPresetWithCallerOptions()
    {
        HnswGeneratedMatrixOptions options = CommandLine.ParseHnswGeneratedMatrix(
            [
                "hnsw-generated-matrix",
                "--preset", "STANDARD",
                "--vectors", "96",
                "--queries", "3",
                "--runs", "2",
                "--warmup-queries", "1",
                "--seed", "0x5EED0370",
                "--output-dir", "VecNet.BenchmarkRunner.Artifacts/hnsw-standard",
                "--manifest", "VecNet.BenchmarkRunner.Artifacts/hnsw-standard/manifest.json"
            ]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(96, options.VectorCount);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(2, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED0370u, options.Seed);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/hnsw-standard", options.OutputDirectory);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/hnsw-standard/manifest.json", options.ManifestPath);
    }

    [Fact]
    public void ExpandCases_SmokePresetCoversDimensionsTopKAndProfiles()
    {
        string outputDirectory = NewArtifactDirectory("expand-smoke");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 2,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0371,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswGeneratedMatrixScenario.HnswMatrixCase[] cases = HnswGeneratedMatrixScenario.ExpandCases(options);

        Assert.Equal(8, cases.Length);
        Assert.Equal([16, 32], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10], cases.Select(item => item.Options.TopK).Distinct().ToArray());
        Assert.Equal(["low-ef-m4", "balanced-m8"], cases.Select(item => item.ProfileName).Distinct().ToArray());
        Assert.Equal(
            Enumerable.Range(0, cases.Length).Select(offset => unchecked(0x5EED0371u + (uint)offset)).ToArray(),
            cases.Select(item => item.Options.Seed).ToArray());
        Assert.Equal(cases.Length, cases.Select(item => item.Options.HnswSeed).Distinct().Count());
        Assert.All(cases, item =>
        {
            Assert.Equal(VectorMetric.SquaredEuclidean, item.Options.Metric);
            Assert.Equal(64, item.Options.VectorCount);
            Assert.Equal(2, item.Options.QueryCount);
            Assert.Equal(3, item.Options.Runs);
            Assert.Equal(1, item.Options.WarmupQueries);
            Assert.True(item.Options.EfSearch >= item.Options.TopK);
            Assert.True(item.Options.EfConstruction >= item.Options.M);
            Assert.StartsWith(outputDirectory, item.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(item.ProfileName, item.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void ExpandCases_StandardPresetIsBroaderAndStillBounded()
    {
        string outputDirectory = NewArtifactDirectory("expand-standard");
        var options = new HnswGeneratedMatrixOptions(
            "standard",
            VectorCount: 96,
            QueryCount: 3,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0xFFFF_FFF0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswGeneratedMatrixScenario.HnswMatrixCase[] cases = HnswGeneratedMatrixScenario.ExpandCases(options);

        Assert.Equal(27, cases.Length);
        Assert.Equal([32, 128, 386], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10, 50], cases.Select(item => item.Options.TopK).Distinct().ToArray());
        Assert.Equal(["low-ef-m4", "balanced-m8", "wide-m16"], cases.Select(item => item.ProfileName).Distinct().ToArray());
        Assert.Contains(cases, item => item.ProfileName == "wide-m16" && item.Options.Dimension == 386 && item.Options.TopK == 50);
        Assert.Equal(0xFFFF_FFF0u, cases[0].Options.Seed);
        Assert.Equal(10u, cases[^1].Options.Seed);
        Assert.EndsWith("case-27-wide-m16-386d-50k.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases, item =>
        {
            Assert.Equal(96, item.Options.VectorCount);
            Assert.Equal(3, item.Options.QueryCount);
            Assert.Equal(2, item.Options.Runs);
            Assert.Equal(1, item.Options.WarmupQueries);
            Assert.True(item.Options.TopK <= item.Options.VectorCount);
            Assert.True(item.Options.EfSearch >= item.Options.TopK);
            Assert.InRange(item.Options.M, 2, 64);
            Assert.InRange(item.Options.EfConstruction, item.Options.M, 4096);
            Assert.InRange(item.Options.EfSearch, 1, 4096);
        });
    }

    [Fact]
    public void CreateCaseArguments_RoundTripThroughHnswGeneratedParser()
    {
        HnswGeneratedMatrixScenario.HnswMatrixCase matrixCase = HnswGeneratedMatrixScenario.ExpandCases(
            new HnswGeneratedMatrixOptions(
                "smoke",
                VectorCount: 64,
                QueryCount: 2,
                Runs: 2,
                WarmupQueries: 1,
                Seed: 0x5EED0372,
                OutputDirectory: NewArtifactDirectory("case-args"),
                ManifestPath: "manifest.json"))[3];

        string[] arguments = HnswGeneratedMatrixScenario.CreateCaseArguments(matrixCase.Options);
        HnswGeneratedOptions parsed = CommandLine.ParseHnswGenerated(arguments);

        Assert.Equal("hnsw-generated", arguments[0]);
        Assert.Equal(matrixCase.Options, parsed);
        Assert.Contains("--m", arguments);
        Assert.Contains("--ef-construction", arguments);
        Assert.Contains("--ef-search", arguments);
        Assert.Contains("--hnsw-seed", arguments);
        Assert.Contains("--output", arguments);
    }

    [Fact]
    public void Run_WritesPerCaseHnswReportsAndDistinctPrivateManifest()
    {
        string outputDirectory = NewArtifactDirectory("run-smoke");
        string manifestPath = Path.Combine(outputDirectory, "hnsw-matrix-manifest.json");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 1,
            Seed: 0x5EED0373,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "hnsw-generated-matrix",
            "--vectors", "64",
            "--queries", "2",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED0373",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(options, arguments);
        HnswGeneratedMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.HnswBenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-037", manifest.TaskId);
        Assert.Equal("hnsw-generated-matrix", manifest.ScenarioName);
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
            Assert.Equal("SquaredEuclidean", matrixCase.Metric);
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.ReportId);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);
            Assert.Equal(64, matrixCase.VectorCount);
            Assert.Equal(2, matrixCase.QueryCount);
            Assert.Equal(1, matrixCase.Runs);
            Assert.Equal(1, matrixCase.WarmupQueries);
            Assert.True(matrixCase.EfSearch >= matrixCase.TopK);
            Assert.Null(matrixCase.ErrorMessage);
        });

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal("VecNet.HnswBenchmarkMatrixManifest", manifestRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-037", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal(8, manifestRoot.GetProperty("caseCount").GetInt32());
        AssertFalseMatrixEligibility(manifestRoot);
        AssertNoComparisonOrBaselineFields(manifestRoot);

        HnswGeneratedMatrixCaseManifest sampledCase = Assert.Single(
            manifest.Cases,
            item => item.Dimension == 32 && item.TopK == 10 && item.ProfileName == "balanced-m8");
        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(sampledCase.ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.HnswBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-036", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("hnsw-generated", reportRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("hnsw-generated", reportRoot.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal(sampledCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
        Assert.Equal(sampledCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
        Assert.Equal(sampledCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
        Assert.Equal(sampledCase.TopK, reportRoot.GetProperty("truth").GetProperty("depth").GetInt32());
        Assert.Equal(sampledCase.M, reportRoot.GetProperty("hnsw").GetProperty("m").GetInt32());
        Assert.Equal(sampledCase.EfConstruction, reportRoot.GetProperty("hnsw").GetProperty("efConstruction").GetInt32());
        Assert.Equal(sampledCase.EfSearch, reportRoot.GetProperty("hnsw").GetProperty("efSearch").GetInt32());
        Assert.Equal(sampledCase.HnswSeed, reportRoot.GetProperty("hnsw").GetProperty("randomSeed").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoComparisonOrBaselineFields(reportRoot);
    }

    [Fact]
    public void Run_RecordsFailedCasesWithoutAbortingEntireMatrix()
    {
        string outputDirectory = NewArtifactDirectory("failed-case");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 5,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0374,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(options, ["hnsw-generated-matrix"]);

        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(4, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(4, manifest.Cases.Count(item => item.Status == "passed"));
        Assert.Equal(4, manifest.Cases.Count(item => item.Status == "failed"));
        Assert.All(manifest.Cases.Where(item => item.TopK == 10), matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.NotNull(matrixCase.ErrorMessage);
            Assert.Contains("top-k", matrixCase.ErrorMessage!, StringComparison.OrdinalIgnoreCase);
        });
        Assert.All(manifest.Cases.Where(item => item.TopK == 1), matrixCase =>
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.ReportId);
            Assert.Null(matrixCase.ErrorMessage);
            Assert.True(File.Exists(matrixCase.ReportPath));
        });
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseMatrix(["exact-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseComparison(["compare-generated-exact", "--baseline", "a.json", "--current", "b.json"]);
        _ = CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--download", "false"]);
        _ = CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--cache-root", "VecNet.DatasetCache"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseMatrix(["exact-generated-matrix", "--m", "4"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--preset", "smoke"]));
        Assert.Equal("hnsw-generated-matrix", HnswGeneratedMatrixOptions.ScenarioName);
        Assert.Equal("external-fashion-mnist-exact", FashionMnistExternalExactBenchmarkOptions.ScenarioName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec037-" + prefix + "-" + Guid.NewGuid().ToString("N"));
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

    private static void AssertNoComparisonOrBaselineFields(JsonElement element)
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
            "workingSetBytes");
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
