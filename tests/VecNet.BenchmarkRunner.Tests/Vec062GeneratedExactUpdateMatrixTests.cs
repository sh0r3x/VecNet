using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec062GeneratedExactUpdateMatrixTests
{
    [Fact]
    public void ParseGeneratedExactUpdateMatrix_UsesBoundedSmokeDefaults()
    {
        GeneratedExactUpdateMatrixOptions options = CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2062u, options.Seed);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(0, options.DuplicateIdsPerQuery);
        Assert.Equal(0, options.UnknownIdsPerQuery);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.Equal(Path.Combine(options.OutputDirectory, "exact-update-matrix-manifest.json"), options.ManifestPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-exact-update-matrix", "--preset", "large")]
    [InlineData("generated-exact-update-matrix", "--runs", "0")]
    [InlineData("generated-exact-update-matrix", "--runs", "6")]
    [InlineData("generated-exact-update-matrix", "--warmup-queries", "-1")]
    [InlineData("generated-exact-update-matrix", "--duplicate-inserts", "-1")]
    [InlineData("generated-exact-update-matrix", "--unknown-deletes", "-1")]
    [InlineData("generated-exact-update-matrix", "--repeated-deletes", "-1")]
    [InlineData("generated-exact-update-matrix", "--duplicate-ids", "-1")]
    [InlineData("generated-exact-update-matrix", "--unknown-ids", "-1")]
    [InlineData("generated-exact-update-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("generated-exact-update-matrix", "--dimension", "32")]
    [InlineData("generated-exact-update-matrix", "--vectors", "64")]
    [InlineData("generated-exact-update-matrix", "--queries", "2")]
    [InlineData("generated-exact-update-matrix", "--top-k", "10")]
    [InlineData("generated-exact-update-matrix", "--insertions", "4")]
    [InlineData("generated-exact-update-matrix", "--deletes", "2")]
    [InlineData("generated-exact-update-matrix", "--allowlist", "broad")]
    [InlineData("generated-exact-update-matrix", "--candidate-set", "selective")]
    [InlineData("generated-exact-update-matrix", "--filter", "broad")]
    [InlineData("generated-exact-update-matrix", "--output", "case.json")]
    [InlineData("generated-exact-update-matrix", "--baseline-report-id", "baseline")]
    [InlineData("generated-exact-update-matrix", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("generated-exact-update-matrix", "--m", "8")]
    [InlineData("generated-exact-update-matrix", "--output-dir")]
    [InlineData("generated-exact-update-matrix", "--output-dir", "--manifest")]
    [InlineData("generated-exact-update-matrix", "--output-dir", "")]
    [InlineData("generated-exact-update-matrix", "--manifest", "")]
    public void ParseGeneratedExactUpdateMatrix_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdateMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseGeneratedExactUpdateMatrix_AcceptsStandardPresetAndCallerOptions()
    {
        GeneratedExactUpdateMatrixOptions options = CommandLine.ParseGeneratedExactUpdateMatrix(
            [
                "generated-exact-update-matrix",
                "--preset", "STANDARD",
                "--runs", "2",
                "--warmup-queries", "3",
                "--seed", "0x5EED0620",
                "--duplicate-inserts", "4",
                "--unknown-deletes", "5",
                "--repeated-deletes", "6",
                "--duplicate-ids", "7",
                "--unknown-ids", "8",
                "--output-dir", "VecNet.BenchmarkRunner.Artifacts/update-standard",
                "--manifest", "VecNet.BenchmarkRunner.Artifacts/update-standard/manifest.json"
            ]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(2, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(0x5EED0620u, options.Seed);
        Assert.Equal(4, options.DuplicateInsertAttempts);
        Assert.Equal(5, options.UnknownDeleteAttempts);
        Assert.Equal(6, options.RepeatedDeleteAttempts);
        Assert.Equal(7, options.DuplicateIdsPerQuery);
        Assert.Equal(8, options.UnknownIdsPerQuery);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/update-standard", options.OutputDirectory);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/update-standard/manifest.json", options.ManifestPath);
    }

    [Fact]
    public void ExpandCases_SmokePresetVariesUpdateWorkloadDimensionsMetricsAndSelectivity()
    {
        string outputDirectory = NewArtifactDirectory("expand-smoke");
        var options = new GeneratedExactUpdateMatrixOptions(
            "smoke",
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0621,
            DuplicateInsertAttempts: 2,
            UnknownDeleteAttempts: 3,
            RepeatedDeleteAttempts: 4,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 2,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactUpdateMatrixScenario.GeneratedExactUpdateMatrixCase[] cases =
            GeneratedExactUpdateMatrixScenario.ExpandCases(options);

        Assert.Equal(4, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine],
            cases.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.True(cases.Select(item => item.Options.BaseVectorCount).Distinct().Count() > 1);
        Assert.True(cases.Select(item => item.Options.InsertedDeltaCount).Distinct().Count() > 1);
        Assert.True(cases.Select(item => item.Options.DeletedBaseCount).Distinct().Count() > 1);
        Assert.True(cases.Select(item => item.Options.TopK).Distinct().Count() > 1);
        Assert.True(cases.Select(item => item.Options.QueryCount).Distinct().Count() > 1);
        Assert.Contains("very-selective", cases.Select(item => item.Options.AllowlistKind));
        Assert.Contains("all", cases.Select(item => item.Options.CandidateSetKind));
        Assert.Equal(
            Enumerable.Range(0, cases.Length).Select(offset => unchecked(0x5EED0621u + (uint)offset)).ToArray(),
            cases.Select(item => item.Options.Seed).ToArray());
        Assert.All(cases, item =>
        {
            Assert.Equal(3, item.Options.Runs);
            Assert.Equal(1, item.Options.WarmupQueries);
            Assert.Equal(2, item.Options.DuplicateInsertAttempts);
            Assert.Equal(3, item.Options.UnknownDeleteAttempts);
            Assert.Equal(4, item.Options.RepeatedDeleteAttempts);
            Assert.Equal(1, item.Options.DuplicateIdsPerQuery);
            Assert.Equal(2, item.Options.UnknownIdsPerQuery);
            Assert.True(item.Options.TopK <= item.Options.BaseVectorCount + item.Options.InsertedDeltaCount - item.Options.DeletedBaseCount);
            Assert.StartsWith(outputDirectory, item.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void ExpandCases_StandardPresetIsBroaderAndStillBounded()
    {
        string outputDirectory = NewArtifactDirectory("expand-standard");
        var options = new GeneratedExactUpdateMatrixOptions(
            "standard",
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0xFFFF_FFF8,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            DuplicateIdsPerQuery: 2,
            UnknownIdsPerQuery: 3,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactUpdateMatrixScenario.GeneratedExactUpdateMatrixCase[] cases =
            GeneratedExactUpdateMatrixScenario.ExpandCases(options);

        Assert.Equal(12, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine],
            cases.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386, 768], cases.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.True(cases.Select(item => item.Options.BaseVectorCount).Distinct().Count() > 3);
        Assert.True(cases.Select(item => item.Options.InsertedDeltaCount).Distinct().Count() > 3);
        Assert.True(cases.Select(item => item.Options.DeletedBaseCount).Distinct().Count() > 3);
        Assert.Contains(100, cases.Select(item => item.Options.TopK));
        Assert.Equal(["all", "broad", "empty", "selective", "very-selective"], cases.Select(item => item.Options.AllowlistKind).Distinct().Order().ToArray());
        Assert.Equal(["all", "broad", "empty", "selective", "very-selective"], cases.Select(item => item.Options.CandidateSetKind).Distinct().Order().ToArray());
        Assert.Equal(0xFFFF_FFF8u, cases[0].Options.Seed);
        Assert.Equal(3u, cases[^1].Options.Seed);
        Assert.EndsWith("case-012-cosine-768d-256b-80i-120d-100k-all-allowlist-broad-candidate-set.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases, item =>
        {
            Assert.True(item.Options.TopK <= item.Options.BaseVectorCount + item.Options.InsertedDeltaCount - item.Options.DeletedBaseCount);
            Assert.True(item.Options.AllowlistKind != "very-selective" || item.Options.TopK > 1);
            Assert.True(item.Options.CandidateSetKind != "very-selective" || item.Options.TopK > 1);
            Assert.Equal(2, item.Options.Runs);
            Assert.Equal(3, item.Options.UnknownIdsPerQuery);
        });
    }

    [Fact]
    public void CreateCaseArguments_RoundTripThroughGeneratedExactUpdateParser()
    {
        GeneratedExactUpdateOptions caseOptions = GeneratedExactUpdateMatrixScenario.ExpandCases(
            new GeneratedExactUpdateMatrixOptions(
                "smoke",
                Runs: 2,
                WarmupQueries: 1,
                Seed: 0x5EED0622,
                DuplicateInsertAttempts: 3,
                UnknownDeleteAttempts: 4,
                RepeatedDeleteAttempts: 5,
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputDirectory: NewArtifactDirectory("case-args"),
                ManifestPath: "manifest.json"))[2].Options;

        string[] arguments = GeneratedExactUpdateMatrixScenario.CreateCaseArguments(caseOptions);
        GeneratedExactUpdateOptions parsed = CommandLine.ParseGeneratedExactUpdate(arguments);

        Assert.Equal("generated-exact-update", arguments[0]);
        Assert.Equal(caseOptions, parsed);
        Assert.Contains("--insertions", arguments);
        Assert.Contains("--deletes", arguments);
        Assert.Contains("--allowlist", arguments);
        Assert.Contains("--candidate-set", arguments);
        Assert.Contains("--output", arguments);
    }

    [Fact]
    public void Run_WritesPerCaseUpdateReportsAndPrivateMatrixManifest()
    {
        string outputDirectory = NewArtifactDirectory("run-smoke");
        string manifestPath = Path.Combine(outputDirectory, "exact-update-matrix-manifest.json");
        var options = new GeneratedExactUpdateMatrixOptions(
            "smoke",
            Runs: 1,
            WarmupQueries: 1,
            Seed: 0x5EED0623,
            DuplicateInsertAttempts: 2,
            UnknownDeleteAttempts: 3,
            RepeatedDeleteAttempts: 2,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 2,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "generated-exact-update-matrix",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED0623",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "2",
            "--duplicate-ids", "1",
            "--unknown-ids", "2",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactUpdateMatrixManifest manifest = GeneratedExactUpdateMatrixScenario.Run(options, arguments);
        GeneratedExactUpdateMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.ExactUpdateBenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-062", manifest.TaskId);
        Assert.Equal("generated-exact-update-matrix", manifest.ScenarioName);
        Assert.Equal("generated-exact-update-matrix", manifest.Command.Scenario);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(4, manifest.Aggregate.PassedCaseCount);
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
            Assert.Equal(matrixCase.BaseVectorCount + matrixCase.InsertedDeltaVectorCount, matrixCase.PhysicalVectorCount);
            Assert.Equal(matrixCase.BaseVectorCount + matrixCase.InsertedDeltaVectorCount - matrixCase.DeletedBaseVectorCount, matrixCase.ExpectedLiveVectorCount);
            Assert.Equal(2, matrixCase.DuplicateInsertAttempts);
            Assert.Equal(3, matrixCase.UnknownDeleteAttempts);
            Assert.Equal(2, matrixCase.RepeatedDeleteAttempts);
            Assert.Equal(1, matrixCase.DuplicateIdCountPerQuery);
            Assert.Equal(2, matrixCase.UnknownIdCountPerQuery);
            Assert.Equal("generated-exact-update", matrixCase.CommandArguments[0]);
            Assert.Null(matrixCase.ErrorMessage);
        });

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal("VecNet.ExactUpdateBenchmarkMatrixManifest", manifestRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-062", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal("generated-exact-update-matrix", manifestRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("smoke", manifestRoot.GetProperty("presetName").GetString());
        Assert.Equal(4, manifestRoot.GetProperty("caseCount").GetInt32());
        Assert.Equal(4, manifestRoot.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, manifestRoot.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("generated-exact-update", manifestRoot.GetProperty("cases")[0].GetProperty("commandArguments")[0].GetString());
        AssertNoForbiddenScopeFields(manifestRoot);

        GeneratedExactUpdateMatrixCaseManifest sampledCase = Assert.Single(
            manifest.Cases,
            item => item.Metric == "Cosine" && item.AllowlistKind == "very-selective");
        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(sampledCase.ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.ExactUpdateBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-061", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("generated-exact-update", reportRoot.GetProperty("scenarioName").GetString());
        Assert.Equal(sampledCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
        Assert.Equal(sampledCase.PhysicalVectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
        Assert.Equal(sampledCase.ExpectedLiveVectorCount, reportRoot.GetProperty("counts").GetProperty("liveVectorCount").GetInt32());
        Assert.Equal(sampledCase.BaseVectorCount, reportRoot.GetProperty("workload").GetProperty("baseVectorCount").GetInt32());
        Assert.Equal(sampledCase.InsertedDeltaVectorCount, reportRoot.GetProperty("workload").GetProperty("insertedDeltaVectorCount").GetInt32());
        Assert.Equal(sampledCase.DeletedBaseVectorCount, reportRoot.GetProperty("counts").GetProperty("tombstoneCount").GetInt32());
        Assert.Equal(sampledCase.AllowlistKind, reportRoot.GetProperty("rawAllowlistInput").GetProperty("kind").GetString());
        Assert.Equal(sampledCase.CandidateSetKind, reportRoot.GetProperty("candidateSetInput").GetProperty("kind").GetString());
        Assert.Equal("constructedAfterMutationsOutsideMeasuredSearch", reportRoot.GetProperty("candidateSet").GetProperty("constructionStatus").GetString());
        Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("mutationLatencyAndAllocation").GetProperty("status").GetString());
        Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("liveViewSave").GetProperty("status").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoForbiddenScopeFields(reportRoot);
    }

    [Fact]
    public void Run_RecordsFailedCasesWithoutAbortingEntireMatrix()
    {
        string outputDirectory = NewArtifactDirectory("failed-case");
        var options = new GeneratedExactUpdateMatrixOptions(
            "smoke",
            Runs: 0,
            WarmupQueries: 0,
            Seed: 0x5EED0624,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactUpdateMatrixManifest manifest = GeneratedExactUpdateMatrixScenario.Run(options, ["generated-exact-update-matrix"]);

        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.NotNull(matrixCase.ErrorMessage);
            Assert.Contains("runs", matrixCase.ErrorMessage!, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(matrixCase.ReportPath));
        });
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndUpdateMatrixModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseMatrix(["exact-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCandidateSetMatrix(["generated-exact-candidate-set-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--preset", "smoke"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseComparison(["compare-generated-exact", "--baseline", "a.json", "--current", "b.json"]);
        _ = CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--download", "false"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdate(["generated-exact-update", "--output-dir", "matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--output", "case.json"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactUpdateMatrix(["generated-exact-update-matrix", "--vectors", "64"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSetMatrix(["generated-exact-candidate-set-matrix", "--insertions", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--unknown-deletes", "1"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--preset", "smoke"]));
        Assert.Equal("generated-exact-update-matrix", GeneratedExactUpdateMatrixOptions.ScenarioName);
        Assert.Equal("generated-exact-update", GeneratedExactUpdateOptions.ScenarioName);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec062-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void AssertNoForbiddenScopeFields(JsonElement element)
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
            "m",
            "hnswSeed",
            "storedLabel",
            "labelFilter",
            "allowlistComparison",
            "checkpointDuration",
            "rebuildDuration",
            "saveCost",
            "mutationLatencyMilliseconds");
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
