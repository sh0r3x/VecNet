using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec076DurableHnswGeneratedMatrixTests
{
    [Fact]
    public void ParseDurableHnswGeneratedMatrix_UsesBoundedSmokeDefaults()
    {
        DurableHnswGeneratedMatrixOptions options = CommandLine.ParseDurableHnswGeneratedMatrix(["hnsw-generated-durable-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(0x5EED0750u, options.Seed);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnsw-generated-durable-matrix"), options.OutputDirectory);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.Equal(Path.Combine(options.OutputDirectory, "durable-hnsw-matrix-manifest.json"), options.ManifestPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnsw-generated-durable-matrix", "--preset", "standard")]
    [InlineData("hnsw-generated-durable-matrix", "--preset", "large")]
    [InlineData("hnsw-generated-durable-matrix", "--seed", "0xNOTHEX")]
    [InlineData("hnsw-generated-durable-matrix", "--output-dir")]
    [InlineData("hnsw-generated-durable-matrix", "--output-dir", "--manifest")]
    [InlineData("hnsw-generated-durable-matrix", "--output-dir", "")]
    [InlineData("hnsw-generated-durable-matrix", "--manifest", "")]
    [InlineData("hnsw-generated-durable-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("hnsw-generated-durable-matrix", "--dimension", "32")]
    [InlineData("hnsw-generated-durable-matrix", "--vectors", "64")]
    [InlineData("hnsw-generated-durable-matrix", "--queries", "2")]
    [InlineData("hnsw-generated-durable-matrix", "--top-k", "10")]
    [InlineData("hnsw-generated-durable-matrix", "--runs", "1")]
    [InlineData("hnsw-generated-durable-matrix", "--warmup-queries", "0")]
    [InlineData("hnsw-generated-durable-matrix", "--m", "8")]
    [InlineData("hnsw-generated-durable-matrix", "--ef-construction", "64")]
    [InlineData("hnsw-generated-durable-matrix", "--ef-search", "64")]
    [InlineData("hnsw-generated-durable-matrix", "--hnsw-seed", "0x484E0DBA43050001")]
    [InlineData("hnsw-generated-durable-matrix", "--output", "case.json")]
    [InlineData("hnsw-generated-durable-matrix", "--snapshot-directory", "snapshot")]
    [InlineData("hnsw-generated-durable-matrix", "--baseline-report-id", "baseline")]
    [InlineData("hnsw-generated-durable-matrix", "--cache-root", "VecNet.DatasetCache")]
    public void ParseDurableHnswGeneratedMatrix_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGeneratedMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseDurableHnswGeneratedMatrix_NormalizesCasedSmokePreset()
    {
        string outputDirectory = NewArtifactDirectory("parse-cased");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        DurableHnswGeneratedMatrixOptions options = CommandLine.ParseDurableHnswGeneratedMatrix(
            [
                "HNSW-GENERATED-DURABLE-MATRIX",
                "--PRESET", "SMOKE",
                "--SEED", "0x5EED0760",
                "--OUTPUT-DIR", outputDirectory,
                "--MANIFEST", manifestPath
            ]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(0x5EED0760u, options.Seed);
        Assert.Equal(outputDirectory, options.OutputDirectory);
        Assert.Equal(manifestPath, options.ManifestPath);
    }

    [Fact]
    public void ExpandCases_SmokePresetMatchesVec075CasesAndSeedDerivation()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        var options = new DurableHnswGeneratedMatrixOptions(
            "smoke",
            Seed: 0x5EED0750,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        DurableHnswGeneratedMatrixScenario.DurableHnswGeneratedMatrixCase[] cases =
            DurableHnswGeneratedMatrixScenario.ExpandCases(options);

        Assert.Equal(4, cases.Length);
        Assert.Equal(
            [
                "case-001-low-ef-m4-16d-64v-3q-5k",
                "case-002-balanced-m8-32d-128v-4q-10k",
                "case-003-wide-m12-128d-192v-5q-25k",
                "case-004-tail-balanced-m8-386d-96v-3q-25k"
            ],
            cases.Select(item => item.CaseId).ToArray());
        Assert.Equal(["low-ef-m4", "balanced-m8", "wide-m12", "tail-balanced-m8"], cases.Select(item => item.ProfileName).ToArray());
        Assert.Equal([16, 32, 128, 386], cases.Select(item => item.Options.Dimension).ToArray());
        Assert.Equal([64, 128, 192, 96], cases.Select(item => item.Options.VectorCount).ToArray());
        Assert.Equal([3, 4, 5, 3], cases.Select(item => item.Options.QueryCount).ToArray());
        Assert.Equal([5, 10, 25, 25], cases.Select(item => item.Options.TopK).ToArray());
        Assert.Equal([4, 8, 12, 8], cases.Select(item => item.Options.M).ToArray());
        Assert.Equal([16, 32, 64, 64], cases.Select(item => item.Options.EfConstruction).ToArray());
        Assert.Equal([8, 24, 64, 64], cases.Select(item => item.Options.EfSearch).ToArray());
        Assert.Equal([1, 1, 2, 1], cases.Select(item => item.Options.Runs).ToArray());
        Assert.Equal([0, 1, 1, 1], cases.Select(item => item.Options.WarmupQueries).ToArray());
        Assert.Equal([0x5EED0751u, 0x5EED0752u, 0x5EED0753u, 0x5EED0754u], cases.Select(item => item.Options.Seed).ToArray());
        Assert.Equal([0x484E0DBA43050001UL, 0x484E0DBA43050002UL, 0x484E0DBA43050003UL, 0x484E0DBA43050004UL], cases.Select(item => item.Options.HnswSeed).ToArray());

        Assert.All(cases, matrixCase =>
        {
            Assert.Equal(VectorMetric.SquaredEuclidean, matrixCase.Options.Metric);
            Assert.True(matrixCase.Options.VectorCount >= matrixCase.Options.TopK);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.True(matrixCase.Options.EfConstruction >= matrixCase.Options.M);
            Assert.InRange(matrixCase.Options.M, 2, 64);
            Assert.InRange(matrixCase.Options.EfConstruction, matrixCase.Options.M, 4096);
            Assert.InRange(matrixCase.Options.EfSearch, matrixCase.Options.TopK, 4096);
            Assert.StartsWith(outputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(outputDirectory, matrixCase.Options.SnapshotDirectory, StringComparison.OrdinalIgnoreCase);
            Assert.False(Path.IsPathRooted(matrixCase.CaseId));
        });
    }

    [Fact]
    public void CreateCaseArguments_RoundTripThroughDurableHnswParser()
    {
        DurableHnswGeneratedMatrixScenario.DurableHnswGeneratedMatrixCase matrixCase =
            DurableHnswGeneratedMatrixScenario.ExpandCases(
                new DurableHnswGeneratedMatrixOptions(
                    "smoke",
                    Seed: 0x5EED0750,
                    OutputDirectory: NewArtifactDirectory("case-args"),
                    ManifestPath: "manifest.json"))[2];

        string[] arguments = DurableHnswGeneratedMatrixScenario.CreateCaseArguments(matrixCase.Options);
        DurableHnswGeneratedOptions parsed = CommandLine.ParseDurableHnswGenerated(arguments);

        Assert.Equal("hnsw-generated-durable", arguments[0]);
        Assert.Equal(matrixCase.Options, parsed);
        Assert.Contains("--snapshot-directory", arguments);
        Assert.Contains("--hnsw-seed", arguments);
        Assert.Contains("--output", arguments);
    }

    [Fact]
    public void Run_WritesDurableHnswLinkedReportsAndPrivateMatrixManifest()
    {
        string outputDirectory = NewArtifactDirectory("run-smoke");
        string manifestPath = Path.Combine(outputDirectory, "durable-hnsw-matrix-manifest.json");
        var options = new DurableHnswGeneratedMatrixOptions(
            "SMOKE",
            Seed: 0x5EED0750,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "hnsw-generated-durable-matrix",
            "--preset", "SMOKE",
            "--seed", "0x5EED0750",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        DurableHnswGeneratedMatrixManifest manifest = DurableHnswGeneratedMatrixScenario.Run(options, arguments);
        DurableHnswGeneratedMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.DurableHnswBenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-076", manifest.TaskId);
        Assert.Equal("hnsw-generated-durable-matrix", manifest.ScenarioName);
        Assert.Equal("hnsw-generated-durable-matrix", manifest.Command.Scenario);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(4, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal("passed", manifest.Validation.Status);
        Assert.Equal(4, manifest.Validation.PassedCaseCount);
        Assert.Equal(0, manifest.Validation.FailedCaseCount);
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", manifest.Validation.LinkedReportSchemaName);
        Assert.Equal("0.1", manifest.Validation.LinkedReportSchemaVersion);
        Assert.Equal("hnsw-generated-durable", manifest.Validation.LinkedReportScenarioName);
        Assert.True(manifest.Validation.AllLinkedReportsValidationPassed);
        Assert.True(manifest.Validation.AllLinkedReportsPrivateRaw);
        Assert.True(manifest.Validation.AllLinkedReportsEligibilityFalse);
        Assert.Equal("local-evidence", manifest.Eligibility.ClaimClass);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("smoke", manifest.Eligibility.EvidenceStatus);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.PreviewReadinessEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonArtifactEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Equal(arguments, manifest.Runner.Arguments);
        Assert.True(File.Exists(manifestPath));

        foreach (DurableHnswGeneratedMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.ReportId);
            Assert.Equal("VecNet.DurableHnswBenchmarkReport", matrixCase.LinkedReportSchemaName);
            Assert.Equal("0.1", matrixCase.LinkedReportSchemaVersion);
            Assert.Equal("VEC-074", matrixCase.LinkedReportTaskId);
            Assert.Equal("hnsw-generated-durable", matrixCase.LinkedReportScenarioName);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);
            Assert.True(Directory.Exists(matrixCase.SnapshotDirectory), matrixCase.SnapshotDirectory);
            Assert.Equal("hnsw-generated-durable", matrixCase.CommandArguments[0]);
            Assert.Null(matrixCase.ErrorType);
            Assert.Null(matrixCase.ErrorMessage);

            DurableHnswGeneratedOptions parsed = CommandLine.ParseDurableHnswGenerated(matrixCase.CommandArguments);
            Assert.Equal(matrixCase.Dimension, parsed.Dimension);
            Assert.Equal(matrixCase.VectorCount, parsed.VectorCount);
            Assert.Equal(matrixCase.QueryCount, parsed.QueryCount);
            Assert.Equal(matrixCase.TopK, parsed.TopK);
            Assert.Equal(matrixCase.Runs, parsed.Runs);
            Assert.Equal(matrixCase.WarmupQueries, parsed.WarmupQueries);
            Assert.Equal(matrixCase.ReportPath, parsed.OutputPath);
            Assert.Equal(matrixCase.SnapshotDirectory, parsed.SnapshotDirectory);
        }

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal("VecNet.DurableHnswBenchmarkMatrixManifest", manifestRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-076", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal("hnsw-generated-durable-matrix", manifestRoot.GetProperty("scenarioName").GetString());
        Assert.Equal("smoke", manifestRoot.GetProperty("presetName").GetString());
        Assert.Equal("passed", manifestRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("hnsw-generated-durable", manifestRoot.GetProperty("cases")[0].GetProperty("commandArguments")[0].GetString());
        AssertNoForbiddenScopeFields(manifestRoot);

        DurableHnswGeneratedMatrixCaseManifest sampledCase = manifest.Cases[2];
        DurableHnswBenchmarkReport linkedReport = ReadReport(sampledCase.ReportPath);
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", linkedReport.SchemaName);
        Assert.Equal("0.1", linkedReport.SchemaVersion);
        Assert.Equal("VEC-074", linkedReport.TaskId);
        Assert.Equal("hnsw-generated-durable", linkedReport.ScenarioName);
        Assert.Equal(sampledCase.ReportId, linkedReport.ReportId);
        Assert.Equal(sampledCase.CommandArguments, linkedReport.Command.Arguments);
        Assert.Equal(sampledCase.Dimension, linkedReport.Workload.Dimension);
        Assert.Equal(sampledCase.VectorCount, linkedReport.Workload.VectorCount);
        Assert.Equal(sampledCase.QueryCount, linkedReport.Workload.QueryCount);
        Assert.Equal(sampledCase.TopK, linkedReport.Workload.TopK);
        Assert.Equal(sampledCase.M, linkedReport.Workload.M);
        Assert.Equal(sampledCase.EfConstruction, linkedReport.Workload.EfConstruction);
        Assert.Equal(sampledCase.EfSearch, linkedReport.Workload.EfSearch);
        Assert.Equal(sampledCase.HnswSeed, linkedReport.Workload.HnswSeed);
        Assert.Contains("internal HnswIndex construction and Add calls", linkedReport.Operations.Build.TimedOperation, StringComparison.Ordinal);
        Assert.Equal("internal HnswIndex.Save(directoryPath)", linkedReport.Operations.Save.TimedOperation);
        Assert.Equal("internal HnswIndex.OpenReadOnly(directoryPath)", linkedReport.Operations.Open.TimedOperation);
        Assert.Equal("internal opened HnswIndex.Search(query, results, workspace)", linkedReport.Operations.OpenedSearch.TimedOperation);
        Assert.Equal("notMeasured", linkedReport.Operations.SourceSearch.Status);
        Assert.Equal("passed", linkedReport.Validation.Status);
        Assert.True(linkedReport.Validation.SavedOpenedParity.AllResultsMatched);
        Assert.True(linkedReport.Validation.ReturnedResultIntegrityPassedForSource);
        Assert.True(linkedReport.Validation.ReturnedResultIntegrityPassedForOpened);
        Assert.Equal("passed", linkedReport.Validation.OpenedReadOnlyMutation.Status);
        Assert.True(linkedReport.Validation.OutputBytesScannedOutsideSaveOpenDuration);
        Assert.True(linkedReport.Outputs.SnapshotOutput.TotalBytes > 0);
        Assert.Equal("outsideSaveAndOpenDuration", linkedReport.Outputs.SnapshotOutput.ScanTimingScope);
        Assert.False(linkedReport.Eligibility.PublicClaimEligible);
        Assert.False(linkedReport.Eligibility.PreviewReadinessEligible);
        Assert.False(linkedReport.Eligibility.BaselineCandidateEligible);
        Assert.False(linkedReport.Eligibility.ComparisonArtifactEligible);
        Assert.False(linkedReport.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ProgramRun_ParseFailureAbortsBeforeManifestCreation()
    {
        string outputDirectory = NewArtifactDirectory("parse-abort");
        string manifestPath = Path.Combine(outputDirectory, "manifest-should-not-exist.json");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable-matrix",
                "--preset", "standard",
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(manifestPath));
    }

    [Fact]
    public void ProgramRun_WithBlockedOutputPathRecordsRecoverableFailuresAndReturnsNonZero()
    {
        string directory = NewArtifactDirectory("blocked-output");
        string blockedOutputDirectory = Path.Combine(directory, "not-a-directory");
        string manifestPath = Path.Combine(directory, "failed-manifest.json");
        File.WriteAllText(blockedOutputDirectory, "this file intentionally blocks per-case report paths");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable-matrix",
                "--preset", "smoke",
                "--seed", "0x5EED0750",
                "--output-dir", blockedOutputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        DurableHnswGeneratedMatrixManifest manifest = ReadManifest(manifestPath);
        Assert.Equal("failed", manifest.Validation.Status);
        Assert.Equal(4, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(4, manifest.Aggregate.FailedCaseCount);
        Assert.False(manifest.Validation.AllLinkedReportsValidationPassed);
        Assert.False(manifest.Validation.AllLinkedReportsPrivateRaw);
        Assert.False(manifest.Validation.AllLinkedReportsEligibilityFalse);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.PreviewReadinessEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonArtifactEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ReportId);
            Assert.Null(matrixCase.LinkedReportSchemaName);
            Assert.NotNull(matrixCase.ErrorType);
            Assert.NotNull(matrixCase.ErrorMessage);
            Assert.StartsWith(blockedOutputDirectory, matrixCase.ReportPath, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(matrixCase.ReportPath));
            _ = CommandLine.ParseDurableHnswGenerated(matrixCase.CommandArguments);
        });
    }

    [Fact]
    public void ExistingRunnerParsersRemainCompatibleAndDurableMatrixModeIsIsolated()
    {
        _ = CommandLine.Parse(["exact-generated", "--vectors", "12", "--queries", "1", "--top-k", "3"]);
        _ = CommandLine.ParseGeneratedExactCheckpoint(["generated-exact-checkpoint", "--vectors", "12", "--queries", "1", "--top-k", "3", "--insertions", "2", "--deletes", "2"]);
        _ = CommandLine.ParseGeneratedExactCheckpointMatrix(["generated-exact-checkpoint-matrix", "--preset", "smoke"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--vectors", "10"]);
        _ = CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--vectors", "12", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseDurableHnswGeneratedMatrix(["hnsw-generated-durable-matrix", "--preset", "smoke"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGenerated(["hnsw-generated-durable", "--output-dir", "matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGeneratedMatrix(["hnsw-generated-durable-matrix", "--output", "case.json"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseDurableHnswGeneratedMatrix(["hnsw-generated-durable-matrix", "--vectors", "64"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--snapshot-directory", "snapshot"]));
        Assert.Equal("hnsw-generated-durable-matrix", DurableHnswGeneratedMatrixOptions.ScenarioName);
        Assert.Equal("hnsw-generated-durable", DurableHnswGeneratedOptions.ScenarioName);
    }

    private static DurableHnswGeneratedMatrixManifest ReadManifest(string path)
    {
        DurableHnswGeneratedMatrixManifest? manifest =
            ReportWriter.Deserialize<DurableHnswGeneratedMatrixManifest>(File.ReadAllText(path));
        Assert.NotNull(manifest);
        return manifest;
    }

    private static DurableHnswBenchmarkReport ReadReport(string path)
    {
        DurableHnswBenchmarkReport? report =
            ReportWriter.Deserialize<DurableHnswBenchmarkReport>(File.ReadAllText(path));
        Assert.NotNull(report);
        return report;
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec076-{prefix}-{Guid.NewGuid():N}"));
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
            "storedLabel",
            "labelFilter",
            "allowlistComparison",
            "previewReadinessPassed",
            "previewReadinessStatus",
            "interruption",
            "filter");
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
