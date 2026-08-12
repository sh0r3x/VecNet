using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec076DurableHnswGeneratedMatrixIndependentTests
{
    [Fact]
    public void Parser_EdgesStaySmokeOnlyAndRejectPerCaseOptions()
    {
        DurableHnswGeneratedMatrixOptions defaults = CommandLine.ParseDurableHnswGeneratedMatrix([]);
        Assert.Equal("smoke", defaults.PresetName);
        Assert.Equal(0x5EED0750u, defaults.Seed);
        AssertUnderArtifactRoot(defaults.OutputDirectory);
        Assert.Equal(Path.Combine(defaults.OutputDirectory, "durable-hnsw-matrix-manifest.json"), defaults.ManifestPath);

        string outputDirectory = NewArtifactDirectory("parser-cased");
        string manifestPath = Path.Combine(outputDirectory, "nested", "manifest.json");
        DurableHnswGeneratedMatrixOptions parsed = CommandLine.ParseDurableHnswGeneratedMatrix(
            [
                "HNSW-GENERATED-DURABLE-MATRIX",
                "--PRESET", "sMoKe",
                "--SEED", "4294967295",
                "--OUTPUT-DIR", outputDirectory,
                "--MANIFEST", manifestPath
            ]);

        Assert.Equal("smoke", parsed.PresetName);
        Assert.Equal(uint.MaxValue, parsed.Seed);
        Assert.Equal(outputDirectory, parsed.OutputDirectory);
        Assert.Equal(manifestPath, parsed.ManifestPath);

        foreach ((string Option, string Value) in new[]
        {
            ("--metric", "SquaredEuclidean"),
            ("--dimension", "32"),
            ("--vectors", "64"),
            ("--queries", "2"),
            ("--top-k", "10"),
            ("--runs", "1"),
            ("--warmup-queries", "0"),
            ("--m", "8"),
            ("--ef-construction", "64"),
            ("--ef-search", "64"),
            ("--hnsw-seed", "0x484E0DBA43050001"),
            ("--output", "case.json"),
            ("--snapshot-directory", "snapshot"),
            ("--baseline", "baseline.json"),
            ("--current", "current.json"),
            ("--cache-root", "VecNet.DatasetCache"),
            ("--download", "false")
        })
        {
            ArgumentException exception = Assert.Throws<ArgumentException>(
                () => CommandLine.ParseDurableHnswGeneratedMatrix(
                    ["hnsw-generated-durable-matrix", Option, Value]));
            Assert.Contains(Option, exception.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Theory]
    [InlineData("hnsw-generated-durable-matrix", "--preset", "smoke ")]
    [InlineData("hnsw-generated-durable-matrix", "--preset", "standard")]
    [InlineData("hnsw-generated-durable-matrix", "--seed", "-1")]
    [InlineData("hnsw-generated-durable-matrix", "--seed", "0x")]
    [InlineData("hnsw-generated-durable-matrix", "--manifest", "--output-dir")]
    public void ProgramRun_ParseFailuresDoNotCreateManifest(params string[] commandPrefix)
    {
        string directory = NewArtifactDirectory("parse-abort");
        string manifestPath = Path.Combine(directory, "should-not-exist.json");
        string[] args = commandPrefix.Concat(["--output-dir", directory, "--manifest", manifestPath]).ToArray();

        int exitCode = BenchmarkRunnerProgram.Run(args);

        Assert.Equal(1, exitCode);
        Assert.False(File.Exists(manifestPath));
        Assert.Empty(Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories));
    }

    [Fact]
    public void ExpandCases_UsesVec075SmokeOrderSeedWrapAndAcceptedCaseArguments()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        var options = new DurableHnswGeneratedMatrixOptions(
            "SMOKE",
            Seed: 0xFFFF_FFFE,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        DurableHnswGeneratedMatrixScenario.DurableHnswGeneratedMatrixCase[] cases =
            DurableHnswGeneratedMatrixScenario.ExpandCases(options);

        (string CaseId, VectorMetric Metric, string Profile, int Dimension, int Vectors, int Queries, int TopK, int Runs, int Warmup, string DataSeed, string HnswSeed)[] expected =
        [
            ("case-001-SquaredEuclidean-low-ef-m4-16d-64v-3q-5k", VectorMetric.SquaredEuclidean, "low-ef-m4", 16, 64, 3, 5, 1, 0, "0xFFFFFFFF", "0x484EACA8BBAB0001"),
            ("case-002-SquaredEuclidean-balanced-m8-32d-128v-4q-10k", VectorMetric.SquaredEuclidean, "balanced-m8", 32, 128, 4, 10, 1, 1, "0x00000000", "0x484EACA8BBAB0002"),
            ("case-003-SquaredEuclidean-wide-m12-128d-192v-5q-25k", VectorMetric.SquaredEuclidean, "wide-m12", 128, 192, 5, 25, 2, 1, "0x00000001", "0x484EACA8BBAB0003"),
            ("case-004-SquaredEuclidean-tail-balanced-m8-386d-96v-3q-25k", VectorMetric.SquaredEuclidean, "tail-balanced-m8", 386, 96, 3, 25, 1, 1, "0x00000002", "0x484EACA8BBAB0004"),
            ("case-005-InnerProduct-low-ef-m4-16d-64v-3q-5k", VectorMetric.InnerProduct, "low-ef-m4", 16, 64, 3, 5, 1, 0, "0x00000003", "0x484EACA8BBAB0005"),
            ("case-006-InnerProduct-balanced-m8-32d-128v-4q-10k", VectorMetric.InnerProduct, "balanced-m8", 32, 128, 4, 10, 1, 1, "0x00000004", "0x484EACA8BBAB0006"),
            ("case-007-InnerProduct-wide-m12-128d-192v-5q-25k", VectorMetric.InnerProduct, "wide-m12", 128, 192, 5, 25, 2, 1, "0x00000005", "0x484EACA8BBAB0007"),
            ("case-008-InnerProduct-tail-balanced-m8-386d-96v-3q-25k", VectorMetric.InnerProduct, "tail-balanced-m8", 386, 96, 3, 25, 1, 1, "0x00000006", "0x484EACA8BBAB0008"),
            ("case-009-Cosine-low-ef-m4-16d-64v-3q-5k", VectorMetric.Cosine, "low-ef-m4", 16, 64, 3, 5, 1, 0, "0x00000007", "0x484EACA8BBAB0009"),
            ("case-010-Cosine-balanced-m8-32d-128v-4q-10k", VectorMetric.Cosine, "balanced-m8", 32, 128, 4, 10, 1, 1, "0x00000008", "0x484EACA8BBAB000A"),
            ("case-011-Cosine-wide-m12-128d-192v-5q-25k", VectorMetric.Cosine, "wide-m12", 128, 192, 5, 25, 2, 1, "0x00000009", "0x484EACA8BBAB000B"),
            ("case-012-Cosine-tail-balanced-m8-386d-96v-3q-25k", VectorMetric.Cosine, "tail-balanced-m8", 386, 96, 3, 25, 1, 1, "0x0000000A", "0x484EACA8BBAB000C")
        ];

        Assert.Equal(expected.Length, cases.Length);
        Assert.Equal(expected, cases.Select(item => (
            item.CaseId,
            item.Options.Metric,
            item.ProfileName,
            item.Options.Dimension,
            item.Options.VectorCount,
            item.Options.QueryCount,
            item.Options.TopK,
            item.Options.Runs,
            item.Options.WarmupQueries,
            FormatHex(item.Options.Seed),
            FormatHex(item.Options.HnswSeed))).ToArray());

        Assert.All(cases, matrixCase =>
        {
            Assert.True(matrixCase.Options.VectorCount >= matrixCase.Options.TopK);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.True(matrixCase.Options.EfConstruction >= matrixCase.Options.M);
            AssertUnderArtifactRoot(matrixCase.Options.OutputPath);
            AssertUnderArtifactRoot(matrixCase.Options.SnapshotDirectory);
            Assert.DoesNotContain(Path.GetFullPath(outputDirectory), matrixCase.CaseId, StringComparison.OrdinalIgnoreCase);

            DurableHnswGeneratedOptions replayed = CommandLine.ParseDurableHnswGenerated(
                DurableHnswGeneratedMatrixScenario.CreateCaseArguments(matrixCase.Options));
            Assert.Equal(matrixCase.Options, replayed);
        });
    }

    [Fact]
    public void ProgramRun_WhenOneCaseReportPathIsBlocked_RecordsFailureAndContinues()
    {
        string outputDirectory = NewArtifactDirectory("single-blocked-case");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        var options = new DurableHnswGeneratedMatrixOptions("smoke", 0x5EED0771, outputDirectory, manifestPath);
        DurableHnswGeneratedMatrixScenario.DurableHnswGeneratedMatrixCase blockedCase =
            DurableHnswGeneratedMatrixScenario.ExpandCases(options)[1];
        Directory.CreateDirectory(blockedCase.Options.OutputPath);

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable-matrix",
                "--preset", "smoke",
                "--seed", "0x5EED0771",
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        DurableHnswGeneratedMatrixManifest manifest = ReadManifest(manifestPath);
        Assert.Equal("failed", manifest.Validation.Status);
        Assert.Equal(12, manifest.CaseCount);
        Assert.Equal(11, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(1, manifest.Aggregate.FailedCaseCount);
        Assert.False(manifest.Validation.AllLinkedReportsValidationPassed);
        Assert.False(manifest.Validation.AllLinkedReportsPrivateRaw);
        Assert.False(manifest.Validation.AllLinkedReportsEligibilityFalse);

        DurableHnswGeneratedMatrixCaseManifest failed = Assert.Single(manifest.Cases, item => item.Status == "failed");
        Assert.Equal(2, failed.CaseNumber);
        Assert.Equal(blockedCase.CaseId, failed.CaseId);
        Assert.Equal(blockedCase.Options.OutputPath, failed.ReportPath);
        Assert.Equal("failed", failed.ValidationStatus);
        Assert.Null(failed.ReportId);
        Assert.Null(failed.LinkedReportSchemaName);
        Assert.False(string.IsNullOrWhiteSpace(failed.ErrorType));
        Assert.False(string.IsNullOrWhiteSpace(failed.ErrorMessage));
        Assert.True(Directory.Exists(failed.ReportPath));

        Assert.Equal([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], manifest.Cases.Where(item => item.Status == "passed").Select(item => item.CaseNumber).ToArray());
        Assert.All(manifest.Cases.Where(item => item.Status == "passed"), passed =>
        {
            Assert.Equal("passed", passed.ValidationStatus);
            Assert.NotNull(passed.ReportId);
            Assert.Equal("VecNet.DurableHnswBenchmarkReport", passed.LinkedReportSchemaName);
            Assert.Equal("VEC-074", passed.LinkedReportTaskId);
            Assert.True(File.Exists(passed.ReportPath), passed.ReportPath);
            Assert.True(Directory.Exists(passed.SnapshotDirectory), passed.SnapshotDirectory);
        });
    }

    [Fact]
    public void ManifestJsonAndLinkedReportsKeepDurableMatrixPrivateAndSchemaSeparated()
    {
        string outputDirectory = NewArtifactDirectory("json-shape");
        string manifestPath = Path.Combine(outputDirectory, "durable-hnsw-matrix-manifest.json");
        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable-matrix",
                "--preset", "SMOKE",
                "--seed", "0x5EED0772",
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(0, exitCode);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.DurableHnswBenchmarkMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-076", root.GetProperty("taskId").GetString());
        Assert.Equal("hnsw-generated-durable-matrix", root.GetProperty("scenarioName").GetString());
        Assert.Equal("smoke", root.GetProperty("presetName").GetString());
        Assert.Equal(12, root.GetProperty("caseCount").GetInt32());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("VecNet.DurableHnswBenchmarkReport", root.GetProperty("validation").GetProperty("linkedReportSchemaName").GetString());
        Assert.True(root.GetProperty("validation").GetProperty("allLinkedReportsValidationPassed").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("allLinkedReportsPrivateRaw").GetBoolean());
        Assert.True(root.GetProperty("validation").GetProperty("allLinkedReportsEligibilityFalse").GetBoolean());

        JsonElement eligibility = root.GetProperty("eligibility");
        Assert.Equal("local-evidence", eligibility.GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", eligibility.GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", eligibility.GetProperty("evidenceStatus").GetString());
        AssertFalseEligibility(eligibility);
        Assert.Contains("not reviewed public evidence", eligibility.GetProperty("publicClaimReason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no durable-HNSW baseline-candidate policy", eligibility.GetProperty("baselineCandidateReason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no durable-HNSW comparison schema", eligibility.GetProperty("comparisonArtifactReason").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no durable-HNSW threshold", eligibility.GetProperty("regressionGateReason").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertNoForbiddenScopeFields(root);

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("hnsw-generated-durable", matrixCase.GetProperty("commandArguments")[0].GetString());
            Assert.Equal("passed", matrixCase.GetProperty("status").GetString());
            Assert.Equal("passed", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal("VecNet.DurableHnswBenchmarkReport", matrixCase.GetProperty("linkedReportSchemaName").GetString());
            Assert.Equal("VEC-074", matrixCase.GetProperty("linkedReportTaskId").GetString());
            Assert.Equal("hnsw-generated-durable", matrixCase.GetProperty("linkedReportScenarioName").GetString());

            string reportPath = matrixCase.GetProperty("reportPath").GetString()!;
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(reportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            Assert.Equal("VecNet.DurableHnswBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("VEC-074", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal("hnsw-generated-durable", reportRoot.GetProperty("scenarioName").GetString());
            Assert.False(reportRoot.TryGetProperty("presetName", out _));
            Assert.False(reportRoot.TryGetProperty("cases", out _));
            Assert.Equal("notMeasured", reportRoot.GetProperty("operations").GetProperty("sourceSearch").GetProperty("status").GetString());
            Assert.Equal("outsideSaveAndOpenDuration", reportRoot.GetProperty("outputs").GetProperty("snapshotOutput").GetProperty("scanTimingScope").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("openedReadOnlyMutation").GetProperty("status").GetString());
            AssertFalseEligibility(reportRoot.GetProperty("evidence"));
            AssertFalseEligibility(reportRoot.GetProperty("validation"));
            AssertFalseEligibility(reportRoot.GetProperty("eligibility"));
        }
    }

    [Fact]
    public void DurableMatrixManifest_IsUnsupportedByGeneratedExactComparison()
    {
        string directory = NewArtifactDirectory("comparison-isolation");
        string baselinePath = Path.Combine(directory, "baseline-durable-matrix.json");
        string currentPath = Path.Combine(directory, "current-durable-matrix.json");
        DurableHnswGeneratedMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "baseline-output")), baselinePath);
        DurableHnswGeneratedMatrixScenario.WriteManifest(CreateFastFailedManifest(Path.Combine(directory, "current-output")), currentPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(baselinePath, currentPath, Path.Combine(directory, "comparison.json")),
            ["compare-generated-exact", "--baseline", baselinePath, "--current", currentPath]);

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Metrics);
        Assert.Empty(comparison.Cases);
        Assert.Null(comparison.MatrixSummary);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal(2, comparison.Compatibility.Reasons.Count(reason => reason.Code == "unsupportedSchema"));
        Assert.All(
            comparison.Compatibility.Reasons.Where(reason => reason.Code == "unsupportedSchema"),
            reason =>
            {
                Assert.Equal("schemaName", reason.Field);
                Assert.Equal("VecNet.DurableHnswBenchmarkMatrixManifest", reason.Actual);
            });
    }

    [Fact]
    public void RepeatedExecutionWithSameSeedKeepsCaseAndReportIdentityStable()
    {
        DurableHnswGeneratedMatrixManifest first = RunSmokeMatrix("deterministic-a", 0x5EED0773);
        DurableHnswGeneratedMatrixManifest second = RunSmokeMatrix("deterministic-b", 0x5EED0773);

        Assert.Equal(first.Cases.Length, second.Cases.Length);
        for (int i = 0; i < first.Cases.Length; i++)
        {
            DurableHnswGeneratedMatrixCaseManifest left = first.Cases[i];
            DurableHnswGeneratedMatrixCaseManifest right = second.Cases[i];

            Assert.Equal(left.CaseNumber, right.CaseNumber);
            Assert.Equal(left.CaseId, right.CaseId);
            Assert.Equal(left.ProfileName, right.ProfileName);
            Assert.Equal(left.Metric, right.Metric);
            Assert.Equal(left.Dimension, right.Dimension);
            Assert.Equal(left.VectorCount, right.VectorCount);
            Assert.Equal(left.QueryCount, right.QueryCount);
            Assert.Equal(left.TopK, right.TopK);
            Assert.Equal(left.Runs, right.Runs);
            Assert.Equal(left.WarmupQueries, right.WarmupQueries);
            Assert.Equal(left.DataSeed, right.DataSeed);
            Assert.Equal(left.HnswSeed, right.HnswSeed);
            Assert.Equal(left.M, right.M);
            Assert.Equal(left.EfConstruction, right.EfConstruction);
            Assert.Equal(left.EfSearch, right.EfSearch);
            Assert.Equal(left.ReportId, right.ReportId);
            Assert.Equal(Path.GetFileName(left.ReportPath), Path.GetFileName(right.ReportPath));
            Assert.Equal(Path.GetFileName(left.SnapshotDirectory), Path.GetFileName(right.SnapshotDirectory));
        }

        Assert.Equal(first.Cases.Select(item => item.ReportId).ToArray(), second.Cases.Select(item => item.ReportId).ToArray());
        Assert.Equal(first.Cases.Length, first.Cases.Select(item => item.ReportId).Distinct(StringComparer.Ordinal).Count());
    }

    private static DurableHnswGeneratedMatrixManifest CreateFastFailedManifest(string outputDirectory)
    {
        File.WriteAllText(outputDirectory, "block per-case output directory");
        var options = new DurableHnswGeneratedMatrixOptions(
            "smoke",
            Seed: 0x5EED0774,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(Path.GetDirectoryName(outputDirectory)!, "unused.json"));

        return DurableHnswGeneratedMatrixScenario.Run(options, ["hnsw-generated-durable-matrix"]);
    }

    private static DurableHnswGeneratedMatrixManifest RunSmokeMatrix(string prefix, uint seed)
    {
        string outputDirectory = NewArtifactDirectory(prefix);
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnsw-generated-durable-matrix",
                "--preset", "smoke",
                "--seed", string.Create(CultureInfo.InvariantCulture, $"0x{seed:X8}"),
                "--output-dir", outputDirectory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(0, exitCode);
        return ReadManifest(manifestPath);
    }

    private static DurableHnswGeneratedMatrixManifest ReadManifest(string path)
    {
        DurableHnswGeneratedMatrixManifest? manifest =
            ReportWriter.Deserialize<DurableHnswGeneratedMatrixManifest>(File.ReadAllText(path));
        Assert.NotNull(manifest);
        return manifest;
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec076-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static void AssertUnderArtifactRoot(string path)
    {
        string fullPath = Path.GetFullPath(path);
        string artifactRoot = Path.GetFullPath("VecNet.BenchmarkRunner.Artifacts");
        Assert.StartsWith(artifactRoot + Path.DirectorySeparatorChar, fullPath, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertFalseEligibility(JsonElement section)
    {
        Assert.False(section.GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(section.GetProperty("previewReadinessEligible").GetBoolean());
        Assert.False(section.GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(section.GetProperty("comparisonArtifactEligible").GetBoolean());
        Assert.False(section.GetProperty("regressionGateEligible").GetBoolean());
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
