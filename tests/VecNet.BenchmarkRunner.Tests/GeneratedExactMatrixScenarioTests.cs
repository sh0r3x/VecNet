using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class GeneratedExactMatrixScenarioTests
{
    [Fact]
    public void ParseMatrix_UsesBoundedSmokeDefaultsUnderPrivateArtifactRoot()
    {
        GeneratedExactMatrixOptions options = CommandLine.ParseMatrix(["exact-generated-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(128, options.VectorCount);
        Assert.Equal(8, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2014u, options.Seed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.Equal(Path.Combine(options.OutputDirectory, "matrix-manifest.json"), options.ManifestPath);
    }

    [Theory]
    [InlineData("exact-generated-matrix", "--preset", "large")]
    [InlineData("exact-generated-matrix", "--preset", "standard", "--vectors", "99")]
    [InlineData("exact-generated-matrix", "--vectors", "9")]
    [InlineData("exact-generated-matrix", "--queries", "0")]
    [InlineData("exact-generated-matrix", "--runs", "-1")]
    [InlineData("exact-generated-matrix", "--warmup-queries", "-1")]
    [InlineData("exact-generated-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("exact-generated-matrix", "--unknown-option", "123")]
    [InlineData("exact-generated-matrix", "--output-dir")]
    [InlineData("exact-generated-matrix", "--output-dir", "--manifest")]
    public void ParseMatrix_RejectsInvalidMatrixCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseMatrix_AcceptsExplicitStandardPresetWithCallerProvidedOptions()
    {
        GeneratedExactMatrixOptions options = CommandLine.ParseMatrix(
            [
                "exact-generated-matrix",
                "--preset", "STANDARD",
                "--vectors", "100",
                "--queries", "3",
                "--runs", "2",
                "--warmup-queries", "1",
                "--seed", "0x5EED0150",
                "--output-dir", "VecNet.BenchmarkRunner.Artifacts/standard",
                "--manifest", "VecNet.BenchmarkRunner.Artifacts/standard/manifest.json"
            ]);

        Assert.Equal("standard", options.PresetName);
        Assert.Equal(100, options.VectorCount);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(2, options.Runs);
        Assert.Equal(1, options.WarmupQueries);
        Assert.Equal(0x5EED0150u, options.Seed);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/standard", options.OutputDirectory);
        Assert.Equal("VecNet.BenchmarkRunner.Artifacts/standard/manifest.json", options.ManifestPath);
    }

    [Fact]
    public void ExpandCases_CoversRequiredMetricsDimensionsAndTopKValues()
    {
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 3,
            WarmupQueries: 1,
            Seed: 0x5EED0140,
            OutputDirectory: "VecNet.BenchmarkRunner.Artifacts/matrix-expand-test",
            ManifestPath: "VecNet.BenchmarkRunner.Artifacts/matrix-expand-test/matrix-manifest.json");

        GeneratedExactSearchOptions[] cases = GeneratedExactMatrixScenario.ExpandCases(options);

        Assert.Equal(18, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine],
            cases.Select(caseOptions => caseOptions.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386], cases.Select(caseOptions => caseOptions.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10], cases.Select(caseOptions => caseOptions.TopK).Distinct().ToArray());
        Assert.All(cases, caseOptions =>
        {
            Assert.Equal(10, caseOptions.VectorCount);
            Assert.Equal(2, caseOptions.QueryCount);
            Assert.Equal(3, caseOptions.Runs);
            Assert.Equal(1, caseOptions.WarmupQueries);
            Assert.StartsWith(options.OutputDirectory, caseOptions.OutputPath);
            Assert.Null(caseOptions.BaselineReportId);
        });
        Assert.Equal(
            Enumerable.Range(0, cases.Length).Select(offset => unchecked(0x5EED0140u + (uint)offset)).ToArray(),
            cases.Select(caseOptions => caseOptions.Seed).ToArray());
    }

    [Fact]
    public void ExpandCases_StandardPresetCoversRequiredShapeAndCallerOptions()
    {
        var options = new GeneratedExactMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 3,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED0151,
            OutputDirectory: "VecNet.BenchmarkRunner.Artifacts/matrix-standard-expand-test",
            ManifestPath: "VecNet.BenchmarkRunner.Artifacts/matrix-standard-expand-test/matrix-manifest.json");

        GeneratedExactSearchOptions[] cases = GeneratedExactMatrixScenario.ExpandCases(options);

        Assert.Equal(36, cases.Length);
        Assert.Equal(
            [VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine],
            cases.Select(caseOptions => caseOptions.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386, 768], cases.Select(caseOptions => caseOptions.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10, 100], cases.Select(caseOptions => caseOptions.TopK).Distinct().ToArray());
        Assert.All(cases, caseOptions =>
        {
            Assert.Equal(100, caseOptions.VectorCount);
            Assert.Equal(3, caseOptions.QueryCount);
            Assert.Equal(2, caseOptions.Runs);
            Assert.Equal(1, caseOptions.WarmupQueries);
            Assert.True(caseOptions.TopK <= caseOptions.VectorCount);
            Assert.StartsWith(options.OutputDirectory, caseOptions.OutputPath);
            Assert.Null(caseOptions.BaselineReportId);
        });
    }

    [Fact]
    public void Run_WritesPerCaseReportsAndManifestWithPrivateSmokeEligibility()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "matrix-test-" + Guid.NewGuid().ToString("N"));
        string manifestPath = Path.Combine(outputDirectory, "matrix-manifest.json");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED0141,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "exact-generated-matrix",
            "--vectors", "10",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "1",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(options, arguments);
        GeneratedExactMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.BenchmarkMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-015", manifest.TaskId);
        Assert.Equal("exact-generated-matrix", manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(18, manifest.CaseCount);
        Assert.Equal(18, manifest.Aggregate.PassedCaseCount);
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
            Assert.True(File.Exists(matrixCase.ReportPath));
            Assert.Equal(10, matrixCase.VectorCount);
            Assert.Equal(2, matrixCase.QueryCount);
            Assert.Equal(2, matrixCase.Runs);
            Assert.Equal(1, matrixCase.WarmupQueries);
            Assert.Null(matrixCase.ErrorMessage);
        });

        using JsonDocument manifestDocument = JsonDocument.Parse(ReportWriter.Serialize(manifest));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal(18, manifestRoot.GetProperty("caseCount").GetInt32());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(manifestRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());

        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(manifest.Cases[0].ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.BenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-014", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
        Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
    }

    [Fact]
    public void Run_StandardPresetWritesManifestWithSelectedPresetAndAllCases()
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "matrix-standard-test-" + Guid.NewGuid().ToString("N"));
        string manifestPath = Path.Combine(outputDirectory, "matrix-manifest.json");
        var options = new GeneratedExactMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0152,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "exact-generated-matrix",
            "--preset", "standard",
            "--vectors", "100",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(options, arguments);
        GeneratedExactMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VEC-015", manifest.TaskId);
        Assert.Equal("standard", manifest.PresetName);
        Assert.Equal(36, manifest.CaseCount);
        Assert.Equal(36, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal([32, 128, 386, 768], manifest.Cases.Select(item => item.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10, 100], manifest.Cases.Select(item => item.TopK).Distinct().ToArray());
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal(100, matrixCase.VectorCount);
            Assert.True(matrixCase.TopK <= matrixCase.VectorCount);
            Assert.NotNull(matrixCase.ReportId);
            Assert.True(File.Exists(matrixCase.ReportPath));
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("standard", root.GetProperty("presetName").GetString());
        Assert.Equal(36, root.GetProperty("caseCount").GetInt32());
        Assert.Equal("local-evidence", root.GetProperty("eligibility").GetProperty("claimClass").GetString());
        Assert.Equal("private-raw", root.GetProperty("eligibility").GetProperty("privacyClass").GetString());
        Assert.Equal("smoke", root.GetProperty("eligibility").GetProperty("evidenceStatus").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }
}
