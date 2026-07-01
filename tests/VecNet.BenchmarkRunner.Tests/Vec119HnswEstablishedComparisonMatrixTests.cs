using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec119HnswEstablishedComparisonMatrixTests
{
    [Fact]
    public void ParseHnswEstablishedComparisonMatrix_UsesPrivateSmokeDefaultsAndPreservesAcceptedDesign()
    {
        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(["hnswlib-generated-comparison-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(256, options.VectorCount);
        Assert.Equal(4, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2119u, options.Seed);
        Assert.EndsWith(Path.Combine("vec-118-tools", "hnswlib-venv", "Scripts", "python.exe"), options.HnswlibPythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("hnswlib-comparison-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);

        HnswEstablishedComparisonMatrixScenario.HnswComparisonMatrixCase[] cases = HnswEstablishedComparisonMatrixScenario.ExpandCases(options);
        Assert.Equal(9, cases.Length);
        Assert.Equal([128, 384, 768], cases.Select(c => c.Options.Dimension).Distinct().ToArray());
        Assert.DoesNotContain(386, cases.Select(c => c.Options.Dimension));
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(c => c.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.All(cases, matrixCase =>
        {
            Assert.Equal(VectorMetric.SquaredEuclidean, matrixCase.Options.Metric);
            Assert.Equal(10, matrixCase.Options.TopK);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.Equal(options.HnswlibPythonPath, matrixCase.Options.HnswlibPythonPath);
        });

        Assert.Contains(cases, c => c.ProfileName == "balanced-m8" && c.Options.M == 8 && c.Options.EfConstruction == 64 && c.Options.EfSearch == 128);
        Assert.Contains(cases, c => c.ProfileName == "wide-m16" && c.Options.M == 16 && c.Options.EfConstruction == 128 && c.Options.EfSearch == 192);
        Assert.Contains(cases, c => c.ProfileName == "default-m16" && c.Options.M == 16 && c.Options.EfConstruction == 200 && c.Options.EfSearch == 200);
    }

    [Fact]
    public void ExpandCases_StandardPresetIncludesRepresentativeDimensionsProfilesAndTopKOneHundred()
    {
        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(
            [
                "hnswlib-generated-comparison-matrix",
                "--preset", "standard",
                "--vectors", "100",
                "--queries", "1"
            ]);

        HnswEstablishedComparisonMatrixScenario.HnswComparisonMatrixCase[] cases = HnswEstablishedComparisonMatrixScenario.ExpandCases(options);

        Assert.Equal(18, cases.Length);
        Assert.Equal([128, 384, 768], cases.Select(c => c.Options.Dimension).Distinct().ToArray());
        Assert.Equal([10, 100], cases.Select(c => c.Options.TopK).Distinct().ToArray());
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], cases.Select(c => c.ProfileName).Distinct().Order(StringComparer.Ordinal).ToArray());
        Assert.DoesNotContain(386, cases.Select(c => c.Options.Dimension));
        Assert.All(cases, matrixCase => Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK));
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnswlib-generated-comparison-matrix", "--preset", "unknown")]
    [InlineData("hnswlib-generated-comparison-matrix", "--preset", "standard", "--vectors", "99")]
    [InlineData("hnswlib-generated-comparison-matrix", "--vectors", "0")]
    [InlineData("hnswlib-generated-comparison-matrix", "--queries", "0")]
    [InlineData("hnswlib-generated-comparison-matrix", "--runs", "0")]
    [InlineData("hnswlib-generated-comparison-matrix", "--runs", "6")]
    [InlineData("hnswlib-generated-comparison-matrix", "--warmup-queries", "-1")]
    [InlineData("hnswlib-generated-comparison-matrix", "--seed", "0xNOTHEX")]
    [InlineData("hnswlib-generated-comparison-matrix", "--output-dir", "")]
    [InlineData("hnswlib-generated-comparison-matrix", "--manifest", "")]
    [InlineData("hnswlib-generated-comparison-matrix", "--hnswlib-python", "")]
    [InlineData("hnswlib-generated-comparison-matrix", "--dimension", "128")]
    [InlineData("hnswlib-generated-comparison-matrix", "--top-k", "10")]
    [InlineData("hnswlib-generated-comparison-matrix", "--m", "8")]
    [InlineData("hnswlib-generated-comparison-matrix", "--ef-search", "128")]
    [InlineData("hnswlib-generated-comparison-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("hnswlib-generated-comparison", "--output-dir", "matrix")]
    [InlineData("hnswlib-generated-comparison", "--manifest", "manifest.json")]
    public void ParseHnswEstablishedComparisonMatrix_RejectsInvalidOrSingleCaseOptions(params string[] args)
    {
        ArgumentException exception = args[0] switch
        {
            "hnswlib-generated-comparison" => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args)),
            _ => Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparisonMatrix(args))
        };

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_WithMissingHnswlibPythonWritesBlockedManifestWithoutFakeReports()
    {
        string directory = NewArtifactDirectory("missing-tool");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string missingPythonPath = Path.Combine(directory, "tools", "python.exe");
        string[] args =
        [
            "hnswlib-generated-comparison-matrix",
            "--preset", "smoke",
            "--vectors", "16",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--hnswlib-python", missingPythonPath,
            "--output-dir", directory,
            "--manifest", manifestPath
        ];
        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(args);

        HnswEstablishedComparisonMatrixManifest manifest = HnswEstablishedComparisonMatrixScenario.Run(options, args);
        HnswEstablishedComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.True(File.Exists(manifestPath));
        Assert.Equal("VecNet.HnswEstablishedComparisonMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-119", manifest.TaskId);
        Assert.Equal(9, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.SkippedCaseCount);
        Assert.Equal(9, manifest.Aggregate.BlockedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("blocked", matrixCase.Status);
            Assert.Equal("blocked", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.False(File.Exists(matrixCase.LinkedReportPath));
            Assert.Contains("unavailable", matrixCase.ErrorMessage, StringComparison.OrdinalIgnoreCase);
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VEC-119", root.GetProperty("taskId").GetString());
        Assert.Equal(9, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal("hnswlib", root.GetProperty("sourcePinning").GetProperty("packageName").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonPublicationEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("\"status\": \"passed\"", File.ReadAllText(manifestPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ProgramRun_WithMissingHnswlibPythonReturnsFailureAndWritesBlockedManifest()
    {
        string directory = NewArtifactDirectory("missing-tool-program");
        string manifestPath = Path.Combine(directory, "manifest.json");

        int exitCode = BenchmarkRunnerProgram.Run(
            [
                "hnswlib-generated-comparison-matrix",
                "--vectors", "16",
                "--queries", "1",
                "--hnswlib-python", Path.Combine(directory, "missing-python.exe"),
                "--output-dir", directory,
                "--manifest", manifestPath
            ]);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        Assert.Equal(9, document.RootElement.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
    }

    [Fact]
    public void Run_WhenPinnedHnswlibExistsWritesManifestAndCompatibleLinkedReports()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!File.Exists(pythonPath))
        {
            return;
        }

        string directory = NewArtifactDirectory("real-matrix");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            "hnswlib-generated-comparison-matrix",
            "--preset", "smoke",
            "--vectors", "16",
            "--queries", "1",
            "--runs", "1",
            "--warmup-queries", "0",
            "--seed", "0x5EED2119",
            "--hnswlib-python", pythonPath,
            "--output-dir", directory,
            "--manifest", manifestPath
        ];
        HnswEstablishedComparisonMatrixOptions options = CommandLine.ParseHnswEstablishedComparisonMatrix(args);

        HnswEstablishedComparisonMatrixManifest manifest = HnswEstablishedComparisonMatrixScenario.Run(options, args);
        HnswEstablishedComparisonMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal(9, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.Equal([128, 384, 768], manifest.Design.RepresentativeGeneratedDimensions);
        Assert.Equal([386], manifest.Design.OptionalAdversarialTailDimensions);
        Assert.Equal(["balanced-m8", "default-m16", "wide-m16"], manifest.Design.Profiles.Select(p => p.Name).Order(StringComparer.Ordinal).ToArray());
        Assert.Equal("hnswlib", manifest.SourcePinning.PackageName);
        Assert.Equal("PyPI", manifest.SourcePinning.PackageSource);
        Assert.Equal("0.8.0", manifest.SourcePinning.PackageVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", manifest.SourcePinning.SourceDistributionSha256);
        Assert.Equal("Apache-2.0", manifest.SourcePinning.License);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.ComparisonPublicationEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.NotNull(matrixCase.LinkedReportId);
            Assert.True(File.Exists(matrixCase.LinkedReportPath));
            Assert.NotEqual(386, matrixCase.Dimension);
            Assert.Equal("representative", matrixCase.DimensionRole);
            Assert.True(matrixCase.EfSearch >= matrixCase.TopK);
        });

        string firstReportJson = File.ReadAllText(manifest.Cases[0].LinkedReportPath);
        using JsonDocument reportDocument = JsonDocument.Parse(firstReportJson);
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VecNet.HnswEstablishedComparisonReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-118", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("hnswlib", reportRoot.GetProperty("sourcePinning").GetProperty("packageName").GetString());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("comparisonPublicationEligible").GetBoolean());

        string manifestJson = File.ReadAllText(manifestPath);
        Assert.DoesNotContain("\"publicClaimEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"baselineCandidateEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"regressionGateEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec119-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
