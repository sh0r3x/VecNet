using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec014IndependentTests
{
    [Theory]
    [InlineData()]
    [InlineData("EXACT-GENERATED-MATRIX", "--PRESET", "SMOKE", "--VECTORS", "10", "--QUERIES", "2", "--RUNS", "2", "--WARMUP-QUERIES", "1", "--SEED", "0X0000002A")]
    [InlineData("exact-generated-matrix", "--vectors", "11", "--vectors", "12", "--queries", "1", "--runs", "1", "--warmup-queries", "0")]
    public void ParseMatrix_EdgeCasesPreserveBoundedSmokeSemantics(params string[] args)
    {
        GeneratedExactMatrixOptions options = CommandLine.ParseMatrix(args);

        Assert.Equal("smoke", options.PresetName, ignoreCase: true);
        Assert.True(options.VectorCount >= GeneratedExactMatrixOptions.MaxTopK);
        Assert.True(options.QueryCount > 0);
        Assert.True(options.Runs > 0);
        Assert.True(options.WarmupQueries >= 0);
        Assert.StartsWith(
            "VecNet.BenchmarkRunner.Artifacts",
            options.OutputDirectory,
            StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.EndsWith("matrix-manifest.json", options.ManifestPath);

        if (args.Count(item => string.Equals(item, "--vectors", StringComparison.OrdinalIgnoreCase)) > 1)
        {
            Assert.Equal(12, options.VectorCount);
        }

        if (args.Contains("--seed", StringComparer.OrdinalIgnoreCase))
        {
            Assert.Equal(42u, options.Seed);
        }
    }

    [Theory]
    [InlineData("exact-generated-matrix", "--preset", " ")]
    [InlineData("exact-generated-matrix", "--preset", "   ")]
    [InlineData("exact-generated-matrix", "--seed", "-1")]
    [InlineData("exact-generated-matrix", "--seed", "0x")]
    [InlineData("exact-generated-matrix", "--manifest", "--output-dir")]
    [InlineData("exact-generated-matrix", "--output", "report.json")]
    [InlineData("exact-generated-matrix", "output-dir", "matrix")]
    public void ParseMatrix_RejectsAdditionalMalformedMatrixEdges(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ExpandCases_HasStableSmokeOrderUniqueReportPathsAndWrappingSeeds()
    {
        string outputDirectory = NewArtifactDirectory("vec014-expand");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0xFFFF_FFF0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactSearchOptions[] cases = GeneratedExactMatrixScenario.ExpandCases(options);

        (VectorMetric Metric, int Dimension, int TopK)[] expected =
        [
            (VectorMetric.SquaredEuclidean, 32, 1),
            (VectorMetric.SquaredEuclidean, 32, 10),
            (VectorMetric.SquaredEuclidean, 128, 1),
            (VectorMetric.SquaredEuclidean, 128, 10),
            (VectorMetric.SquaredEuclidean, 386, 1),
            (VectorMetric.SquaredEuclidean, 386, 10),
            (VectorMetric.InnerProduct, 32, 1),
            (VectorMetric.InnerProduct, 32, 10),
            (VectorMetric.InnerProduct, 128, 1),
            (VectorMetric.InnerProduct, 128, 10),
            (VectorMetric.InnerProduct, 386, 1),
            (VectorMetric.InnerProduct, 386, 10),
            (VectorMetric.Cosine, 32, 1),
            (VectorMetric.Cosine, 32, 10),
            (VectorMetric.Cosine, 128, 1),
            (VectorMetric.Cosine, 128, 10),
            (VectorMetric.Cosine, 386, 1),
            (VectorMetric.Cosine, 386, 10)
        ];

        Assert.Equal(expected.Length, cases.Length);
        Assert.Equal(expected, cases.Select(item => (item.Metric, item.Dimension, item.TopK)).ToArray());
        Assert.Equal(cases.Length, cases.Select(item => item.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.All(cases, item =>
        {
            Assert.Equal(10, item.VectorCount);
            Assert.Equal(1, item.QueryCount);
            Assert.Equal(1, item.Runs);
            Assert.Equal(0, item.WarmupQueries);
            Assert.Null(item.BaselineReportId);
            Assert.StartsWith(outputDirectory, item.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.EndsWith(".json", item.OutputPath);
        });
        Assert.Equal(0xFFFF_FFF0u, cases[0].Seed);
        Assert.Equal(0xFFFF_FFFFu, cases[15].Seed);
        Assert.Equal(0u, cases[16].Seed);
        Assert.Equal(1u, cases[17].Seed);
    }

    [Fact]
    public void Run_ManifestLinksEveryReportAndPerCaseReportsKeepPrivateSmokeMetadata()
    {
        string outputDirectory = NewArtifactDirectory("vec014-linkage");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 3,
            Seed: 0x5EED0142,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));
        string[] arguments =
        [
            "exact-generated-matrix",
            "--vectors", "10",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "3",
            "--seed", "0x5EED0142",
            "--output-dir", outputDirectory,
            "--manifest", options.ManifestPath
        ];

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(options, arguments);
        GeneratedExactMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        string manifestJson = File.ReadAllText(options.ManifestPath);
        using JsonDocument manifestDocument = JsonDocument.Parse(manifestJson);
        JsonElement manifestRoot = manifestDocument.RootElement;

        Assert.Equal("VecNet.BenchmarkMatrixManifest", manifestRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", manifestRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-014", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal(18, manifestRoot.GetProperty("caseCount").GetInt32());
        Assert.Equal(18, manifestRoot.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, manifestRoot.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(outputDirectory, manifestRoot.GetProperty("outputDirectory").GetString());
        Assert.Equal(arguments, manifest.Runner.Arguments);
        AssertFalseMatrixEligibility(manifestRoot);
        AssertNoComparisonMathFields(manifestRoot);

        foreach (GeneratedExactMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.ErrorMessage);
            Assert.NotNull(matrixCase.ReportId);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);
            Assert.StartsWith("exact-generated-", matrixCase.ReportId, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("VEC-014", matrixCase.ReportId, StringComparison.OrdinalIgnoreCase);

            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(matrixCase.ReportPath));
            JsonElement reportRoot = reportDocument.RootElement;

            Assert.Equal(matrixCase.ReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal(matrixCase.Metric, reportRoot.GetProperty("dataset").GetProperty("metric").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal(matrixCase.Runs, reportRoot.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
            Assert.Equal(matrixCase.WarmupQueries, reportRoot.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
            Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
            Assert.Equal("bytesPerQuery", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("unit").GetString());
            Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
            Assert.Equal("absent", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("value").GetString());
            Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
            Assert.Equal("local-evidence", reportRoot.GetProperty("claimClass").GetString());
            Assert.Equal("smoke", reportRoot.GetProperty("evidence").GetProperty("status").GetString());
            Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.True(reportRoot.GetProperty("validation").GetProperty("reportIsPrivateRaw").GetBoolean());
            AssertNoComparisonMathFields(reportRoot);
        }
    }

    [Fact]
    public void Run_WhenOneCaseReportPathIsUnwritable_RecordsFailedCaseAndContinues()
    {
        string outputDirectory = NewArtifactDirectory("vec014-failure");
        var options = new GeneratedExactMatrixOptions(
            "smoke",
            VectorCount: 10,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0143,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));
        string blockedReportPath = Path.Combine(outputDirectory, "case-02-squaredeuclidean-32d-10k.json");
        Directory.CreateDirectory(blockedReportPath);

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(
            options,
            ["exact-generated-matrix", "--output-dir", outputDirectory]);
        GeneratedExactMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        Assert.Equal(18, manifest.CaseCount);
        Assert.Equal(17, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(1, manifest.Aggregate.FailedCaseCount);

        GeneratedExactMatrixCaseManifest failedCase = Assert.Single(manifest.Cases, item => item.Status == "failed");
        Assert.Equal(2, failedCase.CaseNumber);
        Assert.Equal("SquaredEuclidean", failedCase.Metric);
        Assert.Equal(32, failedCase.Dimension);
        Assert.Equal(10, failedCase.TopK);
        Assert.Equal(blockedReportPath, failedCase.ReportPath);
        Assert.Null(failedCase.ReportId);
        Assert.Equal("failed", failedCase.ValidationStatus);
        Assert.False(string.IsNullOrWhiteSpace(failedCase.ErrorMessage));
        Assert.True(Directory.Exists(blockedReportPath));

        Assert.Equal(17, manifest.Cases.Count(item => item.Status == "passed"));
        Assert.All(manifest.Cases.Where(item => item.Status == "passed"), passedCase =>
        {
            Assert.Equal("passed", passedCase.ValidationStatus);
            Assert.NotNull(passedCase.ReportId);
            Assert.Null(passedCase.ErrorMessage);
            Assert.True(File.Exists(passedCase.ReportPath), passedCase.ReportPath);
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(options.ManifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal(17, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(1, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        AssertFalseMatrixEligibility(root);
        AssertNoComparisonMathFields(root);
    }

    [Fact]
    public void ExactGenerated_ReportShapeRemainsCompatibleAfterVec014TaskIdProvenanceChange()
    {
        GeneratedExactSearchOptions options = CommandLine.Parse(
            [
                "exact-generated",
                "--metric", "Cosine",
                "--dimension", "13",
                "--vectors", "20",
                "--queries", "3",
                "--top-k", "4",
                "--runs", "2",
                "--warmup-queries", "5",
                "--seed", "0x5EED0144",
                "--output", "VecNet.BenchmarkRunner.Artifacts/vec014-direct-compat.json",
                "--baseline-report-id", "metadata-only-baseline"
            ]);

        BenchmarkReport report = GeneratedExactSearchScenario.Run(options, ["exact-generated"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.BenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-014", root.GetProperty("taskId").GetString());
        Assert.Equal("exact-generated", root.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal("exact-generated", root.GetProperty("scenario").GetProperty("name").GetString());
        Assert.StartsWith("exact-generated-", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Cosine-13d-20v-3q-4k-2r-5w-5EED0144", report.ReportId, StringComparison.Ordinal);
        Assert.DoesNotContain("VEC-014", report.ReportId, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("metadata-only-baseline", root.GetProperty("baseline").GetProperty("baselineReportId").GetString());
        Assert.Equal("private-raw", root.GetProperty("privacyClass").GetString());
        Assert.Equal("local-evidence", root.GetProperty("claimClass").GetString());
        Assert.Equal("smoke", root.GetProperty("evidence").GetProperty("status").GetString());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.Equal(2, root.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
        Assert.Equal("executed", root.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal(5, root.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal("measured", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", root.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", root.GetProperty("measurement").GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal(2, root.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal(2, root.GetProperty("search").GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("baseline").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoComparisonMathFields(root);
    }

    private static string NewArtifactDirectory(string prefix) =>
        Path.Combine("VecNet.BenchmarkRunner.Artifacts", prefix + "-" + Guid.NewGuid().ToString("N"));

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

    private static void AssertNoComparisonMathFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baselineReportPath",
            "comparisonResult",
            "latencyDeltaMilliseconds",
            "latencyDeltaPercent",
            "qpsRatio",
            "allocationDeltaBytes",
            "allocationRatio",
            "regressionPassed",
            "regressionThreshold",
            "threshold",
            "delta",
            "ratio");
    }

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                bool disallowed = disallowedNames.Any(
                    name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                Assert.False(disallowed, string.Create(CultureInfo.InvariantCulture, $"Unexpected comparison field '{property.Name}' was present."));
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
