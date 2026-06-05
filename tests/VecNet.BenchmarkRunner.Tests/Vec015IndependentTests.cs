using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec015IndependentTests
{
    [Fact]
    public void DefaultAndExplicitSmokeExpandToSameBoundedCaseShape()
    {
        GeneratedExactMatrixOptions defaultOptions = CommandLine.ParseMatrix(
            [
                "exact-generated-matrix",
                "--vectors", "10",
                "--queries", "2",
                "--runs", "3",
                "--warmup-queries", "4",
                "--seed", "0x00000150",
                "--output-dir", NewArtifactDirectory("vec015-default-smoke")
            ]);
        GeneratedExactMatrixOptions explicitSmokeOptions = CommandLine.ParseMatrix(
            [
                "EXACT-GENERATED-MATRIX",
                "--PRESET", "SMOKE",
                "--VECTORS", "10",
                "--QUERIES", "2",
                "--RUNS", "3",
                "--WARMUP-QUERIES", "4",
                "--SEED", "0x00000150",
                "--OUTPUT-DIR", NewArtifactDirectory("vec015-explicit-smoke")
            ]);

        GeneratedExactSearchOptions[] defaultCases = GeneratedExactMatrixScenario.ExpandCases(defaultOptions);
        GeneratedExactSearchOptions[] explicitSmokeCases = GeneratedExactMatrixScenario.ExpandCases(explicitSmokeOptions);

        Assert.Equal("smoke", defaultOptions.PresetName);
        Assert.Equal("smoke", explicitSmokeOptions.PresetName);
        Assert.Equal(18, defaultCases.Length);
        Assert.Equal(18, explicitSmokeCases.Length);
        Assert.Equal(
            defaultCases.Select(item => (item.Metric, item.Dimension, item.TopK)).ToArray(),
            explicitSmokeCases.Select(item => (item.Metric, item.Dimension, item.TopK)).ToArray());
        Assert.Equal([32, 128, 386], defaultCases.Select(item => item.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10], defaultCases.Select(item => item.TopK).Distinct().ToArray());
        Assert.DoesNotContain(defaultCases, item => item.Dimension == 768 || item.TopK == 100);
        Assert.All(defaultCases.Concat(explicitSmokeCases), item =>
        {
            Assert.Equal(10, item.VectorCount);
            Assert.Equal(2, item.QueryCount);
            Assert.Equal(3, item.Runs);
            Assert.Equal(4, item.WarmupQueries);
            Assert.True(item.TopK <= item.VectorCount);
        });
    }

    [Fact]
    public void StandardExpansionCoversFullMatrixAndPropagatesCallerOptions()
    {
        string outputDirectory = NewArtifactDirectory("vec015-standard-expand");
        var options = new GeneratedExactMatrixOptions(
            "standard",
            VectorCount: 123,
            QueryCount: 5,
            Runs: 4,
            WarmupQueries: 2,
            Seed: 0xFFFF_FFE0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        GeneratedExactSearchOptions[] cases = GeneratedExactMatrixScenario.ExpandCases(options);

        Assert.Equal(36, cases.Length);
        Assert.Equal([VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine], cases.Select(item => item.Metric).Distinct().ToArray());
        Assert.Equal([32, 128, 386, 768], cases.Select(item => item.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10, 100], cases.Select(item => item.TopK).Distinct().ToArray());
        Assert.Contains(cases, item => item.Metric == VectorMetric.Cosine && item.Dimension == 768 && item.TopK == 100);
        Assert.Equal(cases.Length, cases.Select(item => item.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.EndsWith("case-36-cosine-768d-100k.json", cases[^1].OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0xFFFF_FFE0u, cases[0].Seed);
        Assert.Equal(3u, cases[^1].Seed);
        Assert.All(cases, item =>
        {
            Assert.Equal(123, item.VectorCount);
            Assert.Equal(5, item.QueryCount);
            Assert.Equal(4, item.Runs);
            Assert.Equal(2, item.WarmupQueries);
            Assert.Null(item.BaselineReportId);
            Assert.StartsWith(outputDirectory, item.OutputPath, StringComparison.OrdinalIgnoreCase);
            Assert.True(item.TopK <= item.VectorCount);
        });
    }

    [Theory]
    [InlineData("baseline")]
    [InlineData("standard ")]
    [InlineData("smoke-standard")]
    public void PresetNamesOutsideCanonicalSetAreRejected(string presetName)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseMatrix(["exact-generated-matrix", "--preset", presetName, "--vectors", "128"]));

        Assert.Contains("Unsupported matrix preset", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StandardPresetRejectsVectorCountsBelowTopKAtParserAndExpansionBoundaries()
    {
        ArgumentException parserException = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseMatrix(["exact-generated-matrix", "--preset", "standard", "--vectors", "99"]));
        Assert.Contains("maximum matrix top-k (100)", parserException.Message, StringComparison.Ordinal);
        Assert.Contains("standard", parserException.Message, StringComparison.Ordinal);

        string outputDirectory = NewArtifactDirectory("vec015-standard-invalid");
        var options = new GeneratedExactMatrixOptions(
            "standard",
            VectorCount: 99,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0153,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "matrix-manifest.json"));

        ArgumentException expansionException = Assert.Throws<ArgumentException>(
            () => GeneratedExactMatrixScenario.ExpandCases(options));
        Assert.Contains("top-k", expansionException.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void StandardRunCanonicalizesManifestPresetAndKeepsPerCaseReportsGeneratedExactSmokeOnly()
    {
        string outputDirectory = NewArtifactDirectory("vec015-standard-run");
        string manifestPath = Path.Combine(outputDirectory, "matrix-manifest.json");
        var options = new GeneratedExactMatrixOptions(
            "STANDARD",
            VectorCount: 100,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 1,
            Seed: 0x5EED0154,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);
        string[] arguments =
        [
            "exact-generated-matrix",
            "--preset", "STANDARD",
            "--vectors", "100",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "1",
            "--seed", "0x5EED0154",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        GeneratedExactMatrixManifest manifest = GeneratedExactMatrixScenario.Run(options, arguments);
        GeneratedExactMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VEC-015", manifest.TaskId);
        Assert.Equal("exact-generated-matrix", manifest.ScenarioName);
        Assert.Equal("standard", manifest.PresetName);
        Assert.Equal(36, manifest.CaseCount);
        Assert.Equal(36, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(arguments, manifest.Runner.Arguments);
        Assert.Equal([32, 128, 386, 768], manifest.Cases.Select(item => item.Dimension).Distinct().ToArray());
        Assert.Equal([1, 10, 100], manifest.Cases.Select(item => item.TopK).Distinct().ToArray());
        Assert.Equal(["SquaredEuclidean", "InnerProduct", "Cosine"], manifest.Cases.Select(item => item.Metric).Distinct().ToArray());
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.Equal(100, matrixCase.VectorCount);
            Assert.Equal(2, matrixCase.QueryCount);
            Assert.Equal(2, matrixCase.Runs);
            Assert.Equal(1, matrixCase.WarmupQueries);
            Assert.True(matrixCase.TopK <= matrixCase.VectorCount);
            Assert.NotNull(matrixCase.ReportId);
            Assert.Null(matrixCase.ErrorMessage);
            Assert.True(File.Exists(matrixCase.ReportPath), matrixCase.ReportPath);
        });

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement manifestRoot = manifestDocument.RootElement;
        Assert.Equal("VEC-015", manifestRoot.GetProperty("taskId").GetString());
        Assert.Equal("standard", manifestRoot.GetProperty("presetName").GetString());
        AssertFalseMatrixEligibility(manifestRoot);
        AssertNoComparisonMathFields(manifestRoot);

        GeneratedExactMatrixCaseManifest sampledCase = Assert.Single(
            manifest.Cases,
            item => item.Metric == "Cosine" && item.Dimension == 768 && item.TopK == 100);
        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(sampledCase.ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;

        Assert.Equal("VecNet.BenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", reportRoot.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-014", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("exact-generated", reportRoot.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal("exact-generated", reportRoot.GetProperty("scenario").GetProperty("name").GetString());
        Assert.Equal("generated-no-external-source", reportRoot.GetProperty("dataset").GetProperty("sourceVerificationStatus").GetString());
        Assert.Equal(sampledCase.Metric, reportRoot.GetProperty("dataset").GetProperty("metric").GetString());
        Assert.Equal(sampledCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
        Assert.Equal(sampledCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
        Assert.Equal(sampledCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
        Assert.Equal(sampledCase.TopK, reportRoot.GetProperty("truth").GetProperty("depth").GetInt32());
        Assert.Equal(sampledCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
        Assert.Equal(1, reportRoot.GetProperty("scenario").GetProperty("concurrency").GetInt32());
        Assert.Equal("Exact", reportRoot.GetProperty("index").GetProperty("profile").GetString());
        Assert.Equal("ExactFlatIndex", reportRoot.GetProperty("index").GetProperty("type").GetString());
        Assert.Equal(2, reportRoot.GetProperty("search").GetProperty("runs").GetArrayLength());
        Assert.Equal(2, reportRoot.GetProperty("search").GetProperty("aggregate").GetProperty("runCount").GetInt32());
        Assert.Equal(2, reportRoot.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
        Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.Equal(1, reportRoot.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
        Assert.Equal("executed", reportRoot.GetProperty("measurement").GetProperty("warmup").GetProperty("status").GetString());
        Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
        Assert.Equal("bytesPerQuery", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("unit").GetString());
        Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("absent", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("value").GetString());
        Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
        Assert.True(reportRoot.GetProperty("validation").GetProperty("finiteVectors").GetBoolean());
        Assert.True(reportRoot.GetProperty("validation").GetProperty("truthGenerated").GetBoolean());
        Assert.Equal(1.0, reportRoot.GetProperty("metrics").GetProperty("recallAtK").GetDouble());
        Assert.Equal(1.0, reportRoot.GetProperty("metrics").GetProperty("orderedAgreement").GetDouble());
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
