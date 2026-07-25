using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec037HnswGeneratedMatrixIndependentTests
{
    [Theory]
    [InlineData()]
    [InlineData("HNSW-GENERATED-MATRIX", "--PRESET", "SMOKE", "--VECTORS", "64", "--QUERIES", "2", "--RUNS", "2", "--WARMUP-QUERIES", "3", "--SEED", "0X0000002A")]
    [InlineData("hnsw-generated-matrix", "--vectors", "63", "--vectors", "64", "--queries", "1", "--runs", "1", "--warmup-queries", "0")]
    public void ParseHnswGeneratedMatrix_EdgeCasesStaySmokeOnlyAndPrivate(params string[] args)
    {
        HnswGeneratedMatrixOptions options = CommandLine.ParseHnswGeneratedMatrix(args);

        Assert.Equal("smoke", options.PresetName);
        Assert.True(options.VectorCount >= HnswGeneratedMatrixScenario.GetMaxTopK(options.PresetName));
        Assert.True(options.QueryCount > 0);
        Assert.InRange(options.Runs, 1, 5);
        Assert.True(options.WarmupQueries >= 0);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.EndsWith("hnsw-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);

        if (args.Count(item => string.Equals(item, "--vectors", StringComparison.OrdinalIgnoreCase)) > 1)
        {
            Assert.Equal(64, options.VectorCount);
        }

        if (args.Contains("--seed", StringComparer.OrdinalIgnoreCase))
        {
            Assert.Equal(42u, options.Seed);
        }
    }

    [Theory]
    [InlineData("hnsw-generated-matrix", "--preset", " ")]
    [InlineData("hnsw-generated-matrix", "--preset", "standard ")]
    [InlineData("hnsw-generated-matrix", "--seed", "-1")]
    [InlineData("hnsw-generated-matrix", "--seed", "0x")]
    [InlineData("hnsw-generated-matrix", "--manifest", "--output-dir")]
    [InlineData("hnsw-generated-matrix", "--output", "single-report.json")]
    [InlineData("hnsw-generated-matrix", "output-dir", "matrix")]
    public void ParseHnswGeneratedMatrix_RejectsMalformedEdgesAndSingleReportOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ExpandCases_HasStableSmokeOrderSeedsProfilesAndReportNames()
    {
        string outputDirectory = NewArtifactDirectory("expand-order");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0xFFFF_FFFC,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "hnsw-matrix-manifest.json"));

        HnswGeneratedMatrixScenario.HnswMatrixCase[] cases = HnswGeneratedMatrixScenario.ExpandCases(options);

        (VectorMetric Metric, string Profile, int Dimension, int TopK, int M, int EfConstruction, int EfSearch)[] expected =
        [
            (VectorMetric.SquaredEuclidean, "low-ef-m4", 16, 1, 4, 16, 10),
            (VectorMetric.SquaredEuclidean, "balanced-m8", 16, 1, 8, 32, 24),
            (VectorMetric.SquaredEuclidean, "low-ef-m4", 16, 10, 4, 16, 10),
            (VectorMetric.SquaredEuclidean, "balanced-m8", 16, 10, 8, 32, 24),
            (VectorMetric.SquaredEuclidean, "low-ef-m4", 32, 1, 4, 16, 10),
            (VectorMetric.SquaredEuclidean, "balanced-m8", 32, 1, 8, 32, 24),
            (VectorMetric.SquaredEuclidean, "low-ef-m4", 32, 10, 4, 16, 10),
            (VectorMetric.SquaredEuclidean, "balanced-m8", 32, 10, 8, 32, 24),
            (VectorMetric.Cosine, "low-ef-m4", 16, 1, 4, 16, 10),
            (VectorMetric.Cosine, "balanced-m8", 16, 1, 8, 32, 24),
            (VectorMetric.Cosine, "low-ef-m4", 16, 10, 4, 16, 10),
            (VectorMetric.Cosine, "balanced-m8", 16, 10, 8, 32, 24),
            (VectorMetric.Cosine, "low-ef-m4", 32, 1, 4, 16, 10),
            (VectorMetric.Cosine, "balanced-m8", 32, 1, 8, 32, 24),
            (VectorMetric.Cosine, "low-ef-m4", 32, 10, 4, 16, 10),
            (VectorMetric.Cosine, "balanced-m8", 32, 10, 8, 32, 24)
        ];

        Assert.Equal(expected.Length, cases.Length);
        Assert.Equal(
            expected,
            cases.Select(item => (item.Options.Metric, item.ProfileName, item.Options.Dimension, item.Options.TopK, item.Options.M, item.Options.EfConstruction, item.Options.EfSearch)).ToArray());
        Assert.Equal(cases.Length, cases.Select(item => item.Options.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal(0xFFFF_FFFCu, cases[0].Options.Seed);
        Assert.Equal(0xFFFF_FFFFu, cases[3].Options.Seed);
        Assert.Equal(0u, cases[4].Options.Seed);
        Assert.Equal(11u, cases[^1].Options.Seed);
        Assert.Equal("0x484EACA8FFFC0001", FormatHex(cases[0].Options.HnswSeed));
        Assert.Equal("0x484EACA8FFFC0010", FormatHex(cases[^1].Options.HnswSeed));
        Assert.EndsWith("case-01-SquaredEuclidean-low-ef-m4-16d-1k.json", cases[0].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("case-16-Cosine-balanced-m8-32d-10k.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases, item =>
        {
            Assert.True(item.Options.EfSearch >= item.Options.TopK);
            Assert.InRange(item.Options.M, 2, 64);
            Assert.InRange(item.Options.EfConstruction, item.Options.M, 4096);
            Assert.InRange(item.Options.EfSearch, 1, 4096);
        });
    }

    [Fact]
    public void Run_WhenOneCaseReportPathIsBlocked_RecordsFailureAndContinuesRemainingCases()
    {
        string outputDirectory = NewArtifactDirectory("blocked-report");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED3701,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "hnsw-matrix-manifest.json"));
        string blockedReportPath = Path.Combine(outputDirectory, "case-02-SquaredEuclidean-balanced-m8-16d-1k.json");
        Directory.CreateDirectory(blockedReportPath);

        HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(
            options,
            ["hnsw-generated-matrix", "--output-dir", outputDirectory]);
        HnswGeneratedMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        Assert.Equal(16, manifest.CaseCount);
        Assert.Equal(15, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(1, manifest.Aggregate.FailedCaseCount);

        HnswGeneratedMatrixCaseManifest failedCase = Assert.Single(manifest.Cases, item => item.Status == "failed");
        Assert.Equal(2, failedCase.CaseNumber);
        Assert.Equal("balanced-m8", failedCase.ProfileName);
        Assert.Equal(16, failedCase.Dimension);
        Assert.Equal(1, failedCase.TopK);
        Assert.Equal(blockedReportPath, failedCase.ReportPath);
        Assert.Null(failedCase.ReportId);
        Assert.Equal("failed", failedCase.ValidationStatus);
        Assert.False(string.IsNullOrWhiteSpace(failedCase.ErrorMessage));
        Assert.True(Directory.Exists(blockedReportPath));

        Assert.All(manifest.Cases.Where(item => item.Status == "passed"), passedCase =>
        {
            Assert.Equal("passed", passedCase.ValidationStatus);
            Assert.NotNull(passedCase.ReportId);
            Assert.Null(passedCase.ErrorMessage);
            Assert.True(File.Exists(passedCase.ReportPath), passedCase.ReportPath);
        });

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(options.ManifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal(15, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(1, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        AssertFalseMatrixEligibility(root);
        AssertNoForbiddenScopeFields(root);
    }

    [Fact]
    public void Run_LinkedReportsKeepV036ShapeAndDoNotInheritMatrixEligibility()
    {
        string outputDirectory = NewArtifactDirectory("linked-reports");
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 2,
            Runs: 2,
            WarmupQueries: 3,
            Seed: 0x5EED3702,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "hnsw-matrix-manifest.json"));
        string[] arguments =
        [
            "hnsw-generated-matrix",
            "--vectors", "64",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "3",
            "--seed", "0x5EED3702",
            "--output-dir", outputDirectory,
            "--manifest", options.ManifestPath
        ];

        HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(options, arguments);
        HnswGeneratedMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        Assert.Equal(16, manifest.Aggregate.PassedCaseCount);
        foreach (HnswGeneratedMatrixCaseManifest matrixCase in manifest.Cases)
        {
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(matrixCase.ReportPath));
            JsonElement reportRoot = reportDocument.RootElement;

            Assert.Equal(matrixCase.ReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal("VecNet.HnswBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("VEC-036", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal("hnsw-generated", reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal("hnsw-generated", reportRoot.GetProperty("command").GetProperty("scenario").GetString());
            Assert.Equal("generated-no-external-source", reportRoot.GetProperty("dataset").GetProperty("sourceVerificationStatus").GetString());
            Assert.Equal(matrixCase.Metric, reportRoot.GetProperty("dataset").GetProperty("metric").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal(matrixCase.Runs, reportRoot.GetProperty("measurement").GetProperty("repeatedRuns").GetProperty("runCount").GetInt32());
            Assert.Equal(matrixCase.WarmupQueries, reportRoot.GetProperty("measurement").GetProperty("warmup").GetProperty("warmupCount").GetInt32());
            Assert.Equal(matrixCase.M, reportRoot.GetProperty("hnsw").GetProperty("m").GetInt32());
            Assert.Equal(matrixCase.EfConstruction, reportRoot.GetProperty("hnsw").GetProperty("efConstruction").GetInt32());
            Assert.Equal(matrixCase.EfSearch, reportRoot.GetProperty("hnsw").GetProperty("efSearch").GetInt32());
            Assert.Equal(matrixCase.HnswSeed, reportRoot.GetProperty("hnsw").GetProperty("randomSeed").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
            Assert.True(reportRoot.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("checkedResultCount").GetInt32() > 0);
            Assert.Equal("estimated", reportRoot.GetProperty("memoryEstimate").GetProperty("status").GetString());
            Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
            Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
            Assert.Equal("local-evidence", reportRoot.GetProperty("claimClass").GetString());
            Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
            AssertNoForbiddenScopeFields(reportRoot);
        }
    }

    [Fact]
    public void HnswMatrixManifestSchema_IsSeparateFromExactMatrixSingleHnswAndComparisonArtifacts()
    {
        string directory = NewArtifactDirectory("schema-separation");
        string hnswMatrixDirectory = Path.Combine(directory, "hnsw-matrix");
        string exactMatrixDirectory = Path.Combine(directory, "exact-matrix");
        Directory.CreateDirectory(hnswMatrixDirectory);
        Directory.CreateDirectory(exactMatrixDirectory);

        HnswGeneratedMatrixManifest hnswMatrix = HnswGeneratedMatrixScenario.Run(
            new HnswGeneratedMatrixOptions(
                "smoke",
                VectorCount: 64,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED3703,
                OutputDirectory: hnswMatrixDirectory,
                ManifestPath: Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")),
            ["hnsw-generated-matrix"]);
        HnswGeneratedMatrixScenario.WriteManifest(hnswMatrix, Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json"));

        GeneratedExactMatrixManifest exactMatrix = GeneratedExactMatrixScenario.Run(
            new GeneratedExactMatrixOptions(
                "smoke",
                VectorCount: 10,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED3703,
                OutputDirectory: exactMatrixDirectory,
                ManifestPath: Path.Combine(exactMatrixDirectory, "matrix-manifest.json")),
            ["exact-generated-matrix"]);
        GeneratedExactMatrixScenario.WriteManifest(exactMatrix, Path.Combine(exactMatrixDirectory, "matrix-manifest.json"));

        using JsonDocument hnswMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")));
        using JsonDocument exactMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(exactMatrixDirectory, "matrix-manifest.json")));
        using JsonDocument singleHnswDocument = JsonDocument.Parse(File.ReadAllText(hnswMatrix.Cases[0].ReportPath));
        JsonElement hnswRoot = hnswMatrixDocument.RootElement;
        JsonElement exactRoot = exactMatrixDocument.RootElement;
        JsonElement singleHnswRoot = singleHnswDocument.RootElement;

        Assert.Equal("VecNet.HnswBenchmarkMatrixManifest", hnswRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.BenchmarkMatrixManifest", exactRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.HnswBenchmarkReport", singleHnswRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-037", hnswRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-015", exactRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-036", singleHnswRoot.GetProperty("taskId").GetString());
        Assert.True(hnswRoot.GetProperty("cases")[0].TryGetProperty("profileName", out _));
        Assert.True(hnswRoot.GetProperty("cases")[0].TryGetProperty("hnswSeed", out _));
        Assert.True(hnswRoot.GetProperty("cases")[0].TryGetProperty("m", out _));
        Assert.False(exactRoot.GetProperty("cases")[0].TryGetProperty("profileName", out _));
        Assert.False(exactRoot.GetProperty("cases")[0].TryGetProperty("hnswSeed", out _));
        Assert.False(singleHnswRoot.TryGetProperty("cases", out _));
        Assert.False(singleHnswRoot.TryGetProperty("presetName", out _));

        string comparisonOutput = Path.Combine(directory, "comparison.json");
        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(
                Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json"),
                Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json"),
                comparisonOutput),
            ["compare-generated-exact"]);

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Cases);
        Assert.Empty(comparison.Metrics);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema");
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Fact]
    public void Run_RepeatedExecutionWithSameSeedKeepsCaseAndReportIdentityStable()
    {
        HnswGeneratedMatrixManifest first = RunSmokeMatrix("deterministic-a", 0x5EED3704);
        HnswGeneratedMatrixManifest second = RunSmokeMatrix("deterministic-b", 0x5EED3704);

        Assert.Equal(first.CaseCount, second.CaseCount);
        Assert.Equal(first.Aggregate.PassedCaseCount, second.Aggregate.PassedCaseCount);
        Assert.Equal(first.Aggregate.FailedCaseCount, second.Aggregate.FailedCaseCount);

        for (int i = 0; i < first.Cases.Length; i++)
        {
            HnswGeneratedMatrixCaseManifest left = first.Cases[i];
            HnswGeneratedMatrixCaseManifest right = second.Cases[i];

            Assert.Equal(left.CaseNumber, right.CaseNumber);
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
            Assert.Equal(left.Status, right.Status);
            Assert.Equal(left.ValidationStatus, right.ValidationStatus);
            Assert.Equal(Path.GetFileName(left.ReportPath), Path.GetFileName(right.ReportPath));
        }

        Assert.Equal(first.Cases.Select(item => item.ReportId).ToArray(), second.Cases.Select(item => item.ReportId).ToArray());
        Assert.Equal(first.Cases.Length, first.Cases.Select(item => item.ReportId).Distinct(StringComparer.Ordinal).Count());
    }

    private static HnswGeneratedMatrixManifest RunSmokeMatrix(string prefix, uint seed)
    {
        string outputDirectory = NewArtifactDirectory(prefix);
        var options = new HnswGeneratedMatrixOptions(
            "smoke",
            VectorCount: 64,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: seed,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "hnsw-matrix-manifest.json"));

        HnswGeneratedMatrixManifest manifest = HnswGeneratedMatrixScenario.Run(options, ["hnsw-generated-matrix"]);
        HnswGeneratedMatrixScenario.WriteManifest(manifest, options.ManifestPath);
        return manifest;
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec037-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

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

    private static void AssertNoForbiddenScopeFields(JsonElement element)
    {
        AssertNoPropertyNamed(
            element,
            "baseline",
            "baselineReportId",
            "candidateEligibility",
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
            "persistedBytes",
            "filter",
            "update",
            "delete");
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
