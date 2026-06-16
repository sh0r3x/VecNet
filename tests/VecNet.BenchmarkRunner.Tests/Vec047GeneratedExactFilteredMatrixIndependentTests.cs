using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec047GeneratedExactFilteredMatrixIndependentTests
{
    [Theory]
    [InlineData()]
    [InlineData("EXACT-GENERATED-FILTERED-MATRIX", "--PRESET", "SMOKE", "--VECTORS", "10", "--QUERIES", "1", "--RUNS", "5", "--WARMUP-QUERIES", "2", "--SEED", "0XFFFFFFFF")]
    [InlineData("exact-generated-filtered-matrix", "--vectors", "11", "--vectors", "12", "--queries", "1", "--runs", "1")]
    public void ParseGeneratedExactFilteredMatrix_AcceptsCaseInsensitiveAliasesAndLastRepeatedValue(params string[] args)
    {
        GeneratedExactFilteredMatrixOptions options = CommandLine.ParseGeneratedExactFilteredMatrix(args);

        Assert.Equal("smoke", options.PresetName);
        Assert.True(options.VectorCount >= GeneratedExactFilteredMatrixScenario.GetMaxTopK(options.PresetName));
        Assert.True(options.QueryCount > 0);
        Assert.InRange(options.Runs, 1, 5);
        Assert.True(options.WarmupQueries >= 0);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("exact-filtered-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);

        if (args.Count(item => string.Equals(item, "--vectors", StringComparison.OrdinalIgnoreCase)) > 1)
        {
            Assert.Equal(12, options.VectorCount);
        }

        if (args.Contains("--seed", StringComparer.OrdinalIgnoreCase))
        {
            Assert.Equal(uint.MaxValue, options.Seed);
        }
    }

    [Theory]
    [InlineData("exact-generated-filtered-matrix", "--preset", " ")]
    [InlineData("exact-generated-filtered-matrix", "--preset", "standard ")]
    [InlineData("exact-generated-filtered-matrix", "--seed", "-1")]
    [InlineData("exact-generated-filtered-matrix", "--seed", "0x100000000")]
    [InlineData("exact-generated-filtered-matrix", "--vectors", "10.5")]
    [InlineData("exact-generated-filtered-matrix", "--manifest", "--output-dir")]
    [InlineData("exact-generated-filtered-matrix", "output-dir", "matrix")]
    [InlineData("exact-generated-filtered-matrix", "--m", "8")]
    [InlineData("exact-generated-filtered-matrix", "--ef-search", "32")]
    [InlineData("exact-generated-filtered-matrix", "--hnsw-seed", "0x47")]
    [InlineData("exact-generated-filtered-matrix", "--truth-depth", "10")]
    [InlineData("exact-generated-filtered-matrix", "--current", "current.json")]
    public void ParseGeneratedExactFilteredMatrix_RejectsMalformedEdgesAndUnrelatedModeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ExpandCases_StandardPresetOrderSeedsAndCaseNamesAreDeterministic()
    {
        string outputDirectory = NewArtifactDirectory("standard-order");
        var options = new GeneratedExactFilteredMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0xFFFF_FFFA,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixScenario.GeneratedExactFilteredMatrixCase[] cases =
            GeneratedExactFilteredMatrixScenario.ExpandCases(options);

        Assert.Equal(120, cases.Length);
        Assert.Equal(
            [
                (VectorMetric.SquaredEuclidean, 32, 10, "all"),
                (VectorMetric.SquaredEuclidean, 32, 10, "broad"),
                (VectorMetric.SquaredEuclidean, 32, 10, "selective"),
                (VectorMetric.SquaredEuclidean, 32, 10, "very-selective"),
                (VectorMetric.SquaredEuclidean, 32, 10, "empty"),
                (VectorMetric.SquaredEuclidean, 32, 100, "all")
            ],
            cases.Take(6).Select(item => (item.Options.Metric, item.Options.Dimension, item.Options.TopK, item.Options.FilterKind)).ToArray());
        Assert.Equal(0xFFFF_FFFAu, cases[0].Options.Seed);
        Assert.Equal(0xFFFF_FFFFu, cases[5].Options.Seed);
        Assert.Equal(0u, cases[6].Options.Seed);
        Assert.Equal(113u, cases[^1].Options.Seed);
        Assert.Equal(cases.Length, cases.Select(item => item.Options.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.EndsWith("case-001-squaredeuclidean-32d-10k-all.json", cases[0].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("case-004-squaredeuclidean-32d-10k-very-selective.json", cases[3].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("case-120-cosine-768d-100k-empty.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases.Where(item => item.Options.FilterKind == "very-selective"), item =>
        {
            Assert.True(item.Options.TopK > 1);
            Assert.True(item.Options.TopK <= item.Options.VectorCount);
        });
    }

    [Fact]
    public void Run_WhenOneCaseReportPathIsBlocked_RecordsFailureAndContinuesRemainingCases()
    {
        string outputDirectory = NewArtifactDirectory("single-blocked-case");
        var options = new GeneratedExactFilteredMatrixOptions(
            "smoke",
            VectorCount: 32,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED4710,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        string blockedReportPath = Path.Combine(outputDirectory, "case-004-squaredeuclidean-128d-10k-selective.json");
        Directory.CreateDirectory(blockedReportPath);

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(
            options,
            ["exact-generated-filtered-matrix", "--output-dir", outputDirectory]);
        GeneratedExactFilteredMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(7, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(1, manifest.Aggregate.FailedCaseCount);

        GeneratedExactFilteredMatrixCaseManifest failedCase = Assert.Single(manifest.Cases, item => item.Status == "failed");
        Assert.Equal(4, failedCase.CaseNumber);
        Assert.Equal("SquaredEuclidean", failedCase.Metric);
        Assert.Equal(128, failedCase.Dimension);
        Assert.Equal("selective", failedCase.FilterKind);
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
        Assert.Equal(7, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(1, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        AssertFalseMatrixEligibility(root);
        AssertNoForbiddenScopeFields(root);
    }

    [Fact]
    public void Run_StandardLinkedReportsPreserveFilterSemanticsAndFalseEligibilityAcrossAllSelectivities()
    {
        string outputDirectory = NewArtifactDirectory("standard-linked-reports");
        var options = new GeneratedExactFilteredMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED4711,
            DuplicateIdsPerQuery: 2,
            UnknownIdsPerQuery: 3,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(options, ["exact-generated-filtered-matrix"]);

        Assert.Equal(120, manifest.CaseCount);
        Assert.Equal(120, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(["all", "broad", "selective", "very-selective", "empty"], manifest.Cases.Select(item => item.FilterKind).Distinct().ToArray());

        foreach (string filterKind in new[] { "all", "broad", "selective", "very-selective", "empty" })
        {
            GeneratedExactFilteredMatrixCaseManifest matrixCase = Assert.Single(
                manifest.Cases,
                item =>
                    item.Metric == "Cosine" &&
                    item.Dimension == 386 &&
                    item.TopK == 10 &&
                    item.FilterKind == filterKind);
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(matrixCase.ReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            JsonElement filter = reportRoot.GetProperty("filter");

            Assert.Equal(matrixCase.ReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal("VecNet.ExactFilteredBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("VEC-046", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal("exact-generated-filtered", reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.Metric, reportRoot.GetProperty("dataset").GetProperty("metric").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal(matrixCase.FilterKind, filter.GetProperty("kind").GetString());
            Assert.Equal(2, filter.GetProperty("duplicateIdCountPerQuery").GetInt32());
            Assert.Equal(3, filter.GetProperty("unknownIdCountPerQuery").GetInt32());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity").GetProperty("status").GetString());
            Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", reportRoot.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
            Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
            Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
            Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
            AssertNoForbiddenScopeFields(reportRoot);

            int expectedVisible = filterKind switch
            {
                "all" => 100,
                "broad" => 50,
                "selective" => 10,
                "very-selective" => 9,
                "empty" => 0,
                _ => throw new InvalidOperationException("Unexpected filter kind.")
            };
            Assert.Equal(expectedVisible, filter.GetProperty("visibleCountPerQuery").GetInt32());
            Assert.Equal(expectedVisible, filter.GetProperty("knownIdCountPerQuery").GetInt32());
            Assert.Equal(expectedVisible + 5, filter.GetProperty("allowlistLengthPerQuery").GetInt32());
            Assert.Equal(Math.Min(10, expectedVisible), reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity").GetProperty("checkedResultCount").GetInt32());
        }
    }

    [Fact]
    public void FilteredMatrixManifestSchemaIsSeparateAndGeneratedExactComparisonRejectsIt()
    {
        string directory = NewArtifactDirectory("schema-separation");
        string filteredMatrixDirectory = Path.Combine(directory, "filtered-matrix");
        string exactMatrixDirectory = Path.Combine(directory, "exact-matrix");
        string hnswMatrixDirectory = Path.Combine(directory, "hnsw-matrix");
        Directory.CreateDirectory(filteredMatrixDirectory);
        Directory.CreateDirectory(exactMatrixDirectory);
        Directory.CreateDirectory(hnswMatrixDirectory);

        GeneratedExactFilteredMatrixManifest filteredMatrix = GeneratedExactFilteredMatrixScenario.Run(
            new GeneratedExactFilteredMatrixOptions(
                "smoke",
                VectorCount: 32,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED4712,
                DuplicateIdsPerQuery: 0,
                UnknownIdsPerQuery: 0,
                OutputDirectory: filteredMatrixDirectory,
                ManifestPath: Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json")),
            ["exact-generated-filtered-matrix"]);
        GeneratedExactFilteredMatrixScenario.WriteManifest(filteredMatrix, Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json"));

        GeneratedExactMatrixManifest exactMatrix = GeneratedExactMatrixScenario.Run(
            new GeneratedExactMatrixOptions(
                "smoke",
                VectorCount: 10,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED4712,
                OutputDirectory: exactMatrixDirectory,
                ManifestPath: Path.Combine(exactMatrixDirectory, "matrix-manifest.json")),
            ["exact-generated-matrix"]);
        GeneratedExactMatrixScenario.WriteManifest(exactMatrix, Path.Combine(exactMatrixDirectory, "matrix-manifest.json"));

        HnswGeneratedMatrixManifest hnswMatrix = HnswGeneratedMatrixScenario.Run(
            new HnswGeneratedMatrixOptions(
                "smoke",
                VectorCount: 64,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED4712,
                OutputDirectory: hnswMatrixDirectory,
                ManifestPath: Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")),
            ["hnsw-generated-matrix"]);
        HnswGeneratedMatrixScenario.WriteManifest(hnswMatrix, Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json"));

        using JsonDocument filteredMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json")));
        using JsonDocument exactMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(exactMatrixDirectory, "matrix-manifest.json")));
        using JsonDocument filteredReportDocument = JsonDocument.Parse(File.ReadAllText(filteredMatrix.Cases[0].ReportPath));
        using JsonDocument hnswMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")));
        JsonElement filteredRoot = filteredMatrixDocument.RootElement;
        JsonElement exactRoot = exactMatrixDocument.RootElement;
        JsonElement filteredReportRoot = filteredReportDocument.RootElement;
        JsonElement hnswRoot = hnswMatrixDocument.RootElement;

        Assert.Equal("VecNet.ExactFilteredBenchmarkMatrixManifest", filteredRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.BenchmarkMatrixManifest", exactRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", filteredReportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.HnswBenchmarkMatrixManifest", hnswRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-047", filteredRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-015", exactRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-046", filteredReportRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-037", hnswRoot.GetProperty("taskId").GetString());
        Assert.True(filteredRoot.GetProperty("cases")[0].TryGetProperty("filterKind", out _));
        Assert.True(filteredRoot.GetProperty("cases")[0].TryGetProperty("duplicateIdCountPerQuery", out _));
        Assert.False(exactRoot.GetProperty("cases")[0].TryGetProperty("filterKind", out _));
        Assert.False(filteredReportRoot.TryGetProperty("cases", out _));
        Assert.False(filteredReportRoot.TryGetProperty("presetName", out _));

        string comparisonOutput = Path.Combine(directory, "comparison.json");
        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(
                Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json"),
                Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json"),
                comparisonOutput),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(comparison, comparisonOutput);

        using JsonDocument comparisonDocument = JsonDocument.Parse(File.ReadAllText(comparisonOutput));
        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Cases);
        Assert.Empty(comparison.Metrics);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema");
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal("VecNet.BenchmarkComparison", comparisonDocument.RootElement.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-020", comparisonDocument.RootElement.GetProperty("taskId").GetString());
    }

    [Fact]
    public void Run_RepeatedExecutionWithSameSeedKeepsCaseAndReportIdentityStable()
    {
        GeneratedExactFilteredMatrixManifest first = RunSmokeMatrix("deterministic-a", 0x5EED4713);
        GeneratedExactFilteredMatrixManifest second = RunSmokeMatrix("deterministic-b", 0x5EED4713);

        Assert.Equal(first.CaseCount, second.CaseCount);
        Assert.Equal(first.Aggregate.PassedCaseCount, second.Aggregate.PassedCaseCount);
        Assert.Equal(first.Aggregate.FailedCaseCount, second.Aggregate.FailedCaseCount);
        Assert.Equal(first.Cases.Length, first.Cases.Select(item => item.ReportId).Distinct(StringComparer.Ordinal).Count());

        for (int i = 0; i < first.Cases.Length; i++)
        {
            GeneratedExactFilteredMatrixCaseManifest left = first.Cases[i];
            GeneratedExactFilteredMatrixCaseManifest right = second.Cases[i];

            Assert.Equal(left.CaseNumber, right.CaseNumber);
            Assert.Equal(left.Metric, right.Metric);
            Assert.Equal(left.Dimension, right.Dimension);
            Assert.Equal(left.VectorCount, right.VectorCount);
            Assert.Equal(left.QueryCount, right.QueryCount);
            Assert.Equal(left.TopK, right.TopK);
            Assert.Equal(left.Runs, right.Runs);
            Assert.Equal(left.WarmupQueries, right.WarmupQueries);
            Assert.Equal(left.Seed, right.Seed);
            Assert.Equal(left.FilterKind, right.FilterKind);
            Assert.Equal(left.DuplicateIdCountPerQuery, right.DuplicateIdCountPerQuery);
            Assert.Equal(left.UnknownIdCountPerQuery, right.UnknownIdCountPerQuery);
            Assert.Equal(left.ReportId, right.ReportId);
            Assert.Equal(left.Status, right.Status);
            Assert.Equal(left.ValidationStatus, right.ValidationStatus);
            Assert.Equal(Path.GetFileName(left.ReportPath), Path.GetFileName(right.ReportPath));
        }
    }

    private static GeneratedExactFilteredMatrixManifest RunSmokeMatrix(string prefix, uint seed)
    {
        string outputDirectory = NewArtifactDirectory(prefix);
        var options = new GeneratedExactFilteredMatrixOptions(
            "smoke",
            VectorCount: 32,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: seed,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 1,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(options, ["exact-generated-filtered-matrix"]);
        GeneratedExactFilteredMatrixScenario.WriteManifest(manifest, options.ManifestPath);
        return manifest;
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec047-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
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
            "m",
            "hnswSeed",
            "retainedIdMap",
            "precompiledFilter",
            "storedLabel",
            "labelFilter");
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
