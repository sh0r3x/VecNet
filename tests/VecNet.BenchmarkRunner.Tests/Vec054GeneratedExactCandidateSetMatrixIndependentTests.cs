using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec054GeneratedExactCandidateSetMatrixIndependentTests
{
    [Theory]
    [InlineData()]
    [InlineData("GENERATED-EXACT-CANDIDATE-SET-MATRIX", "--PRESET", "SMOKE", "--VECTORS", "10", "--QUERIES", "1", "--RUNS", "5", "--WARMUP-QUERIES", "2", "--SEED", "0XFFFFFFFF")]
    [InlineData("generated-exact-candidate-set-matrix", "--vectors", "11", "--vectors", "12", "--queries", "1", "--runs", "1")]
    public void ParseGeneratedExactCandidateSetMatrix_AcceptsCaseInsensitiveAliasesAndLastRepeatedValue(params string[] args)
    {
        GeneratedExactCandidateSetMatrixOptions options = CommandLine.ParseGeneratedExactCandidateSetMatrix(args);

        Assert.Equal("smoke", options.PresetName);
        Assert.True(options.VectorCount >= GeneratedExactCandidateSetMatrixScenario.GetMaxTopK(options.PresetName));
        Assert.True(options.QueryCount > 0);
        Assert.InRange(options.Runs, 1, 5);
        Assert.True(options.WarmupQueries >= 0);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("exact-candidate-set-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);

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
    [InlineData("generated-exact-candidate-set-matrix", "--preset", " ")]
    [InlineData("generated-exact-candidate-set-matrix", "--preset", "standard ")]
    [InlineData("generated-exact-candidate-set-matrix", "--seed", "-1")]
    [InlineData("generated-exact-candidate-set-matrix", "--seed", "0x100000000")]
    [InlineData("generated-exact-candidate-set-matrix", "--vectors", "10.5")]
    [InlineData("generated-exact-candidate-set-matrix", "--manifest", "--output-dir")]
    [InlineData("generated-exact-candidate-set-matrix", "output-dir", "matrix")]
    [InlineData("generated-exact-candidate-set-matrix", "--filter", "broad")]
    [InlineData("generated-exact-candidate-set-matrix", "--metric", "Cosine")]
    [InlineData("generated-exact-candidate-set-matrix", "--top-k", "10")]
    [InlineData("generated-exact-candidate-set-matrix", "--output", "case.json")]
    [InlineData("generated-exact-candidate-set-matrix", "--baseline", "baseline.json")]
    [InlineData("generated-exact-candidate-set-matrix", "--current", "current.json")]
    [InlineData("generated-exact-candidate-set-matrix", "--truth-depth", "10")]
    [InlineData("generated-exact-candidate-set-matrix", "--download", "false")]
    [InlineData("generated-exact-candidate-set-matrix", "--m", "8")]
    [InlineData("generated-exact-candidate-set-matrix", "--ef-construction", "64")]
    [InlineData("generated-exact-candidate-set-matrix", "--ef-search", "32")]
    [InlineData("generated-exact-candidate-set-matrix", "--hnsw-seed", "0x54")]
    public void ParseGeneratedExactCandidateSetMatrix_RejectsMalformedEdgesAndUnrelatedModeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSetMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ExpandCases_StandardPresetOrderSeedsAndCaseNamesAreDeterministic()
    {
        string outputDirectory = NewArtifactDirectory("standard-order");
        var options = new GeneratedExactCandidateSetMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0xFFFF_FFFAu,
            DuplicateIdsPerQuery: 2,
            UnknownIdsPerQuery: 3,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactCandidateSetMatrixScenario.GeneratedExactCandidateSetMatrixCase[] cases =
            GeneratedExactCandidateSetMatrixScenario.ExpandCases(options);

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
            cases.Take(6).Select(item => (item.Options.Metric, item.Options.Dimension, item.Options.TopK, item.Options.CandidateSetKind)).ToArray());
        Assert.Equal(0xFFFF_FFFAu, cases[0].Options.Seed);
        Assert.Equal(0xFFFF_FFFFu, cases[5].Options.Seed);
        Assert.Equal(0u, cases[6].Options.Seed);
        Assert.Equal(113u, cases[^1].Options.Seed);
        Assert.Equal(cases.Length, cases.Select(item => item.Options.OutputPath).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.EndsWith("case-001-squaredeuclidean-32d-10k-all.json", cases[0].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("case-004-squaredeuclidean-32d-10k-very-selective.json", cases[3].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("case-120-cosine-768d-100k-empty.json", cases[^1].Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.All(cases.Where(item => item.Options.CandidateSetKind == "very-selective"), item =>
        {
            Assert.True(item.Options.TopK > 1);
            Assert.True(item.Options.TopK <= item.Options.VectorCount);
        });
    }

    [Fact]
    public void Run_WhenOneCaseReportPathIsBlocked_RecordsFailureAndContinuesRemainingCases()
    {
        string outputDirectory = NewArtifactDirectory("single-blocked-case");
        var options = new GeneratedExactCandidateSetMatrixOptions(
            "smoke",
            VectorCount: 32,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED_5410,
            DuplicateIdsPerQuery: 0,
            UnknownIdsPerQuery: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        string blockedReportPath = Path.Combine(outputDirectory, "case-004-squaredeuclidean-128d-10k-selective.json");
        Directory.CreateDirectory(blockedReportPath);

        GeneratedExactCandidateSetMatrixManifest manifest = GeneratedExactCandidateSetMatrixScenario.Run(
            options,
            ["generated-exact-candidate-set-matrix", "--output-dir", outputDirectory]);
        GeneratedExactCandidateSetMatrixScenario.WriteManifest(manifest, options.ManifestPath);

        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(7, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(1, manifest.Aggregate.FailedCaseCount);

        GeneratedExactCandidateSetMatrixCaseManifest failedCase = Assert.Single(manifest.Cases, item => item.Status == "failed");
        Assert.Equal(4, failedCase.CaseNumber);
        Assert.Equal("SquaredEuclidean", failedCase.Metric);
        Assert.Equal(128, failedCase.Dimension);
        Assert.Equal("selective", failedCase.CandidateSetKind);
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
    public void Run_StandardLinkedReportsPreserveCandidateSetSemanticsAndFalseEligibilityAcrossAllKinds()
    {
        string outputDirectory = NewArtifactDirectory("standard-linked-reports");
        var options = new GeneratedExactCandidateSetMatrixOptions(
            "standard",
            VectorCount: 100,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED_5411,
            DuplicateIdsPerQuery: 2,
            UnknownIdsPerQuery: 3,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        GeneratedExactCandidateSetMatrixManifest manifest = GeneratedExactCandidateSetMatrixScenario.Run(
            options,
            ["generated-exact-candidate-set-matrix"]);

        Assert.Equal(120, manifest.CaseCount);
        Assert.Equal(120, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(["all", "broad", "selective", "very-selective", "empty"], manifest.Cases.Select(item => item.CandidateSetKind).Distinct().ToArray());

        foreach (string candidateSetKind in new[] { "all", "broad", "selective", "very-selective", "empty" })
        {
            GeneratedExactCandidateSetMatrixCaseManifest matrixCase = Assert.Single(
                manifest.Cases,
                item =>
                    item.Metric == "Cosine" &&
                    item.Dimension == 386 &&
                    item.TopK == 10 &&
                    item.CandidateSetKind == candidateSetKind);
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(matrixCase.ReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            JsonElement candidateInput = reportRoot.GetProperty("candidateInput");
            JsonElement candidateSet = reportRoot.GetProperty("candidateSet");

            Assert.Equal(matrixCase.ReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal("VecNet.ExactCandidateSetBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("VEC-053", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal("generated-exact-candidate-set", reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.Metric, reportRoot.GetProperty("dataset").GetProperty("metric").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.VectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.QueryCount, reportRoot.GetProperty("dataset").GetProperty("queryCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal(matrixCase.CandidateSetKind, candidateInput.GetProperty("kind").GetString());
            Assert.Equal(2, candidateInput.GetProperty("duplicateIdCountPerQuery").GetInt32());
            Assert.Equal(3, candidateInput.GetProperty("unknownIdCountPerQuery").GetInt32());
            Assert.Equal("constructedOutsideMeasuredSearch", candidateSet.GetProperty("constructionStatus").GetString());
            Assert.Equal("public ExactFlatIndex.CreateCandidateSet(allowedIds)", candidateSet.GetProperty("constructionOperation").GetString());
            Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", reportRoot.GetProperty("measurement").GetProperty("latency").GetProperty("timedOperation").GetString());
            Assert.Equal("measured", reportRoot.GetProperty("measurement").GetProperty("managedAllocations").GetProperty("status").GetString());
            Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("validation").GetProperty("status").GetString());
            Assert.Equal("passed", reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity").GetProperty("status").GetString());
            Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
            AssertNoForbiddenScopeFields(reportRoot);

            int expectedVisible = candidateSetKind switch
            {
                "all" => 100,
                "broad" => 50,
                "selective" => 10,
                "very-selective" => 9,
                "empty" => 0,
                _ => throw new InvalidOperationException("Unexpected candidate-set kind.")
            };
            Assert.Equal(expectedVisible, candidateInput.GetProperty("knownIdCountPerQuery").GetInt32());
            Assert.Equal(expectedVisible + 5, candidateInput.GetProperty("inputIdCountPerQuery").GetInt32());
            Assert.Equal(expectedVisible, candidateSet.GetProperty("countPerQuery").GetInt32());
            Assert.Equal(expectedVisible, candidateSet.GetProperty("minCount").GetInt32());
            Assert.Equal(expectedVisible, candidateSet.GetProperty("maxCount").GetInt32());
            Assert.Equal(Math.Min(10, expectedVisible), reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity").GetProperty("checkedResultCount").GetInt32());
        }
    }

    [Fact]
    public void CandidateSetMatrixSchemaIsSeparateAndGeneratedExactComparisonRejectsIt()
    {
        string directory = NewArtifactDirectory("schema-separation");
        string candidateMatrixDirectory = Path.Combine(directory, "candidate-matrix");
        string exactMatrixDirectory = Path.Combine(directory, "exact-matrix");
        string filteredMatrixDirectory = Path.Combine(directory, "filtered-matrix");
        string hnswMatrixDirectory = Path.Combine(directory, "hnsw-matrix");
        Directory.CreateDirectory(candidateMatrixDirectory);
        Directory.CreateDirectory(exactMatrixDirectory);
        Directory.CreateDirectory(filteredMatrixDirectory);
        Directory.CreateDirectory(hnswMatrixDirectory);

        GeneratedExactCandidateSetMatrixManifest candidateMatrix = GeneratedExactCandidateSetMatrixScenario.Run(
            new GeneratedExactCandidateSetMatrixOptions(
                "smoke",
                VectorCount: 32,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED_5412,
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputDirectory: candidateMatrixDirectory,
                ManifestPath: Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json")),
            ["generated-exact-candidate-set-matrix"]);
        GeneratedExactCandidateSetMatrixScenario.WriteManifest(candidateMatrix, Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json"));

        GeneratedExactMatrixManifest exactMatrix = GeneratedExactMatrixScenario.Run(
            new GeneratedExactMatrixOptions(
                "smoke",
                VectorCount: 10,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED_5412,
                OutputDirectory: exactMatrixDirectory,
                ManifestPath: Path.Combine(exactMatrixDirectory, "matrix-manifest.json")),
            ["exact-generated-matrix"]);
        GeneratedExactMatrixScenario.WriteManifest(exactMatrix, Path.Combine(exactMatrixDirectory, "matrix-manifest.json"));

        GeneratedExactFilteredMatrixManifest filteredMatrix = GeneratedExactFilteredMatrixScenario.Run(
            new GeneratedExactFilteredMatrixOptions(
                "smoke",
                VectorCount: 32,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED_5412,
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputDirectory: filteredMatrixDirectory,
                ManifestPath: Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json")),
            ["exact-generated-filtered-matrix"]);
        GeneratedExactFilteredMatrixScenario.WriteManifest(filteredMatrix, Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json"));

        HnswGeneratedMatrixManifest hnswMatrix = HnswGeneratedMatrixScenario.Run(
            new HnswGeneratedMatrixOptions(
                "smoke",
                VectorCount: 64,
                QueryCount: 1,
                Runs: 1,
                WarmupQueries: 0,
                Seed: 0x5EED_5412,
                OutputDirectory: hnswMatrixDirectory,
                ManifestPath: Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")),
            ["hnsw-generated-matrix"]);
        HnswGeneratedMatrixScenario.WriteManifest(hnswMatrix, Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json"));

        string candidateReportPath = candidateMatrix.Cases[0].ReportPath;
        string externalStubPath = Path.Combine(directory, "external-report-stub.json");
        File.WriteAllText(externalStubPath, """{"schemaName":"VecNet.ExternalBenchmarkReport"}""");
        string comparisonOutput = Path.Combine(directory, "comparison.json");

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(
                Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json"),
                Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json"),
                comparisonOutput),
            ["compare-generated-exact"]);
        BenchmarkComparisonScenario.Write(comparison, comparisonOutput);

        BenchmarkComparisonArtifact externalComparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(
                Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json"),
                externalStubPath,
                Path.Combine(directory, "external-comparison.json")),
            ["compare-generated-exact"]);

        using JsonDocument candidateMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(candidateMatrixDirectory, "exact-candidate-set-matrix-manifest.json")));
        using JsonDocument candidateReportDocument = JsonDocument.Parse(File.ReadAllText(candidateReportPath));
        using JsonDocument exactMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(exactMatrixDirectory, "matrix-manifest.json")));
        using JsonDocument filteredMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(filteredMatrixDirectory, "exact-filtered-matrix-manifest.json")));
        using JsonDocument hnswMatrixDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(hnswMatrixDirectory, "hnsw-matrix-manifest.json")));
        using JsonDocument comparisonDocument = JsonDocument.Parse(File.ReadAllText(comparisonOutput));
        JsonElement candidateRoot = candidateMatrixDocument.RootElement;
        JsonElement candidateReportRoot = candidateReportDocument.RootElement;
        JsonElement exactRoot = exactMatrixDocument.RootElement;
        JsonElement filteredRoot = filteredMatrixDocument.RootElement;
        JsonElement hnswRoot = hnswMatrixDocument.RootElement;

        Assert.Equal("VecNet.ExactCandidateSetBenchmarkMatrixManifest", candidateRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.ExactCandidateSetBenchmarkReport", candidateReportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.BenchmarkMatrixManifest", exactRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.ExactFilteredBenchmarkMatrixManifest", filteredRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.HnswBenchmarkMatrixManifest", hnswRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VecNet.BenchmarkComparison", comparisonDocument.RootElement.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-054", candidateRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-053", candidateReportRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-015", exactRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-047", filteredRoot.GetProperty("taskId").GetString());
        Assert.Equal("VEC-037", hnswRoot.GetProperty("taskId").GetString());
        Assert.True(candidateRoot.GetProperty("cases")[0].TryGetProperty("candidateSetKind", out _));
        Assert.True(candidateReportRoot.TryGetProperty("candidateInput", out _));
        Assert.True(candidateReportRoot.TryGetProperty("candidateSet", out _));
        Assert.False(candidateRoot.GetProperty("cases")[0].TryGetProperty("filterKind", out _));
        Assert.False(exactRoot.GetProperty("cases")[0].TryGetProperty("candidateSetKind", out _));
        Assert.False(filteredRoot.GetProperty("cases")[0].TryGetProperty("candidateSetKind", out _));
        Assert.False(candidateReportRoot.TryGetProperty("cases", out _));
        Assert.False(candidateReportRoot.TryGetProperty("presetName", out _));

        Assert.Equal("unknown", comparison.ArtifactKind);
        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Empty(comparison.Cases);
        Assert.Empty(comparison.Metrics);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
        Assert.Equal("notComparable", externalComparison.Compatibility.Status);
        Assert.Contains(externalComparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema" && reason.Field == "schemaName");
        AssertNoForbiddenScopeFields(candidateRoot);
        AssertNoForbiddenScopeFields(candidateReportRoot);
    }

    [Fact]
    public void Run_RepeatedExecutionWithSameSeedKeepsCaseAndReportIdentityStable()
    {
        GeneratedExactCandidateSetMatrixManifest first = RunSmokeMatrix("deterministic-a", 0x5EED_5413);
        GeneratedExactCandidateSetMatrixManifest second = RunSmokeMatrix("deterministic-b", 0x5EED_5413);

        Assert.Equal(first.CaseCount, second.CaseCount);
        Assert.Equal(first.Aggregate.PassedCaseCount, second.Aggregate.PassedCaseCount);
        Assert.Equal(first.Aggregate.FailedCaseCount, second.Aggregate.FailedCaseCount);
        Assert.Equal(first.Cases.Length, first.Cases.Select(item => item.ReportId).Distinct(StringComparer.Ordinal).Count());

        for (int i = 0; i < first.Cases.Length; i++)
        {
            GeneratedExactCandidateSetMatrixCaseManifest left = first.Cases[i];
            GeneratedExactCandidateSetMatrixCaseManifest right = second.Cases[i];

            Assert.Equal(left.CaseNumber, right.CaseNumber);
            Assert.Equal(left.Metric, right.Metric);
            Assert.Equal(left.Dimension, right.Dimension);
            Assert.Equal(left.VectorCount, right.VectorCount);
            Assert.Equal(left.QueryCount, right.QueryCount);
            Assert.Equal(left.TopK, right.TopK);
            Assert.Equal(left.Runs, right.Runs);
            Assert.Equal(left.WarmupQueries, right.WarmupQueries);
            Assert.Equal(left.Seed, right.Seed);
            Assert.Equal(left.CandidateSetKind, right.CandidateSetKind);
            Assert.Equal(left.DuplicateIdCountPerQuery, right.DuplicateIdCountPerQuery);
            Assert.Equal(left.UnknownIdCountPerQuery, right.UnknownIdCountPerQuery);
            Assert.Equal(left.ReportId, right.ReportId);
            Assert.Equal(left.Status, right.Status);
            Assert.Equal(left.ValidationStatus, right.ValidationStatus);
            Assert.Equal(Path.GetFileName(left.ReportPath), Path.GetFileName(right.ReportPath));
        }
    }

    private static GeneratedExactCandidateSetMatrixManifest RunSmokeMatrix(string prefix, uint seed)
    {
        string outputDirectory = NewArtifactDirectory(prefix);
        var options = new GeneratedExactCandidateSetMatrixOptions(
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

        GeneratedExactCandidateSetMatrixManifest manifest = GeneratedExactCandidateSetMatrixScenario.Run(options, ["generated-exact-candidate-set-matrix"]);
        GeneratedExactCandidateSetMatrixScenario.WriteManifest(manifest, options.ManifestPath);
        return manifest;
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec054-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
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
            "labelFilter",
            "filterKind",
            "allowlistComparison",
            "rawAllowlistComparison");
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
