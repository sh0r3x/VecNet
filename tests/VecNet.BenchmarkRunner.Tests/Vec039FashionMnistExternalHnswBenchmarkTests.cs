using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec039FashionMnistExternalHnswBenchmarkTests
{
    [Fact]
    public void ParseExternalFashionMnistHnsw_UsesPrivateDefaults()
    {
        FashionMnistExternalHnswBenchmarkOptions options = CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw.json"), options.OutputPath);
        Assert.Equal(3, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(3, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(100, options.EfSearch);
        Assert.Equal(0x484E535700000039UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw", "--truth-depth", "10")]
    [InlineData("external-fashion-mnist-hnsw", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw", "--baseline", "old.json")]
    [InlineData("external-fashion-mnist-hnsw", "--current", "new.json")]
    [InlineData("external-fashion-mnist-hnsw", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnsw", "--runs", "0")]
    [InlineData("external-fashion-mnist-hnsw", "--runs", "6")]
    [InlineData("external-fashion-mnist-hnsw", "--warmup-queries", "-1")]
    [InlineData("external-fashion-mnist-hnsw", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw", "--output", "")]
    [InlineData("external-fashion-mnist-hnsw", "--m", "1")]
    [InlineData("external-fashion-mnist-hnsw", "--m", "65")]
    [InlineData("external-fashion-mnist-hnsw", "--m", "8", "--ef-construction", "7")]
    [InlineData("external-fashion-mnist-hnsw", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnsw", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnsw", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnsw", "--hnsw-seed", "0xNOTHEX")]
    public void ParseExternalFashionMnistHnsw_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnistHnsw(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistHnsw_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
    }

    [Fact]
    public void ParseExternalFashionMnistHnsw_AcceptsInnerProduct()
    {
        FashionMnistExternalHnswBenchmarkOptions options =
            CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--metric", "InnerProduct"]);

        Assert.Equal(VectorMetric.InnerProduct, options.Metric);
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsPrivateExternalHnswReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("report", baseCount: 32, queryCount: 6, truthDepth: 5);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "external-hnsw-report.json");
        var options = new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            QueryCount: 4,
            TopK: 5,
            Runs: 3,
            WarmupQueries: 3,
            VectorMetric.SquaredEuclidean,
            M: 4,
            EfConstruction: 16,
            EfSearch: 8,
            HnswSeed: 0x0000000000000390UL);

        ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-hnsw", "--query-count", "4", "--top-k", "5"]);
        FashionMnistExternalHnswBenchmarkScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalHnswBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-039", report.TaskId);
        Assert.Equal("external-fashion-mnist-hnsw", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("external-hnsw-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal("VecNet.ExternalDatasetManifest", report.Dataset.AdmissionManifest.SchemaName);
        Assert.Equal("0.1", report.Dataset.AdmissionManifest.SchemaVersion);
        Assert.Equal("manifests/fashion-mnist-784-euclidean/dataset-manifest.json", report.Dataset.AdmissionManifest.RelativePath);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Conversion.OutputFiles.Select(file => file.Sha256), report.Dataset.ConvertedMatrices.Select(file => file.Sha256));
        Assert.Equal(admission.Manifest.Truth.Sha256, report.Truth.Sha256);
        Assert.Equal(32, report.Workload.BaseCount);
        Assert.Equal(6, report.Workload.QueryMatrixCount);
        Assert.Equal(4, report.Workload.MeasuredQueryCount);
        Assert.Equal(5, report.Workload.TopK);
        Assert.Equal(5, report.Truth.TopK);
        Assert.Equal("HnswIndex", report.Index.Type);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(8, report.Hnsw.EfSearch);
        Assert.Equal("0x0000000000000390", report.Hnsw.RandomSeed);
        Assert.Contains("admitted base matrix row order", report.Hnsw.InsertionOrder, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Build.Status);
        Assert.True(report.Build.ElapsedMilliseconds >= 0);
        Assert.Equal("measured", report.Build.ManagedAllocations.Status);
        Assert.Equal("bytes", report.Build.ManagedAllocations.Unit);
        Assert.True(long.Parse(report.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Equal(4, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal(4, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("internal HnswIndex.Search(query, results, workspace)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("HNSW build", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.Contains("caller-owned SearchResult[] and HnswSearchWorkspace", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(3, report.Measurement.Warmup.WarmupCount);
        Assert.Equal("estimated", report.MemoryEstimate.Status);
        Assert.Contains("layout-derived", report.MemoryEstimate.EstimateKind, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not a resident/process/GC-heap measurement", report.MemoryEstimate.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.InRange(report.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.ResultCountViolationCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Contains("admitted base-row", report.Metrics.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingTruth);
        Assert.True(report.Validation.AllowsApproximateRecallBelowOne);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("external-hnsw-smoke", root.GetProperty("evidence").GetProperty("scope").GetString());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("estimated", root.GetProperty("memoryEstimate").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("latencyTicks", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_WithSyntheticCosineAdmittedCache_ValidatesCanonicalCosineDistances()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("cosine-report", baseCount: 32, queryCount: 6, truthDepth: 5, metric: VectorMetric.Cosine);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            Path.Combine(cacheRoot, "..", "external-hnsw-cosine-report.json"),
            QueryCount: 4,
            TopK: 5,
            Runs: 1,
            WarmupQueries: 2,
            VectorMetric.Cosine,
            M: 4,
            EfConstruction: 16,
            EfSearch: 16,
            HnswSeed: 0x0000000000002390UL);

        ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(
            options,
            ["external-fashion-mnist-hnsw", "--metric", "Cosine"]);

        Assert.Equal("fashion-mnist-784-cosine", report.Dataset.DatasetId);
        Assert.Equal("Cosine", report.Workload.VecNetMetric);
        Assert.Equal("Cosine", report.Index.Metric);
        Assert.Equal("vecnet-scalar-reference-cosine", report.Truth.Kind);
        Assert.Contains("canonical cosine", report.Truth.DistanceSemantics, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal("passed", report.Validation.Status);
    }

    [Fact]
    public void Run_AllowsPassingApproximateReportWhenRecallIsBelowOne()
    {
        ExternalHnswBenchmarkReport? belowPerfect = null;
        for (ulong seed = 0x390; seed < 0x3D0 && belowPerfect is null; seed++)
        {
            FashionMnistAdmissionResult admission = RunSyntheticAdmission("below-perfect-" + seed.ToString("X", CultureInfo.InvariantCulture), baseCount: 96, queryCount: 8, truthDepth: 10);
            string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
            ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(
                new FashionMnistExternalHnswBenchmarkOptions(
                    cacheRoot,
                    Path.Combine(cacheRoot, "external-hnsw.json"),
                    QueryCount: 8,
                    TopK: 10,
                    Runs: 1,
                    WarmupQueries: 0,
                    VectorMetric.SquaredEuclidean,
                    M: 2,
                    EfConstruction: 2,
                    EfSearch: 10,
                    HnswSeed: seed),
                ["external-fashion-mnist-hnsw"]);

            if (report.Metrics.RecallAtK < 1)
            {
                belowPerfect = report;
            }
        }

        Assert.NotNull(belowPerfect);
        Assert.Equal("passed", belowPerfect.Validation.Status);
        Assert.True(belowPerfect.Validation.AllowsApproximateRecallBelowOne);
        Assert.InRange(belowPerfect.Metrics.RecallAtK, 0, 0.999999);
        Assert.Equal("passed", belowPerfect.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", belowPerfect.Metrics.ReturnedResultIntegrity.Status);
        Assert.True(belowPerfect.Metrics.ReturnedResultIntegrity.CheckedResultCount > 0);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal(0, belowPerfect.Metrics.MissingResultCount);
        Assert.False(belowPerfect.Evidence.PublicClaimEligible);
        Assert.False(belowPerfect.Eligibility.BaselineCandidateEligible);
        Assert.False(belowPerfect.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ValidateReturnedResults_FailsMalformedExternalApproximateResults()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("integrity", baseCount: 8, queryCount: 4, truthDepth: 2);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        FashionMnistExternalHnswBenchmarkOptions options = new(cacheRoot, Path.Combine(cacheRoot, "report.json"), 4, 2, 1, 0, VectorMetric.SquaredEuclidean, 4, 8, 2, 0x39UL);
        ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(options, ["external-fashion-mnist-hnsw"]);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = CreateDatasetForIntegrity(admission, report);
        SearchResult[][] malformed =
        [
            [
                ResultFor(dataset, queryRow: 0, id: 0),
                ResultFor(dataset, queryRow: 0, id: 1),
                ResultFor(dataset, queryRow: 0, id: 2)
            ],
            [
                ResultFor(dataset, queryRow: 1, id: 3),
                ResultFor(dataset, queryRow: 1, id: 3)
            ],
            [
                new SearchResult(99, 1)
            ],
            [
                new SearchResult(4, float.NaN),
                new SearchResult(5, ResultFor(dataset, queryRow: 3, id: 5).Distance + 1)
            ]
        ];

        HnswReturnedResultIntegrityInfo integrity = FashionMnistExternalHnswBenchmarkScenario.ValidateReturnedResults(dataset, malformed, expectedQueryCount: 4, topK: 2);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(8, integrity.CheckedResultCount);
        Assert.Equal(0, integrity.QueryCountMismatchCount);
        Assert.Equal(1, integrity.ResultCountViolationCount);
        Assert.Equal(1, integrity.NonFiniteDistanceCount);
        Assert.Equal(1, integrity.DuplicateIdCount);
        Assert.Equal(1, integrity.UnknownIdCount);
        Assert.Equal(2, integrity.DistanceMismatchCount);
    }

    [Fact]
    public void Run_MissingManifestAndChecksumMismatchFailBeforeReport()
    {
        string missingCacheRoot = CreateArtifactDirectory("missing-manifest");
        string missingOutput = Path.Combine(missingCacheRoot, "report.json");
        Assert.Throws<FileNotFoundException>(() =>
            FashionMnistExternalHnswBenchmarkScenario.Run(
                new FashionMnistExternalHnswBenchmarkOptions(missingCacheRoot, missingOutput, 1, 1, 1, 0, VectorMetric.SquaredEuclidean, 4, 8, 1, 0x39UL),
                ["external-fashion-mnist-hnsw"]));
        Assert.False(File.Exists(missingOutput));

        FashionMnistAdmissionResult admission = RunSyntheticAdmission("checksum", baseCount: 8, queryCount: 2, truthDepth: 2);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "report.json");
        string baseMatrixPath = Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le");
        using (FileStream stream = File.Open(baseMatrixPath, FileMode.Open, FileAccess.ReadWrite))
        {
            stream.Position = stream.Length - 1;
            stream.WriteByte(123);
        }

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalHnswBenchmarkScenario.Run(
                new FashionMnistExternalHnswBenchmarkOptions(cacheRoot, outputPath, 1, 1, 1, 0, VectorMetric.SquaredEuclidean, 4, 8, 1, 0x39UL),
                ["external-fashion-mnist-hnsw"]));

        Assert.Contains("base matrix SHA-256 mismatch", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void CompareGeneratedExact_TreatsExternalHnswReportAsUnsupportedSchema()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("comparison", baseCount: 16, queryCount: 4, truthDepth: 2);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string reportPath = Path.Combine(cacheRoot, "external-hnsw-report.json");
        var options = new FashionMnistExternalHnswBenchmarkOptions(cacheRoot, reportPath, 2, 2, 1, 0, VectorMetric.SquaredEuclidean, 4, 8, 2, 0x39UL);
        ExternalHnswBenchmarkReport report = FashionMnistExternalHnswBenchmarkScenario.Run(options, ["external-fashion-mnist-hnsw"]);
        FashionMnistExternalHnswBenchmarkScenario.Write(report, reportPath);

        BenchmarkComparisonArtifact comparison = BenchmarkComparisonScenario.Run(
            new BenchmarkComparisonOptions(reportPath, reportPath, Path.Combine(cacheRoot, "comparison.json")),
            ["compare-generated-exact"]);

        Assert.Equal("notComparable", comparison.Compatibility.Status);
        Assert.Contains(comparison.Compatibility.Reasons, reason => reason.Code == "unsupportedSchema");
        Assert.Empty(comparison.Metrics);
        Assert.False(comparison.PublicClaimEligible);
        Assert.False(comparison.BaselineCandidateEligible);
        Assert.False(comparison.RegressionGateEligible);
    }

    [Fact]
    public void Program_UnsupportedDownloadOptionDoesNotRunAdmissionOrWriteReport()
    {
        string cacheRoot = CreateArtifactDirectory("program-no-download");
        string outputPath = Path.Combine(cacheRoot, "report.json");
        TextWriter originalOut = Console.Out;
        TextWriter originalError = Console.Error;
        using var output = new StringWriter();
        using var error = new StringWriter();
        try
        {
            Console.SetOut(output);
            Console.SetError(error);

            int exitCode = BenchmarkRunnerProgram.Run(
                ["external-fashion-mnist-hnsw", "--cache-root", cacheRoot, "--output", outputPath, "--download", "true"]);

            Assert.Equal(1, exitCode);
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
        }

        Assert.Contains("Unsupported option '--download'", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "raw")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "converted")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "truth")));
        Assert.False(Directory.Exists(Path.Combine(cacheRoot, "manifests")));
    }

    [Fact]
    public void SyntheticCommandFixture_CreatesAdmittedCacheForRequiredSmokeCommand()
    {
        string cacheRoot = Path.Combine("VecNet.BenchmarkRunner.Artifacts", "vec-039-synthetic-cache");
        if (Directory.Exists(cacheRoot))
        {
            Directory.Delete(cacheRoot, recursive: true);
        }

        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount: 32, queryCount: 4, rows: 4, columns: 4);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: 4, TruthDepth: 5, DownloadRawFiles: false);
        FashionMnistAdmissionResult admission = FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);

        Assert.True(File.Exists(admission.ManifestPath));
        Assert.True(File.Exists(admission.TruthPath));
    }

    private static FashionMnistAdmissionResult RunSyntheticAdmission(
        string prefix,
        int baseCount,
        int queryCount,
        int truthDepth,
        VectorMetric metric = VectorMetric.SquaredEuclidean)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 4, columns: 4);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: queryCount, TruthDepth: truthDepth, DownloadRawFiles: false, metric);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset CreateDatasetForIntegrity(
        FashionMnistAdmissionResult admission,
        ExternalHnswBenchmarkReport report)
    {
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        ExternalConvertedMatrixEntry baseEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "base");
        ExternalConvertedMatrixEntry queryEntry = admission.Manifest.Conversion.OutputFiles.Single(file => file.Role == "query");
        float[] baseVectors = DenseFloat32Matrix.Read(Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le"), (ulong)baseEntry.RowCount, (uint)baseEntry.Dimension);
        float[] queryVectors = DenseFloat32Matrix.Read(Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "query.f32le"), (ulong)queryEntry.RowCount, (uint)queryEntry.Dimension);
        return new FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset(
            new FashionMnistExternalHnswBenchmarkScenario.DatasetPaths(cacheRoot, admission.Manifest.DatasetId, admission.ManifestPath),
            admission.Manifest,
            report.Dataset.AdmissionManifest.Sha256,
            ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(admission.TruthPath))!,
            admission.Manifest.Truth.Sha256,
            baseVectors,
            queryVectors,
            baseEntry.RowCount,
            queryEntry.RowCount,
            baseEntry.Dimension);
    }

    private static SearchResult ResultFor(FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset, int queryRow, ulong id) =>
        new(id, ScalarGroundTruth.CalculateDistance(dataset.GetQueryVector(queryRow), dataset.GetBaseVector(checked((int)id)), VectorMetric.SquaredEuclidean));

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 11)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 29)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount)).ToArray());

        static FashionMnistRawFileSpec Spec(string path, string fileName, string role, int expectedCount) =>
            new(fileName, role, expectedCount, FileChecksum.ComputeMd5(path), downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            datasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: baseCount,
            QueryCount: queryCount,
            ImageRows: rows,
            ImageColumns: columns,
            Dimension: checked(rows * columns),
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", baseCount),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", baseCount),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", queryCount),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", queryCount));
    }

    private static byte[] CreatePixels(int count, int dimension, int offset)
    {
        var payload = new byte[checked(count * dimension)];
        for (int row = 0; row < count; row++)
        {
            for (int column = 0; column < dimension; column++)
            {
                payload[(row * dimension) + column] = (byte)((row * 17 + column * 31 + offset) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)(i % 10);
        }

        return labels;
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2051);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2049);
        WriteInt32BigEndian(decoded, count);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream Gzip(byte[] decoded)
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            gzip.Write(decoded);
        }

        compressed.Position = 0;
        return compressed;
    }

    private static void WriteInt32BigEndian(Stream stream, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, value);
        stream.Write(buffer);
    }

    private static string CreateArtifactDirectory(string prefix)
    {
        string outputDirectory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec039-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
