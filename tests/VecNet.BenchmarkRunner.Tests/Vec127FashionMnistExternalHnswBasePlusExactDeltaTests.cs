using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec127FashionMnistExternalHnswBasePlusExactDeltaTests
{
    [Fact]
    public void ParseExternalFashionMnistHnswBasePlusExactDelta_UsesAcceptedPrivateDefaults()
    {
        FashionMnistExternalHnswBasePlusExactDeltaOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                ["external-fashion-mnist-hnsw-base-plus-exact-delta"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(Path.Combine("VecNet.BenchmarkRunner.Artifacts", "fashion-mnist-external-hnsw-base-plus-exact-delta.json"), options.OutputPath);
        Assert.Equal(50, options.QueryCount);
        Assert.Equal(100, options.TopK);
        Assert.Equal(58_000, options.BaseVectorCount);
        Assert.Equal(1_000, options.InsertedDeltaCount);
        Assert.Equal(59_000, options.PhysicalCandidateVectorCount);
        Assert.Equal(1_000, options.DeletedBaseCount);
        Assert.Equal(100, options.DeletedDeltaCount);
        Assert.Equal(57_900, options.LiveVectorCount);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(0x5EED2127u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(128, options.EfConstruction);
        Assert.Equal(192, options.EfSearch);
        Assert.Equal(0x484E535700012700UL, options.HnswSeed);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--download", "false")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--truth-depth", "100")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--preset", "smoke")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--output-dir", "matrix")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--manifest", "manifest.json")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--snapshot-directory", "snapshot")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--checkpoint-directory", "checkpoint")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--hnswlib-python", "python")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--allowlist", "broad")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--candidate-set", "selective")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--metric", "InnerProduct")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--query-count", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--top-k", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--base-vectors", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--insertions", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--deletes", "11", "--base-vectors", "10")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--delta-deletes", "-1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--delta-deletes", "2", "--insertions", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--deletes", "0", "--delta-deletes", "0", "--repeated-deletes", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--top-k", "10", "--base-vectors", "8", "--insertions", "1", "--deletes", "0", "--delta-deletes", "0")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--top-k", "10", "--ef-search", "9")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--m", "1")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--m", "65")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--m", "8", "--ef-construction", "7")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--ef-construction", "4097")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--ef-search", "4097")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--seed", "0xNOTHEX")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--hnsw-seed", "0xNOTHEX")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--cache-root", "")]
    [InlineData("external-fashion-mnist-hnsw-base-plus-exact-delta", "--output", "")]
    public void ParseExternalFashionMnistHnswBasePlusExactDelta_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(args));

        Assert.NotEmpty(exception.Message);
    }

    [Theory]
    [InlineData("Cosine")]
    [InlineData("cosine")]
    public void ParseExternalFashionMnistHnswBasePlusExactDelta_AcceptsCosine(string metric)
    {
        FashionMnistExternalHnswBasePlusExactDeltaOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, "--metric", metric]);

        Assert.Equal(VectorMetric.Cosine, options.Metric);
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(options.Metric));
    }

    [Fact]
    public void Run_WithSyntheticAdmittedCache_EmitsPrivateExternalCompositeReport()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("report", baseCount: 48, queryCount: 6, truthDepth: 8);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "..", "external-composite-report.json");
        string[] arguments =
        [
            "external-fashion-mnist-hnsw-base-plus-exact-delta",
            "--cache-root", cacheRoot,
            "--output", outputPath,
            "--query-count", "5",
            "--top-k", "6",
            "--base-vectors", "36",
            "--insertions", "8",
            "--deletes", "5",
            "--delta-deletes", "3",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "2",
            "--runs", "3",
            "--warmup-queries", "2",
            "--metric", "squared-euclidean",
            "--seed", "0x5EED1270",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "10",
            "--hnsw-seed", "0x0000000000001270"
        ];
        FashionMnistExternalHnswBasePlusExactDeltaOptions options =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(arguments);

        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(options, arguments);
        FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-127", report.TaskId);
        Assert.Equal("external-fashion-mnist-hnsw-base-plus-exact-delta", report.ScenarioName);
        Assert.Equal("external-fashion-mnist-hnsw-base-plus-exact-delta", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("external-fashion-mnist-hnsw-base-plus-exact-delta-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("fashion-mnist-784-euclidean", report.Dataset.DatasetId);
        Assert.Equal("VecNet.ExternalDatasetManifest", report.Dataset.AdmissionManifest.SchemaName);
        Assert.Equal(FileChecksum.ComputeSha256(admission.ManifestPath), report.Dataset.AdmissionManifest.Sha256);
        Assert.Equal(admission.Manifest.Truth.Sha256, report.ExistingTruthGuard.Sha256);
        Assert.Contains("readiness guard", report.ExistingTruthGuard.DistanceSemantics, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("scalar-reference-external-live-hnsw-base-plus-exact-delta", report.UpdatedTruth.Kind);
        Assert.False(report.UpdatedTruth.Persisted);
        Assert.Equal(5, report.UpdatedTruth.QueryCount);
        Assert.Equal(6, report.UpdatedTruth.TruthDepth);
        Assert.Equal(36 + 8 - 5 - 3, report.UpdatedTruth.LiveVectorCount);
        Assert.Contains("post-update live view", report.UpdatedTruth.Source, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ascending scalar-reference squared", report.UpdatedTruth.TiePolicy, StringComparison.Ordinal);

        Assert.Equal(48, report.Workload.AdmittedBaseMatrixRowCount);
        Assert.Equal(6, report.Workload.QueryMatrixCount);
        Assert.Equal(5, report.Workload.MeasuredQueryCount);
        Assert.Equal(6, report.Workload.TopK);
        Assert.Equal(0, report.Workload.ImmutableBaseStartRow);
        Assert.Equal(35, report.Workload.ImmutableBaseEndRowInclusive);
        Assert.Equal(36, report.Workload.ImmutableBaseRowCount);
        Assert.Equal(36, report.Workload.DeltaStartRow);
        Assert.Equal(43, report.Workload.DeltaEndRowInclusive);
        Assert.Equal(8, report.Workload.DeltaRowCount);
        Assert.Equal(4, report.Workload.UnusedCandidateRowCount);
        Assert.Contains("original Fashion-MNIST base row ordinals", report.Workload.ExternalIdPolicy, StringComparison.Ordinal);
        Assert.Contains("contiguous admitted base-matrix rows", report.Workload.RowSelectionPolicy, StringComparison.Ordinal);

        Assert.Equal("HnswBasePlusExactDeltaIndex", report.Index.Type);
        Assert.Contains("internal", report.Index.Configuration, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(10, report.Hnsw.EfSearch);
        Assert.Equal("0x0000000000001270", report.Hnsw.RandomSeed);
        Assert.Equal("measured", report.Build.Status);
        Assert.True(report.Build.ElapsedMilliseconds >= 0);
        Assert.Equal("measured", report.Build.ManagedAllocations.Status);
        Assert.True(long.Parse(report.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);

        Assert.Equal(36, report.Counts.BasePhysicalVectorCount);
        Assert.Equal(31, report.Counts.BaseLiveVectorCount);
        Assert.Equal(8, report.Counts.DeltaPhysicalVectorCount);
        Assert.Equal(5, report.Counts.DeltaLiveVectorCount);
        Assert.Equal(5, report.Counts.BaseTombstoneCount);
        Assert.Equal(3, report.Counts.DeltaTombstoneCount);
        Assert.Equal(8, report.Counts.TombstoneCount);
        Assert.Equal(36, report.Counts.LiveVectorCount);
        Assert.Equal(8, report.Counts.DeletedReservedIdCount);
        Assert.Equal(16, report.Counts.Generation);

        Assert.Equal(8, report.Mutations.InsertedCount);
        Assert.Equal(5, report.Mutations.DeletedBaseCount);
        Assert.Equal(3, report.Mutations.DeletedDeltaCount);
        Assert.Equal(16, report.Mutations.CommittedMutationCount);
        Assert.Equal(16, report.Mutations.GenerationDelta);
        Assert.True(report.Mutations.GenerationDeltaMatchesCommittedMutations);
        Assert.Equal(16, report.Mutations.StatusCounts.Committed);
        Assert.Equal(2, report.Mutations.StatusCounts.DuplicateId);
        Assert.Equal(3, report.Mutations.StatusCounts.UnknownId);
        Assert.Equal(2, report.Mutations.StatusCounts.AlreadyDeleted);
        Assert.Equal(0, report.Mutations.StatusCounts.ReadOnly);
        Assert.Equal(0, report.Mutations.StatusCounts.Unsupported);

        Assert.Equal(5, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("exact updated truth", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerSearchCall", report.Measurement.ManagedAllocations.Unit);
        Assert.Contains("caller-owned SearchResult[] and HnswBasePlusExactDeltaSearchWorkspace", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);

        Assert.InRange(report.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.TombstonedIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Contains("exact updated top-k", report.Metrics.RecallDefinition, StringComparison.OrdinalIgnoreCase);

        Assert.Equal(5, report.Underfill.QueryCount);
        Assert.Equal(6, report.Underfill.RequestedResultCountPerQuery);
        Assert.Equal(30, report.Underfill.TotalRequestedResultSlots);
        Assert.Equal(30 - report.Underfill.TotalReturnedResults, report.Underfill.UnderfilledSlotCount);
        Assert.Contains("Underfill is recorded", report.Underfill.Policy, StringComparison.Ordinal);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.LoadedExistingTruthGuard);
        Assert.True(report.Validation.UpdatedTruthGeneratedFromLiveView);
        Assert.True(report.Validation.HnswBaseBuilt);
        Assert.True(report.Validation.MutationsApplied);
        Assert.True(report.Validation.MutationStatusCountsMatched);
        Assert.True(report.Validation.GenerationMovementMatchedCommittedMutations);
        Assert.True(report.Validation.FinalRunComparedToUpdatedTruth);
        Assert.True(report.Validation.ReturnedResultsAreLiveAndNotTombstoned);
        Assert.True(report.Validation.AllowsApproximateRecallBelowOne);
        Assert.True(report.Validation.AllowsUnderfill);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("external-fashion-mnist-hnsw-base-plus-exact-delta", root.GetProperty("scenarioName").GetString());
        Assert.Equal("fashion-mnist-784-euclidean", root.GetProperty("dataset").GetProperty("datasetId").GetString());
        Assert.Equal("scalar-reference-external-live-hnsw-base-plus-exact-delta", root.GetProperty("updatedTruth").GetProperty("kind").GetString());
        Assert.Equal(36, root.GetProperty("counts").GetProperty("liveVectorCount").GetInt32());
        Assert.Equal(16, root.GetProperty("mutations").GetProperty("statusCounts").GetProperty("committed").GetInt32());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoPropertyNamed(root, "preset", "manifest", "snapshotDirectory", "checkpointDirectory", "hnswlibPython", "candidateEligibility", "regressionDecision", "publicClaimStatus");
    }

    [Fact]
    public void Run_RejectsCacheShapeThatCannotCoverSelectedBaseAndDeltaRows()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("shape", baseCount: 20, queryCount: 3, truthDepth: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                new FashionMnistExternalHnswBasePlusExactDeltaOptions(
                    cacheRoot,
                    Path.Combine(cacheRoot, "report.json"),
                    QueryCount: 2,
                    TopK: 3,
                    BaseVectorCount: 18,
                    InsertedDeltaCount: 3,
                    DeletedBaseCount: 1,
                    DeletedDeltaCount: 0,
                    DuplicateInsertAttempts: 0,
                    UnknownDeleteAttempts: 0,
                    RepeatedDeleteAttempts: 0,
                    Runs: 1,
                    WarmupQueries: 0,
                    VectorMetric.SquaredEuclidean,
                    Seed: 0x5EED1271,
                    M: 4,
                    EfConstruction: 8,
                    EfSearch: 3,
                    HnswSeed: 0x1271),
                ["external-fashion-mnist-hnsw-base-plus-exact-delta"]));

        Assert.Contains("base vectors plus insertions", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateReturnedResults_FailsUnknownTombstonedDuplicateAndDistanceMismatches()
    {
        FashionMnistAdmissionResult admission = RunSyntheticAdmission("integrity", baseCount: 12, queryCount: 3, truthDepth: 3);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        var options = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            Path.Combine(cacheRoot, "report.json"),
            QueryCount: 3,
            TopK: 2,
            BaseVectorCount: 8,
            InsertedDeltaCount: 2,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED1272,
            M: 4,
            EfConstruction: 8,
            EfSearch: 2,
            HnswSeed: 0x1272);
        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(options, ["external-fashion-mnist-hnsw-base-plus-exact-delta"]);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = CreateDatasetForIntegrity(admission, report);
        SearchResult live = ResultFor(dataset, queryRow: 0, id: 3);

        HnswBasePlusExactDeltaReturnedResultIntegrityInfo integrity =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.ValidateReturnedResults(
                dataset,
                [
                    [
                        live,
                        live,
                        new SearchResult(0, ResultFor(dataset, queryRow: 0, id: 0).Distance),
                        new SearchResult(11, ResultFor(dataset, queryRow: 0, id: 11).Distance),
                        new SearchResult(4, float.NaN)
                    ],
                    [
                        new SearchResult(9, ResultFor(dataset, queryRow: 1, id: 9).Distance)
                    ],
                    [
                        new SearchResult(5, ResultFor(dataset, queryRow: 2, id: 5).Distance + 100)
                    ]
                ],
                options,
                liveIds: [2, 3, 4, 5, 6, 7, 9]);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(7, integrity.CheckedResultCount);
        Assert.Equal(1, integrity.ResultCountViolationCount);
        Assert.Equal(1, integrity.NonFiniteDistanceCount);
        Assert.Equal(1, integrity.DuplicateIdCount);
        Assert.Equal(1, integrity.UnknownIdCount);
        Assert.Equal(1, integrity.TombstonedIdCount);
        Assert.Equal(2, integrity.DistanceMismatchCount);
        Assert.Contains("tombstoned IDs must not be returned", integrity.Policy, StringComparison.OrdinalIgnoreCase);
    }

    private static FashionMnistAdmissionResult RunSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 4, columns: 4);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: queryCount, TruthDepth: truthDepth, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset CreateDatasetForIntegrity(
        FashionMnistAdmissionResult admission,
        ExternalHnswBasePlusExactDeltaBenchmarkReport report)
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
        new(id, SquaredEuclidean(dataset.GetQueryVector(queryRow), dataset.GetBaseVector(checked((int)id))));

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
            "vec127-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
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
