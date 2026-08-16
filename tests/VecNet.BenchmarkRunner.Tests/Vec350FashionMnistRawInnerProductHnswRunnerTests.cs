using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec350FashionMnistRawInnerProductHnswRunnerTests
{
    [Fact]
    public void RawInnerProductAdmissionWritesDistinctIdentityAndNegativeDotTruth()
    {
        Assert.Equal("fashion-mnist-784-euclidean", FashionMnistDatasetSpecification.GetDatasetId(VectorMetric.SquaredEuclidean));
        Assert.Equal("fashion-mnist-784-cosine", FashionMnistDatasetSpecification.GetDatasetId(VectorMetric.Cosine));
        Assert.Equal("fashion-mnist-784-inner-product", FashionMnistDatasetSpecification.GetDatasetId(VectorMetric.InnerProduct));
        Assert.Equal("vecnet-scalar-reference-inner-product", FashionMnistExactTruth.Kind(VectorMetric.InnerProduct));

        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "identity",
            baseCount: 12,
            queryCount: 3,
            truthDepth: 4,
            VectorMetric.InnerProduct);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        ExternalExactTruthArtifact truth = ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(admission.TruthPath))!;

        Assert.Equal("fashion-mnist-784-inner-product", admission.Manifest.DatasetId);
        Assert.Equal("VEC-350", admission.Manifest.AdmittingTaskId);
        Assert.Equal("raw-inner-product", admission.Manifest.Metric.UpstreamName);
        Assert.Equal("InnerProduct", admission.Manifest.Metric.VecNetMetric);
        Assert.Contains("negative-dot", admission.Manifest.Metric.RankingNote, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("vecnet-scalar-reference-inner-product", admission.Manifest.Truth.Kind);
        Assert.Equal(FashionMnistExactTruth.InnerProductTiePolicy, admission.Manifest.Truth.TiePolicy);
        Assert.Equal("fashion-mnist-784-inner-product", truth.DatasetId);
        Assert.Equal("VEC-350", truth.TaskId);
        Assert.Equal("InnerProduct", truth.Metric);
        Assert.Equal(FashionMnistExactTruth.InnerProductTiePolicy, truth.TiePolicy);
        Assert.Contains("inner-product", FashionMnistExactTruth.DistanceSemantics(VectorMetric.InnerProduct), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("converted/fashion-mnist-784-inner-product/base.f32le", admission.Manifest.Conversion.OutputFiles[0].RelativePath, StringComparison.Ordinal);
        Assert.Contains("truth/fashion-mnist-784-inner-product/exact-truth.json", admission.Manifest.Truth.RelativePath, StringComparison.Ordinal);

        ExternalConvertedMatrixEntry baseEntry = admission.Manifest.Conversion.OutputFiles.Single(entry => entry.Role == "base");
        ExternalConvertedMatrixEntry queryEntry = admission.Manifest.Conversion.OutputFiles.Single(entry => entry.Role == "query");
        float[] baseVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "base.f32le"),
            (ulong)baseEntry.RowCount,
            (uint)baseEntry.Dimension);
        float[] queryVectors = DenseFloat32Matrix.Read(
            Path.Combine(cacheRoot, "converted", admission.Manifest.DatasetId, "query.f32le"),
            (ulong)queryEntry.RowCount,
            (uint)queryEntry.Dimension);

        TruthItem[] expected = Enumerable.Range(0, baseEntry.RowCount)
            .Select(row => new TruthItem(
                (ulong)row,
                ScalarGroundTruth.CalculateDistance(
                    queryVectors.AsSpan(0, queryEntry.Dimension),
                    baseVectors.AsSpan(row * baseEntry.Dimension, baseEntry.Dimension),
                    VectorMetric.InnerProduct)))
            .OrderBy(item => item.Distance)
            .ThenBy(item => item.Id)
            .Take(truth.TruthDepth)
            .ToArray();

        Assert.Equal(expected.Select(item => item.Id), truth.Queries[0].Neighbors.Select(item => item.Id));
        Assert.Equal(expected.Select(item => item.Distance), truth.Queries[0].Neighbors.Select(item => item.SquaredDistance));
    }

    [Fact]
    public void InnerProductDoesNotImportExistingEuclideanCacheIdentity()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "euclidean-only",
            baseCount: 12,
            queryCount: 3,
            truthDepth: 4,
            VectorMetric.SquaredEuclidean);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);

        var options = new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            Path.Combine(cacheRoot, "inner-product-report.json"),
            QueryCount: 2,
            TopK: 2,
            Runs: 1,
            WarmupQueries: 0,
            VectorMetric.InnerProduct,
            M: 2,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x350);

        FileNotFoundException exception = Assert.Throws<FileNotFoundException>(
            () => FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(options));

        Assert.Contains("fashion-mnist-784-inner-product", exception.FileName, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("fashion-mnist-784-euclidean", admission.Manifest.DatasetId);
    }

    [Fact]
    public void CommandLineAndMatrixSurfacesAdmitRawInnerProductMetric()
    {
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistExact(["external-fashion-mnist-exact", "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistHnsw(["external-fashion-mnist-hnsw", "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistDurableHnsw(["external-fashion-mnist-hnsw-durable", "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDelta(
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpoint(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistHnswAllowlistFiltering(
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, "--metric", "inner-product"]).Metric);
        Assert.Equal(
            VectorMetric.InnerProduct,
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMemorySmoke(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMemorySmokeOptions.ScenarioName, "--metric", "inner-product"]).Metric);

        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions matrixOptions =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName, "--metric", "inner-product"]);
        FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions checkpointMatrixOptions =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaCheckpointMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixOptions.ScenarioName, "--metric", "inner-product"]);

        Assert.Equal(VectorMetric.InnerProduct, matrixOptions.Metric);
        Assert.Equal(VectorMetric.InnerProduct, checkpointMatrixOptions.Metric);
        Assert.Contains(
            "inner-product",
            FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.CreateCaseArguments(
                FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExpandCases(matrixOptions)[0].Options),
            StringComparer.OrdinalIgnoreCase);
        Assert.Contains(
            "inner-product",
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.CreateCaseArguments(
                FashionMnistExternalHnswBasePlusExactDeltaCheckpointMatrixScenario.ExpandCases(checkpointMatrixOptions)[0].Options),
            StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExternalHnswSurfacesRunOnSyntheticRawInnerProductCache()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "scenarios",
            baseCount: 32,
            queryCount: 5,
            truthDepth: 5,
            VectorMetric.InnerProduct);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputRoot = CreateArtifactDirectory("scenario-output");

        ExternalHnswBenchmarkReport immutable = FashionMnistExternalHnswBenchmarkScenario.Run(
            new FashionMnistExternalHnswBenchmarkOptions(
                cacheRoot,
                Path.Combine(outputRoot, "immutable.json"),
                QueryCount: 3,
                TopK: 3,
                Runs: 1,
                WarmupQueries: 1,
                VectorMetric.InnerProduct,
                M: 4,
                EfConstruction: 32,
                EfSearch: 16,
                HnswSeed: 0x35001),
            [FashionMnistExternalHnswBenchmarkOptions.ScenarioName, "--metric", "inner-product"]);
        AssertInnerProductReport(immutable.Dataset.DatasetId, immutable.Index.Metric, immutable.Truth.Kind, immutable.Truth.DistanceSemantics);
        Assert.Equal("passed", immutable.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, immutable.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);

        ExternalDurableHnswBenchmarkReport durable = FashionMnistExternalDurableHnswBenchmarkScenario.Run(
            new FashionMnistExternalDurableHnswBenchmarkOptions(
                cacheRoot,
                Path.Combine(outputRoot, "durable.json"),
                Path.Combine(outputRoot, "durable-snapshot"),
                QueryCount: 3,
                TopK: 3,
                Runs: 1,
                WarmupQueries: 1,
                VectorMetric.InnerProduct,
                M: 4,
                EfConstruction: 32,
                EfSearch: 16,
                HnswSeed: 0x35002),
            [FashionMnistExternalDurableHnswBenchmarkOptions.ScenarioName, "--metric", "inner-product"]);
        AssertInnerProductReport(durable.Dataset.DatasetId, durable.Index.Metric, durable.Truth.Kind, durable.Truth.DistanceSemantics);
        Assert.Equal("passed", durable.Validation.Status);

        var mutableOptions = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            Path.Combine(outputRoot, "mutable.json"),
            QueryCount: 3,
            TopK: 3,
            BaseVectorCount: 24,
            InsertedDeltaCount: 4,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.InnerProduct,
            Seed: 0x5EED3501,
            M: 4,
            EfConstruction: 32,
            EfSearch: 20,
            HnswSeed: 0x35003);
        ExternalHnswBasePlusExactDeltaBenchmarkReport mutable =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                mutableOptions,
                [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, "--metric", "inner-product"]);
        AssertInnerProductReport(mutable.Dataset.DatasetId, mutable.Index.Metric, mutable.ExistingTruthGuard.Kind, mutable.UpdatedTruth.DistanceSemantics);
        Assert.Equal("passed", mutable.Validation.Status);

        var checkpointOptions = new FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions(
            cacheRoot,
            Path.Combine(outputRoot, "checkpoint.json"),
            Path.Combine(outputRoot, "checkpoint-output"),
            QueryCount: 3,
            TopK: 3,
            BaseVectorCount: 24,
            InsertedDeltaCount: 4,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.InnerProduct,
            Seed: 0x5EED3502,
            M: 4,
            EfConstruction: 32,
            EfSearch: 20,
            HnswSeed: 0x35004);
        ExternalHnswBasePlusExactDeltaCheckpointBenchmarkReport checkpoint =
            FashionMnistExternalHnswBasePlusExactDeltaCheckpointScenario.Run(
                checkpointOptions,
                [FashionMnistExternalHnswBasePlusExactDeltaCheckpointOptions.ScenarioName, "--metric", "inner-product"]);
        AssertInnerProductReport(checkpoint.Dataset.DatasetId, checkpoint.Index.Metric, checkpoint.ExistingTruthGuard.Kind, checkpoint.UpdatedTruth.DistanceSemantics);
        Assert.Equal("passed", checkpoint.Validation.Status);

        ExternalHnswAllowlistFilteringBenchmarkReport allowlist =
            FashionMnistExternalHnswAllowlistFilteringScenario.Run(
                new FashionMnistExternalHnswAllowlistFilteringOptions(
                    cacheRoot,
                    Path.Combine(outputRoot, "allowlist.json"),
                    Path.Combine(outputRoot, "allowlist-opened"),
                    Path.Combine(outputRoot, "allowlist-checkpoint"),
                    QueryCount: 3,
                    TopK: 3,
                    BaseVectorCount: 24,
                    InsertedDeltaCount: 4,
                    DeletedBaseCount: 2,
                    DeletedDeltaCount: 1,
                    DuplicateInsertAttempts: 1,
                    UnknownDeleteAttempts: 1,
                    RepeatedDeleteAttempts: 1,
                    FilterProfile: "fallback-boundary",
                    Runs: 1,
                    WarmupQueries: 1,
                    VectorMetric.InnerProduct,
                    Seed: 0x5EED3503,
                    M: 4,
                    EfConstruction: 32,
                    EfSearch: 20,
                    HnswSeed: 0x35005),
                [FashionMnistExternalHnswAllowlistFilteringOptions.ScenarioName, "--metric", "inner-product"]);
        AssertInnerProductReport(allowlist.Dataset.DatasetId, allowlist.Index.Metric, allowlist.ExistingTruthGuard.Kind, allowlist.FilteredTruth.DistanceSemantics);
        Assert.Equal("passed", allowlist.Validation.Status);
    }

    [Fact]
    public void ExternalExactRunUsesRawInnerProductIndexAndNegativeDotTruth()
    {
        byte[] basePixels =
        [
            1, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            9, 9, 9, 9
        ];
        byte[] queryPixels =
        [
            1, 0, 0, 0,
            0, 1, 0, 0
        ];
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission(
            "exact-inner-product",
            baseCount: 4,
            queryCount: 2,
            truthDepth: 2,
            VectorMetric.InnerProduct,
            rows: 2,
            columns: 2,
            basePixels,
            queryPixels);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string outputPath = Path.Combine(cacheRoot, "exact-inner-product-report.json");

        ExternalBenchmarkReport report = FashionMnistExternalExactBenchmarkScenario.Run(
            new FashionMnistExternalExactBenchmarkOptions(
                cacheRoot,
                outputPath,
                QueryCount: 2,
                TopK: 2,
                Runs: 1,
                WarmupQueries: 1,
                VectorMetric.InnerProduct),
            ["external-fashion-mnist-exact", "--metric", "inner-product"]);

        ExternalExactTruthArtifact truth = ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(admission.TruthPath))!;
        Assert.Equal(3UL, truth.Queries[0].Neighbors[0].Id);
        Assert.Equal(-9f, truth.Queries[0].Neighbors[0].SquaredDistance);
        Assert.Equal(3UL, truth.Queries[1].Neighbors[0].Id);
        Assert.Equal(-9f, truth.Queries[1].Neighbors[0].SquaredDistance);
        AssertInnerProductReport(report.Dataset.DatasetId, report.Index.Metric, report.Truth.Kind, report.Truth.DistanceSemantics);
        Assert.Equal("InnerProduct", report.Workload.VecNetMetric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(1.0, report.Metrics.RecallAtK);
        Assert.Equal(1.0, report.Metrics.OrderedAgreement);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(0, report.Metrics.MissingResultCount);
        Assert.Equal(0, report.Metrics.ExtraResultCount);
    }

    private static void AssertInnerProductReport(string datasetId, string metric, string truthKind, string distanceSemantics)
    {
        Assert.Equal("fashion-mnist-784-inner-product", datasetId);
        Assert.Equal("InnerProduct", metric);
        Assert.Equal("vecnet-scalar-reference-inner-product", truthKind);
        Assert.Contains("-dot", distanceSemantics, StringComparison.OrdinalIgnoreCase);
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(
        string prefix,
        int baseCount,
        int queryCount,
        int truthDepth,
        VectorMetric metric,
        int rows = 3,
        int columns = 5,
        byte[]? basePixels = null,
        byte[]? queryPixels = null)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows, columns, basePixels, queryPixels);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false, metric),
            ["external-fashion-mnist", "--download", "false", "--metric", metric.ToString()],
            spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns,
        byte[]? basePixels = null,
        byte[]? queryPixels = null)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, basePixels ?? CreatePixels(baseCount, rows * columns, offset: 37)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount, offset: 3)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, queryPixels ?? CreatePixels(queryCount, rows * columns, offset: 83)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount, offset: 5)).ToArray());

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
                payload[(row * dimension) + column] = (byte)((row * 29 + column * 17 + offset + ((row + column) % 5) * 11) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count, int offset)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)((i + offset) % 10);
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
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec350-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;
}
