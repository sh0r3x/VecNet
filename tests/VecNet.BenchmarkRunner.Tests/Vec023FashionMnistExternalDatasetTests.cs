using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec023FashionMnistExternalDatasetTests
{
    [Fact]
    public void ParseExternalFashionMnist_UsesPrivateDatasetCacheDefaults()
    {
        FashionMnistExternalDatasetOptions options = CommandLine.ParseExternalFashionMnist(["external-fashion-mnist"]);

        Assert.Equal("VecNet.DatasetCache", options.CacheRoot);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TruthDepth);
        Assert.False(options.DownloadRawFiles);
        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("external-fashion-mnist", "--query-count")]
    [InlineData("external-fashion-mnist", "--unknown", "1")]
    [InlineData("external-fashion-mnist", "--query-count", "0")]
    [InlineData("external-fashion-mnist", "--truth-depth", "-1")]
    [InlineData("external-fashion-mnist", "--download", "maybe")]
    [InlineData("external-fashion-mnist", "--cache-root", "")]
    public void ParseExternalFashionMnist_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseExternalFashionMnist(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseExternalFashionMnist_AcceptsCosineAndInnerProduct()
    {
        FashionMnistExternalDatasetOptions cosine =
            CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--metric", "Cosine"]);
        FashionMnistExternalDatasetOptions innerProduct =
            CommandLine.ParseExternalFashionMnist(["external-fashion-mnist", "--metric", "InnerProduct"]);

        Assert.Equal(VectorMetric.Cosine, cosine.Metric);
        Assert.Equal(VectorMetric.InnerProduct, innerProduct.Metric);
    }

    [Fact]
    public void VerifyRawFile_ChecksMd5AndComputesSha256()
    {
        string directory = CreateArtifactDirectory("checksum");
        string path = Path.Combine(directory, "train-images-idx3-ubyte.gz");
        File.WriteAllBytes(path, [1, 2, 3, 4]);
        var spec = new FashionMnistRawFileSpec(
            "train-images-idx3-ubyte.gz",
            "base-images",
            ExpectedCount: 1,
            FileChecksum.ComputeMd5(path),
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz");

        RawFileVerification verification = FileChecksum.VerifyRawFile(path, spec);

        Assert.Equal("passed", verification.VerificationStatus);
        Assert.Equal(4, verification.ByteSize);
        Assert.Equal(FileChecksum.ComputeSha256(path), verification.ComputedSha256);
        Assert.Throws<InvalidDataException>(() => FileChecksum.VerifyRawFile(path, spec with { OfficialMd5 = "00000000000000000000000000000000" }));
    }

    [Fact]
    public void IdxReaders_ParseTinyGzipFixturesStrictly()
    {
        using MemoryStream images = CreateImageIdxGzip(count: 2, rows: 2, columns: 2, [0, 1, 254, 255, 5, 6, 7, 8]);
        using MemoryStream labels = CreateLabelIdxGzip(count: 2, [0, 9]);

        IdxImageSet imageSet = IdxFileReader.ReadImages(images, expectedCount: 2, expectedRows: 2, expectedColumns: 2);
        IdxLabelSet labelSet = IdxFileReader.ReadLabels(labels, expectedCount: 2);

        Assert.Equal(2, imageSet.Count);
        Assert.Equal(4, imageSet.Dimension);
        Assert.Equal([0, 1, 254, 255], imageSet.GetImage(0).ToArray());
        Assert.Equal(2, labelSet.Count);
        Assert.Equal(0, labelSet.MinValue);
        Assert.Equal(9, labelSet.MaxValue);
        Assert.Equal(1, labelSet.Histogram[0]);
        Assert.Equal(1, labelSet.Histogram[9]);
    }

    [Fact]
    public void IdxReaders_RejectMalformedImageFiles()
    {
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            CreateImageIdxGzip(2, 2, 2, [1, 2, 3], magic: 2051),
            expectedCount: 2,
            expectedRows: 2,
            expectedColumns: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            CreateImageIdxGzip(2, 2, 2, [1, 2, 3, 4, 5, 6, 7, 8, 9], magic: 2051),
            expectedCount: 2,
            expectedRows: 2,
            expectedColumns: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            CreateImageIdxGzip(2, 2, 2, [1, 2, 3, 4, 5, 6, 7, 8], magic: 2049),
            expectedCount: 2,
            expectedRows: 2,
            expectedColumns: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            CreateImageIdxGzip(2, 3, 2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], magic: 2051),
            expectedCount: 2,
            expectedRows: 2,
            expectedColumns: 2));
    }

    [Fact]
    public void IdxReaders_RejectMalformedLabelFiles()
    {
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            CreateLabelIdxGzip(2, [1], magic: 2049),
            expectedCount: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            CreateLabelIdxGzip(2, [1, 2, 3], magic: 2049),
            expectedCount: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            CreateLabelIdxGzip(2, [1, 2], magic: 2051),
            expectedCount: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            CreateLabelIdxGzip(2, [1, 10], magic: 2049),
            expectedCount: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            CreateLabelIdxGzip(2, [1, 2], magic: 2049),
            expectedCount: 3));
    }

    [Fact]
    public void DenseFloat32Matrix_WritesDeterministicLittleEndianHeaderAndPayload()
    {
        using var first = new MemoryStream();
        using var second = new MemoryStream();
        float[] values = [0f, 1f, 254f, 255f];

        DenseFloat32Matrix.Write(first, rowCount: 1, dimension: 4, values);
        DenseFloat32Matrix.Write(second, rowCount: 1, dimension: 4, values);

        Assert.Equal(first.ToArray(), second.ToArray());
        first.Position = 0;
        DenseFloat32MatrixHeader header = DenseFloat32Matrix.ReadHeader(first);
        Assert.Equal("VecNetDenseFloat32Matrix", header.SchemaName);
        Assert.Equal("0.1", header.SchemaVersion);
        Assert.Equal(1UL, header.RowCount);
        Assert.Equal(4U, header.Dimension);
        byte[] bytes = first.ToArray();
        Assert.Equal("VNDM001\0", Encoding.ASCII.GetString(bytes, 0, 8));
        Assert.Equal(BitConverter.SingleToUInt32Bits(255f), BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(36, 4)));
    }

    [Fact]
    public void FashionMnistExactTruth_UsesBoundedQuerySubsetAndBaseIdTieOrder()
    {
        float[] bases =
        [
            0f, 0f,
            2f, 0f,
            0f, 2f
        ];
        float[] queries =
        [
            1f, 0f,
            9f, 9f
        ];

        TruthSet truth = FashionMnistExactTruth.Generate(
            bases,
            baseCount: 3,
            queries,
            queryCount: 2,
            dimension: 2,
            querySubsetCount: 1,
            depth: 3);

        Assert.Single(truth.Results);
        Assert.Equal(
            [
                new TruthItem(0, 1f),
                new TruthItem(1, 1f),
                new TruthItem(2, 5f)
            ],
            truth.Results[0]);
    }

    [Fact]
    public void FashionMnistExactTruth_CosineUsesCanonicalOrderingAndIdTies()
    {
        float[] bases =
        [
            2f, 0f,
            4f, 0f,
            0f, 3f,
            -1f, 0f
        ];
        float[] queries =
        [
            1f, 0f
        ];

        TruthSet truth = FashionMnistExactTruth.Generate(
            bases,
            baseCount: 4,
            queries,
            queryCount: 1,
            dimension: 2,
            querySubsetCount: 1,
            depth: 4,
            VectorMetric.Cosine);

        Assert.Equal(
            [
                new TruthItem(0, 0f),
                new TruthItem(1, 0f),
                new TruthItem(2, 1f),
                new TruthItem(3, 2f)
            ],
            truth.Results[0]);
    }

    [Fact]
    public void Run_WithSyntheticTinyOfficialFileSet_EmitsPrivateManifestConversionTruthAndEvidence()
    {
        string cacheRoot = CreateArtifactDirectory("workflow");
        FashionMnistDatasetSpecification spec = WriteSyntheticFashionMnistRawFiles(cacheRoot);
        var options = new FashionMnistExternalDatasetOptions(
            cacheRoot,
            QueryCount: 2,
            TruthDepth: 2,
            DownloadRawFiles: false);

        FashionMnistAdmissionResult result = FashionMnistExternalDatasetScenario.Run(
            options,
            ["external-fashion-mnist", "--download", "false"],
            spec);

        Assert.True(File.Exists(result.ManifestPath));
        Assert.True(File.Exists(result.EvidencePath));
        Assert.True(File.Exists(result.TruthPath));
        Assert.True(File.Exists(result.ConversionManifestPath));
        Assert.Equal("VecNet.ExternalDatasetManifest", result.Manifest.SchemaName);
        Assert.Equal("0.1", result.Manifest.SchemaVersion);
        Assert.Equal("fashion-mnist-784-euclidean", result.Manifest.DatasetId);
        Assert.Equal("VEC-023", result.Manifest.AdmittingTaskId);
        Assert.False(result.Manifest.Privacy.PublicClaimEligible);
        Assert.False(result.Manifest.Privacy.BaselineCandidateEligible);
        Assert.False(result.Manifest.Privacy.RegressionGateEligible);
        Assert.Equal(4, result.Manifest.RawFiles.Length);
        Assert.All(result.Manifest.RawFiles, file => Assert.Equal("passed", file.VerificationStatus));
        Assert.All(result.Manifest.RawFiles, file => Assert.False(Path.IsPathRooted(file.RelativePath)));
        Assert.False(result.Manifest.Labels.Base.StoredInConvertedVectors);
        Assert.False(result.Manifest.Labels.Base.StoredInTruthArtifact);
        Assert.Equal("VecNet.ExternalExactValidation", result.Evidence.SchemaName);
        Assert.Equal("passed", result.Evidence.Validation.Status);
        Assert.Equal(1.0, result.Evidence.Validation.RecallAtK);
        Assert.Equal(1.0, result.Evidence.Validation.OrderedAgreement);
        Assert.Equal(0, result.Evidence.Validation.MissingResultCount);
        Assert.Equal(0, result.Evidence.Validation.ExtraResultCount);
        Assert.Equal(0, result.Evidence.Validation.DistanceMismatchCount);
        Assert.Equal("notMeasured", result.Evidence.ManagedAllocations.Status);
        Assert.Equal("notMeasured", result.Evidence.Memory.Status);
        Assert.False(result.Evidence.PublicClaimEligible);
        Assert.False(result.Evidence.BaselineCandidateEligible);
        Assert.False(result.Evidence.RegressionGateEligible);

        using FileStream baseMatrix = File.OpenRead(Path.Combine(cacheRoot, "converted", spec.DatasetId, "base.f32le"));
        DenseFloat32MatrixHeader header = DenseFloat32Matrix.ReadHeader(baseMatrix);
        Assert.Equal(4UL, header.RowCount);
        Assert.Equal(4U, header.Dimension);

        string truthJson = File.ReadAllText(result.TruthPath);
        Assert.DoesNotContain("label", truthJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("histogram", truthJson, StringComparison.OrdinalIgnoreCase);

        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(result.ManifestPath));
        Assert.Equal("VecNetDenseFloat32Matrix", manifest.RootElement.GetProperty("conversion").GetProperty("outputFormat").GetString());
        Assert.False(manifest.RootElement.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(manifest.RootElement.GetProperty("evidence").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(manifest.RootElement.GetProperty("evidence").GetProperty("regressionGateEligible").GetBoolean());
    }

    [Fact]
    public void Run_WithCosineMetric_UsesDistinctIdentityAndCosineTruthMetadata()
    {
        string cacheRoot = CreateArtifactDirectory("cosine-workflow");
        FashionMnistDatasetSpecification spec = WriteSyntheticFashionMnistRawFiles(
            cacheRoot,
            basePayload:
            [
                1, 0, 0, 0,
                1, 0, 0, 0,
                0, 1, 0, 0,
                9, 9, 9, 9
            ]);
        var options = new FashionMnistExternalDatasetOptions(
            cacheRoot,
            QueryCount: 2,
            TruthDepth: 2,
            DownloadRawFiles: false,
            VectorMetric.Cosine);

        FashionMnistAdmissionResult result = FashionMnistExternalDatasetScenario.Run(
            options,
            ["external-fashion-mnist", "--metric", "Cosine"],
            spec);

        Assert.Equal("fashion-mnist-784-cosine", result.Manifest.DatasetId);
        Assert.Equal("VEC-239", result.Manifest.AdmittingTaskId);
        Assert.Equal("cosine", result.Manifest.Metric.UpstreamName);
        Assert.Equal("Cosine", result.Manifest.Metric.VecNetMetric);
        Assert.Equal("vecnet-scalar-reference-cosine", result.Manifest.Truth.Kind);
        Assert.Contains("canonical cosine", result.Manifest.Truth.TiePolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("Cosine", result.Evidence.Metric);
        Assert.Equal("cosine", result.Evidence.UpstreamMetric);
        Assert.Equal("passed", result.Evidence.Validation.Status);
        Assert.True(File.Exists(Path.Combine(cacheRoot, "converted", "fashion-mnist-784-cosine", "base.f32le")));
        Assert.True(File.Exists(Path.Combine(cacheRoot, "truth", "fashion-mnist-784-cosine", "exact-truth.json")));

        ExternalExactTruthArtifact truth = ReportWriter.Deserialize<ExternalExactTruthArtifact>(File.ReadAllText(result.TruthPath))!;
        Assert.Equal("fashion-mnist-784-cosine", truth.DatasetId);
        Assert.Equal("VEC-239", truth.TaskId);
        Assert.Equal("Cosine", truth.Metric);
    }

    [Fact]
    public void Run_WithCosineMetric_RejectsZeroBaseOrSelectedQueryRows()
    {
        string zeroBaseCacheRoot = CreateArtifactDirectory("cosine-zero-base");
        FashionMnistDatasetSpecification zeroBaseSpec = WriteSyntheticFashionMnistRawFiles(zeroBaseCacheRoot);
        var cosineOptions = new FashionMnistExternalDatasetOptions(zeroBaseCacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false, VectorMetric.Cosine);

        InvalidDataException baseException = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalDatasetScenario.Run(cosineOptions, ["external-fashion-mnist", "--metric", "Cosine"], zeroBaseSpec));
        Assert.Contains("base row 0 is zero", baseException.Message, StringComparison.OrdinalIgnoreCase);

        string zeroQueryCacheRoot = CreateArtifactDirectory("cosine-zero-query");
        FashionMnistDatasetSpecification zeroQuerySpec = WriteSyntheticFashionMnistRawFiles(
            zeroQueryCacheRoot,
            basePayload:
            [
                1, 0, 0, 0,
                1, 0, 0, 0,
                0, 1, 0, 0,
                9, 9, 9, 9
            ],
            queryPayload:
            [
                0, 0, 0, 0,
                0, 1, 0, 0
            ]);
        cosineOptions = new FashionMnistExternalDatasetOptions(zeroQueryCacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false, VectorMetric.Cosine);

        InvalidDataException queryException = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalDatasetScenario.Run(cosineOptions, ["external-fashion-mnist", "--metric", "Cosine"], zeroQuerySpec));
        Assert.Contains("query row 0 is zero", queryException.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticFashionMnistRawFiles(
        string cacheRoot,
        byte[]? basePayload = null,
        byte[]? queryPayload = null)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(4, 2, 2, basePayload ?? [
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            9, 9, 9, 9
        ]).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(4, [0, 1, 2, 9]).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(2, 2, 2, queryPayload ?? [
            1, 0, 0, 0,
            0, 1, 0, 0
        ]).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(2, [1, 2]).ToArray());

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
            BaseCount: 4,
            QueryCount: 2,
            ImageRows: 2,
            ImageColumns: 2,
            Dimension: 4,
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", 4),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", 4),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", 2),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", 2));
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload, int magic = 2051)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, magic);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload, int magic = 2049)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, magic);
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
            "vec023-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }
}
