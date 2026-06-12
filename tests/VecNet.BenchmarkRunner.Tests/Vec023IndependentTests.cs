using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec023IndependentTests
{
    [Fact]
    public void IdxReaders_RejectInvalidGzipAndTruncatedHeaders()
    {
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            new MemoryStream([0x1f, 0x8b, 0x08, 0x00, 0x01]),
            expectedCount: 1,
            expectedRows: 2,
            expectedColumns: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            new MemoryStream([0x1f, 0x8b, 0x08, 0x00, 0x01]),
            expectedCount: 1));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadImages(
            Gzip([0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 2]),
            expectedCount: 1,
            expectedRows: 2,
            expectedColumns: 2));
        Assert.Throws<InvalidDataException>(() => IdxFileReader.ReadLabels(
            Gzip([0, 0, 8, 1]),
            expectedCount: 1));
    }

    [Fact]
    public void DenseFloat32Matrix_ReadHeaderRejectsCorruptHeaders()
    {
        byte[] valid = MatrixBytes(rowCount: 1, dimension: 2, [7f, 8f]);

        byte[] badMagic = (byte[])valid.Clone();
        badMagic[0] = (byte)'X';
        Assert.Throws<InvalidDataException>(() => DenseFloat32Matrix.ReadHeader(new MemoryStream(badMagic)));

        byte[] badReserved = (byte[])valid.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(badReserved.AsSpan(20, 4), 1);
        Assert.Throws<InvalidDataException>(() => DenseFloat32Matrix.ReadHeader(new MemoryStream(badReserved)));

        Assert.Throws<EndOfStreamException>(() => DenseFloat32Matrix.ReadHeader(new MemoryStream(valid[..20])));
    }

    [Fact]
    public void Scenario_Md5MismatchStopsBeforeDerivedArtifacts()
    {
        string cacheRoot = CreateArtifactDirectory("checksum-hard-stop");
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot);
        FashionMnistDatasetSpecification mismatchedSpec = spec with
        {
            QueryLabels = spec.QueryLabels with { OfficialMd5 = "00000000000000000000000000000000" }
        };
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
            FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist"], mismatchedSpec));

        Assert.Contains("MD5 mismatch", exception.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(cacheRoot, "converted", spec.DatasetId, "base.f32le")));
        Assert.False(File.Exists(Path.Combine(cacheRoot, "converted", spec.DatasetId, "query.f32le")));
        Assert.False(File.Exists(Path.Combine(cacheRoot, "converted", spec.DatasetId, "conversion-manifest.json")));
        Assert.False(File.Exists(Path.Combine(cacheRoot, "truth", spec.DatasetId, "exact-truth.json")));
        Assert.False(File.Exists(Path.Combine(cacheRoot, "evidence", spec.DatasetId, "exact-validation.json")));
        Assert.False(File.Exists(Path.Combine(cacheRoot, "manifests", spec.DatasetId, "dataset-manifest.json")));
    }

    [Fact]
    public void ConversionBytes_AreDeterministicAcrossIndependentRuns()
    {
        FashionMnistAdmissionResult first = RunSyntheticWorkflow("determinism-a");
        FashionMnistAdmissionResult second = RunSyntheticWorkflow("determinism-b");

        string firstRoot = CacheRootFromManifest(first.ManifestPath);
        string secondRoot = CacheRootFromManifest(second.ManifestPath);
        string datasetId = first.Manifest.DatasetId;

        Assert.Equal(
            File.ReadAllBytes(Path.Combine(firstRoot, "converted", datasetId, "base.f32le")),
            File.ReadAllBytes(Path.Combine(secondRoot, "converted", datasetId, "base.f32le")));
        Assert.Equal(
            File.ReadAllBytes(Path.Combine(firstRoot, "converted", datasetId, "query.f32le")),
            File.ReadAllBytes(Path.Combine(secondRoot, "converted", datasetId, "query.f32le")));
    }

    [Fact]
    public void TruthGeneration_IsDeterministicAndOrdersTiesByBaseOrdinal()
    {
        float[] bases =
        [
            2f, 0f,
            0f, 2f,
            -2f, 0f,
            0f, -2f,
            0f, 0f
        ];
        float[] queries =
        [
            0f, 0f,
            2f, 0f,
            99f, 99f
        ];

        TruthSet first = FashionMnistExactTruth.Generate(
            bases,
            baseCount: 5,
            queries,
            queryCount: 3,
            dimension: 2,
            querySubsetCount: 2,
            depth: 4);
        TruthSet second = FashionMnistExactTruth.Generate(
            bases,
            baseCount: 5,
            queries,
            queryCount: 3,
            dimension: 2,
            querySubsetCount: 2,
            depth: 4);

        Assert.Equal(first.Results, second.Results);
        Assert.Equal(
            [
                new TruthItem(4, 0f),
                new TruthItem(0, 4f),
                new TruthItem(1, 4f),
                new TruthItem(2, 4f)
            ],
            first.Results[0]);
        Assert.Equal(
            [
                new TruthItem(0, 0f),
                new TruthItem(4, 4f),
                new TruthItem(1, 8f),
                new TruthItem(3, 8f)
            ],
            first.Results[1]);
    }

    [Fact]
    public void QuerySubset_UsesStableFirstQueriesOnly()
    {
        float[] bases =
        [
            0f, 0f,
            10f, 0f,
            20f, 0f
        ];
        float[] queries =
        [
            0f, 0f,
            10f, 0f,
            20f, 0f
        ];

        TruthSet truth = FashionMnistExactTruth.Generate(
            bases,
            baseCount: 3,
            queries,
            queryCount: 3,
            dimension: 2,
            querySubsetCount: 2,
            depth: 1);

        Assert.Equal(2, truth.Results.Length);
        Assert.Equal(0UL, truth.Results[0][0].Id);
        Assert.Equal(1UL, truth.Results[1][0].Id);
    }

    [Fact]
    public void Artifacts_DoNotLeakLabelsOrPrivatePathsOutsideAllowedMetadata()
    {
        FashionMnistAdmissionResult result = RunSyntheticWorkflow("privacy");
        string cacheRoot = CacheRootFromManifest(result.ManifestPath);
        string datasetId = result.Manifest.DatasetId;
        string baseMatrixPath = Path.Combine(cacheRoot, "converted", datasetId, "base.f32le");
        string queryMatrixPath = Path.Combine(cacheRoot, "converted", datasetId, "query.f32le");

        Assert.False(result.Manifest.Privacy.PublicClaimEligible);
        Assert.False(result.Manifest.Privacy.BaselineCandidateEligible);
        Assert.False(result.Manifest.Privacy.RegressionGateEligible);
        Assert.False(result.Manifest.Evidence.PublicClaimEligible);
        Assert.False(result.Manifest.Evidence.BaselineCandidateEligible);
        Assert.False(result.Manifest.Evidence.RegressionGateEligible);
        Assert.False(result.Evidence.PublicClaimEligible);
        Assert.False(result.Evidence.BaselineCandidateEligible);
        Assert.False(result.Evidence.RegressionGateEligible);
        Assert.Equal("private-raw", result.Manifest.Privacy.PrivacyClass);
        Assert.Equal("local-evidence", result.Manifest.Privacy.EvidenceClass);
        Assert.Equal("private-raw", result.Evidence.PrivacyClass);
        Assert.Equal("local-evidence", result.Evidence.EvidenceClass);

        string manifestJson = File.ReadAllText(result.ManifestPath);
        string evidenceJson = File.ReadAllText(result.EvidencePath);
        string truthJson = File.ReadAllText(result.TruthPath);
        Assert.DoesNotContain(cacheRoot, manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(cacheRoot, evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(cacheRoot, truthJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("histogram", evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("storedInConvertedVectors", evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("storedInTruthArtifact", evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("label", truthJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("histogram", truthJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("publicClaimEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("baselineCandidateEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("regressionGateEligible\": true", manifestJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("publicClaimEligible\": true", evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("baselineCandidateEligible\": true", evidenceJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("regressionGateEligible\": true", evidenceJson, StringComparison.OrdinalIgnoreCase);

        AssertMatrixPayload(baseMatrixPath, rowCount: 4, dimension: 4, expectedValues:
        [
            0f, 0f, 0f, 0f,
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            9f, 9f, 9f, 9f
        ]);
        AssertMatrixPayload(queryMatrixPath, rowCount: 2, dimension: 4, expectedValues:
        [
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f
        ]);
    }

    private static FashionMnistAdmissionResult RunSyntheticWorkflow(string prefix)
    {
        string cacheRoot = CreateArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot);
        var options = new FashionMnistExternalDatasetOptions(cacheRoot, QueryCount: 2, TruthDepth: 2, DownloadRawFiles: false);
        return FashionMnistExternalDatasetScenario.Run(options, ["external-fashion-mnist", "--download", "false"], spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(string cacheRoot)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(4, 2, 2, [
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            9, 9, 9, 9
        ]).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(4, [7, 8, 9, 7]).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(2, 2, 2, [
            1, 0, 0, 0,
            0, 1, 0, 0
        ]).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(2, [8, 9]).ToArray());

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

    private static byte[] MatrixBytes(int rowCount, int dimension, float[] values)
    {
        using var stream = new MemoryStream();
        DenseFloat32Matrix.Write(stream, rowCount, dimension, values);
        return stream.ToArray();
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
            "vec023-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static void AssertMatrixPayload(string path, ulong rowCount, uint dimension, float[] expectedValues)
    {
        byte[] bytes = File.ReadAllBytes(path);
        using var stream = new MemoryStream(bytes);
        DenseFloat32MatrixHeader header = DenseFloat32Matrix.ReadHeader(stream);
        Assert.Equal(rowCount, header.RowCount);
        Assert.Equal(dimension, header.Dimension);

        var actual = new float[expectedValues.Length];
        for (int i = 0; i < actual.Length; i++)
        {
            actual[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(24 + i * 4, 4)));
        }

        Assert.Equal(expectedValues, actual);
    }
}
