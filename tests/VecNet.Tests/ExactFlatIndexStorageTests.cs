using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class ExactFlatIndexStorageTests
{
    [Fact]
    public void Save_RejectsNullWhitespaceNonEmptyRepeatedAndFileBackedPaths()
    {
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        index.Add(1, [1f, 2f, 3f]);

        Assert.Throws<ArgumentNullException>(() => index.Save(null!));
        Assert.Throws<ArgumentException>(() => index.Save(""));
        Assert.Throws<ArgumentException>(() => index.Save("   "));

        using TempIndexDirectory nonEmpty = TempIndexDirectory.Create();
        string markerPath = Path.Combine(nonEmpty.Path, "marker.txt");
        File.WriteAllText(markerPath, "caller-owned");
        Assert.Throws<IOException>(() => index.Save(nonEmpty.Path));
        Assert.Equal("caller-owned", File.ReadAllText(markerPath));
        Assert.False(File.Exists(Path.Combine(nonEmpty.Path, ExactFlatIndexStorage.ManifestFileName)));

        using TempIndexDirectory repeated = TempIndexDirectory.Create();
        index.Save(repeated.Path);
        string manifestBefore = File.ReadAllText(Path.Combine(repeated.Path, ExactFlatIndexStorage.ManifestFileName));
        Assert.Throws<IOException>(() => index.Save(repeated.Path));
        Assert.Equal(manifestBefore, File.ReadAllText(Path.Combine(repeated.Path, ExactFlatIndexStorage.ManifestFileName)));

        using TempIndexDirectory fileParent = TempIndexDirectory.Create();
        string filePath = Path.Combine(fileParent.Path, "not-a-directory.vecnet");
        File.WriteAllText(filePath, "not a directory");
        Assert.Throws<IOException>(() => index.Save(filePath));
        Assert.Equal("not a directory", File.ReadAllText(filePath));
    }

    [Fact]
    public void Save_CreatesMissingTargetDirectoryAndAcceptsExistingEmptyDirectory()
    {
        var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        index.Add(10, [1f, 0.5f]);

        string missingDirectory = Path.Combine(
            Path.GetTempPath(),
            "VecNet.Tests",
            Guid.NewGuid().ToString("N"));

        try
        {
            index.Save(missingDirectory);

            Assert.True(Directory.Exists(missingDirectory));
            Assert.True(File.Exists(Path.Combine(missingDirectory, ExactFlatIndexStorage.ManifestFileName)));
            Assert.True(File.Exists(Path.Combine(missingDirectory, ExactFlatIndexStorage.IdsFileName)));
            Assert.True(File.Exists(Path.Combine(missingDirectory, ExactFlatIndexStorage.VectorsFileName)));
        }
        finally
        {
            if (Directory.Exists(missingDirectory))
            {
                Directory.Delete(missingDirectory, recursive: true);
            }
        }

        using TempIndexDirectory empty = TempIndexDirectory.Create();
        index.Save(empty.Path);
        Assert.True(File.Exists(Path.Combine(empty.Path, ExactFlatIndexStorage.ManifestFileName)));
        Assert.True(File.Exists(Path.Combine(empty.Path, ExactFlatIndexStorage.IdsFileName)));
        Assert.True(File.Exists(Path.Combine(empty.Path, ExactFlatIndexStorage.VectorsFileName)));
    }

    [Fact]
    public void Save_FailedInternalWriteDeletesTemporaryFilesAndCreatedDirectoryWhenEmpty()
    {
        string missingDirectory = Path.Combine(
            Path.GetTempPath(),
            "VecNet.Tests",
            Guid.NewGuid().ToString("N"));

        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                ExactFlatIndexStorage.Save(
                    missingDirectory,
                    dimension: 3,
                    VectorMetric.SquaredEuclidean,
                    [1UL],
                    []));

            Assert.False(Directory.Exists(missingDirectory));
        }
        finally
        {
            if (Directory.Exists(missingDirectory))
            {
                Directory.Delete(missingDirectory, recursive: true);
            }
        }

        using TempIndexDirectory existingEmpty = TempIndexDirectory.Create();
        Assert.Throws<InvalidOperationException>(() =>
            ExactFlatIndexStorage.Save(
                existingEmpty.Path,
                dimension: 3,
                VectorMetric.SquaredEuclidean,
                [1UL],
                []));

        Assert.True(Directory.Exists(existingEmpty.Path));
        Assert.Empty(Directory.EnumerateFileSystemEntries(existingEmpty.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsNullWhitespaceAndFileBackedPaths()
    {
        Assert.Throws<ArgumentNullException>(() => ExactFlatIndex.OpenReadOnly(null!));
        Assert.Throws<ArgumentException>(() => ExactFlatIndex.OpenReadOnly(""));
        Assert.Throws<ArgumentException>(() => ExactFlatIndex.OpenReadOnly("   "));

        using TempIndexDirectory fileParent = TempIndexDirectory.Create();
        string filePath = Path.Combine(fileParent.Path, "not-a-directory.vecnet");
        File.WriteAllText(filePath, "not a directory");

        Assert.Throws<IOException>(() => ExactFlatIndex.OpenReadOnly(filePath));
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void SaveAndOpenReadOnly_RoundTripsSearchResultsAcrossMetrics(VectorMetric metric)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(7, metric);
        AddRoundTripRows(index, metric);

        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Equal(index.Dimension, loaded.Dimension);
        Assert.Equal(index.Metric, loaded.Metric);
        AssertSearchEqual(index, loaded, CreateQuery(metric, index.Dimension), topK: 0);
        AssertSearchEqual(index, loaded, CreateQuery(metric, index.Dimension), topK: 1);
        AssertSearchEqual(index, loaded, CreateQuery(metric, index.Dimension), topK: 4);
        AssertSearchEqual(index, loaded, CreateQuery(metric, index.Dimension), topK: 8);
    }

    [Fact]
    public void Save_WritesPinnedManifestAndBinaryHeaders()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        index.Add(100, [1f, 2f, 3f]);
        index.Add(50, [-1f, 0f, 4f]);

        index.Save(temp.Path);

        string manifestPath = Path.Combine(temp.Path, ExactFlatIndexStorage.ManifestFileName);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        Assert.Equal(
            [
                "schemaName",
                "schemaVersion",
                "formatFamily",
                "createdUtc",
                "createdByTask",
                "writer",
                "index",
                "semantics",
                "files",
                "compatibility"
            ],
            document.RootElement.EnumerateObject().Select(static property => property.Name));

        Assert.Equal(ExactFlatIndexStorage.ManifestSchemaName, document.RootElement.GetProperty("schemaName").GetString());
        Assert.Equal("1.0", document.RootElement.GetProperty("schemaVersion").GetString());
        Assert.Equal("exact-flat", document.RootElement.GetProperty("formatFamily").GetString());
        Assert.Equal("VEC-031", document.RootElement.GetProperty("createdByTask").GetString());

        JsonElement indexJson = document.RootElement.GetProperty("index");
        Assert.Equal(JsonValueKind.Number, indexJson.GetProperty("dimension").ValueKind);
        Assert.Equal(3, indexJson.GetProperty("dimension").GetInt32());
        Assert.Equal("squared-euclidean", indexJson.GetProperty("metric").GetString());
        Assert.Equal(2, indexJson.GetProperty("vectorCount").GetInt32());
        Assert.Equal("uint64", indexJson.GetProperty("idType").GetString());
        Assert.Equal("float32", indexJson.GetProperty("vectorElementType").GetString());
        Assert.Equal("row-major-dense", indexJson.GetProperty("vectorLayout").GetString());
        Assert.Equal("none", indexJson.GetProperty("normalizationState").GetString());

        JsonElement idsJson = document.RootElement.GetProperty("files").GetProperty("ids");
        JsonElement vectorsJson = document.RootElement.GetProperty("files").GetProperty("vectors");
        Assert.Equal(ExactFlatIndexStorage.IdsFileName, idsJson.GetProperty("path").GetString());
        Assert.Equal(ExactFlatIndexStorage.IdsHeaderLength + 2 * sizeof(ulong), idsJson.GetProperty("byteLength").GetInt64());
        Assert.Equal("VNETID01", idsJson.GetProperty("binaryMagic").GetString());
        Assert.Equal("1.0", idsJson.GetProperty("binaryVersion").GetString());
        Assert.Equal(64, idsJson.GetProperty("sha256").GetString()!.Length);
        Assert.Equal(ExactFlatIndexStorage.VectorsFileName, vectorsJson.GetProperty("path").GetString());
        Assert.Equal(ExactFlatIndexStorage.VectorsHeaderLength + 2 * 3 * sizeof(float), vectorsJson.GetProperty("byteLength").GetInt64());
        Assert.Equal("VNETVF01", vectorsJson.GetProperty("binaryMagic").GetString());
        Assert.Equal("1.0", vectorsJson.GetProperty("binaryVersion").GetString());
        Assert.Equal(64, vectorsJson.GetProperty("sha256").GetString()!.Length);

        byte[] idsBytes = File.ReadAllBytes(Path.Combine(temp.Path, ExactFlatIndexStorage.IdsFileName));
        Assert.Equal("VNETID01"u8.ToArray(), idsBytes[..8]);
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(idsBytes.AsSpan(8)));
        Assert.Equal(0, BinaryPrimitives.ReadUInt16LittleEndian(idsBytes.AsSpan(10)));
        Assert.Equal((uint)ExactFlatIndexStorage.IdsHeaderLength, BinaryPrimitives.ReadUInt32LittleEndian(idsBytes.AsSpan(12)));
        Assert.Equal(2UL, BinaryPrimitives.ReadUInt64LittleEndian(idsBytes.AsSpan(16)));
        Assert.Equal(0UL, BinaryPrimitives.ReadUInt64LittleEndian(idsBytes.AsSpan(24)));
        Assert.Equal(100UL, BinaryPrimitives.ReadUInt64LittleEndian(idsBytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength)));
        Assert.Equal(50UL, BinaryPrimitives.ReadUInt64LittleEndian(idsBytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength + 8)));

        byte[] vectorBytes = File.ReadAllBytes(Path.Combine(temp.Path, ExactFlatIndexStorage.VectorsFileName));
        Assert.Equal("VNETVF01"u8.ToArray(), vectorBytes[..8]);
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(vectorBytes.AsSpan(8)));
        Assert.Equal(0, BinaryPrimitives.ReadUInt16LittleEndian(vectorBytes.AsSpan(10)));
        Assert.Equal((uint)ExactFlatIndexStorage.VectorsHeaderLength, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(12)));
        Assert.Equal(2UL, BinaryPrimitives.ReadUInt64LittleEndian(vectorBytes.AsSpan(16)));
        Assert.Equal(3U, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(24)));
        Assert.Equal(ExactFlatIndexStorage.Float32RowMajorRepresentationCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(28)));
        Assert.Equal(ExactFlatIndexStorage.NoNormalizationCode, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(32)));
        Assert.Equal(0U, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(36)));
        Assert.Equal(0UL, BinaryPrimitives.ReadUInt64LittleEndian(vectorBytes.AsSpan(40)));
    }

    [Fact]
    public void OpenReadOnly_ZeroRowIndexRemainsSearchableAndReadOnly()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, VectorMetric.InnerProduct);

        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        var results = new SearchResult[2];
        Assert.Equal(0, loaded.Search([1f, 2f, 3f], results));
        Assert.Throws<InvalidOperationException>(() => loaded.Add(1, [1f, 2f, 3f]));
    }

    [Fact]
    public void OpenReadOnly_CosineHydratesStoredRowsWithoutDoubleNormalization()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(2, VectorMetric.Cosine);
        index.Add(10, [1f, 0f]);
        index.Save(temp.Path);

        PatchVectorFloatAndRefreshManifest(temp.Path, valueIndex: 0, 0.99998f);

        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        var results = new SearchResult[1];
        Assert.Equal(1, loaded.Search([1f, 0f], results));
        Assert.Equal(10UL, results[0].Id);
        Assert.InRange(results[0].Distance, 0.00001f, 0.00003f);
    }

    [Fact]
    public void OpenReadOnly_RejectsMissingDirectoryAndManifest()
    {
        string missingDirectory = Path.Combine(Path.GetTempPath(), "VecNet.Tests", Guid.NewGuid().ToString("N"));
        Assert.Throws<DirectoryNotFoundException>(() => ExactFlatIndex.OpenReadOnly(missingDirectory));

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        Assert.Throws<FileNotFoundException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsMalformedManifestSchemaFeaturesAndTraversal()
    {
        AssertOpenFailsAfterManifestMutation(root => root["schemaName"] = "Wrong.Schema");
        AssertOpenFailsAfterManifestMutation(root => root["schemaVersion"] = "2.0");
        AssertOpenFailsAfterManifestMutation(root => root["formatFamily"] = "wrong-family");
        AssertOpenFailsAfterManifestMutation(root =>
            ((JsonArray)root["compatibility"]!["requiredFeatures"]!).Add("unknown-feature"));
        AssertOpenFailsAfterManifestMutation(root => root["files"]!["ids"]!["path"] = "../exact-flat.ids.u64");

        using TempIndexDirectory malformed = CreateSavedSquaredEuclideanIndex();
        File.WriteAllText(Path.Combine(malformed.Path, ExactFlatIndexStorage.ManifestFileName), "{");
        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(malformed.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsMissingLengthChecksumAndTruncatedFiles()
    {
        using (TempIndexDirectory temp = CreateSavedSquaredEuclideanIndex())
        {
            File.Delete(Path.Combine(temp.Path, ExactFlatIndexStorage.IdsFileName));
            Assert.Throws<FileNotFoundException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        AssertOpenFailsAfterManifestMutation(root =>
            root["files"]!["ids"]!["byteLength"] = (long)root["files"]!["ids"]!["byteLength"]! + 1);

        using (TempIndexDirectory temp = CreateSavedSquaredEuclideanIndex())
        {
            string vectorPath = Path.Combine(temp.Path, ExactFlatIndexStorage.VectorsFileName);
            byte[] bytes = File.ReadAllBytes(vectorPath);
            bytes[^1] ^= 0x01;
            File.WriteAllBytes(vectorPath, bytes);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = CreateSavedSquaredEuclideanIndex())
        {
            string vectorPath = Path.Combine(temp.Path, ExactFlatIndexStorage.VectorsFileName);
            using (var stream = new FileStream(vectorPath, FileMode.Open, FileAccess.Write, FileShare.None))
            {
                stream.SetLength(stream.Length - sizeof(float));
            }

            RefreshManifestFileMetadata(temp.Path, "vectors");
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsBadHeadersAndMismatchedBinaryMetadata()
    {
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.IdsFileName, bytes => bytes[0] = (byte)'X');
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 2));
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(24), 1));
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), 99));
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28), 99));
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), 99));
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(40), 1));
    }

    [Fact]
    public void OpenReadOnly_RejectsDuplicateIdsNonFiniteVectorsAndInvalidCosineRows()
    {
        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.IdsFileName, bytes =>
        {
            ulong firstId = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength));
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength + sizeof(ulong)), firstId);
        });

        AssertOpenFailsAfterFilePatchAndManifestRefresh(ExactFlatIndexStorage.VectorsFileName, bytes =>
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength),
                BitConverter.SingleToInt32Bits(float.NaN)));

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var cosine = new ExactFlatIndex(2, VectorMetric.Cosine);
        cosine.Add(7, [1f, 0f]);
        cosine.Save(temp.Path);
        PatchVectorFloatAndRefreshManifest(temp.Path, valueIndex: 0, 0.9f);
        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_SupportsConcurrentSearchWithCallerOwnedBuffers()
    {
        foreach (VectorMetric metric in new[] { VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine })
        {
            using TempIndexDirectory temp = TempIndexDirectory.Create();
            var index = new ExactFlatIndex(5, metric);
            AddRoundTripRows(index, metric);
            index.Save(temp.Path);
            ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
            float[][] queries = [CreateQuery(metric, 5), CreateAlternateQuery(metric, 5)];
            int[] topKs = [0, 1, 3, 8];

            var expected = queries
                .Select(query => topKs.Select(topK => SearchAll(loaded, query, topK)).ToArray())
                .ToArray();

            Parallel.For(0, 200, iteration =>
            {
                int queryIndex = iteration % queries.Length;
                int topKIndex = iteration % topKs.Length;
                SearchResult[] actual = SearchAll(loaded, queries[queryIndex], topKs[topKIndex]);
                Assert.Equal(expected[queryIndex][topKIndex], actual);
            });
        }
    }

    private static void AssertOpenFailsAfterManifestMutation(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = CreateSavedSquaredEuclideanIndex();
        MutateManifest(temp.Path, mutate);
        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertOpenFailsAfterFilePatchAndManifestRefresh(string fileName, Action<byte[]> patch)
    {
        using TempIndexDirectory temp = CreateSavedSquaredEuclideanIndex();
        string path = Path.Combine(temp.Path, fileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);
        RefreshManifestFileMetadata(temp.Path, fileName == ExactFlatIndexStorage.IdsFileName ? "ids" : "vectors");

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    private static TempIndexDirectory CreateSavedSquaredEuclideanIndex()
    {
        TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 2f, 3f]);
        index.Add(20, [-1f, 0f, 4f]);
        index.Save(temp.Path);
        return temp;
    }

    private static void AddRoundTripRows(ExactFlatIndex index, VectorMetric metric)
    {
        foreach ((ulong id, float[] vector) in GetRoundTripRows(metric, index.Dimension))
        {
            index.Add(id, vector);
        }
    }

    private static (ulong Id, float[] Vector)[] GetRoundTripRows(VectorMetric metric, int dimension) =>
    [
        (90, CreateVector(dimension, 1f)),
        (3, CreateVector(dimension, -1f)),
        (ulong.MaxValue, CreateVector(dimension, 0.25f)),
        (7, CreateTiedVector(metric, dimension, positive: true)),
        (2, CreateTiedVector(metric, dimension, positive: false))
    ];

    private static float[] CreateVector(int dimension, float firstValue)
    {
        var vector = new float[dimension];
        vector[0] = firstValue;
        for (int i = 1; i < dimension; i++)
        {
            vector[i] = (i % 3 - 1) * 0.5f;
        }

        return vector;
    }

    private static float[] CreateTiedVector(VectorMetric metric, int dimension, bool positive)
    {
        var vector = new float[dimension];
        vector[0] = metric == VectorMetric.SquaredEuclidean ? (positive ? 1f : -1f) : 2f;
        if (dimension > 1)
        {
            vector[1] = positive ? 1f : -1f;
        }

        return vector;
    }

    private static float[] CreateQuery(VectorMetric metric, int dimension)
    {
        var query = new float[dimension];
        query[0] = metric == VectorMetric.SquaredEuclidean ? 0f : 1f;
        if (dimension > 1)
        {
            query[1] = metric == VectorMetric.Cosine ? 0.5f : -0.25f;
        }

        return query;
    }

    private static float[] CreateAlternateQuery(VectorMetric metric, int dimension)
    {
        var query = new float[dimension];
        query[0] = metric == VectorMetric.Cosine ? -0.25f : 0.75f;
        for (int i = 1; i < dimension; i++)
        {
            query[i] = (i % 2 == 0 ? 1f : -1f) * 0.125f;
        }

        return query;
    }

    private static void AssertSearchEqual(ExactFlatIndex expectedIndex, ExactFlatIndex actualIndex, float[] query, int topK)
    {
        SearchResult[] expected = SearchAll(expectedIndex, query, topK);
        SearchResult[] actual = SearchAll(actualIndex, query, topK);
        Assert.Equal(expected, actual);
    }

    private static SearchResult[] SearchAll(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static void PatchVectorFloatAndRefreshManifest(string directory, int valueIndex, float value)
    {
        string path = Path.Combine(directory, ExactFlatIndexStorage.VectorsFileName);
        byte[] bytes = File.ReadAllBytes(path);
        BinaryPrimitives.WriteInt32LittleEndian(
            bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength + valueIndex * sizeof(float)),
            BitConverter.SingleToInt32Bits(value));
        File.WriteAllBytes(path, bytes);
        RefreshManifestFileMetadata(directory, "vectors");
    }

    private static void MutateManifest(string directory, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, ExactFlatIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static void RefreshManifestFileMetadata(string directory, string filePropertyName)
    {
        MutateManifest(directory, root =>
        {
            JsonObject file = (JsonObject)root["files"]![filePropertyName]!;
            string relativePath = file["path"]!.GetValue<string>();
            string filePath = Path.Combine(directory, relativePath);
            file["byteLength"] = new FileInfo(filePath).Length;
            file["sha256"] = ComputeSha256Hex(filePath);
        });
    }

    private static string ComputeSha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
