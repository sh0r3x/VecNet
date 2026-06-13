using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class ExactFlatIndexStorageIndependentTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void OpenReadOnly_ZeroAndNonzeroRoundTripsMatchInMemoryForEveryMetric(VectorMetric metric)
    {
        using TempIndexDirectory zeroRows = TempIndexDirectory.Create();
        var empty = new ExactFlatIndex(9, metric);
        empty.Save(zeroRows.Path);
        ExactFlatIndex loadedEmpty = ExactFlatIndex.OpenReadOnly(zeroRows.Path);
        Assert.Equal(empty.Dimension, loadedEmpty.Dimension);
        Assert.Equal(empty.Metric, loadedEmpty.Metric);
        AssertSameSearch(empty, loadedEmpty, Query(metric, 9, alternate: false), topK: 0);
        AssertSameSearch(empty, loadedEmpty, Query(metric, 9, alternate: false), topK: 3);

        using TempIndexDirectory populatedRows = TempIndexDirectory.Create();
        var populated = new ExactFlatIndex(9, metric);
        populated.Add(42, Vector(metric, 9, 0.35f));
        populated.Add(7, Vector(metric, 9, -0.20f));
        populated.Add(99, Vector(metric, 9, 0.80f));
        populated.Add(1, Vector(metric, 9, 0.80f));
        populated.Save(populatedRows.Path);

        ExactFlatIndex loadedPopulated = ExactFlatIndex.OpenReadOnly(populatedRows.Path);
        AssertSameSearch(populated, loadedPopulated, Query(metric, 9, alternate: false), topK: 0);
        AssertSameSearch(populated, loadedPopulated, Query(metric, 9, alternate: false), topK: 2);
        AssertSameSearch(populated, loadedPopulated, Query(metric, 9, alternate: true), topK: 8);
    }

    [Fact]
    public void OpenReadOnly_ValidationOrderMatchesInMemorySearch()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, VectorMetric.Cosine);
        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        AssertSameThrow<ArgumentException>(() => index.Search([1f, 2f], []), () => loaded.Search([1f, 2f], []));
        AssertSameThrow<ArgumentException>(() => index.Search([0f, 0f, 0f], []), () => loaded.Search([0f, 0f, 0f], []));
        AssertSameThrow<ArgumentException>(() => index.Search([1f, float.PositiveInfinity, 0f], []), () => loaded.Search([1f, float.PositiveInfinity, 0f], []));
    }

    [Theory]
    [InlineData("schemaName", "VecNet.OtherManifest")]
    [InlineData("schemaVersion", "1.1")]
    [InlineData("schemaVersion", "0.9")]
    [InlineData("formatFamily", "other-flat")]
    public void OpenReadOnly_RejectsUnsupportedManifestIdentityVersionsAndFamily(string property, string value)
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        MutateManifest(temp.Path, root => root[property] = value);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsUnknownRequiredFeaturesAndNewerReaderRequirement()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            MutateManifest(temp.Path, root => ((JsonArray)root["compatibility"]!["requiredFeatures"]!).Add("vecnet.required.future"));
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            MutateManifest(temp.Path, root => root["compatibility"]!["minimumReaderMajorVersion"] = 2);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_IgnoresUnknownOptionalFeatures()
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        MutateManifest(temp.Path, root => ((JsonArray)root["compatibility"]!["optionalFeatures"]!).Add("vecnet.optional.future"));

        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Equal(3, loaded.Dimension);
    }

    [Theory]
    [InlineData("ids", "../exact-flat.ids.u64")]
    [InlineData("vectors", "subdir/exact-flat.vectors.f32")]
    public void OpenReadOnly_RejectsManifestPathTraversalAndNonPinnedRelativePaths(string fileProperty, string badPath)
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        MutateManifest(temp.Path, root => root["files"]![fileProperty]!["path"] = badPath);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsRootedManifestPaths()
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        string rooted = Path.GetFullPath(Path.Combine(temp.Path, ExactFlatIndexStorage.IdsFileName));
        MutateManifest(temp.Path, root => root["files"]!["ids"]!["path"] = rooted);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsTruncatedAndMalformedManifestFiles()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            File.WriteAllText(Path.Combine(temp.Path, ExactFlatIndexStorage.ManifestFileName), "{\"schemaName\":\"VecNet.ExactFlatIndexManifest\"");
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            File.WriteAllText(Path.Combine(temp.Path, ExactFlatIndexStorage.ManifestFileName), "[]");
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsChecksumMismatchForEachBinaryFile()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            PatchFile(temp.Path, ExactFlatIndexStorage.IdsFileName, bytes => bytes[^1] ^= 0x10, refreshManifest: false);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            PatchFile(temp.Path, ExactFlatIndexStorage.VectorsFileName, bytes => bytes[^1] ^= 0x20, refreshManifest: false);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsManifestByteLengthMismatchForEachBinaryFile()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            MutateManifestFileMetadata(temp.Path, "ids", file => file["byteLength"] = file["byteLength"]!.GetValue<long>() + 1);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            MutateManifestFileMetadata(temp.Path, "vectors", file => file["byteLength"] = file["byteLength"]!.GetValue<long>() - 1);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsTruncatedBinaryHeadersAfterChecksumRefresh()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            TruncateFileAndRefresh(temp.Path, ExactFlatIndexStorage.IdsFileName, ExactFlatIndexStorage.IdsHeaderLength - 1);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            TruncateFileAndRefresh(temp.Path, ExactFlatIndexStorage.VectorsFileName, ExactFlatIndexStorage.VectorsHeaderLength - 1);
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsTruncatedBinaryPayloadsAfterChecksumRefresh()
    {
        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            string idsPath = Path.Combine(temp.Path, ExactFlatIndexStorage.IdsFileName);
            TruncateFileAndRefresh(temp.Path, ExactFlatIndexStorage.IdsFileName, new FileInfo(idsPath).Length - sizeof(ulong));
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }

        using (TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean))
        {
            string vectorsPath = Path.Combine(temp.Path, ExactFlatIndexStorage.VectorsFileName);
            TruncateFileAndRefresh(temp.Path, ExactFlatIndexStorage.VectorsFileName, new FileInfo(vectorsPath).Length - sizeof(float));
            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void OpenReadOnly_RejectsBadIdBinaryHeaderFields()
    {
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => bytes[0] = (byte)'X');
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 0));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(10), 1));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12), 24));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(16), 99));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(24), 1));
    }

    [Fact]
    public void OpenReadOnly_RejectsBadVectorBinaryHeaderFields()
    {
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => bytes[0] = (byte)'X');
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 0));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(10), 1));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12), 40));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(16), 99));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), 99));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28), 99));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), 99));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), 1));
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes => BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(40), 1));
    }

    [Fact]
    public void OpenReadOnly_RejectsManifestBinaryVersionMismatchBeforeReadingHeader()
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        MutateManifestFileMetadata(temp.Path, "vectors", file => file["binaryVersion"] = "1.1");

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsDuplicateIdsAndNonFiniteStoredComponents()
    {
        AssertCorruptBinaryRejected(ExactFlatIndexStorage.IdsFileName, bytes =>
        {
            ulong first = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength));
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength + sizeof(ulong)), first);
        });

        AssertCorruptBinaryRejected(ExactFlatIndexStorage.VectorsFileName, bytes =>
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength + sizeof(float)),
                BitConverter.SingleToInt32Bits(float.PositiveInfinity)));
    }

    [Theory]
    [InlineData(1.00004f)]
    [InlineData(0.99996f)]
    public void OpenReadOnly_AcceptsCosineStoredRowsInsideSquaredLengthTolerance(float component)
    {
        using TempIndexDirectory temp = SavedCosineUnitRow();
        PatchFirstVectorComponent(temp.Path, component);

        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        SearchResult[] results = new SearchResult[1];
        Assert.Equal(1, loaded.Search([1f, 0f], results));
        Assert.Equal(5UL, results[0].Id);
    }

    [Theory]
    [InlineData(1.00020f)]
    [InlineData(0.99980f)]
    public void OpenReadOnly_RejectsCosineStoredRowsOutsideSquaredLengthTolerance(float component)
    {
        using TempIndexDirectory temp = SavedCosineUnitRow();
        PatchFirstVectorComponent(temp.Path, component);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_RejectsZeroLengthCosineStoredRows()
    {
        using TempIndexDirectory temp = SavedCosineUnitRow();
        PatchFile(temp.Path, ExactFlatIndexStorage.VectorsFileName, bytes =>
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength), BitConverter.SingleToInt32Bits(0f));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength + sizeof(float)), BitConverter.SingleToInt32Bits(0f));
        }, refreshManifest: true);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    [Fact]
    public void OpenReadOnly_ReadOnlyAddAlwaysRejectsBeforeDuplicateOrVectorValidation()
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Throws<InvalidOperationException>(() => loaded.Add(11, [1f, 2f, 3f]));
        Assert.Throws<InvalidOperationException>(() => loaded.Add(123, [1f, 2f]));
        Assert.Throws<InvalidOperationException>(() => loaded.Add(123, [1f, float.NaN, 3f]));
    }

    [Fact]
    public void OpenReadOnly_ParallelReadOnlyStressUsesCallerOwnedBuffers()
    {
        foreach (VectorMetric metric in new[] { VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine })
        {
            using TempIndexDirectory temp = TempIndexDirectory.Create();
            var index = new ExactFlatIndex(11, metric);
            for (int row = 0; row < 32; row++)
            {
                index.Add((ulong)(1000 - row), Vector(metric, 11, row / 17f - 0.75f));
            }

            index.Save(temp.Path);
            ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
            float[][] queries =
            [
                Query(metric, 11, alternate: false),
                Query(metric, 11, alternate: true),
                Vector(metric, 11, 0.125f)
            ];
            int[] topKs = [0, 1, 5, 17, 40];
            SearchResult[][] expected = queries
                .SelectMany(query => topKs.Select(topK => Search(index, query, topK)))
                .ToArray();

            Parallel.For(0, 750, iteration =>
            {
                int queryIndex = iteration % queries.Length;
                int topKIndex = (iteration / queries.Length) % topKs.Length;
                var callerOwnedResults = new SearchResult[topKs[topKIndex]];
                int written = loaded.Search(queries[queryIndex], callerOwnedResults);
                Assert.Equal(expected[queryIndex * topKs.Length + topKIndex], callerOwnedResults[..written]);
            });
        }
    }

    private static TempIndexDirectory SavedIndex(VectorMetric metric)
    {
        TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(3, metric);
        index.Add(11, metric == VectorMetric.Cosine ? [1f, 0f, 0f] : [1f, 2f, 3f]);
        index.Add(22, metric == VectorMetric.Cosine ? [0f, 1f, 0f] : [-1f, 0.5f, 4f]);
        index.Save(temp.Path);
        return temp;
    }

    private static TempIndexDirectory SavedCosineUnitRow()
    {
        TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(2, VectorMetric.Cosine);
        index.Add(5, [1f, 0f]);
        index.Save(temp.Path);
        return temp;
    }

    private static void AssertCorruptBinaryRejected(string fileName, Action<byte[]> patch)
    {
        using TempIndexDirectory temp = SavedIndex(VectorMetric.SquaredEuclidean);
        PatchFile(temp.Path, fileName, patch, refreshManifest: true);
        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    private static void PatchFirstVectorComponent(string directory, float component) =>
        PatchFile(
            directory,
            ExactFlatIndexStorage.VectorsFileName,
            bytes => BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(ExactFlatIndexStorage.VectorsHeaderLength),
                BitConverter.SingleToInt32Bits(component)),
            refreshManifest: true);

    private static void PatchFile(string directory, string fileName, Action<byte[]> patch, bool refreshManifest)
    {
        string path = Path.Combine(directory, fileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);

        if (refreshManifest)
        {
            RefreshManifestBinaryMetadata(directory, fileName);
        }
    }

    private static void TruncateFileAndRefresh(string directory, string fileName, long length)
    {
        string path = Path.Combine(directory, fileName);
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.None))
        {
            stream.SetLength(length);
        }

        RefreshManifestBinaryMetadata(directory, fileName);
    }

    private static void RefreshManifestBinaryMetadata(string directory, string fileName)
    {
        string filePropertyName = fileName == ExactFlatIndexStorage.IdsFileName ? "ids" : "vectors";
        MutateManifestFileMetadata(directory, filePropertyName, file =>
        {
            string relativePath = file["path"]!.GetValue<string>();
            string binaryPath = Path.Combine(directory, relativePath);
            file["byteLength"] = new FileInfo(binaryPath).Length;
            file["sha256"] = Sha256Hex(binaryPath);
        });
    }

    private static void MutateManifestFileMetadata(string directory, string filePropertyName, Action<JsonObject> mutate) =>
        MutateManifest(directory, root => mutate((JsonObject)root["files"]![filePropertyName]!));

    private static void MutateManifest(string directory, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, ExactFlatIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static string Sha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static float[] Vector(VectorMetric metric, int dimension, float seed)
    {
        var vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = seed + (i % 5 - 2) * 0.125f;
        }

        if (metric == VectorMetric.Cosine && vector.All(static value => value == 0))
        {
            vector[0] = 1f;
        }

        return vector;
    }

    private static float[] Query(VectorMetric metric, int dimension, bool alternate)
    {
        float seed = alternate ? -0.35f : 0.45f;
        float[] query = Vector(metric, dimension, seed);
        if (metric == VectorMetric.Cosine && query.All(static value => value == 0))
        {
            query[0] = 1f;
        }

        return query;
    }

    private static void AssertSameSearch(ExactFlatIndex expectedIndex, ExactFlatIndex actualIndex, float[] query, int topK) =>
        Assert.Equal(Search(expectedIndex, query, topK), Search(actualIndex, query, topK));

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static void AssertSameThrow<TException>(Action expected, Action actual)
        where TException : Exception
    {
        TException expectedException = Assert.Throws<TException>(expected);
        TException actualException = Assert.Throws<TException>(actual);
        Assert.Equal(expectedException.GetType(), actualException.GetType());
        Assert.Equal(expectedException.Message, actualException.Message);
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
