using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;

namespace VecNet.Tests;

public sealed class ExactFlatIndexBoundedPersistenceTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Save_NoTombstonePlannedCapacityWritesCompactSearchEquivalentOutput(VectorMetric metric)
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(3, metric, initialCapacity: 16);
        index.Add(30, CreateVector(metric, 30));
        index.Add(10, CreateVector(metric, 10));
        index.Add(20, CreateVector(metric, 20));
        index.Add(40, CreateVector(metric, 40));
        SearchResult[] expected = SearchAll(index, CreateQuery(metric), topK: 10);

        Assert.Equal(16, index.Capacity);

        index.Save(saved.Path);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(saved.Path);
        Assert.Equal(4, ReadManifestVectorCount(saved.Path));
        Assert.Equal(ExactFlatIndexStorage.IdsHeaderLength + 4 * sizeof(ulong), FileLength(saved.Path, ExactFlatIndexStorage.IdsFileName));
        Assert.Equal(ExactFlatIndexStorage.VectorsHeaderLength + 4 * index.Dimension * sizeof(float), FileLength(saved.Path, ExactFlatIndexStorage.VectorsFileName));
        Assert.Equal(16, index.Capacity);
        AssertCompactOpened(opened, metric, expectedLiveCount: 4);
        Assert.Equal(expected, SearchAll(opened, CreateQuery(metric), topK: 10));
    }

    [Fact]
    public void Save_AfterTombstonesWritesOnlyLiveRowsAndOpensCompactSearchEquivalentOutput()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, initialCapacity: 8);
        index.Add(10, [1f, 0f]);
        index.Add(20, [2f, 0f]);
        index.Add(30, [-1f, 0f]);
        index.Add(40, [3f, 0f]);
        index.Add(50, [0f, 1f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(40).Status);
        SearchResult[] expected = SearchAll(index, [0f, 0f], topK: 10);

        index.Save(saved.Path);

        Assert.Equal([10UL, 30UL, 50UL], ReadIdsPayload(saved.Path));
        Assert.Equal(3, ReadManifestVectorCount(saved.Path));
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(saved.Path);
        AssertCompactOpened(opened, VectorMetric.SquaredEuclidean, expectedLiveCount: 3);
        Assert.Equal(expected, SearchAll(opened, [0f, 0f], topK: 10));
        Assert.Equal(expected, SearchRaw(opened, [10, 20, 30, 40, 50], [0f, 0f], topK: 10));
    }

    [Fact]
    public void Checkpoint_AfterAdditionsPublishesCompactMemoryAndCompactOpenedOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.InnerProduct, initialCapacity: 12);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(10, [1f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [0f, 1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [1f, 1f]).Status);
        SearchResult[] expected = SearchAll(index, [1f, 1f], topK: 10);
        long beforeGeneration = index.Generation;

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(3, result.FoldedDeltaVectorCount);
        Assert.Equal(0, result.FoldedTombstoneCount);
        Assert.Equal(3, index.PhysicalVectorCount);
        Assert.Equal(3, index.LiveVectorCount);
        Assert.Equal(3, index.BaseVectorCount);
        Assert.Equal(0, index.DeltaVectorCount);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(3, index.Capacity);
        Assert.Equal(expected, SearchAll(index, [1f, 1f], topK: 10));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        AssertCompactOpened(opened, VectorMetric.InnerProduct, expectedLiveCount: 3);
        Assert.Equal(expected, SearchAll(opened, [1f, 1f], topK: 10));
    }

    [Fact]
    public void Checkpoint_AfterTombstonesPublishesCompactMemoryAndCompactOpenedOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, initialCapacity: 10);
        for (int i = 0; i < 6; i++)
        {
            index.Add((ulong)(10 + i), [i - 2f]);
        }

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(11).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(14).Status);
        SearchResult[] expected = SearchAll(index, [0f], topK: 10);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(4, result.PhysicalVectorCount);
        Assert.Equal(4, result.LiveVectorCount);
        Assert.Equal(4, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(0, result.FoldedDeltaVectorCount);
        Assert.Equal(2, result.FoldedTombstoneCount);
        Assert.Equal(4, index.Capacity);
        Assert.Equal(expected, SearchAll(index, [0f], topK: 10));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        AssertCompactOpened(opened, VectorMetric.SquaredEuclidean, expectedLiveCount: 4);
        Assert.Equal(expected, SearchAll(opened, [0f], topK: 10));
    }

    [Fact]
    public void Checkpoint_AfterMixedChurnPreservesReservationAndStaleCandidateSetBehavior()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean, initialCapacity: 12);
        for (int i = 0; i < 5; i++)
        {
            index.Add((ulong)(10 + i), [i + 1f]);
        }

        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(60, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(70, [7f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(11).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(60).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(80, [0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(13).Status);
        ExactFlatCandidateSet staleCandidates = index.CreateCandidateSet([10, 11, 12, 13, 14, 60, 70, 80]);
        SearchResult[] expected = SearchAll(index, [0f], topK: 10);
        SearchResult[] sentinel = [new(123, 456f), new(789, 999f)];

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(5, result.PhysicalVectorCount);
        Assert.Equal(5, result.LiveVectorCount);
        Assert.Equal(5, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(3, result.DeletedReservedIdCount);
        Assert.Equal(2, result.FoldedDeltaVectorCount);
        Assert.Equal(3, result.FoldedTombstoneCount);
        Assert.Throws<InvalidOperationException>(() => index.Search([0f], staleCandidates, sentinel));
        Assert.Equal([new SearchResult(123, 456f), new SearchResult(789, 999f)], sentinel);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(11, [11f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(13, [13f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(60, [60f]).Status);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(60).Status);
        Assert.Equal(expected, SearchAll(index, [0f], topK: 10));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        AssertCompactOpened(opened, VectorMetric.SquaredEuclidean, expectedLiveCount: 5);
        Assert.Equal(expected, SearchAll(opened, [0f], topK: 10));
    }

    [Fact]
    public void FileMetadataHashesRemainCompatibleWithExistingReadersAndBoundedValidation()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 2f]);
        index.Add(20, [3f, 4f]);

        index.Save(saved.Path);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(saved.Path, ExactFlatIndexStorage.ManifestFileName)));
        AssertManifestHashMatchesFile(document, saved.Path, "ids");
        AssertManifestHashMatchesFile(document, saved.Path, "vectors");
        ExactFlatIndexStorage.ValidateSavedCompactSnapshot(
            saved.Path,
            dimension: 2,
            VectorMetric.SquaredEuclidean,
            [10UL, 20UL],
            [1f, 2f, 3f, 4f]);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(saved.Path);
        Assert.Equal(SearchAll(index, [0f, 0f], topK: 2), SearchAll(opened, [0f, 0f], topK: 2));
    }

    [Fact]
    public void StreamingSaveFailureDeletesTempFilesAndCreatedDirectoryWhenPractical()
    {
        using TempIndexDirectory existingEmpty = TempIndexDirectory.Create();
        Assert.Throws<InvalidOperationException>(() =>
            ExactFlatIndexStorage.Save(
                existingEmpty.Path,
                dimension: 1,
                VectorMetric.SquaredEuclidean,
                liveRowCount: 1,
                sourceIds: [10UL, 20UL],
                sourceVectors: [1f, 2f],
                sourceRowDeleted: [1, 1]));
        Assert.Empty(Directory.EnumerateFileSystemEntries(existingEmpty.Path));

        using TempIndexDirectory missing = TempIndexDirectory.CreateMissing();
        Assert.Throws<InvalidOperationException>(() =>
            ExactFlatIndexStorage.Save(
                missing.Path,
                dimension: 1,
                VectorMetric.SquaredEuclidean,
                liveRowCount: 1,
                sourceIds: [10UL, 20UL],
                sourceVectors: [1f, 2f],
                sourceRowDeleted: [1, 1]));
        Assert.False(Directory.Exists(missing.Path));
    }

    private static void AssertCompactOpened(ExactFlatIndex opened, VectorMetric metric, int expectedLiveCount)
    {
        Assert.Equal(metric, opened.Metric);
        Assert.Equal(expectedLiveCount, opened.PhysicalVectorCount);
        Assert.Equal(expectedLiveCount, opened.LiveVectorCount);
        Assert.Equal(expectedLiveCount, opened.BaseVectorCount);
        Assert.Equal(0, opened.DeltaVectorCount);
        Assert.Equal(0, opened.TombstoneCount);
        Assert.Equal(0, opened.DeletedReservedIdCount);
        Assert.Equal(expectedLiveCount, opened.Capacity);
    }

    private static SearchResult[] SearchAll(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] SearchRaw(ExactFlatIndex index, ulong[] allowedIds, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowedIds, results, new ExactFlatSearchFilterWorkspace(index.VectorCount));
        return results[..written];
    }

    private static float[] CreateVector(VectorMetric metric, ulong id) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => [(float)(id % 5), (float)(id % 7 - 3), (float)(id % 11 - 5)],
            VectorMetric.InnerProduct => [(float)(id % 3 + 1), (float)(id % 5 - 2), (float)(id % 7 + 0.5)],
            VectorMetric.Cosine => [(float)(id % 3 + 1), (float)(id % 5 + 1), (float)(id % 7 + 1)],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float[] CreateQuery(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? [1f, 0.25f, 0.5f] : [0.5f, -1f, 2f];

    private static int ReadManifestVectorCount(string directory)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, ExactFlatIndexStorage.ManifestFileName)));
        return document.RootElement.GetProperty("index").GetProperty("vectorCount").GetInt32();
    }

    private static long FileLength(string directory, string fileName) =>
        new FileInfo(Path.Combine(directory, fileName)).Length;

    private static ulong[] ReadIdsPayload(string directory)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, ExactFlatIndexStorage.IdsFileName));
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(16));
        var ids = new ulong[rowCount];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = BinaryPrimitives.ReadUInt64LittleEndian(
                bytes.AsSpan(ExactFlatIndexStorage.IdsHeaderLength + i * sizeof(ulong)));
        }

        return ids;
    }

    private static void AssertManifestHashMatchesFile(JsonDocument document, string directory, string propertyName)
    {
        JsonElement file = document.RootElement.GetProperty("files").GetProperty(propertyName);
        string relativePath = file.GetProperty("path").GetString()!;
        string expectedHash = file.GetProperty("sha256").GetString()!;
        string path = Path.Combine(directory, relativePath);
        using FileStream stream = File.OpenRead(path);
        string actualHash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        Assert.Equal(new FileInfo(path).Length, file.GetProperty("byteLength").GetInt64());
        Assert.Equal(expectedHash, actualHash);
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory Create()
        {
            string path = CreatePath();
            Directory.CreateDirectory(path);
            return new TempIndexDirectory(path);
        }

        public static TempIndexDirectory CreateMissing() => new(CreatePath());

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
    }
}
