using System.Buffers.Binary;
using System.Text.Json;

namespace VecNet.Tests;

public sealed class ExactFlatIndexBoundedPersistenceTestsIndependent
{
    private const string ManifestFileName = "exact-flat.manifest.json";
    private const string IdsFileName = "exact-flat.ids.u64";
    private const string VectorsFileName = "exact-flat.vectors.f32";
    private const int IdsHeaderLength = 32;
    private const int VectorsHeaderLength = 48;

    [Fact]
    public void Vec170_SaveAfterTombstoneChurnPersistsOnlyLiveRowsAndMatchesSourceLiveView()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(4, VectorMetric.SquaredEuclidean, initialCapacity: 32);
        for (int row = 0; row < 14; row++)
        {
            index.Add((ulong)(100 + row), Vector4(row));
        }

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(101).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(106).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(112).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(250, [0.25f, -0.5f, 0.75f, 1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(260, [-1f, 0.25f, 0.5f, -0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(250).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(270, [2f, -1f, 0.5f, 0.25f]).Status);

        ulong[] deletedIds = [101, 106, 112, 250];
        ulong[] expectedLiveIds =
        [
            100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113, 260, 270
        ];
        SearchResult[][] expectedSearches =
        [
            Search(index, [0f, 0f, 0f, 0f], topK: 16),
            Search(index, [1f, -1f, 0.5f, 0.25f], topK: 7)
        ];

        index.Save(saved.Path);

        Assert.Equal(expectedLiveIds.Length, ReadManifestVectorCount(saved.Path));
        Assert.Equal(IdsHeaderLength + expectedLiveIds.Length * sizeof(ulong), FileLength(saved.Path, IdsFileName));
        Assert.Equal(VectorsHeaderLength + expectedLiveIds.Length * index.Dimension * sizeof(float), FileLength(saved.Path, VectorsFileName));
        Assert.Equal(expectedLiveIds.Order().ToArray(), ReadIdsPayload(saved.Path).Order().ToArray());
        Assert.DoesNotContain(ReadIdsPayload(saved.Path), id => deletedIds.Contains(id));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(saved.Path);
        AssertCompactOpened(opened, expectedLiveIds.Length);
        Assert.Equal(expectedSearches[0], Search(opened, [0f, 0f, 0f, 0f], topK: 16));
        Assert.Equal(expectedSearches[1], Search(opened, [1f, -1f, 0.5f, 0.25f], topK: 7));
        Assert.Equal(
            expectedSearches[0],
            SearchRaw(opened, [.. expectedLiveIds, .. deletedIds, 999_999], [0f, 0f, 0f, 0f], topK: 16));
    }

    [Fact]
    public void Vec170_SaveWithPlannedExtraCapacityDoesNotPersistSlackAsRows()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(3, VectorMetric.InnerProduct, initialCapacity: 64);
        index.Add(30, [3f, 0f, 1f]);
        index.Add(10, [1f, 1f, 0f]);
        index.Add(20, [2f, -1f, 1f]);
        index.EnsureCapacity(128);
        SearchResult[] expected = Search(index, [1f, 0.5f, 1f], topK: 8);

        index.Save(saved.Path);

        Assert.Equal(128, index.Capacity);
        Assert.Equal(3, ReadManifestVectorCount(saved.Path));
        Assert.Equal(IdsHeaderLength + 3 * sizeof(ulong), FileLength(saved.Path, IdsFileName));
        Assert.Equal(VectorsHeaderLength + 3 * index.Dimension * sizeof(float), FileLength(saved.Path, VectorsFileName));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(saved.Path);
        Assert.Equal(3, opened.Capacity);
        Assert.Equal(3, opened.PhysicalVectorCount);
        Assert.Equal(expected, Search(opened, [1f, 0.5f, 1f], topK: 8));
        Assert.Equal([10UL, 20UL, 30UL], ReadIdsPayload(saved.Path).Order().ToArray());
        Assert.Equal(expected, SearchRaw(opened, [10, 20, 30], [1f, 0.5f, 1f], topK: 8));
        Assert.Throws<ArgumentException>(() =>
            opened.Search(
                [1f, 0.5f, 1f],
                [10UL, 20UL, 30UL],
                new SearchResult[8],
                new ExactFlatSearchFilterWorkspace(opened.PhysicalVectorCount - 1)));
    }

    [Fact]
    public void Vec170_CheckpointAfterMixedChurnCompactsRowsAndPreservesDeletedReservation()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, initialCapacity: 20);
        for (int row = 0; row < 9; row++)
        {
            index.Add((ulong)(10 + row), [(float)row, (row % 3) - 1f]);
        }

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(11).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(14).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(100, [0.25f, 0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(110, [5.5f, -0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(100).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(120, [-0.25f, 1f]).Status);
        ExactFlatCandidateSet stale = index.CreateCandidateSet([10, 11, 14, 100, 110, 120]);
        SearchResult[] expected = Search(index, [0f, 0f], topK: 12);
        long generationBefore = index.Generation;

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(generationBefore + 1, result.Generation);
        Assert.Equal(9, result.LiveVectorCount);
        Assert.Equal(9, result.PhysicalVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(3, result.DeletedReservedIdCount);
        Assert.Equal(2, result.FoldedDeltaVectorCount);
        Assert.Equal(3, result.FoldedTombstoneCount);
        Assert.Equal(9, index.Capacity);
        Assert.Equal(9, index.PhysicalVectorCount);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(3, index.DeletedReservedIdCount);
        Assert.Throws<InvalidOperationException>(() => index.Search([0f, 0f], stale, new SearchResult[4]));
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(11, [11f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(14, [14f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(100, [100f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(100).Status);
        Assert.Equal(expected, Search(index, [0f, 0f], topK: 12));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);
        AssertCompactOpened(opened, expected.Length);
        Assert.Equal(expected, Search(opened, [0f, 0f], topK: 12));
        Assert.Equal(expected.Select(static result => result.Id).Order().ToArray(), ReadIdsPayload(checkpoint.Path).Order().ToArray());
    }

    [Fact]
    public void Vec170_CheckpointOutputSupportsFreshFiltersCandidateSetsAndReadOnlySearchEquivalence()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(3, VectorMetric.Cosine, initialCapacity: 12);
        index.Add(10, [1f, 0f, 0f]);
        index.Add(20, [0f, 1f, 0f]);
        index.Add(30, [0f, 0f, 1f]);
        index.Add(40, [1f, 1f, 0f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(50, [1f, 0f, 1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(60, [0f, 1f, 1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(50).Status);

        float[] query = [1f, 0.25f, 0.5f];
        ulong[] scope = [10, 20, 30, 40, 50, 60, 999];
        SearchResult[] expectedAll = Search(index, query, topK: 8);
        SearchResult[] expectedFiltered = SearchRaw(index, scope, query, topK: 8);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(expectedAll, Search(index, query, topK: 8));
        Assert.Equal(expectedAll, Search(opened, query, topK: 8));
        Assert.Equal(expectedFiltered, SearchRaw(index, scope, query, topK: 8));
        Assert.Equal(expectedFiltered, SearchRaw(opened, scope, query, topK: 8));

        ExactFlatCandidateSet sourceCandidates = index.CreateCandidateSet(scope);
        ExactFlatCandidateSet openedCandidates = opened.CreateCandidateSet(scope);
        Assert.Equal(expectedFiltered, SearchCandidateSet(index, sourceCandidates, query, topK: 8));
        Assert.Equal(expectedFiltered, SearchCandidateSet(opened, openedCandidates, query, topK: 8));
        Assert.Throws<InvalidOperationException>(() => opened.Add(70, [1f, 1f, 1f]));
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryDelete(10).Status);
    }

    [Fact]
    public void Vec170_SaveAndCheckpointKeepDurableManifestAndBinaryFormatOpenable()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, initialCapacity: 10);
        index.Add(5, [1f, 2f]);
        index.Add(3, [-1f, 0.5f]);
        index.Add(7, [0f, -2f]);
        index.Save(saved.Path);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(5).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(11, [0.25f, 0.25f]).Status);
        _ = index.Checkpoint(checkpoint.Path);

        AssertDurableFormat(saved.Path, expectedDimension: 2, expectedMetric: "squared-euclidean", expectedCount: 3);
        AssertDurableFormat(checkpoint.Path, expectedDimension: 2, expectedMetric: "squared-euclidean", expectedCount: 3);
        Assert.Equal(Search(ExactFlatIndex.OpenReadOnly(saved.Path), [0f, 0f], topK: 8), Search(BuildSavedReference(), [0f, 0f], topK: 8));
        Assert.Equal(Search(index, [0f, 0f], topK: 8), Search(ExactFlatIndex.OpenReadOnly(checkpoint.Path), [0f, 0f], topK: 8));
    }

    [Fact]
    public void Vec170_SaveAndCheckpointRejectedTargetsLeaveCallerFilesUntouched()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [1f, 0f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(20, [2f, 0f]).Status);

        using TempIndexDirectory saveTarget = TempIndexDirectory.Create();
        string saveMarker = Path.Combine(saveTarget.Path, "marker.txt");
        File.WriteAllText(saveMarker, "save marker");
        Assert.Throws<IOException>(() => index.Save(saveTarget.Path));
        Assert.Equal(["marker.txt"], Directory.EnumerateFileSystemEntries(saveTarget.Path).Select(static path => Path.GetFileName(path)!).Order().ToArray());
        Assert.Equal("save marker", File.ReadAllText(saveMarker));

        using TempIndexDirectory checkpointTarget = TempIndexDirectory.Create();
        string checkpointMarker = Path.Combine(checkpointTarget.Path, "marker.txt");
        File.WriteAllText(checkpointMarker, "checkpoint marker");
        Assert.Throws<IOException>(() => index.Checkpoint(checkpointTarget.Path));
        Assert.Equal(["marker.txt"], Directory.EnumerateFileSystemEntries(checkpointTarget.Path).Select(static path => Path.GetFileName(path)!).Order().ToArray());
        Assert.Equal("checkpoint marker", File.ReadAllText(checkpointMarker));

        using TempIndexDirectory noChangesTarget = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory published = TempIndexDirectory.CreateMissing();
        _ = index.Checkpoint(published.Path);
        ExactFlatCheckpointResult noChanges = index.Checkpoint(noChangesTarget.Path);
        Assert.Equal(ExactFlatCheckpointStatus.NoChanges, noChanges.Status);
        Assert.False(Directory.Exists(noChangesTarget.Path));
    }

    private static ExactFlatIndex BuildSavedReference()
    {
        var reference = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        reference.Add(5, [1f, 2f]);
        reference.Add(3, [-1f, 0.5f]);
        reference.Add(7, [0f, -2f]);
        return reference;
    }

    private static float[] Vector4(int row) =>
    [
        ((row * 3) % 11) - 5f,
        ((row * 5) % 13) - 6f,
        ((row * 7) % 17) - 8f,
        ((row * 11) % 19) - 9f
    ];

    private static void AssertCompactOpened(ExactFlatIndex opened, int expectedLiveCount)
    {
        Assert.Equal(expectedLiveCount, opened.PhysicalVectorCount);
        Assert.Equal(expectedLiveCount, opened.VectorCount);
        Assert.Equal(expectedLiveCount, opened.LiveVectorCount);
        Assert.Equal(expectedLiveCount, opened.BaseVectorCount);
        Assert.Equal(0, opened.DeltaVectorCount);
        Assert.Equal(0, opened.TombstoneCount);
        Assert.Equal(0, opened.DeletedReservedIdCount);
        Assert.Equal(expectedLiveCount, opened.Capacity);
    }

    private static void AssertDurableFormat(string directory, int expectedDimension, string expectedMetric, int expectedCount)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, ManifestFileName)));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.ExactFlatIndexManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("1.0", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("exact-flat", root.GetProperty("formatFamily").GetString());
        Assert.Equal("VEC-031", root.GetProperty("createdByTask").GetString());
        Assert.Equal(expectedDimension, root.GetProperty("index").GetProperty("dimension").GetInt32());
        Assert.Equal(expectedMetric, root.GetProperty("index").GetProperty("metric").GetString());
        Assert.Equal(expectedCount, root.GetProperty("index").GetProperty("vectorCount").GetInt32());
        Assert.Equal(IdsFileName, root.GetProperty("files").GetProperty("ids").GetProperty("path").GetString());
        Assert.Equal(VectorsFileName, root.GetProperty("files").GetProperty("vectors").GetProperty("path").GetString());
        Assert.Equal(IdsHeaderLength + expectedCount * sizeof(ulong), FileLength(directory, IdsFileName));
        Assert.Equal(VectorsHeaderLength + expectedCount * expectedDimension * sizeof(float), FileLength(directory, VectorsFileName));

        byte[] idsBytes = File.ReadAllBytes(Path.Combine(directory, IdsFileName));
        byte[] vectorBytes = File.ReadAllBytes(Path.Combine(directory, VectorsFileName));
        Assert.Equal("VNETID01"u8.ToArray(), idsBytes[..8]);
        Assert.Equal((ulong)expectedCount, BinaryPrimitives.ReadUInt64LittleEndian(idsBytes.AsSpan(16)));
        Assert.Equal("VNETVF01"u8.ToArray(), vectorBytes[..8]);
        Assert.Equal((ulong)expectedCount, BinaryPrimitives.ReadUInt64LittleEndian(vectorBytes.AsSpan(16)));
        Assert.Equal((uint)expectedDimension, BinaryPrimitives.ReadUInt32LittleEndian(vectorBytes.AsSpan(24)));
        _ = ExactFlatIndex.OpenReadOnly(directory);
    }

    private static int ReadManifestVectorCount(string directory)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, ManifestFileName)));
        return document.RootElement.GetProperty("index").GetProperty("vectorCount").GetInt32();
    }

    private static ulong[] ReadIdsPayload(string directory)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, IdsFileName));
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(16));
        var ids = new ulong[rowCount];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(IdsHeaderLength + i * sizeof(ulong)));
        }

        return ids;
    }

    private static long FileLength(string directory, string fileName) =>
        new FileInfo(Path.Combine(directory, fileName)).Length;

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] SearchRaw(ExactFlatIndex index, ulong[] allowedIds, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(
            query,
            allowedIds,
            results,
            new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));
        return results[..written];
    }

    private static SearchResult[] SearchCandidateSet(
        ExactFlatIndex index,
        ExactFlatCandidateSet candidates,
        float[] query,
        int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, candidates, results);
        return results[..written];
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
