namespace VecNet.Tests;

public sealed class ExactFlatIndexRowLocalTombstoneTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    public void UnfilteredSearch_SuppressesTombstonesAndPreservesOrderingForRequiredTopK(int topK)
    {
        Row[] rows = CreateRows(180);
        var index = CreateIndex(rows);
        HashSet<ulong> deleted = rows
            .Where(static (_, row) => row % 7 == 0 || row % 19 == 0)
            .Select(static row => row.Id)
            .ToHashSet();

        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        var results = new SearchResult[topK];
        int written = index.Search([0f], results);

        SearchResult[] expected = Expected(rows, deleted, topK);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void RawAllowlistSearch_SuppressesTombstonesAndPreservesOrdering()
    {
        Row[] rows = CreateRows(96);
        var index = CreateIndex(rows);
        HashSet<ulong> deleted = [rows[0].Id, rows[5].Id, rows[17].Id, rows[42].Id, rows[71].Id];
        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        ulong[] allowedIds = rows
            .Where(static (_, row) => row % 2 == 0 || row % 5 == 0)
            .Select(static row => row.Id)
            .Reverse()
            .Concat(deleted)
            .Concat([ulong.MaxValue, rows[10].Id, rows[10].Id])
            .ToArray();
        var results = new SearchResult[30];
        int written = index.Search(
            [0f],
            allowedIds,
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        SearchResult[] expected = Expected(rows, deleted, allowedIds, topK: 30);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void ReusableCandidateSetSearch_SuppressesTombstonesAndPreservesOrdering()
    {
        Row[] rows = CreateRows(128);
        var index = CreateIndex(rows);
        HashSet<ulong> deleted = [rows[1].Id, rows[11].Id, rows[22].Id, rows[63].Id];
        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        ulong[] allowedIds = rows
            .Where(static (_, row) => row % 3 == 0 || row % 11 == 0)
            .Select(static row => row.Id)
            .Concat(deleted)
            .Concat([rows[33].Id, rows[33].Id, 987_654_321UL])
            .ToArray();
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowedIds);
        var results = new SearchResult[40];
        int written = index.Search([0f], candidates, results);

        SearchResult[] expected = Expected(rows, deleted, allowedIds, topK: 40);
        Assert.Equal(CountUniqueLiveAllowed(rows, deleted, allowedIds), candidates.Count);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void CountsAndDeleteReservationBehavior_RemainExternalIdBasedAcrossCheckpoint()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        index.Add(30, [30f]);
        index.Add(40, [40f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(50, [5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(60, [6f]).Status);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(50).Status);

        Assert.Equal(6, index.VectorCount);
        Assert.Equal(6, index.PhysicalVectorCount);
        Assert.Equal(4, index.LiveVectorCount);
        Assert.Equal(3, index.BaseVectorCount);
        Assert.Equal(1, index.DeltaVectorCount);
        Assert.Equal(2, index.TombstoneCount);
        Assert.Equal(2, index.DeletedReservedIdCount);

        long generation = index.Generation;
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.UnknownId, index.TryDelete(999).Status);
        Assert.Equal(generation, index.Generation);
        Assert.Equal(2, index.TombstoneCount);
        Assert.Equal(2, index.DeletedReservedIdCount);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(4, result.PhysicalVectorCount);
        Assert.Equal(4, result.LiveVectorCount);
        Assert.Equal(4, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(2, result.DeletedReservedIdCount);
        Assert.Equal(1, result.FoldedDeltaVectorCount);
        Assert.Equal(2, result.FoldedTombstoneCount);
        Assert.Equal(4, index.PhysicalVectorCount);
        Assert.Equal(4, index.LiveVectorCount);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(2, index.DeletedReservedIdCount);

        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(20, [2f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, index.TryAdd(50, [5f]).Status);
        Assert.Equal(VectorMutationStatus.AlreadyDeleted, index.TryDelete(50).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(70, [7f]).Status);
        Assert.Equal(5, index.LiveVectorCount);
        Assert.Equal(1, index.DeltaVectorCount);
    }

    [Fact]
    public void SaveOpenAndCheckpoint_ProduceCompactAllLiveSearchParity()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        Row[] rows = CreateRows(72);
        var index = CreateIndex(rows);
        HashSet<ulong> deleted = [rows[4].Id, rows[9].Id, rows[37].Id, rows[55].Id];
        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        SearchResult[] expected = SearchAll(index, topK: 30);
        int liveCount = rows.Length - deleted.Count;
        index.Save(saved.Path);
        ExactFlatIndex openedSaved = ExactFlatIndex.OpenReadOnly(saved.Path);

        Assert.Equal(liveCount, openedSaved.PhysicalVectorCount);
        Assert.Equal(liveCount, openedSaved.LiveVectorCount);
        Assert.Equal(liveCount, openedSaved.BaseVectorCount);
        Assert.Equal(0, openedSaved.DeltaVectorCount);
        Assert.Equal(0, openedSaved.TombstoneCount);
        Assert.Equal(0, openedSaved.DeletedReservedIdCount);
        Assert.Equal(expected, SearchAll(openedSaved, topK: 30));
        Assert.Equal(expected, SearchRaw(openedSaved, rows.Select(static row => row.Id).ToArray(), topK: 30));
        Assert.Equal(expected, SearchCandidates(openedSaved, rows.Select(static row => row.Id).ToArray(), topK: 30));

        ExactFlatCheckpointResult checkpointResult = index.Checkpoint(checkpoint.Path);
        ExactFlatIndex openedCheckpoint = ExactFlatIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(liveCount, index.PhysicalVectorCount);
        Assert.Equal(liveCount, index.LiveVectorCount);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(expected, SearchAll(index, topK: 30));
        Assert.Equal(liveCount, openedCheckpoint.PhysicalVectorCount);
        Assert.Equal(liveCount, openedCheckpoint.LiveVectorCount);
        Assert.Equal(0, openedCheckpoint.TombstoneCount);
        Assert.Equal(expected, SearchAll(openedCheckpoint, topK: 30));
    }

    [Fact]
    public void GrowthAfterDeletes_InitializesNewRowsLiveAndPreservesExistingTombstones()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        for (int i = 0; i < 4; i++)
        {
            index.Add((ulong)(100 + i), [i + 10f]);
        }

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(100).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(102).Status);

        for (int i = 4; i < 12; i++)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryAdd((ulong)(100 + i), [i - 4f]).Status);
        }

        var results = new SearchResult[12];
        int written = index.Search([0f], results);

        Assert.Equal(10, written);
        Assert.Equal(12, index.PhysicalVectorCount);
        Assert.Equal(10, index.LiveVectorCount);
        Assert.Equal(2, index.TombstoneCount);
        Assert.DoesNotContain(results[..written], static result => result.Id is 100 or 102);
        Assert.Contains(results[..written], static result => result.Id is 104);
        Assert.Contains(results[..written], static result => result.Id is 111);
    }

    [Fact]
    public void TombstonedPublicExactSearch_DoesNotAllocateWithReusedResultBuffer()
    {
        Row[] rows = CreateRows(512);
        var index = CreateIndex(rows);
        for (int i = 0; i < rows.Length; i += 5)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(rows[i].Id).Status);
        }

        var results = new SearchResult[100];
        Assert.Equal(100, index.Search([0f], results));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 256; i++)
        {
            int written = index.Search([0f], results);
            if (written != 100)
            {
                throw new InvalidOperationException("Unexpected tombstoned exact search result count.");
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    private static ExactFlatIndex CreateIndex(IEnumerable<Row> rows)
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        foreach (Row row in rows)
        {
            index.Add(row.Id, [row.Value]);
        }

        return index;
    }

    private static Row[] CreateRows(int count)
    {
        var rows = new Row[count];
        for (int i = 0; i < rows.Length; i++)
        {
            int bucket = i % 53;
            float value = (i & 1) == 0 ? bucket : -bucket;
            ulong id = 1_000_000UL + (ulong)((i * 7_919) % 999_983);
            rows[i] = new Row(id, value);
        }

        return rows;
    }

    private static SearchResult[] Expected(IEnumerable<Row> rows, HashSet<ulong> deleted, int topK) =>
        Expected(rows, deleted, null, topK);

    private static SearchResult[] Expected(
        IEnumerable<Row> rows,
        HashSet<ulong> deleted,
        IEnumerable<ulong>? allowedIds,
        int topK)
    {
        HashSet<ulong>? allowed = allowedIds?.ToHashSet();
        return rows
            .Where(row => !deleted.Contains(row.Id))
            .Where(row => allowed is null || allowed.Contains(row.Id))
            .Select(static row => new SearchResult(row.Id, row.Value * row.Value))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static int CountUniqueLiveAllowed(IEnumerable<Row> rows, HashSet<ulong> deleted, IEnumerable<ulong> allowedIds)
    {
        HashSet<ulong> knownLiveIds = rows
            .Where(row => !deleted.Contains(row.Id))
            .Select(static row => row.Id)
            .ToHashSet();
        var matched = new HashSet<ulong>();
        foreach (ulong id in allowedIds)
        {
            if (knownLiveIds.Contains(id))
            {
                matched.Add(id);
            }
        }

        return matched.Count;
    }

    private static SearchResult[] SearchAll(ExactFlatIndex index, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search([0f], results);
        return results[..written];
    }

    private static SearchResult[] SearchRaw(ExactFlatIndex index, ulong[] allowedIds, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(
            [0f],
            allowedIds,
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));
        return results[..written];
    }

    private static SearchResult[] SearchCandidates(ExactFlatIndex index, ulong[] allowedIds, int topK)
    {
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allowedIds);
        var results = new SearchResult[topK];
        int written = index.Search([0f], candidates, results);
        return results[..written];
    }

    private readonly record struct Row(ulong Id, float Value);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

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
