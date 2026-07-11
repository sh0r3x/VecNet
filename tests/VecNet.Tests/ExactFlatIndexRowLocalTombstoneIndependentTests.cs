namespace VecNet.Tests;

public sealed class ExactFlatIndexRowLocalTombstoneIndependentTests
{
    [Fact]
    public void Vec165_UnfilteredHeapPathSkipsDeletedRowsWithMultiDimensionalRows()
    {
        Row[] rows = CreateRows(96);
        var index = BuildIndex(rows);
        HashSet<ulong> deleted =
        [
            rows[3].Id,
            rows[8].Id,
            rows[11].Id,
            rows[47].Id,
            rows[72].Id
        ];

        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        var actual = SentinelResults(16);
        int written = index.Search(Query, actual);

        SearchResult[] expected = Expected(rows, deleted, allowedIds: null, topK: actual.Length);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, actual[..written]);
        Assert.DoesNotContain(actual[..written], result => deleted.Contains(result.Id));
    }

    [Fact]
    public void Vec165_RawAllowlistIgnoresDuplicateDeletedIdsAndPreservesHeapPathOrdering()
    {
        Row[] rows = CreateRows(40);
        var index = BuildIndex(rows);
        HashSet<ulong> deleted = [rows[0].Id, rows[5].Id, rows[12].Id, rows[31].Id];
        foreach (ulong id in deleted)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(id).Status);
        }

        ulong[] allowedIds =
        [
            rows[12].Id,
            rows[12].Id,
            UnknownId(1),
            rows[3].Id,
            rows[5].Id,
            rows[3].Id,
            rows[9].Id,
            rows[31].Id,
            rows[18].Id,
            rows[0].Id,
            rows[24].Id,
            UnknownId(2),
            rows[18].Id,
            rows[30].Id
        ];
        var actual = SentinelResults(12);

        int written = index.Search(
            Query,
            allowedIds,
            actual,
            new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));

        SearchResult[] expected = Expected(rows, deleted, allowedIds, actual.Length);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, actual[..written]);
        AssertUnwrittenSentinels(actual, written);
        Assert.DoesNotContain(actual[..written], result => deleted.Contains(result.Id));
    }

    [Fact]
    public void Vec165_CandidateSetsBeforeAndAfterDeletionFollowGenerationRules()
    {
        Row[] rows = CreateRows(32);
        var index = BuildIndex(rows);
        ulong deletedId = rows[6].Id;
        ulong liveId = rows[7].Id;
        ulong anotherLiveId = rows[10].Id;

        ExactFlatCandidateSet beforeDelete = index.CreateCandidateSet([deletedId, liveId, liveId, UnknownId(10)]);
        var destination = SentinelResults(10);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(deletedId).Status);
        Assert.Throws<InvalidOperationException>(() => index.Search(Query, beforeDelete, destination));
        AssertUnwrittenSentinels(destination, written: 0);

        ExactFlatCandidateSet afterDelete = index.CreateCandidateSet(
            [deletedId, deletedId, liveId, anotherLiveId, UnknownId(11), liveId]);
        var afterDeleteResults = SentinelResults(10);
        int afterDeleteWritten = index.Search(Query, afterDelete, afterDeleteResults);

        Assert.Equal(2, afterDelete.Count);
        Assert.Equal(
            Expected(rows, [deletedId], [deletedId, liveId, anotherLiveId], topK: 10),
            afterDeleteResults[..afterDeleteWritten]);

        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(9_900, [0.125f, 0.125f, 0.125f]).Status);
        Assert.Throws<InvalidOperationException>(() => index.Search(Query, afterDelete, afterDeleteResults));

        ExactFlatCandidateSet afterInsert = index.CreateCandidateSet([deletedId, liveId, 9_900, 9_900]);
        var afterInsertResults = SentinelResults(10);
        int afterInsertWritten = index.Search(Query, afterInsert, afterInsertResults);

        Assert.Equal(2, afterInsert.Count);
        Assert.Equal([9_900UL, liveId], afterInsertResults[..afterInsertWritten].Select(static result => result.Id));
    }

    [Fact]
    public void Vec165_CountsAndPostTombstoneAddsStayCorrectAcrossBaseAndDeltaRows()
    {
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f, 0f, 0f]);
        index.Add(20, [20f, 0f, 0f]);
        index.Add(30, [30f, 0f, 0f]);
        index.Add(40, [40f, 0f, 0f]);
        index.Add(50, [50f, 0f, 0f]);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(60, [0.5f, 0f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(70, [0.25f, 0f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(60).Status);

        Assert.Equal(7, index.PhysicalVectorCount);
        Assert.Equal(5, index.LiveVectorCount);
        Assert.Equal(4, index.BaseVectorCount);
        Assert.Equal(1, index.DeltaVectorCount);
        Assert.Equal(2, index.TombstoneCount);
        Assert.Equal(2, index.DeletedReservedIdCount);

        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(80, [0f, 0f, 0f]).Status);
        Assert.Equal(8, index.PhysicalVectorCount);
        Assert.Equal(6, index.LiveVectorCount);
        Assert.Equal(4, index.BaseVectorCount);
        Assert.Equal(2, index.DeltaVectorCount);
        Assert.Equal(2, index.TombstoneCount);

        var results = new SearchResult[8];
        int written = index.Search([0f, 0f, 0f], results);

        Assert.Equal(6, written);
        Assert.Equal([80UL, 70UL, 10UL, 30UL, 40UL, 50UL], results[..written].Select(static result => result.Id));
        Assert.DoesNotContain(results[..written], static result => result.Id is 20 or 60);
    }

    [Fact]
    public void Vec165_SaveOpenAndCheckpointPublishCompactLiveParityFromIndependentRows()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        Row[] baseRows = CreateRows(28);
        var index = BuildIndex(baseRows);

        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(baseRows[2].Id).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(baseRows[19].Id).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(8_001, [0.5f, 0.25f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(8_002, [0.75f, 0f, 0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(8_001).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(8_003, [0f, 0.5f, 0.25f]).Status);

        Row[] allRows =
        [
            ..baseRows,
            new(8_001, [0.5f, 0.25f, 0f]),
            new(8_002, [0.75f, 0f, 0.25f]),
            new(8_003, [0f, 0.5f, 0.25f])
        ];
        HashSet<ulong> deleted = [baseRows[2].Id, baseRows[19].Id, 8_001];
        SearchResult[] expected = Expected(allRows, deleted, allowedIds: null, topK: 20);

        index.Save(saved.Path);
        ExactFlatIndex openedSaved = ExactFlatIndex.OpenReadOnly(saved.Path);
        AssertCompactAllLive(openedSaved, expected, liveCount: allRows.Length - deleted.Count);

        ExactFlatCheckpointResult checkpointResult = index.Checkpoint(checkpoint.Path);
        ExactFlatIndex openedCheckpoint = ExactFlatIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(0, index.TombstoneCount);
        Assert.Equal(allRows.Length - deleted.Count, index.PhysicalVectorCount);
        Assert.Equal(3, index.DeletedReservedIdCount);
        AssertCompactAllLive(index, expected, liveCount: allRows.Length - deleted.Count);
        AssertCompactAllLive(openedCheckpoint, expected, liveCount: allRows.Length - deleted.Count);
    }

    [Fact]
    public void Vec165_TombstonedHeapPathAllocationSmokeUsesReusedBuffersOnly()
    {
        Row[] rows = CreateRows(768);
        var index = BuildIndex(rows);
        for (int i = 1; i < rows.Length; i += 6)
        {
            Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(rows[i].Id).Status);
        }

        var results = new SearchResult[32];
        Assert.Equal(results.Length, index.Search(Query, results));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 128; i++)
        {
            int written = index.Search(Query, results);
            if (written != results.Length)
            {
                throw new InvalidOperationException("Unexpected result count from tombstoned heap-path search.");
            }
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    private static readonly float[] s_query = [0f, 0f, 0f];

    private static float[] Query => s_query;

    private static ExactFlatIndex BuildIndex(IEnumerable<Row> rows)
    {
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static Row[] CreateRows(int count)
    {
        var rows = new Row[count];
        for (int row = 0; row < rows.Length; row++)
        {
            float x = ((row * 17) % 23) - 11;
            float y = ((row * 31) % 19) - 9;
            float z = ((row * 43) % 29) - 14;
            ulong id = 50_000UL + (ulong)(row * 97);
            rows[row] = new Row(id, [x * 0.25f, y * 0.125f, z * 0.0625f]);
        }

        return rows;
    }

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
            .GroupBy(static row => row.Id)
            .Select(static group => group.First())
            .Select(static row => new SearchResult(row.Id, SquaredDistance(row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float SquaredDistance(float[] vector)
    {
        float sum = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        return sum;
    }

    private static SearchResult[] SentinelResults(int length)
    {
        var results = new SearchResult[length];
        for (int i = 0; i < results.Length; i++)
        {
            results[i] = new SearchResult(ulong.MaxValue - (ulong)i, -1_000f - i);
        }

        return results;
    }

    private static void AssertUnwrittenSentinels(SearchResult[] results, int written)
    {
        for (int i = written; i < results.Length; i++)
        {
            Assert.Equal(new SearchResult(ulong.MaxValue - (ulong)i, -1_000f - i), results[i]);
        }
    }

    private static void AssertCompactAllLive(ExactFlatIndex index, SearchResult[] expected, int liveCount)
    {
        Assert.Equal(liveCount, index.PhysicalVectorCount);
        Assert.Equal(liveCount, index.LiveVectorCount);
        Assert.Equal(liveCount, index.BaseVectorCount);
        Assert.Equal(0, index.DeltaVectorCount);
        Assert.Equal(0, index.TombstoneCount);

        var actual = new SearchResult[expected.Length];
        int written = index.Search(Query, actual);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, actual);
    }

    private static ulong UnknownId(int salt) => ulong.MaxValue - (ulong)salt;

    private sealed record Row(ulong Id, float[] Vector);

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
