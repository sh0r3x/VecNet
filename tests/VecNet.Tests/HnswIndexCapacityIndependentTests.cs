namespace VecNet.Tests;

public sealed class HnswIndexCapacityIndependentTests
{
    [Fact]
    public void CapacityTracksReservationNotCountAcrossEmptyPartialExactGrowthAndCompactOpen()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(8, 32, 32, 0x5174_0001UL);
        int[] levels = Enumerable.Repeat(0, 7).ToArray();
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            options,
            initialCapacity: 5,
            () => levels[nextLevel++]);

        Assert.Equal(0, index.Count);
        Assert.Equal(5, index.Capacity);
        Assert.Empty(Search(index, [0f, 0f], topK: 4));

        Add(index, 10, 0f);
        Add(index, 20, 1f);
        Add(index, 30, 2f);
        Assert.Equal(3, index.Count);
        Assert.Equal(5, index.Capacity);
        Assert.Equal([10UL, 20UL, 30UL], SearchIds(index, [0f, 0f], topK: 5));

        Add(index, 40, 3f);
        Add(index, 50, 4f);
        Assert.Equal(5, index.Count);
        Assert.Equal(5, index.Capacity);
        Assert.Equal([10UL, 20UL, 30UL, 40UL, 50UL], SearchIds(index, [0f, 0f], topK: 5));

        Add(index, 60, 5f);
        Add(index, 70, 6f);
        Assert.Equal(7, index.Count);
        Assert.True(index.Capacity > 5);
        Assert.Equal(index.Capacity, index.DebugGetLayerCapacity(0));

        SearchResult[] sourceResults = Search(index, [1.25f, 0f], topK: 7);
        index.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(index.Count, opened.Count);
        Assert.Equal(opened.Count, opened.Capacity);
        Assert.Equal(sourceResults, Search(opened, [1.25f, 0f], topK: 7));
    }

    [Fact]
    public void EnsureCapacityFailuresAndOverplanningPreserveGraphSearchAndDuplicateState()
    {
        var options = new HnswIndexOptions(4, 24, 24, 0x5174_0002UL);
        int[] levels = [1, 0, 1, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            3,
            VectorMetric.SquaredEuclidean,
            options,
            initialCapacity: 6,
            () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(100 + i), [i, i % 2, -i]);
        }

        int capacity = index.Capacity;
        SearchResult[] baseline = Search(index, [1f, 1f, -1f], topK: 6);
        string graphBefore = GraphSnapshot(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            index.EnsureCapacity((Array.MaxLength / (options.M * 2)) + 1));
        Assert.Throws<ArgumentException>(() => index.Add(102, [99f, 99f, 99f]));

        Assert.Equal(capacity, index.Capacity);
        Assert.Equal(levels.Length, index.Count);
        Assert.Equal(baseline, Search(index, [1f, 1f, -1f], topK: 6));
        Assert.Equal(graphBefore, GraphSnapshot(index));

        index.EnsureCapacity(19);

        Assert.Equal(19, index.Capacity);
        Assert.Equal(levels.Length, index.Count);
        Assert.Equal(baseline, Search(index, [1f, 1f, -1f], topK: 6));
        Assert.Equal(graphBefore, GraphSnapshot(index));
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            Assert.Equal(19, index.DebugGetLayerCapacity(layer));
        }
    }

    [Fact]
    public void OpenedReadOnlyEnsureCapacityRejectsBeforeRequestValidationAndKeepsCompactSearch()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 16, 16, 0x5174_0003UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, initialCapacity: 12, () => 0);
        Add(index, 10, 0f);
        Add(index, 20, 1f);
        Add(index, 30, 2f);

        SearchResult[] baseline = Search(index, [0f, 0f], topK: 3);
        index.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(3, opened.Capacity);
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(-1));
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(int.MaxValue));
        Assert.Equal(3, opened.Capacity);
        Assert.Equal(3, opened.Count);
        Assert.Equal(baseline, Search(opened, [0f, 0f], topK: 3));
    }

    [Fact]
    public void PlannedUpperLayerBuildKeepsCapacityAndGraphIntegrityWhileMatchingExactTruth()
    {
        var options = new HnswIndexOptions(12, 48, 48, 0x5174_0004UL);
        int[] levels = [0, 2, 0, 1, 3, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0];
        int nextLevel = 0;
        var hnsw = new HnswIndex(
            4,
            VectorMetric.SquaredEuclidean,
            options,
            initialCapacity: levels.Length,
            () => levels[nextLevel++]);
        var exact = new ExactFlatIndex(4, VectorMetric.SquaredEuclidean, initialCapacity: levels.Length);

        for (int i = 0; i < levels.Length; i++)
        {
            float[] vector = [i % 4, i / 4, (i % 3) - 1f, i * 0.125f];
            ulong id = (ulong)(1_000 + i);
            hnsw.Add(id, vector);
            exact.Add(id, vector);

            Assert.Equal(levels.Length, hnsw.Capacity);
            for (int layer = 0; layer <= hnsw.MaxLayer; layer++)
            {
                Assert.Equal(levels.Length, hnsw.DebugGetLayerCapacity(layer));
            }
        }

        Assert.Equal(3, hnsw.MaxLayer);
        AssertGraphInvariants(hnsw);

        float[] query = [1f, 2f, 0f, 0.5f];
        Assert.Equal(Search(exact, query, topK: 10), Search(hnsw, query, topK: 10));
    }

    [Fact]
    public void PlannedCapacityAllowlistUsesCountSizedCallerWorkspaceAndPreservesCallerAllowlist()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(8, 32, 32, 0x5174_0005UL);
        var hnsw = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, initialCapacity: 24, () => 0);
        var exact = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean, initialCapacity: 6);
        Row[] rows =
        [
            new(10, [0f, 0f]),
            new(20, [1f, 0f]),
            new(30, [2f, 0f]),
            new(40, [3f, 0f]),
            new(50, [4f, 0f]),
            new(60, [5f, 0f])
        ];

        foreach (Row row in rows)
        {
            hnsw.Add(row.Id, row.Vector);
            exact.Add(row.Id, row.Vector);
        }

        ulong[] allowlist = [999, 50, 20, 50, 10, 777];
        ulong[] allowlistBefore = allowlist.ToArray();
        float[] query = [0f, 0f];
        SearchResult[] expected = Search(exact, query, allowlist, topK: 3);
        SearchResult[] actual = Search(hnsw, query, allowlist, topK: 3);

        Assert.Equal(24, hnsw.Capacity);
        Assert.Equal(6, hnsw.Count);
        Assert.Equal(allowlistBefore, allowlist);
        Assert.Equal(expected, actual);

        hnsw.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(6, opened.Capacity);
        Assert.Equal(expected, Search(opened, query, allowlist, topK: 3));
        Assert.Equal(allowlistBefore, allowlist);
    }

    [Fact]
    public void MutableCheckpointFromPlannedBaseKeepsLiveSearchFilterAndReservationParity()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(8, 32, 32, 0x5174_0006UL);
        var baseIndex = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, initialCapacity: 16, () => 0);
        Row[] baseRows =
        [
            new(10, [0f, 0f]),
            new(20, [1f, 0f]),
            new(30, [2f, 0f]),
            new(40, [3f, 0f]),
            new(50, [4f, 0f])
        ];
        foreach (Row row in baseRows)
        {
            baseIndex.Add(row.Id, row.Vector);
        }

        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(35, [2.5f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(35).Status);

        float[] query = [0f, 0f];
        ulong[] allowlist = [999, 35, 20, 15, 10, 50, 15];
        Row[] liveRows =
        [
            new(10, [0f, 0f]),
            new(30, [2f, 0f]),
            new(40, [3f, 0f]),
            new(50, [4f, 0f]),
            new(15, [0.5f, 0f])
        ];
        SearchResult[] expectedUnfiltered = ExactTruth(liveRows, query, topK: 5);
        SearchResult[] expectedFiltered = ExactTruth(liveRows, query, allowlist, topK: 4);

        Assert.Equal(expectedUnfiltered, Search(mutable, query, topK: 5));
        Assert.Equal(expectedFiltered, Search(mutable, query, allowlist, topK: 4));

        HnswMutableCheckpointResult checkpointResult = mutable.Checkpoint(checkpoint.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(5, checkpointResult.RebuiltBaseVectorCount);
        Assert.Equal(5, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(5, opened.Capacity);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(35, [9f, 9f]).Status);
        Assert.Equal(expectedUnfiltered, Search(mutable, query, topK: 5));
        Assert.Equal(expectedFiltered, Search(mutable, query, allowlist, topK: 4));
        Assert.Equal(expectedFiltered, Search(opened, query, allowlist, topK: 4));
    }

    private static void Add(HnswIndex index, ulong id, float x) => index.Add(id, [x, 0f]);

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static ulong[] SearchIds(HnswIndex index, float[] query, int topK) =>
        Search(index, query, topK).Select(static result => result.Id).ToArray();

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, int topK) =>
        rows
            .Select(row => new SearchResult(row.Id, SquaredEuclidean(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, ulong[] allowlist, int topK)
    {
        HashSet<ulong> allowed = allowlist.ToHashSet();
        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, SquaredEuclidean(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float SquaredEuclidean(float[] query, float[] vector)
    {
        float sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            float difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return sum;
    }

    private static string GraphSnapshot(HnswIndex index)
    {
        var parts = new List<string>
        {
            $"count={index.Count};entry={index.EntryPoint};max={index.MaxLayer}"
        };

        for (int ordinal = 0; ordinal < index.Count; ordinal++)
        {
            parts.Add($"level[{ordinal}]={index.DebugGetLevel(ordinal)}");
        }

        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                parts.Add($"l{layer}n{ordinal}={string.Join(",", GetNeighbors(index, layer, ordinal))}");
            }
        }

        return string.Join("|", parts);
    }

    private static void AssertGraphInvariants(HnswIndex index)
    {
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            int degreeLimit = layer == 0 ? index.Options.M * 2 : index.Options.M;
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                int[] neighbors = GetNeighbors(index, layer, ordinal);
                if (index.DebugGetLevel(ordinal) < layer)
                {
                    Assert.Empty(neighbors);
                    continue;
                }

                Assert.InRange(neighbors.Length, 0, degreeLimit);
                Assert.DoesNotContain(ordinal, neighbors);
                Assert.Equal(neighbors.Length, neighbors.Distinct().Count());
                foreach (int neighbor in neighbors)
                {
                    Assert.InRange(neighbor, 0, index.Count - 1);
                    Assert.True(index.DebugGetLevel(neighbor) >= layer);
                }
            }
        }
    }

    private static int[] GetNeighbors(HnswIndex index, int layer, int ordinal)
    {
        Span<int> buffer = stackalloc int[128];
        int count = index.DebugGetNeighbors(layer, ordinal, buffer);
        return buffer[..count].ToArray();
    }

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
                "VecNet-HnswIndexCapacityIndependentTests-" + Guid.NewGuid().ToString("N"));
    }
}
