namespace VecNet.Tests;

public sealed class HnswIndexCapacityTests
{
    [Fact]
    public void ConstructorCapacity_ValidatesNegativeOverflowAndContiguousGraphLimit()
    {
        var options = new HnswIndexOptions(4, 16, 16, 0x1740UL);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, -1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, options, -1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex((int.MaxValue / 2) + 1, VectorMetric.SquaredEuclidean, options, 2));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(1, VectorMetric.SquaredEuclidean, options, (Array.MaxLength / (options.M * 2)) + 1));
    }

    [Fact]
    public void EnsureCapacity_ValidatesAndPreallocatesWithoutChangingLogicalState()
    {
        var options = new HnswIndexOptions(4, 16, 16, 0x1741UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options);
        index.Add(10, [1f, 0f]);
        int count = index.Count;

        index.EnsureCapacity(12);

        Assert.Equal(12, index.Capacity);
        Assert.Equal(count, index.Count);
        Assert.Equal([10UL], SearchIds(index, [0f, 0f], topK: 4));

        index.EnsureCapacity(3);
        Assert.Equal(12, index.Capacity);

        Assert.Throws<ArgumentOutOfRangeException>(() => index.EnsureCapacity(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HnswIndex((int.MaxValue / 2) + 1, VectorMetric.SquaredEuclidean).EnsureCapacity(2));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HnswIndex(1, VectorMetric.SquaredEuclidean, options).EnsureCapacity((Array.MaxLength / (options.M * 2)) + 1));

        Assert.Equal(12, index.Capacity);
        Assert.Equal(count, index.Count);
        Assert.Equal([10UL], SearchIds(index, [0f, 0f], topK: 4));
    }

    [Fact]
    public void NoHintConstructionBehavior_StartsEmptyAndGrowsOnFirstAppend()
    {
        int[] levels = [0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1742UL),
            () => levels[nextLevel++]);

        Assert.Equal(0, index.Capacity);

        index.Add(10, [0f]);

        Assert.Equal(4, index.Capacity);
        Assert.Equal(4, index.DebugGetLayerCapacity(0));
        Assert.Null(index.DebugBuildSearchWorkspace);

        index.Add(20, [1f]);

        Assert.Equal(4, index.Capacity);
        Assert.Equal(4, index.DebugGetLayerCapacity(0));
        Assert.Equal(4, index.DebugBuildSearchWorkspace!.MaxElements);
    }

    [Fact]
    public void PlannedCapacityBuild_AvoidsRowGraphAndScratchGrowthForPlannedRows()
    {
        int[] levels = [0, 2, 1, 0, 0, 0];
        int nextLevel = 0;
        var options = new HnswIndexOptions(2, 8, 8, 0x1743UL);
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            options,
            initialCapacity: levels.Length,
            () => levels[nextLevel++]);

        Assert.Equal(levels.Length, index.Capacity);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(100 + i), [i]);
            Assert.Equal(levels.Length, index.Capacity);
            for (int layer = 0; layer <= index.MaxLayer; layer++)
            {
                Assert.Equal(levels.Length, index.DebugGetLayerCapacity(layer));
            }

            if (i > 0)
            {
                Assert.Equal(levels.Length, index.DebugBuildSearchWorkspace!.MaxElements);
            }
        }

        Assert.Equal(levels, Enumerable.Range(0, index.Count).Select(index.DebugGetLevel));
    }

    [Fact]
    public void GraphLayersCreatedAfterCapacityPlanning_UsePlannedRowCapacity()
    {
        int[] levels = [0, 3, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1744UL),
            initialCapacity: 10,
            () => levels[nextLevel++]);

        index.Add(10, [0f]);
        Assert.Equal(10, index.DebugGetLayerCapacity(0));

        index.Add(20, [1f]);

        Assert.Equal(3, index.MaxLayer);
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            Assert.Equal(10, index.DebugGetLayerCapacity(layer));
        }
    }

    [Fact]
    public void EnsureCapacityAfterBuild_GrowsExistingGraphLayersAndBuildScratch()
    {
        int[] levels = [0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1745UL),
            () => levels[nextLevel++]);
        index.Add(10, [0f]);
        index.Add(20, [1f]);

        Assert.Equal(4, index.Capacity);
        Assert.Equal(4, index.DebugBuildSearchWorkspace!.MaxElements);

        index.EnsureCapacity(12);

        Assert.Equal(12, index.Capacity);
        Assert.Equal(12, index.DebugGetLayerCapacity(0));
        Assert.Equal(12, index.DebugBuildSearchWorkspace!.MaxElements);

        index.Add(30, [2f]);
        Assert.Equal(12, index.Capacity);
        Assert.Equal([10UL, 20UL, 30UL], SearchIds(index, [0f], topK: 3));
    }

    [Fact]
    public void AddBeyondPlannedCapacity_GrowsAndPreservesGraphSearchAndIds()
    {
        int[] levels = [0, 0, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            1,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1746UL),
            initialCapacity: 2,
            () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(10 + i), [i]);
        }

        Assert.True(index.Capacity >= levels.Length);
        Assert.Equal(index.Capacity, index.DebugGetLayerCapacity(0));
        Assert.Throws<ArgumentException>(() => index.Add(12, [12f]));
        Assert.Equal([10UL, 11UL, 12UL, 13UL, 14UL], SearchIds(index, [0f], topK: 8));
    }

    [Fact]
    public void ReadOnlyOpenedIndex_RejectsEnsureCapacityWithoutChangingCompactView()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 16, 16, 0x1747UL), 8);
        index.Add(10, [1f, 0f]);
        index.Add(20, [0f, 1f]);
        index.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(2, opened.Capacity);
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(2));
        Assert.Throws<InvalidOperationException>(() => opened.EnsureCapacity(10));
        Assert.Equal(2, opened.Capacity);
        Assert.Equal(2, opened.Count);
        Assert.Equal([10UL, 20UL], SearchIds(opened, [0f, 0f], topK: 4));
    }

    [Fact]
    public void PlannedCapacitySearch_SaveOpenAndAllowlistParity()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 32, 32, 0x1748UL);
        var hnsw = new HnswIndex(3, VectorMetric.SquaredEuclidean, options, initialCapacity: 12);
        var exact = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean, initialCapacity: 12);

        for (int i = 0; i < 12; i++)
        {
            float[] vector = [i % 4, i / 4, (i % 3) * 0.25f];
            ulong id = (ulong)(1000 + i);
            hnsw.Add(id, vector);
            exact.Add(id, vector);
        }

        float[] query = [1f, 1f, 0.25f];
        SearchResult[] truth = Search(exact, query, topK: 8);
        SearchResult[] source = Search(hnsw, query, topK: 8);
        Assert.Equal(truth, source);
        Assert.Equal(12, hnsw.Capacity);

        ulong[] allowlist = [9999, 1007, 1001, 1003, 1001, 1009];
        Assert.Equal(
            SearchExactAllowlist(exact, query, allowlist, topK: 4),
            SearchAllowlist(hnsw, query, allowlist, topK: 4));

        hnsw.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(12, opened.Capacity);
        Assert.Equal(source, Search(opened, query, topK: 8));
        Assert.Equal(
            SearchExactAllowlist(exact, query, allowlist, topK: 4),
            SearchAllowlist(opened, query, allowlist, topK: 4));
    }

    [Fact]
    public void MutableCheckpointRebuild_PreservesSearchOpenedParityAndCompactLiveCapacity()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 32, 32, 0x1749UL);
        var baseIndex = new HnswIndex(2, VectorMetric.SquaredEuclidean, options, initialCapacity: 8);
        baseIndex.Add(10, [0f, 0f]);
        baseIndex.Add(20, [1f, 0f]);
        baseIndex.Add(30, [2f, 0f]);
        baseIndex.Add(40, [3f, 0f]);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(35, [2.5f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);

        SearchResult[] before = SearchComposite(composite, [0f, 0f], topK: 6);

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(5, result.RebuiltBaseVectorCount);
        Assert.Equal(5, composite.BasePhysicalVectorCount);
        Assert.Equal(0, composite.DeltaPhysicalVectorCount);
        Assert.Equal(5, opened.Capacity);
        Assert.Equal(before, SearchComposite(composite, [0f, 0f], topK: 6));
        Assert.Equal(SearchComposite(composite, [0f, 0f], topK: 6), Search(opened, [0f, 0f], topK: 6));
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static ulong[] SearchIds(HnswIndex index, float[] query, int topK) =>
        Search(index, query, topK).Select(static result => result.Id).ToArray();

    private static SearchResult[] SearchAllowlist(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] SearchExactAllowlist(ExactFlatIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));
        return results[..written];
    }

    private static SearchResult[] SearchComposite(HnswBasePlusExactDeltaIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        var workspace = new HnswBasePlusExactDeltaSearchWorkspace(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK);
        int written = index.Search(query, results, workspace);
        return results[..written];
    }

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
                "VecNet-HnswIndexCapacityTests-" + Guid.NewGuid().ToString("N"));
    }
}
