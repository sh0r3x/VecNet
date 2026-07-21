namespace VecNet.Tests;

public sealed class Vec208WorkspaceAndCountApiTests
{
    [Fact]
    public void ExactFlatSearchFilterWorkspaceFactory_SizesFromPhysicalRowsAfterTombstones()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);

        Assert.True(index.PhysicalVectorCount > index.LiveVectorCount);
        Assert.Equal(index.PhysicalVectorCount, index.VectorCount);

        ExactFlatSearchFilterWorkspace workspace = index.CreateSearchFilterWorkspace();

        Assert.Equal(index.PhysicalVectorCount, workspace.MaxVectorCount);

        SearchResult[] results = new SearchResult[3];
        int written = index.Search([0f], [10UL, 20UL, 30UL], results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([30UL, 10UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void ExactFlatSearchFilterWorkspace_ManualConstructorRemainsUsable()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);

        var workspace = new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount);
        SearchResult[] results = new SearchResult[1];

        int written = index.Search([0f], [10UL, 20UL], results, workspace);

        Assert.Equal(1, written);
        Assert.Equal(10UL, results[0].Id);
    }

    [Fact]
    public void HnswSearchWorkspaceFactory_SizesFromCountAndEfSearch()
    {
        var options = new HnswIndexOptions(2, 8, 6, 0x208UL);
        HnswIndex index = CreateHnsw(options);

        HnswSearchWorkspace workspace = index.CreateSearchWorkspace();

        Assert.Equal(index.Count, workspace.MaxElements);
        Assert.Equal(index.Options.EfSearch, workspace.MaxEf);

        SearchResult[] results = new SearchResult[2];
        int written = index.Search([0f], results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void HnswSearchWorkspace_ManualConstructorRemainsUsable()
    {
        var options = new HnswIndexOptions(2, 8, 6, 0x209UL);
        HnswIndex index = CreateHnsw(options);
        var workspace = new HnswSearchWorkspace(index.Count, index.Options.EfSearch);
        SearchResult[] results = new SearchResult[2];

        int written = index.Search([0f], results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 20UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void CountAliases_RemainCompatibleWithExplicitCountNames()
    {
        var exact = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        exact.Add(10, [10f]);
        exact.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, exact.TryDelete(20).Status);

        Assert.Equal(exact.PhysicalVectorCount, exact.VectorCount);
        Assert.NotEqual(exact.LiveVectorCount, exact.VectorCount);

        var constructedMutation = new VectorMutationResult(
            VectorMutationStatus.Committed,
            Generation: 7,
            LiveVectorCount: 3,
            DeltaVectorCount: 2,
            TombstoneCount: 1);

        Assert.Equal(constructedMutation.LiveVectorCount, constructedMutation.VectorCount);
        Assert.Equal(constructedMutation.DeltaVectorCount, constructedMutation.DeltaCount);

        VectorMutationResult operationMutation = exact.TryAdd(30, [30f]);
        Assert.Equal(operationMutation.LiveVectorCount, operationMutation.VectorCount);
        Assert.Equal(operationMutation.DeltaVectorCount, operationMutation.DeltaCount);

        HnswIndex baseIndex = CreateHnsw(new HnswIndexOptions(2, 8, 6, 0x210UL));
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(40, [4f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);

        Assert.Equal(mutable.LiveVectorCount, mutable.Count);
        Assert.Equal(3, mutable.LiveVectorCount);
        Assert.Equal(1, mutable.BaseTombstoneCount);
        Assert.Equal(1, mutable.TombstoneCount);
        Assert.Equal(1, mutable.DeletedReservedIdCount);
    }

    private static HnswIndex CreateHnsw(HnswIndexOptions options)
    {
        var index = new HnswIndex(1, VectorMetric.SquaredEuclidean, options);
        index.Add(10, [0f]);
        index.Add(20, [1f]);
        index.Add(30, [4f]);
        return index;
    }
}
