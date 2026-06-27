using System.Reflection;

namespace VecNet.Tests;

public sealed class ExactFlatIndexCountInspectionTests
{
    [Fact]
    public void CountInspection_DistinguishesPhysicalLiveDeltaTombstoneAndWorkspaceCounts()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);

        AssertCounts(
            index,
            physical: 2,
            live: 2,
            @base: 2,
            delta: 0,
            tombstones: 0,
            reserved: 0);
        Assert.Equal(index.PhysicalVectorCount, index.VectorCount);

        VectorMutationResult added = index.TryAdd(30, [1f]);
        Assert.Equal(VectorMutationStatus.Committed, added.Status);
        Assert.Equal(3, added.LiveVectorCount);
        Assert.Equal(added.LiveVectorCount, added.VectorCount);
        Assert.Equal(1, added.DeltaVectorCount);
        Assert.Equal(1, added.DeltaCount);
        Assert.Equal(0, added.TombstoneCount);

        VectorMutationResult deleted = index.TryDelete(20);
        Assert.Equal(VectorMutationStatus.Committed, deleted.Status);
        Assert.Equal(2, deleted.LiveVectorCount);
        Assert.Equal(deleted.LiveVectorCount, deleted.VectorCount);
        Assert.Equal(1, deleted.DeltaVectorCount);
        Assert.Equal(1, deleted.DeltaCount);
        Assert.Equal(1, deleted.TombstoneCount);

        AssertCounts(
            index,
            physical: 3,
            live: 2,
            @base: 1,
            delta: 1,
            tombstones: 1,
            reserved: 1);
        Assert.Equal(index.PhysicalVectorCount, index.VectorCount);

        var destination = new SearchResult[3];
        Assert.Throws<ArgumentException>(
            () => index.Search([0f], [10, 20, 30], destination, new ExactFlatSearchFilterWorkspace(index.LiveVectorCount)));

        int written = index.Search(
            [0f],
            [10, 20, 30],
            destination,
            new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount));
        Assert.Equal(2, written);
        Assert.Equal([30UL, 10UL], destination[..written].Select(static result => result.Id));
    }

    [Fact]
    public void CountInspection_SaveAndOpenExposeCompactedLiveView()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);
        AssertCounts(index, physical: 3, live: 2, @base: 1, delta: 1, tombstones: 1, reserved: 1);

        index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        AssertCounts(opened, physical: 2, live: 2, @base: 2, delta: 0, tombstones: 0, reserved: 0);
        Assert.Equal(opened.PhysicalVectorCount, opened.VectorCount);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryAdd(40, [4f]).Status);
        Assert.Equal(VectorMutationStatus.ReadOnly, opened.TryDelete(10).Status);
    }

    [Fact]
    public void CountInspection_CheckpointPublishesPhysicalLiveAndMutationCountsConsistently()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(10, [10f]);
        index.Add(20, [20f]);
        Assert.Equal(VectorMutationStatus.Committed, index.TryAdd(30, [1f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, index.TryDelete(20).Status);

        ExactFlatCheckpointResult result = index.Checkpoint(checkpoint.Path);

        Assert.Equal(ExactFlatCheckpointStatus.Published, result.Status);
        Assert.Equal(2, result.PhysicalVectorCount);
        Assert.Equal(2, result.LiveVectorCount);
        Assert.Equal(2, result.BaseVectorCount);
        Assert.Equal(0, result.DeltaVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(1, result.DeletedReservedIdCount);
        Assert.Equal(1, result.FoldedDeltaVectorCount);
        Assert.Equal(1, result.FoldedTombstoneCount);
        AssertCounts(index, physical: 2, live: 2, @base: 2, delta: 0, tombstones: 0, reserved: 1);
    }

    [Fact]
    public void CountInspection_PublicSurfaceExposesUnambiguousExactCountNames()
    {
        string[] exactCountProperties = typeof(ExactFlatIndex)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(static property => property.Name)
            .Where(static name => name.Contains("Count", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(
            [
                "BaseVectorCount",
                "DeletedReservedIdCount",
                "DeltaVectorCount",
                "LiveVectorCount",
                "PhysicalVectorCount",
                "TombstoneCount",
                "VectorCount"
            ],
            exactCountProperties);

        string[] mutationCountProperties = typeof(VectorMutationResult)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(static property => property.Name)
            .Where(static name => name.Contains("Count", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(
            [
                "DeltaCount",
                "DeltaVectorCount",
                "LiveVectorCount",
                "TombstoneCount",
                "VectorCount"
            ],
            mutationCountProperties);
    }

    private static void AssertCounts(
        ExactFlatIndex index,
        int physical,
        int live,
        int @base,
        int delta,
        int tombstones,
        int reserved)
    {
        Assert.Equal(physical, index.PhysicalVectorCount);
        Assert.Equal(physical, index.VectorCount);
        Assert.Equal(live, index.LiveVectorCount);
        Assert.Equal(@base, index.BaseVectorCount);
        Assert.Equal(delta, index.DeltaVectorCount);
        Assert.Equal(tombstones, index.TombstoneCount);
        Assert.Equal(reserved, index.DeletedReservedIdCount);
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
