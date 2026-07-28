using System.Reflection;

namespace VecNet.Tests;

public sealed class HnswMutableIndexPublicPreviewApiTests
{
    [Fact]
    public void PublicSurface_ExposesMutableWrapperWithoutDiagnosticCheckpointApi()
    {
        Assert.True(typeof(HnswMutableIndex).IsPublic);
        Assert.True(typeof(HnswMutableSearchWorkspace).IsPublic);
        Assert.True(typeof(HnswMutableCheckpointResult).IsPublic);
        Assert.True(typeof(HnswMutableCheckpointStatus).IsPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointStatus).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnosticResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseStatus).IsNotPublic);

        Assert.Equal(
            [
                "Checkpoint",
                "CreateSearchWorkspace",
                "CreateSearchWorkspace",
                "Search",
                "Search",
                "Search",
                "Search",
                "TryAdd",
                "TryDelete"
            ],
            typeof(HnswMutableIndex)
                .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Where(static method => !method.IsSpecialName)
                .Select(static method => method.Name)
                .Order()
                .ToArray());

        Assert.DoesNotContain(
            typeof(HnswMutableIndex).GetMethods(BindingFlags.Public | BindingFlags.Instance),
            static method => method.Name.Contains("Diagnostic", StringComparison.OrdinalIgnoreCase));

        Assert.Equal(
            ["NoChanges", "Published"],
            Enum.GetNames<HnswMutableCheckpointStatus>().Order(StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void PublicWrapper_ReportsCountsAndSearchesBasePlusExactDelta()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })]);
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Equal(1, mutable.Dimension);
        Assert.Equal(VectorMetric.SquaredEuclidean, mutable.Metric);
        Assert.Equal(baseIndex.Options, mutable.Options);
        Assert.Equal(2, mutable.Count);
        Assert.Equal(2, mutable.LiveVectorCount);
        Assert.Equal(2, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);
        Assert.Equal(0, mutable.Generation);

        VectorMutationResult added = mutable.TryAdd(15, [0.5f]);
        VectorMutationResult deleted = mutable.TryDelete(20);

        Assert.Equal(VectorMutationStatus.Committed, added.Status);
        Assert.Equal(VectorMutationStatus.Committed, deleted.Status);
        Assert.Equal(2, mutable.Generation);
        Assert.Equal(2, mutable.LiveVectorCount);
        Assert.Equal(1, mutable.DeltaLiveVectorCount);
        Assert.Equal(1, mutable.BaseTombstoneCount);
        Assert.Equal(1, mutable.TombstoneCount);
        Assert.Equal(1, mutable.DeletedReservedIdCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [2f]).Status);

        var results = new SearchResult[3];
        var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: results.Length);
        int written = mutable.Search([0f], results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 15UL], results[..written].Select(static result => result.Id));
        Assert.All(results[..written], static result => Assert.True(float.IsFinite(result.Distance)));
    }

    [Fact]
    public void PublicWrapper_AllowlistSearchCoalescesUnknownDuplicatesAndTombstones()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f }), (30UL, new[] { 4f })]);
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);

        ulong[] allowed = [999, 20, 15, 15, 10];
        var results = new SearchResult[4];
        var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: results.Length);

        int written = mutable.Search([0f], allowed, results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 15UL], results[..written].Select(static result => result.Id));
    }

    [Fact]
    public void PublicWrapper_CheckpointPublishesRebuiltBaseAndInvalidatesOldWorkspace()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex(
            [(10UL, new[] { 0f }), (20UL, new[] { 2f })]);
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);

        var staleWorkspace = new HnswMutableSearchWorkspace(mutable, maxResults: 2);
        SearchResult[] before = Search(mutable, [0f], topK: 2);
        long beforeGeneration = mutable.Generation;

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(result.Generation, mutable.Generation);
        Assert.Equal(2, result.RebuiltBaseVectorCount);
        Assert.Equal(2, result.LiveVectorCount);
        Assert.Equal(0, result.DeltaPhysicalVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(1, result.FoldedDeltaVectorCount);
        Assert.Equal(1, result.FoldedBaseTombstoneCount);
        Assert.Equal(0, result.FoldedDeltaTombstoneCount);
        Assert.Equal(2, mutable.BasePhysicalVectorCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);

        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], new SearchResult[2], staleWorkspace));
        Assert.Equal(before, Search(mutable, [0f], topK: 2));

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(2, opened.Count);
        Assert.Equal(before.Select(static result => result.Id), Search(opened, [0f], topK: 2).Select(static result => result.Id));
    }

    [Fact]
    public void PublicWrapper_CheckpointNoChangesReturnsNarrowStatusWithoutWritingOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f })]);
        var mutable = new HnswMutableIndex(baseIndex);

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.NoChanges, result.Status);
        Assert.Equal(0, result.Generation);
        Assert.Equal(1, result.LiveVectorCount);
        Assert.False(Directory.Exists(checkpoint.Path));
    }

    [Fact]
    public void PublicWorkspace_ValidatesShapeAndGenerationBeforeDestinationWrites()
    {
        HnswIndex baseIndex = CreateBaseIndex([(10UL, new[] { 0f })]);
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswMutableSearchWorkspace(mutable, maxResults: -1));

        var tooSmall = new HnswMutableSearchWorkspace(mutable, maxResults: 1);
        SearchResult[] destination = [new(111, 111), new(222, 222)];
        Assert.Throws<ArgumentException>(() => mutable.Search([0f], destination, tooSmall));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);

        var stale = new HnswMutableSearchWorkspace(mutable, maxResults: 2);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(20, [1f]).Status);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], destination, stale));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222)], destination);
    }

    private static HnswIndex CreateBaseIndex(IEnumerable<(ulong Id, float[] Vector)> rows)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x153UL));

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswMutablePublicTests-" + Guid.NewGuid().ToString("N"));
            return new TempIndexDirectory(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
            else if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }
    }
}
