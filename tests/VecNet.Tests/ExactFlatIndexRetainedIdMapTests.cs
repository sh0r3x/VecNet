using System.Reflection;

namespace VecNet.Tests;

public sealed class ExactFlatIndexRetainedIdMapTests
{
    [Fact]
    public void Add_RejectsDuplicateUsingRetainedIdMapNotBackingIdScan()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(42, [1f]);

        GetBackingIds(index)[0] = 99;

        Assert.Throws<ArgumentException>(() => index.Add(42, [2f]));
        Assert.Equal(1, index.VectorCount);
        Assert.Equal(0, GetRetainedIdMap(index)[42]);
    }

    [Fact]
    public void Add_RetainedIdMapTracksOrdinalsAcrossGrowth()
    {
        var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        for (int row = 0; row < 9; row++)
        {
            index.Add((ulong)(100 + row * 7), [row, row + 1]);
        }

        Dictionary<ulong, int> map = GetRetainedIdMap(index);
        Assert.Equal(index.VectorCount, map.Count);
        for (int row = 0; row < index.VectorCount; row++)
        {
            Assert.Equal(row, map[(ulong)(100 + row * 7)]);
        }
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Search_WithRetainedMapAllowlistMatchesExpectedFilteredResults(VectorMetric metric)
    {
        var index = CreateIndex(metric);
        ulong[] allowedIds = [35, 999, 14, 35, 7, 888, 28, 14];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var results = new SearchResult[4];

        int written = index.Search(CreateQuery(metric), allowedIds, results, workspace);

        SearchResult[] expected = ExpectedFiltered(metric);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void Search_WithRetainedMapReusesWorkspaceAcrossUnknownAndDuplicateAllowlists()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var results = new SearchResult[3];

        int firstWritten = index.Search([1f, 1f], [7, 14, 999, 14], results, workspace);
        Assert.Equal(2, firstWritten);
        Assert.Equal([14UL, 7UL], results[..firstWritten].Select(static result => result.Id));

        int secondWritten = index.Search([1f, 1f], [28, 888, 28], results, workspace);
        Assert.Equal(1, secondWritten);
        Assert.Equal(28UL, results[0].Id);

        int thirdWritten = index.Search([1f, 1f], [777, 888, 777], results, workspace);
        Assert.Equal(0, thirdWritten);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void OpenReadOnly_RebuildsRetainedIdMapAndPreservesFilteredBehavior(VectorMetric metric)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = CreateIndex(metric);
        index.Save(temp.Path);

        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        Dictionary<ulong, int> map = GetRetainedIdMap(loaded);
        Assert.Equal(5, map.Count);
        Assert.Equal(0, map[7]);
        Assert.Equal(4, map[35]);
        Assert.Throws<InvalidOperationException>(() => loaded.Add(7, CreateQuery(metric)));

        var results = new SearchResult[4];
        int written = loaded.Search(
            CreateQuery(metric),
            [35, 14, 999, 7, 35, 28],
            results,
            new ExactFlatSearchFilterWorkspace(loaded.VectorCount));

        SearchResult[] expected = ExpectedFiltered(metric);
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, results[..written]);
    }

    [Fact]
    public void Search_WithRetainedMapDoesNotAllocateWhenCallerBuffersAreReusedAfterWarmup()
    {
        var index = new ExactFlatIndex(8, VectorMetric.SquaredEuclidean);
        for (int row = 0; row < 64; row++)
        {
            var vector = new float[8];
            vector[0] = row % 11;
            vector[1] = row / 11;
            index.Add((ulong)(10_000 + row * 3), vector);
        }

        float[] query = [4f, 2f, 0f, 0f, 0f, 0f, 0f, 0f];
        ulong[] allowedIds =
        [
            10_000, 10_003, 10_003, 10_030, 10_060, 10_090, 10_120, 10_150,
            10_180, 777_777, 10_030, 888_888
        ];
        var results = new SearchResult[6];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);

        Assert.Equal(6, index.Search(query, allowedIds, results, workspace));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 256; i++)
        {
            int written = index.Search(query, allowedIds, results, workspace);
            if (written != 6)
            {
                throw new InvalidOperationException("Unexpected filtered result count during allocation measurement.");
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    private static ExactFlatIndex CreateIndex(VectorMetric metric)
    {
        var index = new ExactFlatIndex(2, metric);
        index.Add(7, metric == VectorMetric.Cosine ? [-1f, 0f] : [0f, 0f]);
        index.Add(14, metric == VectorMetric.Cosine ? [1f, 0f] : [1f, 0f]);
        index.Add(21, metric == VectorMetric.Cosine ? [0f, 1f] : [0f, 2f]);
        index.Add(28, metric == VectorMetric.Cosine ? [1f, 1f] : [2f, 0f]);
        index.Add(35, metric == VectorMetric.Cosine ? [2f, 1f] : [3f, 1f]);
        return index;
    }

    private static float[] CreateQuery(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? [1f, 0f] : [1f, 1f];

    private static SearchResult[] ExpectedFiltered(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean =>
            [
                new SearchResult(14, 1f),
                new SearchResult(7, 2f),
                new SearchResult(28, 2f),
                new SearchResult(35, 4f)
            ],
            VectorMetric.InnerProduct =>
            [
                new SearchResult(35, -4f),
                new SearchResult(28, -2f),
                new SearchResult(14, -1f),
                new SearchResult(7, 0f)
            ],
            VectorMetric.Cosine =>
            [
                new SearchResult(14, 0f),
                new SearchResult(35, 0.10557282f),
                new SearchResult(28, 0.29289323f),
                new SearchResult(7, 2f)
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static Dictionary<ulong, int> GetRetainedIdMap(ExactFlatIndex index)
    {
        FieldInfo field = typeof(ExactFlatIndex).GetField(
            "_idToOrdinal",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        return Assert.IsType<Dictionary<ulong, int>>(field.GetValue(index));
    }

    private static ulong[] GetBackingIds(ExactFlatIndex index)
    {
        FieldInfo field = typeof(ExactFlatIndex).GetField(
            "_ids",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        return Assert.IsType<ulong[]>(field.GetValue(index));
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
