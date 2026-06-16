namespace VecNet.Tests;

public sealed class ExactFlatIndexFilterTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Search_WithAllowlistMatchesUnfilteredRowsRestrictedToKnownDistinctAllowedIds(VectorMetric metric)
    {
        var index = CreateIndex(metric);
        ulong[] allowedA = [70, 999, 10, 70, 30, 888, 50];
        ulong[] allowedB = [50, 30, 70, 10, 10, 999];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var actualA = new SearchResult[4];
        var actualB = new SearchResult[4];

        SearchResult[] expected = ExpectedFromUnfiltered(index, CreateQuery(metric), allowedA, topK: 4);
        int writtenA = index.Search(CreateQuery(metric), allowedA, actualA, workspace);
        int writtenB = index.Search(CreateQuery(metric), allowedB, actualB, workspace);

        Assert.Equal(expected.Length, writtenA);
        Assert.Equal(expected, actualA[..writtenA]);
        Assert.Equal(expected.Length, writtenB);
        Assert.Equal(expected, actualB[..writtenB]);
    }

    [Fact]
    public void Search_WithAllIdsPreservesUnfilteredBehavior()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        ulong[] allowedIds = [10, 20, 30, 40, 50, 60, 70];
        var unfiltered = new SearchResult[3];
        var filtered = new SearchResult[3];

        int unfilteredWritten = index.Search(CreateQuery(VectorMetric.SquaredEuclidean), unfiltered);
        int filteredWritten = index.Search(
            CreateQuery(VectorMetric.SquaredEuclidean),
            allowedIds,
            filtered,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        Assert.Equal(unfilteredWritten, filteredWritten);
        Assert.Equal(unfiltered, filtered);
    }

    [Fact]
    public void Search_WithAllowlistReturnsZeroForEmptyFilterEmptyIndexAndEmptyDestinationAfterValidation()
    {
        var index = CreateIndex(VectorMetric.InnerProduct);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);

        Assert.Equal(0, index.Search(CreateQuery(VectorMetric.InnerProduct), [], new SearchResult[3], workspace));
        Assert.Equal(0, index.Search(CreateQuery(VectorMetric.InnerProduct), [10, 20], [], workspace));

        var empty = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        Assert.Equal(
            0,
            empty.Search(
                CreateQuery(VectorMetric.InnerProduct),
                [10],
                new SearchResult[1],
                new ExactFlatSearchFilterWorkspace(empty.VectorCount)));
    }

    [Fact]
    public void Search_WithAllowlistUsesReturnedWrittenCountForUnderfill()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        var results = new SearchResult[5];

        int written = index.Search(
            CreateQuery(VectorMetric.SquaredEuclidean),
            [30, 999, 30],
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        Assert.Equal(1, written);
        Assert.Equal(30UL, results[0].Id);
    }

    [Fact]
    public void Search_WithAllowlistPreservesEqualDistanceTieOrderByExternalId()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(30, [-1f]);
        index.Add(10, [1f]);
        index.Add(20, [-1f]);
        var results = new SearchResult[3];

        int written = index.Search(
            [0f],
            [30, 20, 10],
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        Assert.Equal(3, written);
        Assert.Equal([10UL, 20UL, 30UL], results[..written].Select(static result => result.Id));
        Assert.Equal(results[0].Distance, results[1].Distance);
        Assert.Equal(results[1].Distance, results[2].Distance);
    }

    [Fact]
    public void Search_WithAllowlistRejectsInvalidInputsBeforeMutatingDestination()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        var destination = new[]
        {
            new SearchResult(123, 456),
            new SearchResult(789, 101)
        };

        Assert.Throws<ArgumentException>(
            () => index.Search(
                [float.NaN, 0f],
                [10],
                destination,
                new ExactFlatSearchFilterWorkspace(index.VectorCount)));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);

        Assert.Throws<ArgumentNullException>(
            () => index.Search(CreateQuery(VectorMetric.SquaredEuclidean), [10], destination, null!));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);

        Assert.Throws<ArgumentException>(
            () => index.Search(
                CreateQuery(VectorMetric.SquaredEuclidean),
                [10],
                destination,
                new ExactFlatSearchFilterWorkspace(index.VectorCount - 1)));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);
    }

    [Fact]
    public void Search_WithAllowlistPreservesCosineZeroQueryValidationForEmptyFilterAndDestination()
    {
        var index = CreateIndex(VectorMetric.Cosine);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);

        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], [], new SearchResult[3], workspace));
        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], [10], [], workspace));
    }

    [Fact]
    public void Workspace_RejectsNegativeCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExactFlatSearchFilterWorkspace(-1));
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Search_WithAllowlistMatchesSavedAndOpenedReadOnlyIndex(VectorMetric metric)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = CreateIndex(metric);
        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        var expected = new SearchResult[3];
        var actual = new SearchResult[3];
        ulong[] allowedIds = [70, 10, 999, 30, 70];

        int expectedWritten = index.Search(
            CreateQuery(metric),
            allowedIds,
            expected,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));
        int actualWritten = loaded.Search(
            CreateQuery(metric),
            allowedIds,
            actual,
            new ExactFlatSearchFilterWorkspace(loaded.VectorCount));

        Assert.Equal(index.VectorCount, loaded.VectorCount);
        Assert.Equal(expectedWritten, actualWritten);
        Assert.Equal(expected[..expectedWritten], actual[..actualWritten]);
    }

    [Fact]
    public void Search_WithAllowlistSupportsReadOnlyParallelSearchWithIndependentWorkspaces()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        ulong[][] allowlists =
        [
            [10, 30, 50, 999],
            [70, 20, 20, 888],
            [60, 40, 10, 70]
        ];

        SearchResult[][] expected = allowlists
            .Select(allowlist => SearchFiltered(loaded, allowlist, topK: 3))
            .ToArray();

        Parallel.For(0, 200, iteration =>
        {
            int allowlistIndex = iteration % allowlists.Length;
            SearchResult[] actual = SearchFiltered(loaded, allowlists[allowlistIndex], topK: 3);
            Assert.Equal(expected[allowlistIndex], actual);
        });
    }

    [Fact]
    public void Search_WithAllowlistDoesNotAllocateWhenWorkspaceAndResultsAreSizedAndWarmed()
    {
        var index = new ExactFlatIndex(8, VectorMetric.SquaredEuclidean);
        for (int row = 0; row < 32; row++)
        {
            var vector = new float[8];
            vector[0] = row;
            vector[1] = row % 3;
            index.Add((ulong)(100 + row), vector);
        }

        float[] query = [3f, 1f, 0f, 0f, 0f, 0f, 0f, 0f];
        ulong[] allowedIds = [100, 101, 101, 105, 110, 999, 115, 120, 125, 131];
        var results = new SearchResult[5];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);

        Assert.Equal(5, index.Search(query, allowedIds, results, workspace));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 128; i++)
        {
            int written = index.Search(query, allowedIds, results, workspace);
            if (written != 5)
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
        foreach ((ulong id, float[] vector) in CreateRows(metric))
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static (ulong Id, float[] Vector)[] CreateRows(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean =>
            [
                (10, [0f, 0f]),
                (20, [2f, 0f]),
                (30, [1f, 0f]),
                (40, [-1f, 0f]),
                (50, [0f, 3f]),
                (60, [4f, 0f]),
                (70, [1f, 1f])
            ],
            VectorMetric.InnerProduct =>
            [
                (10, [1f, 0f]),
                (20, [0f, 2f]),
                (30, [2f, 1f]),
                (40, [-1f, 0f]),
                (50, [1f, 3f]),
                (60, [0.5f, 0.5f]),
                (70, [3f, 0f])
            ],
            VectorMetric.Cosine =>
            [
                (10, [1f, 0f]),
                (20, [0f, 1f]),
                (30, [1f, 1f]),
                (40, [-1f, 0f]),
                (50, [1f, 2f]),
                (60, [2f, 1f]),
                (70, [3f, 1f])
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float[] CreateQuery(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? [1f, 0.5f] : [1f, 1f];

    private static SearchResult[] ExpectedFromUnfiltered(
        ExactFlatIndex index,
        float[] query,
        ulong[] allowedIds,
        int topK)
    {
        var allowlist = allowedIds.ToHashSet();
        var allResults = new SearchResult[index.VectorCount];
        int written = index.Search(query, allResults);
        return allResults[..written]
            .Where(result => allowlist.Contains(result.Id))
            .Take(topK)
            .ToArray();
    }

    private static SearchResult[] SearchFiltered(ExactFlatIndex index, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(
            CreateQuery(index.Metric),
            allowlist,
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));
        return results[..written];
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
