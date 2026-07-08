namespace VecNet.Tests;

public sealed class HnswAllowlistFilteringTests
{
    [Fact]
    public void ImmutableHnswFilteredSearch_CoversFallbackBoundariesAndBroadEmissionUnderfill()
    {
        var options = new HnswIndexOptions(4, 8, 4, 0x147UL);
        Row[] rows =
        [
            new(10, [0f]),
            new(20, [1f]),
            new(30, [2f]),
            new(40, [3f]),
            new(50, [4f]),
            new(60, [5f])
        ];
        HnswIndex index = CreateHnsw(rows, options);
        float[] query = [0f];

        AssertFilteredEquals(rows, [], index, query, [999, 888], topK: 3);
        AssertFilteredEquals(rows, [10, 20], index, query, [20, 999, 10, 20], topK: 3);
        AssertFilteredEquals(rows, [10, 20, 30], index, query, [30, 20, 10, 10], topK: 3);
        AssertFilteredEquals(rows, [10, 20, 30], index, query, [40, 30, 20, 10, 777, 40], topK: 3);

        SearchResult[] allResults = FilteredSearch(index, query, [60, 50, 40, 30, 20, 10], topK: 3);
        AssertResultsAreValid(allResults, rows, query, [10, 20, 30, 40, 50, 60]);

        var narrow = new HnswIndex(1, VectorMetric.SquaredEuclidean, new HnswIndexOptions(4, 8, 1, 0x148UL), () => 0);
        narrow.Add(10, [0f]);
        narrow.Add(20, [10f]);
        narrow.Add(30, [20f]);

        SearchResult[] underfilled = FilteredSearch(narrow, query, [20, 30], topK: 1);
        SearchResult[] exactTruth = BruteForce([new(20, [10f]), new(30, [20f])], query, topK: 1);

        Assert.Empty(underfilled);
        Assert.NotEmpty(exactTruth);
    }

    [Fact]
    public void OpenedReadOnlyHnswFilteredSearch_MatchesSourceAndExactFallbackTruth()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(4, 16, 8, 0x149UL);
        Row[] rows =
        [
            new(100, [0f, 0f]),
            new(200, [1f, 0f]),
            new(300, [0f, 2f]),
            new(400, [2f, 2f]),
            new(500, [4f, 4f])
        ];
        HnswIndex source = CreateHnsw(rows, options);
        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        float[] query = [0.25f, 0.25f];
        ulong[] allowlist = [999, 400, 100, 300, 100, 888];

        SearchResult[] expected = BruteForce(rows, query, allowlist, topK: 4);
        SearchResult[] sourceResults = FilteredSearch(source, query, allowlist, topK: 4);
        SearchResult[] openedResults = FilteredSearch(opened, query, allowlist, topK: 4);

        Assert.Equal(expected, sourceResults);
        Assert.Equal(expected, openedResults);
        Assert.Throws<InvalidOperationException>(() => opened.Add(600, [1f, 1f]));
    }

    [Fact]
    public void CompositeFilteredSearch_SourceRebuiltAndCheckpointOpenedRespectTombstonesDeltaAndReservations()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(2, 8, 8, 0x14AUL);
        Row[] baseRows =
        [
            new(10, [0f]),
            new(20, [1f]),
            new(30, [2f]),
            new(40, [3f])
        ];
        HnswBasePlusExactDeltaIndex composite = new(CreateHnsw(baseRows, options));

        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(25, [1.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(35, [2.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(25).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(25, [9f]).Status);

        float[] query = [0f];
        ulong[] allowlist = [999, 20, 25, 15, 35, 10, 15, 30];
        Row[] sourceLiveRows =
        [
            new(10, [0f]),
            new(30, [2f]),
            new(40, [3f]),
            new(15, [0.5f]),
            new(35, [2.5f])
        ];
        SearchResult[] expected = BruteForce(sourceLiveRows, query, allowlist, topK: 5);

        SearchResult[] sourceResults = FilteredSearch(composite, query, allowlist, topK: 5);
        Assert.Equal(expected, sourceResults);
        AssertResultsAreValid(sourceResults, sourceLiveRows, query, [10, 15, 30, 35]);

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(25, [9f]).Status);

        SearchResult[] rebuiltResults = FilteredSearch(composite, query, allowlist, topK: 5);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        SearchResult[] openedResults = FilteredSearch(opened, query, allowlist, topK: 5);

        Assert.Equal(expected, rebuiltResults);
        Assert.Equal(expected, openedResults);
    }

    [Fact]
    public void CompositeFilteredSearch_BroadEmissionUnderfillsAndStillScansAllowedDeltaExactly()
    {
        var options = new HnswIndexOptions(4, 8, 1, 0x14BUL);
        HnswBasePlusExactDeltaIndex baseOnly = new(CreateHnsw(
            [new(10, [0f]), new(20, [10f]), new(30, [20f])],
            options));

        SearchResult[] underfilled = FilteredSearch(baseOnly, [0f], [20, 30], topK: 1);
        Assert.Empty(underfilled);
        Assert.NotEmpty(BruteForce([new Row(20, [10f]), new Row(30, [20f])], [0f], topK: 1));

        HnswBasePlusExactDeltaIndex withDelta = new(CreateHnsw(
            [new(10, [0f]), new(20, [10f]), new(30, [20f])],
            options));
        Assert.Equal(VectorMutationStatus.Committed, withDelta.TryAdd(5, [0.25f]).Status);

        SearchResult[] deltaResults = FilteredSearch(withDelta, [0f], [20, 5], topK: 1);

        Assert.Equal([new SearchResult(5, 0.0625f)], deltaResults);
    }

    [Fact]
    public void FilteredSearchValidationAndWorkspaceFailuresRejectBeforeDestinationWrites()
    {
        var options = new HnswIndexOptions(2, 8, 2, 0x14CUL);
        HnswIndex hnsw = CreateHnsw([new(10, [0f]), new(20, [1f])], options);
        SearchResult[] hnswDestination = [new(111, 111), new(222, 222), new(333, 333)];

        Assert.Throws<ArgumentException>(() => hnsw.Search(
            [0f],
            [10],
            hnswDestination.AsSpan(0, 1),
            new HnswSearchWorkspace(1, options.EfSearch)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222), new SearchResult(333, 333)], hnswDestination);

        Assert.Throws<ArgumentException>(() => hnsw.Search(
            [0f],
            [10],
            hnswDestination.AsSpan(0, 1),
            new HnswSearchWorkspace(hnsw.Count, options.EfSearch - 1)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222), new SearchResult(333, 333)], hnswDestination);

        Assert.Throws<ArgumentOutOfRangeException>(() => hnsw.Search(
            [0f],
            [10, 20],
            hnswDestination,
            new HnswSearchWorkspace(hnsw.Count, options.EfSearch)));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222), new SearchResult(333, 333)], hnswDestination);

        Assert.Throws<ArgumentException>(() => hnsw.Search(
            [float.NaN],
            [],
            Span<SearchResult>.Empty,
            new HnswSearchWorkspace(hnsw.Count, options.EfSearch)));

        HnswBasePlusExactDeltaIndex composite = new(hnsw);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(30, [0.5f]).Status);
        SearchResult[] compositeDestination = [new(444, 444), new(555, 555)];

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            [30],
            compositeDestination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch,
                Math.Min(composite.BasePhysicalVectorCount, composite.Options.EfSearch),
                compositeDestination.Length)));
        Assert.Equal([new SearchResult(444, 444), new SearchResult(555, 555)], compositeDestination);

        var workspace = CreateWorkspace(composite, topK: compositeDestination.Length);
        Assert.Equal(1, composite.Search([0f], [30], compositeDestination, workspace));
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(40, [2f]).Status);

        SearchResult[] staleDestination = [new(666, 666), new(777, 777)];
        Assert.Throws<InvalidOperationException>(() => composite.Search([0f], [30, 40], staleDestination, workspace));
        Assert.Equal([new SearchResult(666, 666), new SearchResult(777, 777)], staleDestination);
    }

    private static HnswIndex CreateHnsw(IEnumerable<Row> rows, HnswIndexOptions options)
    {
        Row[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options, () => 0);
        foreach (Row row in materialized)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static SearchResult[] FilteredSearch(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] FilteredSearch(
        HnswBasePlusExactDeltaIndex index,
        float[] query,
        ulong[] allowlist,
        int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, CreateWorkspace(index, topK));
        return results[..written];
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK,
            index.DeltaPhysicalVectorCount);

    private static void AssertFilteredEquals(
        Row[] rows,
        ulong[] expectedAllowedLiveIds,
        HnswIndex index,
        float[] query,
        ulong[] allowlist,
        int topK)
    {
        SearchResult[] expected = BruteForce(rows, query, allowlist, topK);
        SearchResult[] actual = FilteredSearch(index, query, allowlist, topK);

        Assert.Equal(expected, actual);
        AssertResultsAreValid(actual, rows, query, expectedAllowedLiveIds);
    }

    private static SearchResult[] BruteForce(Row[] rows, float[] query, ulong[] allowlist, int topK)
    {
        HashSet<ulong> allowed = allowlist.ToHashSet();
        return BruteForce(rows.Where(row => allowed.Contains(row.Id)).ToArray(), query, topK);
    }

    private static SearchResult[] BruteForce(Row[] rows, float[] query, int topK) =>
        rows.Select(row => new SearchResult(row.Id, SquaredEuclidean(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static void AssertResultsAreValid(
        SearchResult[] results,
        Row[] liveRows,
        float[] query,
        IEnumerable<ulong> expectedAllowedLiveIds)
    {
        HashSet<ulong> allowed = expectedAllowedLiveIds.ToHashSet();
        Dictionary<ulong, float> expectedDistanceById = liveRows
            .Where(row => allowed.Contains(row.Id))
            .ToDictionary(row => row.Id, row => SquaredEuclidean(query, row.Vector));

        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in results)
        {
            Assert.True(allowed.Contains(result.Id), $"Unexpected filtered result ID {result.Id}.");
            Assert.True(float.IsFinite(result.Distance), $"Distance for ID {result.Id} was not finite.");
            Assert.True(expectedDistanceById.TryGetValue(result.Id, out float expectedDistance));
            Assert.Equal(expectedDistance, result.Distance);
        }
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

    private sealed record Row(ulong Id, float[] Vector);

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
            else if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswAllowlistFilteringTests-" + Guid.NewGuid().ToString("N"));
    }
}
