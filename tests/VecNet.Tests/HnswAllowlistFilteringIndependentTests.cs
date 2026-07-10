namespace VecNet.Tests;

public sealed class HnswAllowlistFilteringIndependentTests
{
    [Fact]
    public void ImmutableAndOpenedHnswUseExactFallbackAtEfBoundaryAndBroadEmissionUnderfillsHonestly()
    {
        var options = new HnswIndexOptions(4, 8, 2, 0x5EED_0147UL);
        Row[] rows =
        [
            new(10, [0f]),
            new(20, [10f]),
            new(30, [20f]),
            new(40, [30f])
        ];
        HnswIndex index = CreateHnsw(rows, options);
        float[] query = [0f];

        SearchResult[] unknownOnlyDestination = [new(901, 901), new(902, 902)];
        int unknownOnlyWritten = index.Search(
            query,
            [999_001, 999_002, 999_001],
            unknownOnlyDestination,
            new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(0, unknownOnlyWritten);
        Assert.Equal([new SearchResult(901, 901), new SearchResult(902, 902)], unknownOnlyDestination);

        SearchResult[] boundaryDestination = [new(801, 801), new(802, 802), new(803, 803)];
        int boundaryWritten = index.Search(
            query,
            [40, 30, 40, 999_003],
            boundaryDestination.AsSpan(0, 2),
            new HnswSearchWorkspace(index.Count, options.EfSearch));

        SearchResult[] boundaryExpected = ExactTruth(rows, query, [40, 30, 40, 999_003], topK: 2);
        Assert.Equal(2, boundaryWritten);
        Assert.Equal(boundaryExpected, boundaryDestination[..boundaryWritten]);
        Assert.Equal(new SearchResult(803, 803), boundaryDestination[2]);
        AssertResultIntegrity(boundaryDestination[..boundaryWritten], rows, query, [30, 40]);

        SearchResult[] broadDestination = [new(701, 701), new(702, 702)];
        int broadWritten = index.Search(
            query,
            [20, 30, 40],
            broadDestination,
            new HnswSearchWorkspace(index.Count, options.EfSearch));

        SearchResult[] broadTruth = ExactTruth(rows, query, [20, 30, 40], topK: 2);
        Assert.True(broadWritten < broadTruth.Length, "Broad emission filtering should expose underfill through written count.");
        AssertResultIntegrity(broadDestination[..broadWritten], rows, query, [20, 30, 40]);
        Assert.Equal(new SearchResult(702, 702), broadDestination[1]);

        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        index.Save(saved.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        SearchResult[] openedDestination = new SearchResult[3];
        int openedWritten = opened.Search(
            query,
            [999_004, 40, 30, 30],
            openedDestination.AsSpan(0, 2),
            new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch));

        Assert.Equal(boundaryExpected.Length, openedWritten);
        Assert.Equal(boundaryExpected, openedDestination[..openedWritten]);
    }

    [Fact]
    public void CompositeAllowlistMatchesExactLiveViewBeforeAndAfterCheckpointAndOpenedOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 16, 4, 0x5EED_2147UL);
        Row[] baseRows =
        [
            new(10, [0f, 0f]),
            new(20, [1f, 0f]),
            new(30, [2f, 0f]),
            new(40, [3f, 0f]),
            new(50, [4f, 0f])
        ];
        HnswBasePlusExactDeltaIndex composite = new(CreateHnsw(baseRows, options));

        AssertCommitted(composite.TryAdd(15, [0.5f, 0f]));
        AssertCommitted(composite.TryAdd(35, [2.5f, 0f]));
        AssertCommitted(composite.TryAdd(45, [3.5f, 0f]));
        AssertCommitted(composite.TryDelete(20));
        AssertCommitted(composite.TryDelete(35));
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(35, [9f, 9f]).Status);

        Row[] liveRowsBeforeCheckpoint =
        [
            new(10, [0f, 0f]),
            new(30, [2f, 0f]),
            new(40, [3f, 0f]),
            new(50, [4f, 0f]),
            new(15, [0.5f, 0f]),
            new(45, [3.5f, 0f])
        ];
        float[] query = [0f, 0f];
        ulong[] allowlist = [999, 35, 20, 45, 15, 10, 45, 888];
        SearchResult[] expected = ExactTruth(liveRowsBeforeCheckpoint, query, allowlist, topK: 4);

        SearchResult[] beforeCheckpoint = CompositeSearch(composite, query, allowlist, topK: 4);
        Assert.Equal(expected, beforeCheckpoint);
        AssertResultIntegrity(beforeCheckpoint, liveRowsBeforeCheckpoint, query, [10, 15, 45]);

        HnswBasePlusExactDeltaCheckpointResult checkpointResult = composite.Checkpoint(checkpoint.Path);
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, checkpointResult.Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(35, [9f, 9f]).Status);

        SearchResult[] rebuiltComposite = CompositeSearch(composite, query, allowlist, topK: 4);
        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        SearchResult[] openedResults = HnswSearch(opened, query, allowlist, topK: 4);

        Assert.Equal(expected, rebuiltComposite);
        Assert.Equal(expected, openedResults);
        AssertResultIntegrity(openedResults, liveRowsBeforeCheckpoint, query, [10, 15, 45]);
    }

    [Fact]
    public void CompositeBroadEmissionKeepsDeltaExactAndReportsUnderfillWithoutLeakingTombstones()
    {
        var options = new HnswIndexOptions(4, 8, 3, 0x5EED_3147UL);
        Row[] baseRows =
        [
            new(100, [0f, 0f]),
            new(200, [10f, 0f]),
            new(300, [20f, 0f]),
            new(400, [30f, 0f]),
            new(500, [40f, 0f])
        ];
        HnswBasePlusExactDeltaIndex composite = new(CreateHnsw(baseRows, options));
        AssertCommitted(composite.TryAdd(50, [0.25f, 0f]));
        AssertCommitted(composite.TryAdd(60, [0.5f, 0f]));
        AssertCommitted(composite.TryDelete(60));

        Row[] liveRows =
        [
            new(100, [0f, 0f]),
            new(200, [10f, 0f]),
            new(300, [20f, 0f]),
            new(400, [30f, 0f]),
            new(500, [40f, 0f]),
            new(50, [0.25f, 0f])
        ];
        float[] query = [0f, 0f];
        ulong[] allowlist = [300, 400, 500, 50, 60, 60, 999];
        SearchResult[] exactTruth = ExactTruth(liveRows, query, allowlist, topK: 3);
        SearchResult[] actual = CompositeSearch(composite, query, allowlist, topK: 3);

        Assert.Contains(new SearchResult(50, 0.0625f), actual);
        Assert.DoesNotContain(actual, static result => result.Id == 60);
        Assert.True(actual.Length < exactTruth.Length, "Composite broad filtering should honestly underfill through written count.");
        AssertResultIntegrity(actual, liveRows, query, [50, 300, 400, 500]);
    }

    [Fact]
    public void FilteredCompositeWorkspaceValidationAndStalenessHappenBeforeDestinationWrites()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 16, 4, 0x5EED_4147UL);
        HnswBasePlusExactDeltaIndex composite = new(CreateHnsw(
            [
                new(10, [0f]),
                new(20, [1f]),
                new(30, [2f]),
                new(40, [3f]),
                new(50, [4f])
            ],
            options));
        AssertCommitted(composite.TryAdd(15, [0.5f]));
        AssertCommitted(composite.TryAdd(25, [1.5f]));

        SearchResult[] destination = [new(611, 611), new(622, 622), new(633, 633)];

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            [10, 15],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch,
                maxBaseCandidates: Math.Min(composite.BasePhysicalVectorCount, composite.Options.EfSearch) - 1,
                maxDeltaCandidates: destination.Length,
                maxDeltaFilterElements: composite.DeltaPhysicalVectorCount)));
        Assert.Equal([new SearchResult(611, 611), new SearchResult(622, 622), new SearchResult(633, 633)], destination);

        Assert.Throws<ArgumentException>(() => composite.Search(
            [0f],
            [10, 15],
            destination,
            new HnswBasePlusExactDeltaSearchWorkspace(
                composite.BasePhysicalVectorCount,
                composite.Options.EfSearch,
                Math.Min(composite.BasePhysicalVectorCount, composite.Options.EfSearch),
                maxDeltaCandidates: destination.Length - 1,
                maxDeltaFilterElements: composite.DeltaPhysicalVectorCount)));
        Assert.Equal([new SearchResult(611, 611), new SearchResult(622, 622), new SearchResult(633, 633)], destination);

        Assert.Throws<ArgumentException>(() => composite.Search(
            [float.NaN],
            ReadOnlySpan<ulong>.Empty,
            Span<SearchResult>.Empty,
            CreateCompositeWorkspace(composite, topK: 0)));

        HnswBasePlusExactDeltaSearchWorkspace reusable = CreateCompositeWorkspace(composite, topK: 3);
        SearchResult[] warmDestination = new SearchResult[3];
        Assert.True(composite.Search([0f], [10, 15, 25], warmDestination, reusable) > 0);

        AssertCommitted(composite.TryDelete(30));
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, composite.Checkpoint(checkpoint.Path).Status);

        SearchResult[] staleDestination = [new(711, 711), new(722, 722), new(733, 733)];
        Assert.Throws<InvalidOperationException>(() => composite.Search([0f], [10, 15], staleDestination, reusable));
        Assert.Equal([new SearchResult(711, 711), new SearchResult(722, 722), new SearchResult(733, 733)], staleDestination);
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

    private static SearchResult[] HnswSearch(HnswIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var destination = new SearchResult[topK];
        int written = index.Search(query, allowlist, destination, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return destination[..written];
    }

    private static SearchResult[] CompositeSearch(
        HnswBasePlusExactDeltaIndex index,
        float[] query,
        ulong[] allowlist,
        int topK)
    {
        var destination = new SearchResult[topK];
        int written = index.Search(query, allowlist, destination, CreateCompositeWorkspace(index, topK));
        return destination[..written];
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK,
            index.DeltaPhysicalVectorCount);

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

    private static void AssertResultIntegrity(
        SearchResult[] results,
        Row[] liveRows,
        float[] query,
        IEnumerable<ulong> allowedIds)
    {
        HashSet<ulong> allowed = allowedIds.ToHashSet();
        Dictionary<ulong, float> distanceById = liveRows
            .Where(row => allowed.Contains(row.Id))
            .ToDictionary(row => row.Id, row => SquaredEuclidean(query, row.Vector));

        Assert.Equal(results.Length, results.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in results)
        {
            Assert.Contains(result.Id, allowed);
            Assert.True(float.IsFinite(result.Distance), $"Distance for {result.Id} was not finite.");
            Assert.True(distanceById.TryGetValue(result.Id, out float expectedDistance));
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

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

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
            else if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswAllowlistFilteringIndependentTests-" + Guid.NewGuid().ToString("N"));
    }
}
