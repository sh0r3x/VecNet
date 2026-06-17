using System.Reflection;

namespace VecNet.Tests;

public sealed class ExactFlatIndexCandidateSetTests
{
    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void CandidateSetSearch_MatchesRawAllowlistAcrossMetricsAndInputOrder(VectorMetric metric)
    {
        var index = CreateIndex(metric);
        float[] query = CreateQuery(metric);
        ulong[] allowlistA = [70, 999, 10, 70, 30, 888, 50];
        ulong[] allowlistB = [50, 30, 70, 10, 10, 999];
        ExactFlatCandidateSet candidatesA = index.CreateCandidateSet(allowlistA);
        ExactFlatCandidateSet candidatesB = index.CreateCandidateSet(allowlistB);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var rawResults = new SearchResult[4];
        var candidateResultsA = new SearchResult[4];
        var candidateResultsB = new SearchResult[4];

        int rawWritten = index.Search(query, allowlistA, rawResults, workspace);
        int candidateWrittenA = index.Search(query, candidatesA, candidateResultsA);
        int candidateWrittenB = index.Search(query, candidatesB, candidateResultsB);

        Assert.Equal(4, candidatesA.Count);
        Assert.Equal(4, candidatesB.Count);
        Assert.Equal(rawWritten, candidateWrittenA);
        Assert.Equal(rawResults[..rawWritten], candidateResultsA[..candidateWrittenA]);
        Assert.Equal(rawWritten, candidateWrittenB);
        Assert.Equal(rawResults[..rawWritten], candidateResultsB[..candidateWrittenB]);
    }

    [Fact]
    public void CandidateSetSearch_WithAllIdsPreservesUnfilteredBehavior()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        ulong[] allIds = [10, 20, 30, 40, 50, 60, 70];
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(allIds);
        var unfiltered = new SearchResult[5];
        var filtered = new SearchResult[5];

        int unfilteredWritten = index.Search(CreateQuery(VectorMetric.SquaredEuclidean), unfiltered);
        int filteredWritten = index.Search(CreateQuery(VectorMetric.SquaredEuclidean), candidates, filtered);

        Assert.Equal(index.VectorCount, candidates.Count);
        Assert.Equal(unfilteredWritten, filteredWritten);
        Assert.Equal(unfiltered, filtered);
    }

    [Fact]
    public void CandidateSetSearch_ReturnsZeroForEmptyCandidatesEmptyIndexAndEmptyDestinationAfterValidation()
    {
        var index = CreateIndex(VectorMetric.InnerProduct);

        Assert.Equal(0, index.Search(CreateQuery(VectorMetric.InnerProduct), index.CreateCandidateSet([]), new SearchResult[3]));
        Assert.Equal(
            0,
            index.Search(
                CreateQuery(VectorMetric.InnerProduct),
                index.CreateCandidateSet([10, 20]),
                []));

        var empty = new ExactFlatIndex(2, VectorMetric.InnerProduct);
        ExactFlatCandidateSet emptyCandidates = empty.CreateCandidateSet([10, 20, 10]);
        Assert.Equal(0, emptyCandidates.Count);
        Assert.Equal(0, empty.Search(CreateQuery(VectorMetric.InnerProduct), emptyCandidates, new SearchResult[1]));
    }

    [Fact]
    public void CandidateSetSearch_UsesReturnedWrittenCountForUnderfillAndNeverDuplicatesResults()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet([30, 999, 30]);
        var results = new SearchResult[5];

        int written = index.Search(CreateQuery(VectorMetric.SquaredEuclidean), candidates, results);

        Assert.Equal(1, candidates.Count);
        Assert.Equal(1, written);
        Assert.Equal(30UL, results[0].Id);
    }

    [Fact]
    public void CandidateSetSearch_PreservesEqualDistanceTieOrderByExternalId()
    {
        var index = new ExactFlatIndex(1, VectorMetric.SquaredEuclidean);
        index.Add(30, [-1f]);
        index.Add(10, [1f]);
        index.Add(20, [-1f]);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet([30, 20, 10, 20]);
        var results = new SearchResult[3];

        int written = index.Search([0f], candidates, results);

        Assert.Equal(3, written);
        Assert.Equal([10UL, 20UL, 30UL], results[..written].Select(static result => result.Id));
        Assert.Equal(results[0].Distance, results[1].Distance);
        Assert.Equal(results[1].Distance, results[2].Distance);
    }

    [Fact]
    public void CandidateSetSearch_RejectsInvalidInputsBeforeMutatingDestination()
    {
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        ExactFlatCandidateSet candidates = index.CreateCandidateSet([10]);
        var destination = new[]
        {
            new SearchResult(123, 456),
            new SearchResult(789, 101)
        };

        Assert.Throws<ArgumentException>(() => index.Search([float.NaN, 0f], candidates, destination));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);

        Assert.Throws<ArgumentException>(() => index.Search([float.NaN, 0f], null!, destination));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);

        Assert.Throws<ArgumentNullException>(() => index.Search(CreateQuery(VectorMetric.SquaredEuclidean), null!, destination));
        Assert.Equal(new SearchResult(123, 456), destination[0]);
        Assert.Equal(new SearchResult(789, 101), destination[1]);
    }

    [Fact]
    public void CandidateSetSearch_PreservesCosineZeroQueryValidationForEmptyCandidatesAndDestination()
    {
        var index = CreateIndex(VectorMetric.Cosine);

        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], index.CreateCandidateSet([]), new SearchResult[3]));
        Assert.Throws<ArgumentException>(() => index.Search([0f, 0f], index.CreateCandidateSet([10]), []));
    }

    [Fact]
    public void CandidateSetSearch_RejectsWrongIndexAndStaleGeneration()
    {
        var first = CreateIndex(VectorMetric.SquaredEuclidean);
        var second = CreateIndex(VectorMetric.SquaredEuclidean);
        ExactFlatCandidateSet candidates = first.CreateCandidateSet([10, 30]);
        var results = new SearchResult[2];

        InvalidOperationException wrongIndex = Assert.Throws<InvalidOperationException>(
            () => second.Search(CreateQuery(VectorMetric.SquaredEuclidean), candidates, results));
        Assert.Contains("different exact flat index", wrongIndex.Message, StringComparison.OrdinalIgnoreCase);

        Assert.Throws<ArgumentException>(() => first.Add(10, [100f, 100f]));
        Assert.Equal(2, first.Search(CreateQuery(VectorMetric.SquaredEuclidean), candidates, results));

        first.Add(777, [100f, 100f]);
        InvalidOperationException stale = Assert.Throws<InvalidOperationException>(
            () => first.Search(CreateQuery(VectorMetric.SquaredEuclidean), candidates, results));
        Assert.Contains("older exact flat index generation", stale.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void CandidateSetSearch_MatchesSavedAndOpenedReadOnlyIndex(VectorMetric metric)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = CreateIndex(metric);
        ulong[] allowedIds = [70, 10, 999, 30, 70];
        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        ExactFlatCandidateSet freshCandidates = index.CreateCandidateSet(allowedIds);
        ExactFlatCandidateSet openedCandidates = loaded.CreateCandidateSet(allowedIds);
        var expected = new SearchResult[3];
        var actual = new SearchResult[3];

        int expectedWritten = index.Search(CreateQuery(metric), freshCandidates, expected);
        int actualWritten = loaded.Search(CreateQuery(metric), openedCandidates, actual);

        Assert.Equal(index.VectorCount, loaded.VectorCount);
        Assert.Equal(freshCandidates.Count, openedCandidates.Count);
        Assert.Equal(expectedWritten, actualWritten);
        Assert.Equal(expected[..expectedWritten], actual[..actualWritten]);
    }

    [Fact]
    public void CandidateSetSearch_SupportsReadOnlyParallelSearchWithSeparateResultBuffers()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = CreateIndex(VectorMetric.SquaredEuclidean);
        index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);
        ExactFlatCandidateSet[] candidateSets =
        [
            loaded.CreateCandidateSet([10, 30, 50, 999]),
            loaded.CreateCandidateSet([70, 20, 20, 888]),
            loaded.CreateCandidateSet([60, 40, 10, 70])
        ];

        SearchResult[][] expected = candidateSets
            .Select(candidates => SearchWithCandidateSet(loaded, candidates, topK: 3))
            .ToArray();

        Parallel.For(0, 200, iteration =>
        {
            int candidateIndex = iteration % candidateSets.Length;
            SearchResult[] actual = SearchWithCandidateSet(loaded, candidateSets[candidateIndex], topK: 3);
            Assert.Equal(expected[candidateIndex], actual);
        });
    }

    [Fact]
    public void CandidateSetSearch_DoesNotAllocateWhenCandidatesAndResultsAreReusedAfterWarmup()
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
        ExactFlatCandidateSet candidates = index.CreateCandidateSet(
        [
            10_000, 10_003, 10_003, 10_030, 10_060, 10_090, 10_120, 10_150,
            10_180, 777_777, 10_030, 888_888
        ]);
        var results = new SearchResult[6];

        Assert.Equal(6, index.Search(query, candidates, results));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 256; i++)
        {
            int written = index.Search(query, candidates, results);
            if (written != 6)
            {
                throw new InvalidOperationException("Unexpected candidate-set result count during allocation measurement.");
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void CandidateSetPublicApiKeepsRowsAndOrdinalsHidden()
    {
        Assert.Empty(typeof(ExactFlatCandidateSet).GetConstructors(BindingFlags.Public | BindingFlags.Instance));
        Assert.Equal(["Count"], typeof(ExactFlatCandidateSet)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(static property => property.Name)
            .Order(StringComparer.Ordinal)
            .ToArray());
        Assert.DoesNotContain(
            typeof(ExactFlatCandidateSet).GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly),
            static member => member.Name.Contains("row", StringComparison.OrdinalIgnoreCase) ||
                member.Name.Contains("ordinal", StringComparison.OrdinalIgnoreCase));
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

    private static SearchResult[] SearchWithCandidateSet(
        ExactFlatIndex index,
        ExactFlatCandidateSet candidates,
        int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(CreateQuery(index.Metric), candidates, results);
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
