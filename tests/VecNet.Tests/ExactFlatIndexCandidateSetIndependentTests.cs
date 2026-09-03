using System.Numerics;
using System.Reflection;
using System.Text.Json;

namespace VecNet.Tests;

public sealed class ExactFlatIndexCandidateSetIndependentTests
{
    private const int Seed = 0x52_0CA;

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Vec052_CandidateSetSearchMatchesIndependentBruteForceAcrossMetricsDimensionsAndScopes(VectorMetric metric)
    {
        int[] dimensions =
        [
            1,
            2,
            3,
            Math.Max(1, Vector<float>.Count - 1),
            Vector<float>.Count,
            Vector<float>.Count + 1,
            32,
            96,
            128,
            386,
            768
        ];

        foreach (int dimension in dimensions.Distinct())
        {
            float[] query = CreateQuery(metric, dimension);
            (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 73);
            ulong[] knownIds = fixture.Rows.Select(static row => row.Id).ToArray();
            var workspace = new ExactFlatSearchFilterWorkspace(fixture.Index.VectorCount);

            ulong[][] scopes =
            [
                [],
                knownIds.Reverse().ToArray(),
                knownIds.Where((_, index) => index % 2 == 0).Concat([UnknownId(dimension, 1)]).ToArray(),
                knownIds.Skip(9).Take(11).Concat([UnknownId(dimension, 2), UnknownId(dimension, 3)]).ToArray(),
                [knownIds[5], UnknownId(dimension, 4), knownIds[5], UnknownId(dimension, 5)],
                CreateDuplicateHeavyScope(knownIds, dimension),
                Enumerable.Range(0, 19).Select(i => UnknownId(dimension, 100 + i)).ToArray()
            ];

            foreach (ulong[] scope in scopes)
            {
                ExactFlatCandidateSet candidates = fixture.Index.CreateCandidateSet(scope);
                Assert.Equal(UniqueKnownCount(scope, knownIds), candidates.Count);

                ulong[] reorderedScope = scope
                    .OrderBy(id => unchecked((long)(id * 6_364_136_223_846_793_005UL)))
                    .ToArray();
                ExactFlatCandidateSet reorderedCandidates = fixture.Index.CreateCandidateSet(reorderedScope);
                Assert.Equal(candidates.Count, reorderedCandidates.Count);

                foreach (int topK in new[] { 0, 1, 7, 17, knownIds.Length + 13 })
                {
                    var candidateResults = new SearchResult[topK];
                    int candidateWritten = fixture.Index.Search(query, candidates, candidateResults);
                    SearchResult[] expected = BruteForce(fixture.Rows, query, metric, scope, topK);
                    AssertResultsEqual(expected, candidateResults.AsSpan(0, candidateWritten), metric, dimension);

                    var reorderedResults = new SearchResult[topK];
                    int reorderedWritten = fixture.Index.Search(query, reorderedCandidates, reorderedResults);
                    AssertResultsEqual(expected, reorderedResults.AsSpan(0, reorderedWritten), metric, dimension);

                    var rawAllowlistResults = new SearchResult[topK];
                    int rawAllowlistWritten = fixture.Index.Search(query, scope, rawAllowlistResults, workspace);
                    AssertResultsEqual(expected, rawAllowlistResults.AsSpan(0, rawAllowlistWritten), metric, dimension);
                }
            }

            ExactFlatCandidateSet allCandidates = fixture.Index.CreateCandidateSet(knownIds);
            foreach (int topK in new[] { 0, 1, 23, knownIds.Length + 13 })
            {
                var unfiltered = new SearchResult[topK];
                var filteredAll = new SearchResult[topK];

                int unfilteredWritten = fixture.Index.Search(query, unfiltered);
                int filteredAllWritten = fixture.Index.Search(query, allCandidates, filteredAll);

                Assert.Equal(unfilteredWritten, filteredAllWritten);
                AssertResultsEqual(
                    unfiltered.AsSpan(0, unfilteredWritten).ToArray(),
                    filteredAll.AsSpan(0, filteredAllWritten),
                    metric,
                    dimension);
            }
        }
    }

    [Fact]
    public void Vec052_WrongIndexStaleGenerationAndFailedAddsHaveConservativeLifecycle()
    {
        var first = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        var second = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        foreach (var index in new[] { first, second })
        {
            index.Add(10, [1f, 0f, 0f]);
            index.Add(20, [2f, 0f, 0f]);
            index.Add(30, [3f, 0f, 0f]);
        }

        ExactFlatCandidateSet candidates = first.CreateCandidateSet([10, 30]);
        var results = new SearchResult[4];

        InvalidOperationException wrongIndex = Assert.Throws<InvalidOperationException>(
            () => second.Search([0f, 0f, 0f], candidates, results));
        Assert.Contains("different exact flat index", wrongIndex.Message, StringComparison.OrdinalIgnoreCase);

        Assert.Throws<ArgumentException>(() => first.Add(10, [9f, 9f, 9f]));
        Assert.Throws<ArgumentException>(() => first.Add(40, [1f, 2f]));
        Assert.Throws<ArgumentException>(() => first.Add(40, [1f, float.NaN, 3f]));
        Assert.Equal(2, first.Search([0f, 0f, 0f], candidates, results));
        Assert.Equal([10UL, 30UL], results[..2].Select(static result => result.Id));

        first.Add(40, [0.5f, 0f, 0f]);
        InvalidOperationException stale = Assert.Throws<InvalidOperationException>(
            () => first.Search([0f, 0f, 0f], candidates, results));
        Assert.Contains("older exact flat index generation", stale.Message, StringComparison.OrdinalIgnoreCase);

        var empty = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        ExactFlatCandidateSet emptyCandidates = empty.CreateCandidateSet([]);
        Assert.Equal(0, empty.Search([0f, 0f, 0f], emptyCandidates, results));
        Assert.Throws<ArgumentException>(() => empty.Add(100, [float.PositiveInfinity, 0f, 0f]));
        Assert.Equal(0, empty.Search([0f, 0f, 0f], emptyCandidates, results));
        empty.Add(100, [1f, 0f, 0f]);
        Assert.Throws<InvalidOperationException>(() => empty.Search([0f, 0f, 0f], emptyCandidates, results));
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, 33)]
    [InlineData(VectorMetric.InnerProduct, 96)]
    [InlineData(VectorMetric.Cosine, 129)]
    public void Vec052_OpenedReadOnlyCandidateSetsMatchFreshAndBruteForceButRemainIndexBound(
        VectorMetric metric,
        int dimension)
    {
        float[] query = CreateQuery(metric, dimension);
        (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 67);
        ulong[] knownIds = fixture.Rows.Select(static row => row.Id).ToArray();
        ulong[] scope = CreateDuplicateHeavyScope(knownIds, dimension);

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        fixture.Index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        ExactFlatCandidateSet freshCandidates = fixture.Index.CreateCandidateSet(scope);
        ExactFlatCandidateSet openedCandidates = opened.CreateCandidateSet(scope);
        var freshResults = new SearchResult[80];
        var openedResults = new SearchResult[80];

        int freshWritten = fixture.Index.Search(query, freshCandidates, freshResults);
        int openedWritten = opened.Search(query, openedCandidates, openedResults);
        SearchResult[] expected = BruteForce(fixture.Rows, query, metric, scope, topK: 80);

        Assert.Equal(freshCandidates.Count, openedCandidates.Count);
        AssertResultsEqual(expected, freshResults.AsSpan(0, freshWritten), metric, dimension);
        AssertResultsEqual(expected, openedResults.AsSpan(0, openedWritten), metric, dimension);

        Assert.Throws<InvalidOperationException>(() => opened.Search(query, freshCandidates, openedResults));
        Assert.Throws<InvalidOperationException>(() => fixture.Index.Search(query, openedCandidates, freshResults));
        Assert.Throws<InvalidOperationException>(() => opened.Add(999, CreateVector(metric, query, 99)));

        Array.Clear(openedResults);
        int afterFailedAddWritten = opened.Search(query, openedCandidates, openedResults);
        AssertResultsEqual(expected, openedResults.AsSpan(0, afterFailedAddWritten), metric, dimension);
    }

    [Fact]
    public void Vec052_PublicSurfaceAndDurableFormatDoNotExposeOrdinalsSidecarsLabelsOrAnnFilters()
    {
        Assert.Empty(typeof(ExactFlatCandidateSet).GetConstructors(BindingFlags.Public | BindingFlags.Instance));
        Assert.Empty(typeof(ExactFlatCandidateSet).GetFields(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static));
        Assert.Empty(typeof(ExactFlatCandidateSet).GetEvents(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static));
        Assert.Equal(["Count"], typeof(ExactFlatCandidateSet)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(static property => property.Name)
            .Order(StringComparer.Ordinal)
            .ToArray());
        Assert.DoesNotContain(
            typeof(ExactFlatCandidateSet).GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly),
            static member => ContainsScopeCreepTerm(member.Name));

        MethodInfo[] exactFlatMethods = typeof(ExactFlatIndex)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Where(static method => !method.IsSpecialName)
            .ToArray();
        Assert.Equal(
            [
                "Add",
                "Checkpoint",
                "CreateCandidateSet",
                "CreateSearchFilterWorkspace",
                "EnsureCapacity",
                "OpenReadOnly",
                "Save",
                "Search",
                "Search",
                "Search",
                "TryAdd",
                "TryDelete"
            ],
            exactFlatMethods.Select(static method => method.Name).Order(StringComparer.Ordinal).ToArray());
        Assert.DoesNotContain(
            exactFlatMethods.SelectMany(static method => method.GetParameters()),
            static parameter => ContainsScopeCreepTerm(parameter.Name ?? string.Empty));
        Assert.DoesNotContain(
            typeof(ExactFlatIndex).Assembly.GetExportedTypes(),
            static type => type.FullName is not null &&
                (type.FullName.Contains("VectorData", StringComparison.OrdinalIgnoreCase) ||
                 type.FullName.Contains("HnswFilter", StringComparison.OrdinalIgnoreCase) ||
                 type.FullName.Contains("Label", StringComparison.OrdinalIgnoreCase) ||
                 type.FullName.Contains("Bitset", StringComparison.OrdinalIgnoreCase) ||
                 type.FullName.Contains("Bitmap", StringComparison.OrdinalIgnoreCase)));

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(4, VectorMetric.Cosine);
        index.Add(10, [1f, 0f, 0f, 0f]);
        index.Add(20, [1f, 1f, 0f, 0f]);
        _ = index.CreateCandidateSet([10, 20, 999, 10]);

        index.Save(temp.Path);

        Assert.Equal(
            [
                ExactFlatIndexStorage.IdsFileName,
                ExactFlatIndexStorage.ManifestFileName,
                ExactFlatIndexStorage.VectorsFileName
            ],
            Directory.EnumerateFiles(temp.Path)
                .Select(static path => Path.GetFileName(path)!)
                .Order(StringComparer.Ordinal)
                .ToArray());

        string manifestText = File.ReadAllText(Path.Combine(temp.Path, ExactFlatIndexStorage.ManifestFileName));
        foreach (string forbidden in new[] { "candidate", "filter", "ordinal", "rowOrdinal", "label", "sidecar", "VectorData", "bitmap", "bitset" })
        {
            Assert.DoesNotContain(forbidden, manifestText, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument manifest = JsonDocument.Parse(manifestText);
        Assert.Equal(
            [
                "schemaName",
                "schemaVersion",
                "formatFamily",
                "contentDigest",
                "createdUtc",
                "writer",
                "index",
                "semantics",
                "files",
                "compatibility"
            ],
            manifest.RootElement.EnumerateObject().Select(static property => property.Name).ToArray());
        Assert.Equal(ExactFlatIndexStorage.ManifestSchemaName, manifest.RootElement.GetProperty("schemaName").GetString());
        Assert.Equal(ExactFlatIndexStorage.ManifestSchemaVersion, manifest.RootElement.GetProperty("schemaVersion").GetString());
        Assert.False(manifest.RootElement.TryGetProperty("createdByTask", out _));
    }

    private static (ExactFlatIndex Index, List<Row> Rows) CreateFixture(
        VectorMetric metric,
        int dimension,
        int rowCount)
    {
        var index = new ExactFlatIndex(dimension, metric);
        var rows = new List<Row>(rowCount);
        float[] query = CreateQuery(metric, dimension);
        var random = new Random(Seed + (int)metric * 2_003 + dimension * 97 + rowCount);
        int[] insertionOrder = Enumerable.Range(0, rowCount).OrderBy(_ => random.Next()).ToArray();

        foreach (int rank in insertionOrder)
        {
            ulong id = (ulong)(700_000 + rank * 12_989 + dimension * 193);
            float[] vector = CreateVector(metric, query, rank);
            rows.Add(new Row(id, vector));
            index.Add(id, vector);
        }

        return (index, rows);
    }

    private static float[] CreateQuery(VectorMetric metric, int dimension)
    {
        var query = new float[dimension];
        query[0] = 1f;
        if (metric == VectorMetric.SquaredEuclidean)
        {
            for (int i = 1; i < dimension; i++)
            {
                query[i] = ((i % 5) - 2) * 0.125f;
            }
        }
        else if (metric == VectorMetric.InnerProduct)
        {
            for (int i = 1; i < dimension; i++)
            {
                query[i] = ((i & 1) == 0 ? 0.25f : -0.125f);
            }
        }

        return query;
    }

    private static float[] CreateVector(VectorMetric metric, float[] query, int rank)
    {
        var vector = new float[query.Length];
        switch (metric)
        {
            case VectorMetric.SquaredEuclidean:
                float offset = 0.25f + rank * 0.625f;
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] = query[i] + offset + (i % 7) * 0.015625f;
                }

                break;

            case VectorMetric.InnerProduct:
                vector[0] = 0.5f + rank * 0.75f;
                for (int i = 1; i < vector.Length; i++)
                {
                    vector[i] = ((rank + i) % 11 - 5) * 0.03125f;
                }

                break;

            case VectorMetric.Cosine:
                vector[0] = 1f;
                if (vector.Length > 1)
                {
                    vector[1] = 0.02f + rank * 0.04f;
                }

                for (int i = 2; i < vector.Length; i++)
                {
                    vector[i] = ((rank + i) % 5 - 2) * 0.0025f;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(metric));
        }

        return vector;
    }

    private static ulong[] CreateDuplicateHeavyScope(ulong[] knownIds, int dimension) =>
    [
        UnknownId(dimension, 10),
        knownIds[41],
        knownIds[3],
        knownIds[41],
        knownIds[3],
        UnknownId(dimension, 11),
        knownIds[17],
        knownIds[52],
        knownIds[17],
        knownIds[6],
        UnknownId(dimension, 12),
        knownIds[6],
        knownIds[63],
        knownIds[63],
        UnknownId(dimension, 13)
    ];

    private static SearchResult[] BruteForce(
        IEnumerable<Row> rows,
        float[] query,
        VectorMetric metric,
        IEnumerable<ulong> scope,
        int topK)
    {
        var allowed = scope.ToHashSet();
        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, Distance(query, row.Vector, metric)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float Distance(float[] query, float[] vector, VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
            VectorMetric.InnerProduct => InnerProduct(query, vector),
            VectorMetric.Cosine => Cosine(query, vector),
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float SquaredEuclidean(float[] query, float[] vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static float InnerProduct(float[] query, float[] vector)
    {
        double dot = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
        }

        return (float)-dot;
    }

    private static float Cosine(float[] query, float[] vector)
    {
        double dot = 0;
        double queryMagnitudeSquared = 0;
        double vectorMagnitudeSquared = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
            queryMagnitudeSquared += (double)query[i] * query[i];
            vectorMagnitudeSquared += (double)vector[i] * vector[i];
        }

        return (float)(1 - dot / (Math.Sqrt(queryMagnitudeSquared) * Math.Sqrt(vectorMagnitudeSquared)));
    }

    private static void AssertResultsEqual(
        SearchResult[] expected,
        ReadOnlySpan<SearchResult> actual,
        VectorMetric metric,
        int dimension)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            float tolerance = metric == VectorMetric.SquaredEuclidean
                ? D026Tolerance(dimension, expected[i].Distance)
                : 2e-5f;
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, tolerance);
        }
    }

    private static int UniqueKnownCount(IEnumerable<ulong> scope, IEnumerable<ulong> knownIds)
    {
        HashSet<ulong> known = knownIds.ToHashSet();
        return scope.Where(known.Contains).Distinct().Count();
    }

    private static bool ContainsScopeCreepTerm(string name) =>
        name.Contains("row", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("ordinal", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("bitset", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("bitmap", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("label", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("VectorData", StringComparison.OrdinalIgnoreCase);

    private static float D026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static ulong UnknownId(int dimension, int offset) =>
        ulong.MaxValue - (ulong)(dimension * 313 + offset);

    private sealed record Row(ulong Id, float[] Vector);

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
