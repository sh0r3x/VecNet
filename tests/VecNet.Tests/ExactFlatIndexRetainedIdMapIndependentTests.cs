using System.Numerics;
using System.Reflection;
using System.Text.Json;

namespace VecNet.Tests;

public sealed class ExactFlatIndexRetainedIdMapIndependentTests
{
    private const int Seed = 0x50_1D;

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void Vec050_FilteredSearchMatchesBruteForceAcrossMetricsDimensionsAndAllowlists(VectorMetric metric)
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
            386
        ];

        foreach (int dimension in dimensions.Distinct())
        {
            float[] query = CreateQuery(dimension);
            (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 73);
            ulong[] knownIds = fixture.Rows.Select(static row => row.Id).ToArray();
            var workspace = new ExactFlatSearchFilterWorkspace(fixture.Index.VectorCount);

            ulong[][] allowlists =
            [
                knownIds.OrderByDescending(static id => id).ToArray(),
                [],
                [UnknownId(dimension, 1), knownIds[11], knownIds[11], UnknownId(dimension, 2), knownIds[3]],
                CreateDuplicateAndUnknownHeavyAllowlist(knownIds, dimension),
                knownIds.Concat(Enumerable.Range(0, 19).Select(i => UnknownId(dimension, 20 + i))).ToArray()
            ];

            foreach (ulong[] allowlist in allowlists)
            {
                foreach (int topK in new[] { 0, 1, 7, knownIds.Length + 23 })
                {
                    var actual = new SearchResult[topK];
                    int written = fixture.Index.Search(query, allowlist, actual, workspace);
                    SearchResult[] expected = BruteForce(fixture.Rows, query, metric, allowlist, topK);

                    AssertResultsEqual(expected, actual.AsSpan(0, written), metric, dimension);
                }
            }
        }
    }

    [Fact]
    public void Vec050_UnknownDuplicateHighKAndUnderfilledAllowlistsReuseWorkspace()
    {
        var index = new ExactFlatIndex(5, VectorMetric.SquaredEuclidean);
        index.Add(90, [9f, 0f, 0f, 0f, 0f]);
        index.Add(10, [1f, 0f, 0f, 0f, 0f]);
        index.Add(50, [5f, 0f, 0f, 0f, 0f]);
        index.Add(30, [3f, 0f, 0f, 0f, 0f]);

        float[] query = [0f, 0f, 0f, 0f, 0f];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        var results = new SearchResult[12];

        int first = index.Search(query, [999, 50, 999, 50, 10, 888, 10], results, workspace);
        Assert.Equal(2, first);
        Assert.Equal([10UL, 50UL], results[..first].Select(static result => result.Id));

        int second = index.Search(query, [777, 888, 777], results, workspace);
        Assert.Equal(0, second);

        int third = index.Search(query, [30], results, workspace);
        Assert.Equal(1, third);
        Assert.Equal(30UL, results[0].Id);

        int fourth = index.Search(query, [90, 10, 30, 50, 30, 10, 90], results, workspace);
        Assert.Equal(4, fourth);
        Assert.Equal([10UL, 30UL, 50UL, 90UL], results[..fourth].Select(static result => result.Id));
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, 33)]
    [InlineData(VectorMetric.InnerProduct, 64)]
    [InlineData(VectorMetric.Cosine, 97)]
    public void Vec050_OpenedReadOnlyIndexRebuiltMapMatchesFreshFilteredSearch(VectorMetric metric, int dimension)
    {
        float[] query = CreateQuery(dimension);
        (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 57);
        ulong[] knownIds = fixture.Rows.Select(static row => row.Id).ToArray();
        ulong[] allowlist = CreateDuplicateAndUnknownHeavyAllowlist(knownIds, dimension);

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        fixture.Index.Save(temp.Path);
        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        var freshResults = new SearchResult[100];
        var openedResults = new SearchResult[100];
        int freshWritten = fixture.Index.Search(
            query,
            allowlist,
            freshResults,
            new ExactFlatSearchFilterWorkspace(fixture.Index.VectorCount));
        int openedWritten = opened.Search(
            query,
            allowlist,
            openedResults,
            new ExactFlatSearchFilterWorkspace(opened.VectorCount));

        Assert.Equal(fixture.Index.VectorCount, opened.VectorCount);
        Assert.Throws<InvalidOperationException>(() => opened.Add(knownIds[0], CreateVector(metric, dimension, 1)));
        Assert.Equal(freshWritten, openedWritten);
        AssertResultsEqual(freshResults.AsSpan(0, freshWritten).ToArray(), openedResults.AsSpan(0, openedWritten), metric, dimension);

        SearchResult[] expected = BruteForce(fixture.Rows, query, metric, allowlist, topK: 100);
        AssertResultsEqual(expected, openedResults.AsSpan(0, openedWritten), metric, dimension);
    }

    [Fact]
    public void Vec050_GrowthInsertionOrderDuplicateIdsAndUnfilteredSearchRemainStable()
    {
        var index = new ExactFlatIndex(3, VectorMetric.SquaredEuclidean);
        var rows = new List<Row>();
        int[] insertionOrder = [15, 2, 27, 0, 8, 19, 4, 31, 1, 22, 6, 11, 29, 13, 24, 17, 35, 9];

        foreach (int rank in insertionOrder)
        {
            ulong id = (ulong)(1_000 + rank * 17);
            float[] vector = [rank, rank % 5, rank % 7];
            rows.Add(new Row(id, vector));
            index.Add(id, vector);
        }

        Assert.Throws<ArgumentException>(() => index.Add((ulong)(1_000 + 19 * 17), [100f, 100f, 100f]));
        Assert.Equal(insertionOrder.Length, index.VectorCount);

        index.Add(42_424, [2f, 2f, 2f]);
        rows.Add(new Row(42_424, [2f, 2f, 2f]));

        float[] query = [2f, 1f, 4f];
        foreach (int topK in new[] { 1, 5, 30 })
        {
            var unfiltered = new SearchResult[topK];
            int unfilteredWritten = index.Search(query, unfiltered);
            SearchResult[] expectedUnfiltered = BruteForce(rows, query, VectorMetric.SquaredEuclidean, rows.Select(static row => row.Id), topK);
            AssertResultsEqual(expectedUnfiltered, unfiltered.AsSpan(0, unfilteredWritten), VectorMetric.SquaredEuclidean, dimension: 3);

            var filteredAll = new SearchResult[topK];
            int filteredWritten = index.Search(
                query,
                rows.Select(static row => row.Id).Reverse().Concat([42_424UL, 777_777UL]).ToArray(),
                filteredAll,
                new ExactFlatSearchFilterWorkspace(index.VectorCount));
            AssertResultsEqual(unfiltered.AsSpan(0, unfilteredWritten).ToArray(), filteredAll.AsSpan(0, filteredWritten), VectorMetric.SquaredEuclidean, dimension: 3);
        }
    }

    [Fact]
    public void Vec050_PublicApiAndDurableExactFlatFormatDoNotExposeRetainedMap()
    {
        Assert.DoesNotContain(
            typeof(ExactFlatIndex).Assembly.GetExportedTypes(),
            static type => type.Name.Contains("Ordinal", StringComparison.OrdinalIgnoreCase) ||
                type.Name.Contains("Map", StringComparison.OrdinalIgnoreCase));

        MethodInfo[] publicMethods = typeof(ExactFlatIndex)
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
            publicMethods.Select(static method => method.Name).Order(StringComparer.Ordinal).ToArray());
        Assert.DoesNotContain(
            publicMethods.SelectMany(static method => method.GetParameters()),
            static parameter => parameter.Name?.Contains("ordinal", StringComparison.OrdinalIgnoreCase) == true);

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var index = new ExactFlatIndex(4, VectorMetric.InnerProduct);
        index.Add(7, [1f, 2f, 3f, 4f]);
        index.Add(3, [4f, 3f, 2f, 1f]);
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
        Assert.DoesNotContain("ordinal", manifestText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("idToOrdinal", manifestText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("map", manifestText, StringComparison.OrdinalIgnoreCase);

        using JsonDocument manifest = JsonDocument.Parse(manifestText);
        Assert.Equal(
            [
                "schemaName",
                "schemaVersion",
                "formatFamily",
                "createdUtc",
                "createdByTask",
                "writer",
                "index",
                "semantics",
                "files",
                "compatibility"
            ],
            manifest.RootElement.EnumerateObject().Select(static property => property.Name).ToArray());
        Assert.Equal(ExactFlatIndexStorage.ManifestSchemaName, manifest.RootElement.GetProperty("schemaName").GetString());
        Assert.Equal(ExactFlatIndexStorage.ManifestSchemaVersion, manifest.RootElement.GetProperty("schemaVersion").GetString());
    }

    private static (ExactFlatIndex Index, List<Row> Rows) CreateFixture(VectorMetric metric, int dimension, int rowCount)
    {
        var index = new ExactFlatIndex(dimension, metric);
        var rows = new List<Row>(rowCount);
        var random = new Random(Seed + (int)metric * 1_009 + dimension * 17 + rowCount);
        int[] insertionOrder = Enumerable.Range(0, rowCount).OrderBy(_ => random.Next()).ToArray();

        foreach (int rank in insertionOrder)
        {
            ulong id = (ulong)(50_000 + rank * 9_973 + dimension * 131);
            float[] vector = CreateVector(metric, dimension, rank);
            rows.Add(new Row(id, vector));
            index.Add(id, vector);
        }

        return (index, rows);
    }

    private static float[] CreateQuery(int dimension)
    {
        var query = new float[dimension];
        query[0] = 1f;
        return query;
    }

    private static float[] CreateVector(VectorMetric metric, int dimension, int rank)
    {
        var vector = new float[dimension];
        switch (metric)
        {
            case VectorMetric.SquaredEuclidean:
                vector[0] = rank + 1f;
                for (int i = 1; i < dimension; i++)
                {
                    vector[i] = ((rank + i) % 5 - 2) * 0.03125f;
                }

                break;

            case VectorMetric.InnerProduct:
                vector[0] = rank + 1f;
                for (int i = 1; i < dimension; i++)
                {
                    vector[i] = ((rank + i) % 7 - 3) * 0.015625f;
                }

                break;

            case VectorMetric.Cosine:
                vector[0] = 1f;
                if (dimension > 1)
                {
                    vector[1] = rank * 0.05f;
                }

                for (int i = 2; i < dimension; i++)
                {
                    vector[i] = ((rank + i) % 3 - 1) * 0.01f;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(metric));
        }

        return vector;
    }

    private static ulong[] CreateDuplicateAndUnknownHeavyAllowlist(ulong[] knownIds, int dimension) =>
    [
        UnknownId(dimension, 30),
        knownIds[41],
        knownIds[3],
        knownIds[41],
        UnknownId(dimension, 31),
        knownIds[17],
        knownIds[3],
        knownIds[52],
        UnknownId(dimension, 32),
        knownIds[17],
        knownIds[6],
        knownIds[6],
        UnknownId(dimension, 33)
    ];

    private static SearchResult[] BruteForce(
        IEnumerable<Row> rows,
        float[] query,
        VectorMetric metric,
        IEnumerable<ulong> allowlist,
        int topK)
    {
        var allowed = allowlist.ToHashSet();
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

    private static float D026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static ulong UnknownId(int dimension, int offset) =>
        ulong.MaxValue - (ulong)(dimension * 257 + offset);

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
