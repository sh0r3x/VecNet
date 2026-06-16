using System.Numerics;
using System.Reflection;

namespace VecNet.Tests;

public sealed class ExactFlatIndexFilterIndependentTests
{
    private const int RandomSeed = 0x45F_1A7;

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void FilteredSearch_MatchesIndependentBruteForceAcrossDimensionsAndAllowlistShapes(VectorMetric metric)
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
            384,
            386,
            768
        ];

        foreach (int dimension in dimensions.Distinct())
        {
            var random = new Random(RandomSeed + (int)metric * 10_007 + dimension * 97);
            float[] query = CreateQuery(metric, dimension, random);
            (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 41, query, random);
            ulong[] knownIds = fixture.Rows.Select(static row => row.Id).ToArray();
            var workspace = new ExactFlatSearchFilterWorkspace(fixture.Index.VectorCount);
            var results = new SearchResult[fixture.Index.VectorCount + 9];
            var reorderedResults = new SearchResult[fixture.Index.VectorCount + 9];

            ulong[][] allowlists =
            [
                knownIds.Reverse().ToArray(),
                [],
                [UnknownId(dimension, 1), knownIds[17], UnknownId(dimension, 2)],
                CreateUnknownHeavyAllowlist(knownIds, dimension),
                CreateDuplicateHeavyAllowlist(knownIds, dimension),
                knownIds.Concat(Enumerable.Range(0, 15).Select(i => UnknownId(dimension, i + 20))).ToArray()
            ];

            foreach (ulong[] allowlist in allowlists)
            {
                SearchResult[] expected = BruteForceFiltered(fixture.Rows, query, metric, allowlist, results.Length);
                int written = fixture.Index.Search(query, allowlist, results, workspace);

                AssertResultsEqual(expected, results.AsSpan(0, written), metric, dimension);

                ulong[] reordered = allowlist
                    .OrderBy(id => unchecked((long)(id * 1_103_515_245UL)))
                    .ToArray();
                int reorderedWritten = fixture.Index.Search(query, reordered, reorderedResults, workspace);

                Assert.Equal(written, reorderedWritten);
                AssertResultsEqual(expected, reorderedResults.AsSpan(0, reorderedWritten), metric, dimension);
            }
        }
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean, 33)]
    [InlineData(VectorMetric.InnerProduct, 97)]
    [InlineData(VectorMetric.Cosine, 129)]
    public void FilteredSearch_OnOpenedReadOnlyIndexMatchesBruteForceReference(VectorMetric metric, int dimension)
    {
        var random = new Random(RandomSeed + (int)metric * 4099 + dimension);
        float[] query = CreateQuery(metric, dimension, random);
        (ExactFlatIndex Index, List<Row> Rows) fixture = CreateFixture(metric, dimension, rowCount: 29, query, random);
        ulong[] allowlist = CreateUnknownHeavyAllowlist(fixture.Rows.Select(static row => row.Id).ToArray(), dimension);

        using TempIndexDirectory temp = TempIndexDirectory.Create();
        fixture.Index.Save(temp.Path);
        ExactFlatIndex loaded = ExactFlatIndex.OpenReadOnly(temp.Path);

        var actual = new SearchResult[fixture.Rows.Count + 5];
        int written = loaded.Search(
            query,
            allowlist,
            actual,
            new ExactFlatSearchFilterWorkspace(loaded.VectorCount));

        SearchResult[] expected = BruteForceFiltered(fixture.Rows, query, metric, allowlist, actual.Length);

        Assert.Equal(fixture.Index.VectorCount, loaded.VectorCount);
        AssertResultsEqual(expected, actual.AsSpan(0, written), metric, dimension);
    }

    [Theory]
    [InlineData(VectorMetric.SquaredEuclidean)]
    [InlineData(VectorMetric.InnerProduct)]
    [InlineData(VectorMetric.Cosine)]
    public void FilteredSearch_CoalescesDuplicateAllowlistIdsWithoutChangingEqualDistanceTieOrder(VectorMetric metric)
    {
        var index = new ExactFlatIndex(4, metric);
        float[] query = metric == VectorMetric.SquaredEuclidean ? [0f, 0f, 0f, 0f] : [1f, 0f, 0f, 0f];
        float[] tiedA = CreateTiedVector(metric, positive: true);
        float[] tiedB = CreateTiedVector(metric, positive: false);

        index.Add(90, tiedA);
        index.Add(20, tiedB);
        index.Add(70, tiedA);
        index.Add(10, tiedB);
        index.Add(50, CreateFartherVector(metric));

        ulong[] duplicateHeavyAllowlist = [70, 999_001, 20, 70, 10, 20, 90, 10, 999_002, 50, 50, 90];
        var results = new SearchResult[8];

        int written = index.Search(
            query,
            duplicateHeavyAllowlist,
            results,
            new ExactFlatSearchFilterWorkspace(index.VectorCount));

        Assert.Equal(5, written);
        Assert.Equal([10UL, 20UL, 70UL, 90UL, 50UL], results[..written].Select(static result => result.Id));
        Assert.Equal(results[0].Distance, results[1].Distance);
        Assert.Equal(results[1].Distance, results[2].Distance);
        Assert.Equal(results[2].Distance, results[3].Distance);
    }

    [Fact]
    public void FilteredSearch_ValidationKeepsQueryFailuresBeforeWorkspaceAndZeroResultShortcuts()
    {
        var index = new ExactFlatIndex(3, VectorMetric.Cosine);
        index.Add(10, [1f, 0f, 0f]);
        var destination = new[] { new SearchResult(444, 5), new SearchResult(555, 6) };

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f, 0f], [], destination, null!));
        Assert.Equal([444UL, 555UL], destination.Select(static result => result.Id));

        Assert.Throws<ArgumentException>(
            () => index.Search([1f, float.NaN, 0f], [10], Span<SearchResult>.Empty, null!));
        Assert.Equal([444UL, 555UL], destination.Select(static result => result.Id));

        Assert.Throws<ArgumentException>(
            () => index.Search([1f, 0f], [10], destination, new ExactFlatSearchFilterWorkspace(index.VectorCount)));
        Assert.Equal([444UL, 555UL], destination.Select(static result => result.Id));
    }

    [Fact]
    public void FilteredSearch_WorkspaceGenerationResetDoesNotExposeStaleRows()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(10, [0f, 0f]);
        index.Add(20, [1f, 0f]);
        index.Add(30, [2f, 0f]);
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        Array.Fill(workspace.RowMarks, int.MaxValue);

        FieldInfo? markField = typeof(ExactFlatSearchFilterWorkspace)
            .GetField("_searchMark", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(markField);
        markField.SetValue(workspace, int.MaxValue);

        var results = new SearchResult[3];
        int written = index.Search([0f, 0f], [20], results, workspace);

        Assert.Equal(1, written);
        Assert.Equal(20UL, results[0].Id);
        Assert.Equal(0, workspace.RowMarks[0]);
        Assert.Equal(1, workspace.RowMarks[1]);
        Assert.Equal(0, workspace.RowMarks[2]);
    }

    private static (ExactFlatIndex Index, List<Row> Rows) CreateFixture(
        VectorMetric metric,
        int dimension,
        int rowCount,
        float[] query,
        Random random)
    {
        var index = new ExactFlatIndex(dimension, metric);
        var rows = new List<Row>(rowCount);

        int[] insertionOrder = Enumerable.Range(0, rowCount).OrderBy(_ => random.Next()).ToArray();
        foreach (int rank in insertionOrder)
        {
            ulong id = (ulong)(10_000 + rank * 1_009 + dimension * 17);
            float[] vector = CreateVectorForRank(metric, query, rank, random);
            rows.Add(new Row(id, vector));
            index.Add(id, vector);
        }

        return (index, rows);
    }

    private static float[] CreateQuery(VectorMetric metric, int dimension, Random random)
    {
        var query = new float[dimension];
        for (int i = 0; i < query.Length; i++)
        {
            query[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }

        if (metric == VectorMetric.Cosine && query.All(static value => value == 0f))
        {
            query[0] = 1f;
        }

        return query;
    }

    private static float[] CreateVectorForRank(VectorMetric metric, float[] query, int rank, Random random)
    {
        var vector = new float[query.Length];
        switch (metric)
        {
            case VectorMetric.SquaredEuclidean:
                float offset = 0.125f + rank * 0.75f;
                for (int i = 0; i < vector.Length; i++)
                {
                    float sign = ((rank + i) & 1) == 0 ? 1f : -1f;
                    vector[i] = query[i] + sign * offset * (1f + (i % 7) * 0.03125f);
                }

                break;

            case VectorMetric.InnerProduct:
                float scale = 0.25f + rank * 0.5f;
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] = query[i] * scale + (float)(random.NextDouble() * 0.01 - 0.005);
                }

                break;

            case VectorMetric.Cosine:
                for (int i = 0; i < vector.Length; i++)
                {
                    float perturbation = (float)(random.NextDouble() * 0.5 - 0.25);
                    vector[i] = query[i] + perturbation + rank * 0.001f;
                }

                if (vector.All(static value => value == 0f))
                {
                    vector[0] = 1f;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(metric));
        }

        return vector;
    }

    private static SearchResult[] BruteForceFiltered(
        IEnumerable<Row> rows,
        float[] query,
        VectorMetric metric,
        ReadOnlySpan<ulong> allowlist,
        int topK)
    {
        var allowed = new HashSet<ulong>();
        foreach (ulong id in allowlist)
        {
            allowed.Add(id);
        }

        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, CalculateReferenceDistance(query, row.Vector, metric)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static float CalculateReferenceDistance(float[] query, float[] vector, VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => (float)query
                .Zip(vector, static (left, right) => (double)(left - right) * (left - right))
                .Sum(),
            VectorMetric.InnerProduct => (float)-query
                .Zip(vector, static (left, right) => (double)left * right)
                .Sum(),
            VectorMetric.Cosine => CalculateReferenceCosineDistance(query, vector),
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float CalculateReferenceCosineDistance(float[] query, float[] vector)
    {
        double dotProduct = 0;
        double queryMagnitudeSquared = 0;
        double vectorMagnitudeSquared = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (double)query[i] * vector[i];
            queryMagnitudeSquared += (double)query[i] * query[i];
            vectorMagnitudeSquared += (double)vector[i] * vector[i];
        }

        return (float)(1 - dotProduct / (Math.Sqrt(queryMagnitudeSquared) * Math.Sqrt(vectorMagnitudeSquared)));
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
                ? CalculateD026Tolerance(dimension, expected[i].Distance)
                : 2e-5f;
            Assert.InRange(MathF.Abs(expected[i].Distance - actual[i].Distance), 0f, tolerance);
        }
    }

    private static ulong[] CreateUnknownHeavyAllowlist(ulong[] knownIds, int dimension) =>
    [
        UnknownId(dimension, 10),
        UnknownId(dimension, 11),
        knownIds[3],
        UnknownId(dimension, 12),
        UnknownId(dimension, 13),
        knownIds[19],
        UnknownId(dimension, 14),
        knownIds[5],
        UnknownId(dimension, 15)
    ];

    private static ulong[] CreateDuplicateHeavyAllowlist(ulong[] knownIds, int dimension) =>
    [
        knownIds[7],
        knownIds[7],
        UnknownId(dimension, 30),
        knownIds[12],
        knownIds[7],
        knownIds[12],
        knownIds[1],
        knownIds[1],
        UnknownId(dimension, 31),
        knownIds[12],
        knownIds[20],
        knownIds[20],
        knownIds[1]
    ];

    private static ulong UnknownId(int dimension, int offset) =>
        (ulong.MaxValue - (ulong)(dimension * 100 + offset));

    private static float[] CreateTiedVector(VectorMetric metric, bool positive) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => positive ? [1f, 0f, 0f, 0f] : [-1f, 0f, 0f, 0f],
            VectorMetric.InnerProduct => positive ? [0f, 1f, 0f, 0f] : [0f, -1f, 0f, 0f],
            VectorMetric.Cosine => positive ? [0f, 1f, 0f, 0f] : [0f, -1f, 0f, 0f],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float[] CreateFartherVector(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => [4f, 0f, 0f, 0f],
            VectorMetric.InnerProduct => [-1f, 0f, 0f, 0f],
            VectorMetric.Cosine => [-1f, 0f, 0f, 0f],
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static float CalculateD026Tolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

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
