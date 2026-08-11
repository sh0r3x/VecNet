using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class HnswInnerProductMetricIndependentTests
{
    [Fact]
    public void FullyReachableSearchMatchesExactFlatForNormSkewMixedSignsTiesNearTiesAndZeroVectors()
    {
        Row[] rows =
        [
            new(70, [1_000f, 0f, -10f, 0f, 1f]),
            new(20, [0.000_001f, -0.000_001f, 0f, 0f, 0f]),
            new(30, [0f, 0f, 0f, 0f, 0f]),
            new(40, [-5f, 4f, 3f, -2f, 1f]),
            new(10, [2f, 2f, 2f, 2f, 2f]),
            new(50, [2f, 2f, 2f, 2f, 2f]),
            new(60, [2f, 2f, 2f, 2f, 2.000_001f]),
            new(80, [-100f, 100f, -100f, 100f, -100f]),
            new(90, [3f, -7f, 0.125f, -0.25f, 9f])
        ];
        var options = new HnswIndexOptions(16, 64, 64, 0x3440_1001UL);
        HnswIndex hnsw = CreateHnsw(rows, options, static ordinal => 0);
        ExactFlatIndex exact = CreateExact(rows);
        float[][] queries =
        [
            [1f, -2f, 0.5f, 4f, -1f],
            [-0.25f, 0.125f, -8f, 16f, 0.5f],
            [0f, 0f, 0f, 0f, 0f]
        ];

        foreach (float[] query in queries)
        {
            SearchResult[] expected = Search(exact, query, rows.Length);
            SearchResult[] actual = Search(hnsw, query, rows.Length, efSearch: rows.Length);

            AssertResultsEqual(expected, actual);
            AssertResultDistancesMatchExactRows(actual, rows, query);
        }

        SearchResult[] zeroQueryResults = Search(hnsw, queries[2], rows.Length, efSearch: rows.Length);
        Assert.Equal(rows.Select(static row => row.Id).Order().ToArray(), zeroQueryResults.Select(static result => result.Id).ToArray());
        Assert.All(zeroQueryResults, static result => Assert.Equal(0f, result.Distance));
    }

    [Fact]
    public void AllowlistCoversEmptyUnknownDuplicatesFallbackBoundaryAndBroadUnderfill()
    {
        Row[] rows =
        [
            new(10, [10f]),
            new(20, [9f]),
            new(30, [1f]),
            new(40, [0f]),
            new(50, [-1f])
        ];
        var options = new HnswIndexOptions(2, 8, 2, 0x3440_2001UL);
        HnswIndex hnsw = CreateHnsw(rows, options, static ordinal => 0);
        ExactFlatIndex exact = CreateExact(rows);
        float[] query = [1f];

        SearchResult[] emptyDestination = FilledResults(2, 900);
        int emptyWritten = hnsw.Search(
            query,
            ReadOnlySpan<ulong>.Empty,
            emptyDestination,
            hnsw.CreateSearchWorkspace(options.EfSearch),
            options.EfSearch);
        Assert.Equal(0, emptyWritten);
        Assert.Equal(FilledResults(2, 900), emptyDestination);

        SearchResult[] unknownDestination = FilledResults(2, 800);
        int unknownWritten = hnsw.Search(
            query,
            [999, 998, 999],
            unknownDestination,
            hnsw.CreateSearchWorkspace(options.EfSearch),
            options.EfSearch);
        Assert.Equal(0, unknownWritten);
        Assert.Equal(FilledResults(2, 800), unknownDestination);

        AssertResultsEqual(
            Search(exact, query, [20, 20, 20, 999], topK: 2),
            Search(hnsw, query, [20, 20, 20, 999], topK: 2, efSearch: options.EfSearch));

        AssertResultsEqual(
            Search(exact, query, [30, 50, 30, 999], topK: 2),
            Search(hnsw, query, [30, 50, 30, 999], topK: 2, efSearch: options.EfSearch));

        ulong[] broadAllowlist = [30, 40, 50];
        SearchResult[] broadTruth = Search(exact, query, broadAllowlist, topK: 2);
        SearchResult[] unfilteredCandidates = Search(hnsw, query, topK: options.EfSearch, efSearch: options.EfSearch);
        SearchResult[] expectedBroad = unfilteredCandidates
            .Where(result => broadAllowlist.Contains(result.Id))
            .Take(2)
            .ToArray();
        SearchResult[] actualBroad = Search(hnsw, query, broadAllowlist, topK: 2, efSearch: options.EfSearch);

        Assert.True(actualBroad.Length < broadTruth.Length, "Broad allowlist emission filtering should report underfill.");
        AssertResultsEqual(expectedBroad, actualBroad);
        Assert.All(actualBroad, result => Assert.Contains(result.Id, broadAllowlist));
    }

    [Fact]
    public void OpenedReadOnlySearchAndAllowlistSearchMatchSequentialBaselinesUnderConcurrency()
    {
        using TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        Row[] rows = CreateConcurrentRows();
        var options = new HnswIndexOptions(8, 48, 32, 0x3440_3001UL);
        HnswIndex source = CreateHnsw(rows, options, DeterministicLevel);
        ExactFlatIndex exact = CreateExact(rows);
        float[][] queries =
        [
            CreateConcurrentQuery(3, 0x3440_3011),
            CreateConcurrentQuery(11, 0x3440_3012),
            CreateConcurrentQuery(29, 0x3440_3013),
            CreateConcurrentQuery(47, 0x3440_3014),
            CreateConcurrentQuery(61, 0x3440_3015),
            CreateConcurrentQuery(74, 0x3440_3016)
        ];
        ulong[][] allowlists =
        [
            [999_001, RowId(3), RowId(11), RowId(29), RowId(47), RowId(47)],
            rows.Where(static (_, index) => index % 2 == 0).Select(static row => row.Id).Concat([999_002UL]).ToArray(),
            rows.Where(static (_, index) => index % 3 != 0).Select(static row => row.Id).Concat([RowId(5), RowId(5)]).ToArray()
        ];

        source.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        for (int i = 0; i < queries.Length; i++)
        {
            AssertResultsEqual(Search(source, queries[i], 10, options.EfSearch), Search(opened, queries[i], 10, options.EfSearch));
            AssertResultDistancesMatchExactIndex(Search(opened, queries[i], 10, options.EfSearch), exact, queries[i], rows.Length);
        }

        SearchResult[][] unfilteredBaselines = queries
            .Select(query => Search(opened, query, topK: 10, efSearch: options.EfSearch))
            .ToArray();
        SearchResult[][] allowlistBaselines = queries
            .Select((query, index) => Search(opened, query, allowlists[index % allowlists.Length], topK: 7, efSearch: options.EfSearch))
            .ToArray();

        Parallel.For(0, 360, iteration =>
        {
            int queryIndex = iteration % queries.Length;
            float[] query = queries[queryIndex].ToArray();

            SearchResult unfilteredSentinel = new(ulong.MaxValue - (ulong)iteration, -1_000f - iteration);
            SearchResult[] unfilteredResults = Enumerable.Repeat(unfilteredSentinel, 12).ToArray();
            HnswSearchWorkspace unfilteredWorkspace = opened.CreateSearchWorkspace(options.EfSearch);
            int unfilteredWritten = opened.Search(query, unfilteredResults.AsSpan(0, 10), unfilteredWorkspace, options.EfSearch);

            AssertMatchesBaseline(unfilteredBaselines[queryIndex], unfilteredResults, unfilteredWritten);
            Assert.Equal(unfilteredSentinel, unfilteredResults[10]);
            Assert.Equal(unfilteredSentinel, unfilteredResults[11]);

            ulong[] allowlist = allowlists[queryIndex % allowlists.Length].ToArray();
            SearchResult allowlistSentinel = new(ulong.MaxValue - 1_000UL - (ulong)iteration, -2_000f - iteration);
            SearchResult[] allowlistResults = Enumerable.Repeat(allowlistSentinel, 9).ToArray();
            HnswSearchWorkspace allowlistWorkspace = opened.CreateSearchWorkspace(options.EfSearch);
            int allowlistWritten = opened.Search(query, allowlist, allowlistResults.AsSpan(0, 7), allowlistWorkspace, options.EfSearch);

            AssertMatchesBaseline(allowlistBaselines[queryIndex], allowlistResults, allowlistWritten);
            Assert.Equal(allowlistSentinel, allowlistResults[7]);
            Assert.Equal(allowlistSentinel, allowlistResults[8]);
        });
    }

    [Fact]
    public void OpenReadOnlyRejectsInnerProductHostileMetricNormalizationPayloadIdsGraphAndTruncation()
    {
        Action<TempIndexDirectory>[] mutations =
        [
            temp => MutateManifest(temp.Path, root => root["index"]!["metric"] = "cosine"),
            temp => MutateManifest(temp.Path, root => root["index"]!["normalizationState"] = "cosine-unit-normalized"),
            temp => PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), HnswIndexStorage.CosineMetricCode)),
            temp => PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36), HnswIndexStorage.CosineUnitNormalizedCode)),
            temp => PatchFile(
                temp.Path,
                HnswIndexStorage.IdsFileName,
                bytes =>
                {
                    ulong first = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(HnswIndexStorage.IdsHeaderLength));
                    BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(HnswIndexStorage.IdsHeaderLength + sizeof(ulong)), first);
                }),
            temp => PatchFile(
                temp.Path,
                HnswIndexStorage.VectorsFileName,
                bytes => BinaryPrimitives.WriteInt32LittleEndian(
                    bytes.AsSpan(HnswIndexStorage.VectorsHeaderLength + sizeof(float)),
                    BitConverter.SingleToInt32Bits(float.NaN))),
            temp => PatchFile(
                temp.Path,
                HnswIndexStorage.GraphFileName,
                bytes =>
                {
                    (_, int countsOffset, int neighborsOffset) = Layer(bytes, 0);
                    BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(countsOffset), 1);
                    BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(neighborsOffset), 999);
                }),
            temp => TruncatePayload(temp.Path, HnswIndexStorage.VectorsFileName, sizeof(float)),
            temp => TruncatePayload(temp.Path, HnswIndexStorage.GraphFileName, sizeof(int))
        ];

        foreach (Action<TempIndexDirectory> mutate in mutations)
        {
            using TempIndexDirectory temp = SavedInnerProductIndex();
            mutate(temp);
            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
        }
    }

    [Fact]
    public void MutableInnerProductMergesTombstonesRetriesCheckpointsReopensAndRejectsStaleWorkspaces()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        Row[] baseRows =
        [
            new(10, [2f, 0f]),
            new(20, [1f, 0f]),
            new(30, [0f, 0f]),
            new(40, [-1f, 0f])
        ];
        HnswMutableIndex mutable = new(CreateHnsw(baseRows, new HnswIndexOptions(8, 32, 16, 0x3440_5001UL), static ordinal => 0));
        HnswMutableSearchWorkspace staleAfterAdd = mutable.CreateSearchWorkspace(maxResults: 4, maxEfSearch: 16);

        AssertCommitted(mutable.TryAdd(5, [2f, 0f]));
        SearchResult[] staleDestination = FilledResults(4, 700);
        Assert.Throws<InvalidOperationException>(() => mutable.Search([1f, 0f], staleDestination, staleAfterAdd, efSearch: 16));
        Assert.Equal(FilledResults(4, 700), staleDestination);

        AssertCommitted(mutable.TryAdd(25, [3f, 0f]));
        AssertCommitted(mutable.TryDelete(20));
        AssertCommitted(mutable.TryDelete(25));
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [99f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(25, [99f, 0f]).Status);

        Row[] liveRows =
        [
            new(10, [2f, 0f]),
            new(30, [0f, 0f]),
            new(40, [-1f, 0f]),
            new(5, [2f, 0f])
        ];
        float[] query = [1f, 0f];
        SearchResult[] expected = ExactTruth(liveRows, query, topK: 4);
        Assert.Equal([5UL, 10UL, 30UL, 40UL], expected.Select(static result => result.Id));
        AssertResultsEqual(expected, Search(mutable, query, topK: 4, efSearch: 16));
        AssertResultsEqual(
            ExactTruth(liveRows, query, [999, 25, 20, 30, 10, 5, 5], topK: 3),
            Search(mutable, query, [999, 25, 20, 30, 10, 5, 5], topK: 3, efSearch: 16));

        HnswMutableSearchWorkspace staleAfterCheckpoint = mutable.CreateSearchWorkspace(maxResults: 4, maxEfSearch: 16);
        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(20, [99f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, mutable.TryAdd(25, [99f, 0f]).Status);
        Assert.Throws<InvalidOperationException>(() => mutable.Search(query, staleDestination, staleAfterCheckpoint, efSearch: 16));
        Assert.Equal(FilledResults(4, 700), staleDestination);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(VectorMetric.InnerProduct, opened.Metric);
        Assert.DoesNotContain(20UL, opened.InternalIds.ToArray());
        Assert.DoesNotContain(25UL, opened.InternalIds.ToArray());
        AssertResultsEqual(expected, Search(mutable, query, topK: 4, efSearch: 16));
        AssertResultsEqual(expected, Search(opened, query, topK: 4, efSearch: 16));

        HnswMutableIndex retryMutable = new(CreateHnsw(
            [
                new(101, [10f]),
                new(102, [9f]),
                new(103, [8f]),
                new(201, [1f]),
                new(202, [0.5f]),
                new(203, [0f])
            ],
            new HnswIndexOptions(6, 16, 3, 0x3440_5002UL),
            static ordinal => 0));
        AssertCommitted(retryMutable.TryDelete(101));
        AssertCommitted(retryMutable.TryDelete(102));
        AssertCommitted(retryMutable.TryDelete(103));

        SearchResult[] tight = FilledResults(3, 600);
        int tightWritten = retryMutable.Search(
            [1f],
            tight,
            retryMutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 3),
            efSearch: 3);

        SearchResult[] retried = FilledResults(3, 500);
        int retriedWritten = retryMutable.Search(
            [1f],
            retried,
            retryMutable.CreateSearchWorkspace(maxResults: 3, maxEfSearch: 6),
            efSearch: 3);

        Assert.True(tightWritten < retriedWritten, $"Expected adaptive retry to improve inner-product base tombstone underfill; tight wrote {tightWritten}, retried wrote {retriedWritten}.");
        Assert.Equal(3, retriedWritten);
        Assert.Equal([201UL, 202UL, 203UL], retried[..retriedWritten].Select(static result => result.Id));
        Assert.DoesNotContain(retried[..retriedWritten], static result => result.Id is 101 or 102 or 103);
        AssertResultDistancesMatchExactRows(retried[..retriedWritten], [new(201, [1f]), new(202, [0.5f]), new(203, [0f])], [1f]);
    }

    private static HnswIndex CreateHnsw(Row[] rows, HnswIndexOptions options, Func<int, int> levelProvider)
    {
        int nextOrdinal = 0;
        int dimension = rows.Length == 0 ? 1 : rows[0].Vector.Length;
        var index = new HnswIndex(dimension, VectorMetric.InnerProduct, options, () => levelProvider(nextOrdinal++));
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static ExactFlatIndex CreateExact(IEnumerable<Row> rows)
    {
        Row[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new ExactFlatIndex(dimension, VectorMetric.InnerProduct);
        foreach (Row row in materialized)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, ulong[] allowlist, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchWorkspace(topK, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, ulong[] allowlist, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, allowlist, results, index.CreateSearchFilterWorkspace());
        return results[..written];
    }

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, int topK) =>
        rows
            .Select(row => new SearchResult(row.Id, NegativeDot(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();

    private static SearchResult[] ExactTruth(Row[] rows, float[] query, ulong[] allowlist, int topK)
    {
        HashSet<ulong> allowed = allowlist.ToHashSet();
        return rows
            .Where(row => allowed.Contains(row.Id))
            .Select(row => new SearchResult(row.Id, NegativeDot(query, row.Vector)))
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Id)
            .Take(topK)
            .ToArray();
    }

    private static void AssertResultsEqual(SearchResult[] expected, SearchResult[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Id, actual[i].Id);
            Assert.Equal(expected[i].Distance, actual[i].Distance, precision: 6);
        }
    }

    private static void AssertMatchesBaseline(SearchResult[] expected, SearchResult[] destination, int written)
    {
        Assert.Equal(expected.Length, written);
        for (int i = 0; i < written; i++)
        {
            Assert.Equal(expected[i].Id, destination[i].Id);
            Assert.Equal(expected[i].Distance, destination[i].Distance, precision: 6);
        }
    }

    private static void AssertResultDistancesMatchExactIndex(SearchResult[] actual, ExactFlatIndex exact, float[] query, int count)
    {
        var allExact = new SearchResult[count];
        int written = exact.Search(query, allExact);
        Dictionary<ulong, float> distanceById = allExact[..written]
            .ToDictionary(static result => result.Id, static result => result.Distance);

        Assert.Equal(actual.Length, actual.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in actual)
        {
            Assert.True(float.IsFinite(result.Distance), $"Distance for {result.Id} was not finite.");
            Assert.Equal(distanceById[result.Id], result.Distance, precision: 6);
        }
    }

    private static void AssertResultDistancesMatchExactRows(SearchResult[] actual, Row[] rows, float[] query)
    {
        Dictionary<ulong, float> distanceById = rows.ToDictionary(row => row.Id, row => NegativeDot(query, row.Vector));
        Assert.Equal(actual.Length, actual.Select(static result => result.Id).Distinct().Count());
        foreach (SearchResult result in actual)
        {
            Assert.True(float.IsFinite(result.Distance), $"Distance for {result.Id} was not finite.");
            Assert.Equal(distanceById[result.Id], result.Distance, precision: 6);
        }
    }

    private static SearchResult[] FilledResults(int count, ulong firstId) =>
        Enumerable.Range(0, count)
            .Select(offset => new SearchResult(firstId + (ulong)offset, (float)(firstId + (ulong)offset)))
            .ToArray();

    private static float NegativeDot(float[] query, float[] vector)
    {
        double dot = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dot += (double)query[i] * vector[i];
        }

        return (float)-dot;
    }

    private static Row[] CreateConcurrentRows() =>
        Enumerable.Range(0, 80)
            .Select(ordinal => new Row(RowId(ordinal), CreateConcurrentVector(ordinal)))
            .ToArray();

    private static float[] CreateConcurrentVector(int ordinal)
    {
        var vector = new float[7];
        int normBand = ordinal % 10;
        float norm = normBand switch
        {
            0 => 0f,
            1 => 0.0005f,
            8 => 75f,
            9 => -50f,
            _ => normBand - 4.5f
        };
        for (int i = 0; i < vector.Length; i++)
        {
            int lane = ((ordinal * 13) + (i * 17)) % 23;
            float sign = ((ordinal + i) & 1) == 0 ? 1f : -1f;
            vector[i] = norm * sign + (lane - 11) * 0.03125f;
        }

        return vector;
    }

    private static float[] CreateConcurrentQuery(int ordinal, int seed)
    {
        float[] vector = CreateConcurrentVector(ordinal);
        var random = new Random(seed);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = (vector[i] * 0.125f) + ((random.NextSingle() - 0.5f) * 0.25f);
        }

        return vector;
    }

    private static int DeterministicLevel(int ordinal)
    {
        if (ordinal == 0 || ordinal % 37 == 0)
        {
            return 2;
        }

        return ordinal % 11 == 0 ? 1 : 0;
    }

    private static TempIndexDirectory SavedInnerProductIndex()
    {
        TempIndexDirectory temp = TempIndexDirectory.CreateMissing();
        CreateHnsw(
            [
                new(10, [2f, 0f]),
                new(20, [1f, -1f]),
                new(30, [0f, 0f]),
                new(40, [-3f, 4f]),
                new(50, [5f, 5f])
            ],
            new HnswIndexOptions(4, 16, 16, 0x3440_4001UL),
            ordinal => ordinal == 0 ? 1 : 0).Save(temp.Path);
        return temp;
    }

    private static void PatchFile(string directory, string fileName, Action<byte[]> patch)
    {
        string path = Path.Combine(directory, fileName);
        byte[] bytes = File.ReadAllBytes(path);
        patch(bytes);
        File.WriteAllBytes(path, bytes);
        RefreshManifestBinaryMetadata(directory, fileName);
    }

    private static void TruncatePayload(string directory, string fileName, int bytesToRemove)
    {
        string path = Path.Combine(directory, fileName);
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.None))
        {
            stream.SetLength(stream.Length - bytesToRemove);
        }

        RefreshManifestBinaryMetadata(directory, fileName);
    }

    private static void RefreshManifestBinaryMetadata(string directory, string fileName)
    {
        MutateManifest(directory, root =>
        {
            JsonObject file = (JsonObject)root["files"]![FilePropertyName(fileName)]!;
            string relativePath = file["path"]!.GetValue<string>();
            string binaryPath = Path.Combine(directory, relativePath);
            file["byteLength"] = new FileInfo(binaryPath).Length;
            file["sha256"] = Sha256Hex(binaryPath);
        });
    }

    private static void MutateManifest(string directory, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, HnswIndexStorage.ManifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static (int Stride, int CountsOffset, int NeighborsOffset) Layer(byte[] graphBytes, int layer)
    {
        int entryOffset = HnswIndexStorage.GraphHeaderLength + layer * HnswIndexStorage.GraphLayerDirectoryEntryLength;
        int stride = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(graphBytes.AsSpan(entryOffset + 4)));
        int countsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 24)));
        int neighborsOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(graphBytes.AsSpan(entryOffset + 32)));
        return (stride, countsOffset, neighborsOffset);
    }

    private static string FilePropertyName(string fileName) =>
        fileName switch
        {
            HnswIndexStorage.IdsFileName => "ids",
            HnswIndexStorage.VectorsFileName => "vectors",
            HnswIndexStorage.LevelsFileName => "levels",
            HnswIndexStorage.GraphFileName => "graph",
            _ => throw new ArgumentOutOfRangeException(nameof(fileName))
        };

    private static string Sha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static ulong RowId(int ordinal) => 10_000UL + (ulong)ordinal * 13UL;

    private static void AssertCommitted(VectorMutationResult result) =>
        Assert.Equal(VectorMutationStatus.Committed, result.Status);

    private sealed record Row(ulong Id, float[] Vector);

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing() =>
            new(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswInnerProductMetricIndependentTests-" + Guid.NewGuid().ToString("N")));

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
