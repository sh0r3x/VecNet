namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaCheckpointIndependentTests
{
    public static TheoryData<int, int> RandomizedCheckpointCases() =>
        new()
        {
            { 1, unchecked((int)0x5EED0131) },
            { 3, unchecked((int)0x5EED1131) },
            { 32, unchecked((int)0x5EED2131) },
            { 386, unchecked((int)0x5EED3131) }
        };

    [Theory]
    [MemberData(nameof(RandomizedCheckpointCases))]
    public void FixedSeedRandomizedSequencesCheckpointAgainstIndependentLiveModel(
        int dimension,
        int seed)
    {
        var random = new Random(seed);
        var options = new HnswIndexOptions(4, 32, 32, unchecked(0x484E535700013100UL + (uint)dimension));
        Row[] initialRows = CreateInitialRows(dimension, random);
        var model = new ReferenceModel(dimension, initialRows);
        HnswIndex baseIndex = CreateBaseIndex(dimension, options, initialRows);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        int addOrdinal = 0;
        int checkpointOrdinal = 0;

        AssertCompositeMatchesModel(composite, model);
        ApplyAdd(composite, model, 4, CreateVector(dimension, 200, random));
        ApplyDelete(composite, model, 1);
        ulong deletedDeltaId = ulong.MaxValue - 10;
        ApplyAdd(composite, model, deletedDeltaId, CreateVector(dimension, 201, random));
        ApplyDelete(composite, model, deletedDeltaId);
        CheckpointAndValidate(composite, model, dimension, ref checkpointOrdinal);

        for (int step = 0; step < 36; step++)
        {
            int selector = step % 9 == 8 ? 2 : random.Next(0, 3);
            if (selector == 0)
            {
                ulong id = SelectAddId(model, ref addOrdinal, random);
                ApplyAdd(composite, model, id, CreateVector(dimension, 300 + step, random));
            }
            else if (selector == 1)
            {
                ulong id = SelectDeleteId(model, random);
                ApplyDelete(composite, model, id);
            }
            else
            {
                CheckpointAndValidate(composite, model, dimension, ref checkpointOrdinal);
            }

            AssertCompositeMatchesModel(composite, model);
            AssertCompositeSearchReturnsOnlyLiveFiniteIds(composite, model, CreateQuery(dimension, step), topK: 8);
        }

        CheckpointAndValidate(composite, model, dimension, ref checkpointOrdinal);
    }

    [Fact]
    public void NoChangesWritesNoOutputDoesNotAdvanceGenerationOrInvalidateWorkspace()
    {
        using TempIndexDirectory missing = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 32, 16, 0x131AUL);
        Row[] rows =
        [
            new(0, [0f, 0f, 0f]),
            new(ulong.MaxValue, [1f, 1f, 1f])
        ];
        var composite = new HnswBasePlusExactDeltaIndex(CreateBaseIndex(3, options, rows));
        var workspace = CreateWorkspace(composite, topK: 2);
        var destination = new SearchResult[2];
        Assert.Equal(2, composite.Search([0f, 0f, 0f], destination, workspace));
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(missing.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, result.Status);
        Assert.Equal(beforeGeneration, result.Generation);
        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.False(Directory.Exists(missing.Path));

        var reusedDestination = new SearchResult[2];
        Assert.Equal(2, composite.Search([0f, 0f, 0f], reusedDestination, workspace));
    }

    [Fact]
    public void FailedCheckpointLeavesOldCompositeSearchableAndUnchanged()
    {
        using TempIndexDirectory nonEmpty = TempIndexDirectory.Create();
        File.WriteAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt"), "do-not-touch");
        var options = new HnswIndexOptions(4, 32, 16, 0x131BUL);
        Row[] rows =
        [
            new(0, [0f, 0f, 0f]),
            new(1, [1f, 1f, 1f]),
            new(ulong.MaxValue, [2f, 2f, 2f])
        ];
        var model = new ReferenceModel(3, rows);
        var composite = new HnswBasePlusExactDeltaIndex(CreateBaseIndex(3, options, rows));
        ApplyAdd(composite, model, 7, [0.25f, 0.25f, 0.25f]);
        ApplyDelete(composite, model, 1);
        SearchResult[] before = SearchComposite(composite, [0f, 0f, 0f], topK: 4);
        long beforeGeneration = composite.Generation;

        Assert.Throws<IOException>(() => composite.Checkpoint(nonEmpty.Path));

        Assert.Equal(beforeGeneration, composite.Generation);
        AssertCompositeMatchesModel(composite, model);
        Assert.Equal(before, SearchComposite(composite, [0f, 0f, 0f], topK: 4));
        Assert.Equal("do-not-touch", File.ReadAllText(Path.Combine(nonEmpty.Path, "caller-owned.txt")));
        Assert.False(File.Exists(Path.Combine(nonEmpty.Path, HnswIndexStorage.ManifestFileName)));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AllDeletedAndNearAllDeletedCheckpointsFoldTombstonesAndKeepReservations(bool deleteEverything)
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var options = new HnswIndexOptions(4, 32, 16, 0x131CUL);
        Row[] rows =
        [
            new(0, [0f, 0f, 0f]),
            new(1, [0f, 0f, 0f]),
            new(ulong.MaxValue - 1, [1f, 1f, 1f]),
            new(ulong.MaxValue, [1f, 1f, 1f])
        ];
        var model = new ReferenceModel(3, rows);
        var composite = new HnswBasePlusExactDeltaIndex(CreateBaseIndex(3, options, rows));
        ApplyAdd(composite, model, 2, [0.5f, 0.5f, 0.5f]);
        ApplyAdd(composite, model, ulong.MaxValue - 10, [0.5f, 0.5f, 0.5f]);

        foreach (ulong id in deleteEverything
                     ? model.LiveIds.ToArray()
                     : model.LiveIds.Where(static id => id != ulong.MaxValue).ToArray())
        {
            ApplyDelete(composite, model, id);
        }

        Row[] expectedRows = model.CheckpointRows().ToArray();
        long beforeGeneration = composite.Generation;
        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(beforeGeneration + 1, result.Generation);
        Assert.Equal(expectedRows.Length, result.RebuiltBaseVectorCount);
        Assert.Equal(expectedRows.Length, result.LiveVectorCount);
        Assert.Equal(0, result.DeltaPhysicalVectorCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.True(result.FoldedBaseTombstoneCount > 0);
        Assert.True(result.FoldedDeltaTombstoneCount > 0);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        AssertRowsEqual(expectedRows, opened, dimension: 3);
        model.Generation++;
        model.PublishCheckpoint();
        AssertCompositeMatchesModel(composite, model);
        AssertCompositeAndOpenedSearchParity(composite, opened, dimension: 3);

        foreach (ulong id in model.DeletedIds)
        {
            long beforeDuplicate = composite.Generation;
            VectorMutationResult duplicate = composite.TryAdd(id, [9f, 9f, 9f]);
            Assert.Equal(VectorMutationStatus.DuplicateId, duplicate.Status);
            Assert.Equal(beforeDuplicate, duplicate.Generation);
            Assert.Equal(beforeDuplicate, composite.Generation);
        }
    }

    private static void CheckpointAndValidate(
        HnswBasePlusExactDeltaIndex composite,
        ReferenceModel model,
        int dimension,
        ref int checkpointOrdinal)
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        bool pending = model.HasPendingCheckpoint;
        Row[] expectedRows = model.CheckpointRows().ToArray();
        int foldedDeltaLive = model.DeltaLiveCount;
        int foldedBaseTombstones = model.BaseTombstoneCount;
        int foldedDeltaTombstones = model.DeltaTombstoneCount;
        long beforeGeneration = model.Generation;

        var staleWorkspace = CreateWorkspace(composite, topK: 3);
        _ = composite.Search(CreateQuery(dimension, checkpointOrdinal), new SearchResult[3], staleWorkspace);

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);
        checkpointOrdinal++;

        if (!pending)
        {
            Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, result.Status);
            Assert.Equal(beforeGeneration, result.Generation);
            Assert.Equal(beforeGeneration, composite.Generation);
            Assert.False(Directory.Exists(checkpoint.Path));
            _ = composite.Search(CreateQuery(dimension, checkpointOrdinal), new SearchResult[3], staleWorkspace);
            return;
        }

        model.Generation++;
        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Status);
        Assert.Equal(model.Generation, result.Generation);
        Assert.Equal(model.Generation, composite.Generation);
        Assert.Equal(expectedRows.Length, result.RebuiltBaseVectorCount);
        Assert.Equal(expectedRows.Length, result.LiveVectorCount);
        Assert.Equal(expectedRows.Length, result.BasePhysicalVectorCount);
        Assert.Equal(expectedRows.Length, result.BaseLiveVectorCount);
        Assert.Equal(0, result.DeltaPhysicalVectorCount);
        Assert.Equal(0, result.DeltaLiveVectorCount);
        Assert.Equal(0, result.BaseTombstoneCount);
        Assert.Equal(0, result.DeltaTombstoneCount);
        Assert.Equal(0, result.TombstoneCount);
        Assert.Equal(model.DeletedIds.Count, result.DeletedReservedIdCount);
        Assert.Equal(foldedDeltaLive, result.FoldedDeltaVectorCount);
        Assert.Equal(foldedBaseTombstones, result.FoldedBaseTombstoneCount);
        Assert.Equal(foldedDeltaTombstones, result.FoldedDeltaTombstoneCount);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        AssertRowsEqual(expectedRows, opened, dimension);
        AssertCompositeAndOpenedSearchParity(composite, opened, dimension);

        var sentinel = new[] { new SearchResult(111, 111), new SearchResult(222, 222), new SearchResult(333, 333) };
        float[] staleQuery = CreateQuery(dimension, checkpointOrdinal);
        Assert.Throws<InvalidOperationException>(() => composite.Search(staleQuery, sentinel, staleWorkspace));
        Assert.Equal([new SearchResult(111, 111), new SearchResult(222, 222), new SearchResult(333, 333)], sentinel);

        model.PublishCheckpoint();
        AssertCompositeMatchesModel(composite, model);

        foreach (ulong deletedId in model.DeletedIds.Take(2))
        {
            long beforeDuplicate = model.Generation;
            VectorMutationResult duplicate = composite.TryAdd(deletedId, CreateVector(dimension, (int)(deletedId & 0xFF), new Random(17)));
            Assert.Equal(VectorMutationStatus.DuplicateId, duplicate.Status);
            Assert.Equal(beforeDuplicate, duplicate.Generation);
            Assert.Equal(beforeDuplicate, composite.Generation);
        }
    }

    private static void ApplyAdd(
        HnswBasePlusExactDeltaIndex composite,
        ReferenceModel model,
        ulong id,
        float[] vector)
    {
        long beforeGeneration = model.Generation;
        bool duplicate = model.IsKnownOrReserved(id);
        VectorMutationResult result = composite.TryAdd(id, vector);
        if (duplicate)
        {
            Assert.Equal(VectorMutationStatus.DuplicateId, result.Status);
            Assert.Equal(beforeGeneration, result.Generation);
        }
        else
        {
            model.Add(id, vector);
            model.Generation++;
            Assert.Equal(VectorMutationStatus.Committed, result.Status);
            Assert.Equal(model.Generation, result.Generation);
        }

        AssertCompositeMatchesModel(composite, model);
    }

    private static void ApplyDelete(
        HnswBasePlusExactDeltaIndex composite,
        ReferenceModel model,
        ulong id)
    {
        long beforeGeneration = model.Generation;
        VectorMutationStatus expected = model.DeleteStatus(id);
        VectorMutationResult result = composite.TryDelete(id);
        Assert.Equal(expected, result.Status);
        if (expected == VectorMutationStatus.Committed)
        {
            model.Delete(id);
            model.Generation++;
            Assert.Equal(model.Generation, result.Generation);
        }
        else
        {
            Assert.Equal(beforeGeneration, result.Generation);
        }

        AssertCompositeMatchesModel(composite, model);
    }

    private static ulong SelectAddId(ReferenceModel model, ref int addOrdinal, Random random)
    {
        if (addOrdinal % 7 == 3 && model.DeletedIds.Count > 0)
        {
            addOrdinal++;
            return model.DeletedIds.ElementAt(random.Next(model.DeletedIds.Count));
        }

        if (addOrdinal % 7 == 5 && model.LiveIds.Count > 0)
        {
            addOrdinal++;
            return model.LiveIds[random.Next(model.LiveIds.Count)];
        }

        ulong candidate = (addOrdinal % 4) switch
        {
            0 => checked((ulong)(5 + addOrdinal)),
            1 => checked(10_000UL + (ulong)addOrdinal),
            2 => ulong.MaxValue - 100 - checked((ulong)addOrdinal),
            _ => checked(1UL << 40) + (ulong)addOrdinal
        };
        addOrdinal++;
        while (model.IsKnownOrReserved(candidate))
        {
            candidate++;
        }

        return candidate;
    }

    private static ulong SelectDeleteId(ReferenceModel model, Random random)
    {
        if (model.LiveIds.Count > 0 && random.Next(0, 5) != 0)
        {
            return model.LiveIds[random.Next(model.LiveIds.Count)];
        }

        if (model.DeletedIds.Count > 0 && random.Next(0, 2) == 0)
        {
            return model.DeletedIds.ElementAt(random.Next(model.DeletedIds.Count));
        }

        return 0xDEAD_0000UL + checked((ulong)random.Next(0, 10_000));
    }

    private static Row[] CreateInitialRows(int dimension, Random random)
    {
        ulong[] ids =
        [
            0,
            1,
            2,
            3,
            42,
            65_535,
            1UL << 40,
            ulong.MaxValue - 3,
            ulong.MaxValue - 1,
            ulong.MaxValue
        ];
        var rows = new Row[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            float[] vector = CreateVector(dimension, i, random);
            if (i is 3 or 8)
            {
                vector = rows[i - 1].Vector.ToArray();
            }

            rows[i] = new Row(ids[i], vector);
        }

        return rows;
    }

    private static HnswIndex CreateBaseIndex(int dimension, HnswIndexOptions options, IEnumerable<Row> rows)
    {
        var index = new HnswIndex(dimension, VectorMetric.SquaredEuclidean, options, () => 0);
        foreach (Row row in rows)
        {
            index.Add(row.Id, row.Vector);
        }

        return index;
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK);

    private static SearchResult[] SearchComposite(HnswBasePlusExactDeltaIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, CreateWorkspace(index, topK));
        return results[..written];
    }

    private static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static void AssertCompositeAndOpenedSearchParity(
        HnswBasePlusExactDeltaIndex composite,
        HnswIndex opened,
        int dimension)
    {
        int topK = Math.Min(12, composite.Options.EfSearch);
        for (int i = 0; i < 4; i++)
        {
            float[] query = CreateQuery(dimension, i);
            Assert.Equal(SearchHnsw(opened, query, topK), SearchComposite(composite, query, topK));
        }
    }

    private static void AssertCompositeSearchReturnsOnlyLiveFiniteIds(
        HnswBasePlusExactDeltaIndex composite,
        ReferenceModel model,
        float[] query,
        int topK)
    {
        HashSet<ulong> live = model.LiveIds.ToHashSet();
        foreach (SearchResult result in SearchComposite(composite, query, topK))
        {
            Assert.Contains(result.Id, live);
            Assert.True(float.IsFinite(result.Distance), $"Distance for {result.Id} was not finite.");
        }
    }

    private static void AssertCompositeMatchesModel(HnswBasePlusExactDeltaIndex composite, ReferenceModel model)
    {
        Assert.Equal(model.Dimension, composite.Dimension);
        Assert.Equal(VectorMetric.SquaredEuclidean, composite.Metric);
        Assert.Equal(model.Generation, composite.Generation);
        Assert.Equal(model.BasePhysicalCount, composite.BasePhysicalVectorCount);
        Assert.Equal(model.BaseLiveCount, composite.BaseLiveVectorCount);
        Assert.Equal(model.DeltaPhysicalCount, composite.DeltaPhysicalVectorCount);
        Assert.Equal(model.DeltaLiveCount, composite.DeltaLiveVectorCount);
        Assert.Equal(model.BaseTombstoneCount, composite.BaseTombstoneCount);
        Assert.Equal(model.DeltaTombstoneCount, composite.DeltaTombstoneCount);
        Assert.Equal(model.BaseTombstoneCount + model.DeltaTombstoneCount, composite.TombstoneCount);
        Assert.Equal(model.LiveIds.Count, composite.LiveVectorCount);
        Assert.Equal(model.DeletedIds.Count, composite.DeletedReservedIdCount);
    }

    private static void AssertRowsEqual(Row[] expectedRows, HnswIndex opened, int dimension)
    {
        Assert.Equal(expectedRows.Length, opened.Count);
        Assert.Equal(expectedRows.Select(static row => row.Id).ToArray(), opened.InternalIds.ToArray());
        float[] expectedVectors = expectedRows.SelectMany(static row => row.Vector).ToArray();
        Assert.Equal(expectedVectors.Length, checked(opened.Count * dimension));
        Assert.Equal(expectedVectors, opened.InternalVectors.ToArray());
    }

    private static float[] CreateVector(int dimension, int salt, Random random)
    {
        var vector = new float[dimension];
        if (salt % 11 == 0)
        {
            for (int i = 0; i < dimension; i++)
            {
                vector[i] = 1.25f + (i % 3) * 0.125f;
            }

            return vector;
        }

        for (int i = 0; i < dimension; i++)
        {
            vector[i] =
                ((salt % 17) - 8) * 0.25f +
                (i % 13) * 0.03125f +
                random.Next(-3, 4) * 0.00390625f;
        }

        return vector;
    }

    private static float[] CreateQuery(int dimension, int salt)
    {
        var query = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            query[i] = ((salt + i) % 9 - 4) * 0.0625f;
        }

        return query;
    }

    private sealed class ReferenceModel
    {
        public ReferenceModel(int dimension, IEnumerable<Row> baseRows)
        {
            Dimension = dimension;
            BaseRows = baseRows.Select(static row => row.Copy()).ToList();
        }

        public int Dimension { get; }

        public long Generation { get; set; }

        public List<Row> BaseRows { get; private set; }

        public List<Row> DeltaRows { get; } = [];

        public HashSet<ulong> DeletedIds { get; } = [];

        public List<ulong> LiveIds => BaseRows.Concat(DeltaRows)
            .Where(row => !DeletedIds.Contains(row.Id))
            .Select(static row => row.Id)
            .ToList();

        public int BasePhysicalCount => BaseRows.Count;

        public int BaseLiveCount => BaseRows.Count(row => !DeletedIds.Contains(row.Id));

        public int DeltaPhysicalCount => DeltaRows.Count;

        public int DeltaLiveCount => DeltaRows.Count(row => !DeletedIds.Contains(row.Id));

        public int BaseTombstoneCount => BaseRows.Count(row => DeletedIds.Contains(row.Id));

        public int DeltaTombstoneCount => DeltaRows.Count(row => DeletedIds.Contains(row.Id));

        public bool HasPendingCheckpoint => DeltaRows.Count > 0 || BaseTombstoneCount > 0;

        public bool IsKnownOrReserved(ulong id) =>
            DeletedIds.Contains(id) || BaseRows.Any(row => row.Id == id) || DeltaRows.Any(row => row.Id == id);

        public void Add(ulong id, float[] vector) => DeltaRows.Add(new Row(id, vector.ToArray()));

        public VectorMutationStatus DeleteStatus(ulong id)
        {
            if (DeletedIds.Contains(id))
            {
                return VectorMutationStatus.AlreadyDeleted;
            }

            return BaseRows.Concat(DeltaRows).Any(row => row.Id == id)
                ? VectorMutationStatus.Committed
                : VectorMutationStatus.UnknownId;
        }

        public void Delete(ulong id) => DeletedIds.Add(id);

        public IEnumerable<Row> CheckpointRows() =>
            BaseRows.Concat(DeltaRows)
                .Where(row => !DeletedIds.Contains(row.Id))
                .Select(static row => row.Copy());

        public void PublishCheckpoint()
        {
            BaseRows = CheckpointRows().ToList();
            DeltaRows.Clear();
        }
    }

    private sealed record Row(ulong Id, float[] Vector)
    {
        public Row Copy() => new(Id, Vector.ToArray());
    }

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
                "VecNet-HnswCompositeCheckpointIndependentTests-" + Guid.NewGuid().ToString("N"));
    }
}
