namespace VecNet.Tests;

public sealed class HnswBasePlusExactDeltaCheckpointDiagnosticsIndependentTests
{
    [Fact]
    public void CheckpointAndCheckpointWithDiagnosticsPublishEquivalentResultsForEquivalentStates()
    {
        using TempPath plainOutput = TempPath.CreateMissingDirectory();
        using TempPath diagnosticOutput = TempPath.CreateMissingDirectory();
        HnswBasePlusExactDeltaIndex plain = CreateMutatedComposite();
        HnswBasePlusExactDeltaIndex measured = CreateMutatedComposite();

        HnswBasePlusExactDeltaCheckpointResult plainResult = plain.Checkpoint(plainOutput.Path);
        HnswBasePlusExactDeltaCheckpointDiagnosticResult diagnosticResult =
            measured.CheckpointWithDiagnostics(diagnosticOutput.Path);

        Assert.Equal(plainResult, diagnosticResult.Result);
        Assert.Equal(plainResult.Generation, plain.Generation);
        Assert.Equal(diagnosticResult.Result.Generation, measured.Generation);
        AssertMeasured(diagnosticResult.Diagnostics);

        HnswIndex plainOpened = HnswIndex.OpenReadOnly(plainOutput.Path);
        HnswIndex diagnosticOpened = HnswIndex.OpenReadOnly(diagnosticOutput.Path);
        Assert.Equal(plainOpened.InternalIds.ToArray(), diagnosticOpened.InternalIds.ToArray());
        Assert.Equal(plainOpened.InternalVectors.ToArray(), diagnosticOpened.InternalVectors.ToArray());
        Assert.Equal(SearchComposite(plain, [0f, 0f, 0f], 8), SearchComposite(measured, [0f, 0f, 0f], 8));
        Assert.Equal(SearchHnsw(plainOpened, [0f, 0f, 0f], 8), SearchHnsw(diagnosticOpened, [0f, 0f, 0f], 8));
    }

    [Fact]
    public void PublishedDiagnosticsMeasureEveryCheckpointPhaseWithNonNegativeValues()
    {
        using TempPath output = TempPath.CreateMissingDirectory();
        HnswBasePlusExactDeltaIndex composite = CreateMutatedComposite();

        HnswBasePlusExactDeltaCheckpointDiagnosticResult result = composite.CheckpointWithDiagnostics(output.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, result.Result.Status);
        AssertMeasured(result.Diagnostics.LiveSnapshot);
        AssertMeasured(result.Diagnostics.RebuildBuild);
        AssertMeasured(result.Diagnostics.Save);
        AssertMeasured(result.Diagnostics.OpenValidation);
        AssertMeasured(result.Diagnostics.Publication);
        Assert.True(File.Exists(Path.Combine(output.Path, HnswIndexStorage.ManifestFileName)));
    }

    [Fact]
    public void NoChangesDiagnosticsMarkAllPhasesNotExecutedWithoutOutputOrGenerationMovement()
    {
        using TempPath output = TempPath.CreateMissingDirectory();
        HnswBasePlusExactDeltaIndex composite = new(CreateBaseIndex());
        long beforeGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointDiagnosticResult result = composite.CheckpointWithDiagnostics(output.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, result.Result.Status);
        Assert.Equal(beforeGeneration, result.Result.Generation);
        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.False(Directory.Exists(output.Path));
        AssertNotExecuted(result.Diagnostics.LiveSnapshot);
        AssertNotExecuted(result.Diagnostics.RebuildBuild);
        AssertNotExecuted(result.Diagnostics.Save);
        AssertNotExecuted(result.Diagnostics.OpenValidation);
        AssertNotExecuted(result.Diagnostics.Publication);
    }

    [Fact]
    public void SaveFailureBeforePublicationPreservesOldCompositeState()
    {
        using TempPath blockingFile = TempPath.CreateFile();
        HnswBasePlusExactDeltaIndex composite = CreateMutatedComposite();
        SearchResult[] beforeSearch = SearchComposite(composite, [0f, 0f, 0f], 8);
        long beforeGeneration = composite.Generation;
        int beforeBasePhysical = composite.BasePhysicalVectorCount;
        int beforeBaseLive = composite.BaseLiveVectorCount;
        int beforeDeltaPhysical = composite.DeltaPhysicalVectorCount;
        int beforeDeltaLive = composite.DeltaLiveVectorCount;
        int beforeTombstones = composite.TombstoneCount;
        int beforeReserved = composite.DeletedReservedIdCount;
        string blockedChildPath = Path.Combine(blockingFile.Path, "checkpoint");

        Assert.ThrowsAny<IOException>(() => composite.CheckpointWithDiagnostics(blockedChildPath));

        Assert.Equal(beforeGeneration, composite.Generation);
        Assert.Equal(beforeBasePhysical, composite.BasePhysicalVectorCount);
        Assert.Equal(beforeBaseLive, composite.BaseLiveVectorCount);
        Assert.Equal(beforeDeltaPhysical, composite.DeltaPhysicalVectorCount);
        Assert.Equal(beforeDeltaLive, composite.DeltaLiveVectorCount);
        Assert.Equal(beforeTombstones, composite.TombstoneCount);
        Assert.Equal(beforeReserved, composite.DeletedReservedIdCount);
        Assert.Equal(beforeSearch, SearchComposite(composite, [0f, 0f, 0f], 8));
        Assert.False(Directory.Exists(blockedChildPath));
    }

    [Fact]
    public void DiagnosticsPathPreservesDeletedIdReservationBehavior()
    {
        using TempPath output = TempPath.CreateMissingDirectory();
        HnswBasePlusExactDeltaIndex composite = CreateMutatedComposite();
        long beforeCheckpointGeneration = composite.Generation;

        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpoint = composite.CheckpointWithDiagnostics(output.Path);

        Assert.Equal(beforeCheckpointGeneration + 1, checkpoint.Result.Generation);
        Assert.Equal(2, checkpoint.Result.DeletedReservedIdCount);
        long afterCheckpointGeneration = composite.Generation;
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f, 9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(35, [9f, 9f, 9f]).Status);
        Assert.Equal(afterCheckpointGeneration, composite.Generation);
    }

    [Fact]
    public void DiagnosticsPathPreservesRebuiltCompositeAndOpenedHnswSearchParity()
    {
        using TempPath output = TempPath.CreateMissingDirectory();
        HnswBasePlusExactDeltaIndex composite = CreateMutatedComposite();

        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpoint = composite.CheckpointWithDiagnostics(output.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(output.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, checkpoint.Result.Status);
        float[][] queries =
        [
            [0f, 0f, 0f],
            [1f, 1f, 1f],
            [2.25f, 2.25f, 2.25f],
            [4f, 4f, 4f]
        ];
        foreach (float[] query in queries)
        {
            Assert.Equal(SearchComposite(composite, query, 8), SearchHnsw(opened, query, 8));
        }
    }

    [Fact]
    public void CheckpointDiagnosticTypesRemainInternal()
    {
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnosticResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointPhaseStatus).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointResult).IsNotPublic);
        Assert.True(typeof(HnswBasePlusExactDeltaCheckpointStatus).IsNotPublic);
    }

    private static HnswBasePlusExactDeltaIndex CreateMutatedComposite()
    {
        var composite = new HnswBasePlusExactDeltaIndex(CreateBaseIndex());
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f, 0.5f, 0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(35, [2.5f, 2.5f, 2.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(45, [3.5f, 3.5f, 3.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(35).Status);
        return composite;
    }

    private static HnswIndex CreateBaseIndex()
    {
        var index = new HnswIndex(
            3,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(4, 32, 32, 0x5133UL),
            () => 0);
        index.Add(10, [0f, 0f, 0f]);
        index.Add(20, [1f, 1f, 1f]);
        index.Add(30, [2f, 2f, 2f]);
        index.Add(40, [3f, 3f, 3f]);
        index.Add(50, [4f, 4f, 4f]);
        return index;
    }

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

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(
        HnswBasePlusExactDeltaIndex index,
        int topK) =>
        new(
            index.BasePhysicalVectorCount,
            index.Options.EfSearch,
            Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
            topK);

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointDiagnostics diagnostics)
    {
        AssertMeasured(diagnostics.LiveSnapshot);
        AssertMeasured(diagnostics.RebuildBuild);
        AssertMeasured(diagnostics.Save);
        AssertMeasured(diagnostics.OpenValidation);
        AssertMeasured(diagnostics.Publication);
    }

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        Assert.Equal(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured, diagnostics.Status);
        Assert.True(diagnostics.ElapsedTicks >= 0);
        Assert.True(diagnostics.ManagedAllocatedBytes >= 0);
    }

    private static void AssertNotExecuted(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        Assert.Equal(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted, diagnostics.Status);
        Assert.Equal(0, diagnostics.ElapsedTicks);
        Assert.Equal(0, diagnostics.ManagedAllocatedBytes);
    }

    private sealed class TempPath : IDisposable
    {
        private TempPath(string path) => Path = path;

        public string Path { get; }

        public static TempPath CreateMissingDirectory() => new(CreatePath());

        public static TempPath CreateFile()
        {
            string path = CreatePath();
            File.WriteAllText(path, "blocking file");
            return new TempPath(path);
        }

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
                "VecNet-HnswCompositeDiagnosticsIndependentTests-" + Guid.NewGuid().ToString("N"));
    }
}
