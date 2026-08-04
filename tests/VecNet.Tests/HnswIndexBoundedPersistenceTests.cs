using System.Buffers.Binary;
using System.Text.Json;

namespace VecNet.Tests;

public sealed class HnswIndexBoundedPersistenceTests
{
    [Fact]
    public void SaveOpen_StreamingWriterPreservesDurableFormatGraphAndSearchParity()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex();
        string expectedGraph = GraphSnapshot(source);
        SearchResult[] expectedSearch = Search(source, [2.25f, 0.5f], topK: 5);

        source.Save(saved.Path);

        using JsonDocument manifest = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(saved.Path, HnswIndexStorage.ManifestFileName)));
        Assert.False(manifest.RootElement.TryGetProperty("createdByTask", out _));
        Assert.False(manifest.RootElement.TryGetProperty("evidence", out _));
        Assert.Equal(
            "dense-layer0-sparse-upper-v1",
            manifest.RootElement.GetProperty("hnsw").GetProperty("graph").GetProperty("adjacencyLayout").GetString());
        Assert.Equal(
            "SplitMix64",
            manifest.RootElement.GetProperty("hnsw").GetProperty("graph").GetProperty("levelGenerator").GetString());

        HnswIndex opened = HnswIndex.OpenReadOnly(saved.Path);

        Assert.Equal(source.InternalIds.ToArray(), opened.InternalIds.ToArray());
        Assert.Equal(expectedGraph, GraphSnapshot(opened));
        Assert.Equal(expectedSearch, Search(opened, [2.25f, 0.5f], topK: 5));
    }

    [Fact]
    public void ValidateSavedIndex_RejectsCorruptOutputWithoutHydratingOpenedIndex()
    {
        using TempIndexDirectory saved = TempIndexDirectory.CreateMissing();
        HnswIndex source = CreateLayeredIndex();
        source.Save(saved.Path);
        HnswIndexStorage.ValidateSavedIndex(saved.Path, source);

        string graphPath = Path.Combine(saved.Path, HnswIndexStorage.GraphFileName);
        byte[] graph = File.ReadAllBytes(graphPath);
        int layerZeroEntry = HnswIndexStorage.GraphHeaderLength;
        int neighborOffset = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(
            graph.AsSpan(layerZeroEntry + 32)));
        BinaryPrimitives.WriteInt32LittleEndian(graph.AsSpan(neighborOffset), 123456);
        File.WriteAllBytes(graphPath, graph);

        Assert.Throws<InvalidDataException>(() => HnswIndexStorage.ValidateSavedIndex(saved.Path, source));
        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(saved.Path));
    }

    [Fact]
    public void MutableCheckpoint_StreamsValidationFoldsTombstonesPreservesReservationsAndCleansTemps()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        HnswIndex baseIndex = CreateBaseIndex();
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(15, [0.5f, 0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryAdd(35, [2.5f, 2.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(20).Status);
        Assert.Equal(VectorMutationStatus.Committed, composite.TryDelete(35).Status);

        HnswBasePlusExactDeltaCheckpointDiagnosticResult measured =
            composite.CheckpointWithDiagnostics(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.Published, measured.Result.Status);
        Assert.Equal(1, measured.Result.FoldedDeltaVectorCount);
        Assert.Equal(1, measured.Result.FoldedBaseTombstoneCount);
        Assert.Equal(1, measured.Result.FoldedDeltaTombstoneCount);
        Assert.Equal(2, measured.Result.DeletedReservedIdCount);
        Assert.Equal(0, measured.Result.TombstoneCount);
        Assert.Equal(0, measured.Result.DeltaPhysicalVectorCount);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(20, [9f, 9f]).Status);
        Assert.Equal(VectorMutationStatus.DuplicateId, composite.TryAdd(35, [9f, 9f]).Status);
        AssertMeasured(measured.Diagnostics.LiveSnapshot);
        AssertMeasured(measured.Diagnostics.RebuildBuild);
        AssertMeasured(measured.Diagnostics.Save);
        AssertMeasured(measured.Diagnostics.OpenValidation);
        AssertMeasured(measured.Diagnostics.Publication);
        Assert.DoesNotContain(
            Directory.EnumerateFileSystemEntries(checkpoint.Path),
            path => Path.GetFileName(path).Contains(".tmp-", StringComparison.Ordinal));

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal([10UL, 30UL, 15UL], opened.InternalIds.ToArray());
        Assert.Equal(SearchComposite(composite, [0f, 0f], topK: 3), Search(opened, [0f, 0f], topK: 3));
    }

    [Fact]
    public void MutableCheckpoint_NoChangesReturnsWithoutCreatingOutput()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        var composite = new HnswBasePlusExactDeltaIndex(CreateBaseIndex());

        HnswBasePlusExactDeltaCheckpointResult result = composite.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswBasePlusExactDeltaCheckpointStatus.NoChanges, result.Status);
        Assert.False(Directory.Exists(checkpoint.Path));
    }

    private static HnswIndex CreateLayeredIndex()
    {
        int[] levels = [0, 2, 0, 1, 0, 2, 0, 1];
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1800UL),
            levels.Length,
            () => levels[nextLevel++]);

        for (int i = 0; i < levels.Length; i++)
        {
            index.Add((ulong)(100 + i), [i, i % 3]);
        }

        return index;
    }

    private static HnswIndex CreateBaseIndex()
    {
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x1801UL),
            initialCapacity: 3);
        index.Add(10, [0f, 0f]);
        index.Add(20, [1f, 1f]);
        index.Add(30, [2f, 2f]);
        return index;
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results.AsSpan(0, written).ToArray();
    }

    private static SearchResult[] SearchComposite(
        HnswBasePlusExactDeltaIndex index,
        float[] query,
        int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(
            query,
            results,
            new HnswBasePlusExactDeltaSearchWorkspace(
                index.BasePhysicalVectorCount,
                index.Options.EfSearch,
                Math.Min(index.BasePhysicalVectorCount, index.Options.EfSearch),
                topK,
                index.DeltaPhysicalVectorCount));
        return results.AsSpan(0, written).ToArray();
    }

    private static string GraphSnapshot(HnswIndex index)
    {
        using var writer = new StringWriter();
        writer.Write(index.EntryPoint);
        writer.Write('|');
        writer.Write(index.MaxLayer);
        int[] neighbors = new int[index.Options.M * 2];
        for (int layer = 0; layer <= index.MaxLayer; layer++)
        {
            writer.Write("|L");
            writer.Write(layer);
            for (int ordinal = 0; ordinal < index.Count; ordinal++)
            {
                int count = index.DebugGetNeighbors(layer, ordinal, neighbors);
                writer.Write(':');
                writer.Write(ordinal);
                writer.Write('[');
                for (int i = 0; i < count; i++)
                {
                    if (i > 0)
                    {
                        writer.Write(',');
                    }

                    writer.Write(neighbors[i]);
                }

                writer.Write(']');
            }
        }

        return writer.ToString();
    }

    private static void AssertMeasured(HnswBasePlusExactDeltaCheckpointPhaseDiagnostics diagnostics)
    {
        Assert.Equal(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured, diagnostics.Status);
        Assert.True(diagnostics.ElapsedTicks >= 0);
        Assert.True(diagnostics.ManagedAllocatedBytes >= 0);
    }

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
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), "VecNetTests-" + Guid.NewGuid().ToString("N"));
    }
}
