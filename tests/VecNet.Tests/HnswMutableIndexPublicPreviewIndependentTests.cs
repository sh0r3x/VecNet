using System.Reflection;
using System.Xml.Linq;

namespace VecNet.Tests;

public sealed class HnswMutableIndexPublicPreviewIndependentTests
{
    [Fact]
    public void PublicMutableWrapper_ReportsConflictStatusesAndReservesDeletedDeltaIdsAcrossCheckpoint()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory noChangesCheckpoint = TempIndexDirectory.CreateMissing();
        var mutable = new HnswMutableIndex(CreateBaseIndex(
            [(10UL, new[] { 0f, 0f }), (20UL, new[] { 2f, 0f })],
            efSearch: 8));

        AssertMutation(VectorMutationStatus.DuplicateId, generation: 0, live: 2, delta: 0, tombstones: 0, mutable.TryAdd(10, [9f, 9f]));
        AssertMutation(VectorMutationStatus.UnknownId, generation: 0, live: 2, delta: 0, tombstones: 0, mutable.TryDelete(999));

        AssertMutation(VectorMutationStatus.Committed, generation: 1, live: 3, delta: 1, tombstones: 0, mutable.TryAdd(30, [0.25f, 0f]));
        AssertMutation(VectorMutationStatus.DuplicateId, generation: 1, live: 3, delta: 1, tombstones: 0, mutable.TryAdd(30, [0.25f, 0f]));

        AssertMutation(VectorMutationStatus.Committed, generation: 2, live: 2, delta: 0, tombstones: 1, mutable.TryDelete(30));
        AssertMutation(VectorMutationStatus.AlreadyDeleted, generation: 2, live: 2, delta: 0, tombstones: 1, mutable.TryDelete(30));
        AssertMutation(VectorMutationStatus.DuplicateId, generation: 2, live: 2, delta: 0, tombstones: 1, mutable.TryAdd(30, [0.25f, 0f]));

        HnswMutableCheckpointResult published = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, published.Status);
        Assert.Equal(3, published.Generation);
        Assert.Equal(2, published.LiveVectorCount);
        Assert.Equal(0, published.FoldedDeltaVectorCount);
        Assert.Equal(0, published.FoldedBaseTombstoneCount);
        Assert.Equal(1, published.FoldedDeltaTombstoneCount);
        Assert.Equal(1, published.DeletedReservedIdCount);
        Assert.Equal(0, mutable.DeltaPhysicalVectorCount);
        Assert.Equal(0, mutable.TombstoneCount);
        Assert.Equal(1, mutable.DeletedReservedIdCount);

        AssertMutation(VectorMutationStatus.DuplicateId, generation: 3, live: 2, delta: 0, tombstones: 0, mutable.TryAdd(30, [0.25f, 0f]));

        HnswMutableCheckpointResult noChanges = mutable.Checkpoint(noChangesCheckpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.NoChanges, noChanges.Status);
        Assert.Equal(3, noChanges.Generation);
        Assert.Equal(2, noChanges.LiveVectorCount);
        Assert.False(Directory.Exists(noChangesCheckpoint.Path));
    }

    [Fact]
    public void PublicMutableWrapper_BroadAllowlistSuppressesUnknownDuplicatesAndTombstones()
    {
        HnswIndex baseIndex = CreateBaseIndex(
            [
                (10UL, new[] { 0f, 0f }),
                (20UL, new[] { 1f, 0f }),
                (30UL, new[] { 4f, 0f }),
                (40UL, new[] { 8f, 0f })
            ],
            efSearch: 3);
        var mutable = new HnswMutableIndex(baseIndex);

        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.1f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(35, [7f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);

        ulong[] allowed = [999, 20, 10, 15, 15, 30, 35, 40];
        var results = new SearchResult[2];
        var workspace = new HnswMutableSearchWorkspace(mutable, maxResults: results.Length);

        int written = mutable.Search([0f, 0f], allowed, results, workspace);

        Assert.Equal(2, written);
        Assert.Equal([10UL, 15UL], results[..written].Select(static result => result.Id).OrderBy(static id => id));
        Assert.All(results[..written], result =>
        {
            Assert.Contains(result.Id, new[] { 10UL, 15UL, 30UL, 35UL, 40UL });
            Assert.NotEqual(20UL, result.Id);
            Assert.NotEqual(999UL, result.Id);
            Assert.True(float.IsFinite(result.Distance));
        });
        Assert.Equal(results[..written].Length, results[..written].Select(static result => result.Id).Distinct().Count());
    }

    [Fact]
    public void PublicMutableAllowlistSearch_RejectsStaleOrUndersizedWorkspaceBeforeWritingDestination()
    {
        var mutable = new HnswMutableIndex(CreateBaseIndex([(10UL, new[] { 0f }), (20UL, new[] { 2f })], efSearch: 8));
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.5f]).Status);

        SearchResult[] destination = [new(701, -701), new(702, -702)];
        SearchResult[] original = destination.ToArray();

        var tooSmall = new HnswMutableSearchWorkspace(mutable, maxResults: 1);
        Assert.Throws<ArgumentException>(() => mutable.Search([0f], [10UL, 15UL], destination, tooSmall));
        Assert.Equal(original, destination);

        var stale = new HnswMutableSearchWorkspace(mutable, maxResults: 2);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);

        Assert.Throws<InvalidOperationException>(() => mutable.Search([0f], [10UL, 15UL], destination, stale));
        Assert.Equal(original, destination);
    }

    [Fact]
    public void PublicMutableWrapper_DoesNotExposeInternalCompositeOrDiagnosticsInPublicSignatures()
    {
        Assembly assembly = typeof(HnswMutableIndex).Assembly;
        Type[] publicHnswTypes = assembly.GetExportedTypes()
            .Where(static type => type.Namespace == "VecNet" && type.Name.Contains("Hnsw", StringComparison.Ordinal))
            .ToArray();

        Assert.DoesNotContain(publicHnswTypes, static type => type.Name.Contains("BasePlusExactDelta", StringComparison.Ordinal));
        Assert.DoesNotContain(publicHnswTypes, static type => type.Name.Contains("Diagnostic", StringComparison.Ordinal));

        foreach (MethodInfo method in typeof(HnswMutableIndex).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            Assert.DoesNotContain("Diagnostic", method.Name, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("BasePlusExactDelta", method.ReturnType.Name, StringComparison.Ordinal);
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                Assert.DoesNotContain("BasePlusExactDelta", parameter.ParameterType.Name, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void PublicHnswReadmeAndXml_DoNotIntroduceForbiddenNumericOrComparisonClaims()
    {
        string readme = File.ReadAllText(FindRepositoryFile("README.md"));
        string xml = File.ReadAllText(Path.ChangeExtension(typeof(HnswIndex).Assembly.Location, ".xml"));
        string combined = readme + Environment.NewLine + XDocument.Parse(xml);

        string[] forbiddenPhrases =
        [
            "zero allocation",
            "faster than hnswlib",
            "faster than FAISS",
            "beats hnswlib",
            "beats FAISS",
            "recall@",
            "QPS",
            "queries per second",
            "bytes per vector"
        ];

        foreach (string phrase in forbiddenPhrases)
        {
            Assert.DoesNotContain(phrase, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("does not make public HNSW recall", readme, StringComparison.Ordinal);
        Assert.Contains("not public", xml, StringComparison.Ordinal);
        Assert.Contains("allocation", xml, StringComparison.Ordinal);
        Assert.Contains("storage-size", xml, StringComparison.Ordinal);
    }

    private static HnswIndex CreateBaseIndex(IEnumerable<(ulong Id, float[] Vector)> rows, int efSearch)
    {
        (ulong Id, float[] Vector)[] materialized = rows.ToArray();
        int dimension = materialized.Length == 0 ? 1 : materialized[0].Vector.Length;
        var index = new HnswIndex(
            dimension,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(M: 2, EfConstruction: 8, EfSearch: efSearch, RandomSeed: 0x153A11UL));

        foreach ((ulong id, float[] vector) in materialized)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static void AssertMutation(
        VectorMutationStatus status,
        long generation,
        int live,
        int delta,
        int tombstones,
        VectorMutationResult result)
    {
        Assert.Equal(status, result.Status);
        Assert.Equal(generation, result.Generation);
        Assert.Equal(live, result.LiveVectorCount);
        Assert.Equal(delta, result.DeltaVectorCount);
        Assert.Equal(tombstones, result.TombstoneCount);
    }

    private static string FindRepositoryFile(string fileName)
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, fileName);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException($"Could not locate {fileName} from {AppContext.BaseDirectory}.");
    }

    private sealed class TempIndexDirectory : IDisposable
    {
        private TempIndexDirectory(string path) => Path = path;

        public string Path { get; }

        public static TempIndexDirectory CreateMissing()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet-HnswMutableIndependent-" + Guid.NewGuid().ToString("N"));
            return new TempIndexDirectory(path);
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
    }
}
