using System.Reflection;
using System.Xml.Linq;

namespace VecNet.Tests;

public sealed class HnswIndexPublicPreviewIndependentTests
{
    [Fact]
    public void PublicHnswExportedTypes_DoNotExposeDebugStorageOrGraphInternals()
    {
        Type[] exportedHnswTypes = typeof(HnswIndex).Assembly
            .GetExportedTypes()
            .Where(static type => type.Namespace == "VecNet" && type.Name.Contains("Hnsw", StringComparison.Ordinal))
            .OrderBy(static type => type.FullName, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(
            [
                typeof(HnswIndex),
                typeof(HnswIndexOptions),
                typeof(HnswMutableCheckpointResult),
                typeof(HnswMutableCheckpointStatus),
                typeof(HnswMutableIndex),
                typeof(HnswMutableSearchWorkspace),
                typeof(HnswSearchWorkspace)
            ],
            exportedHnswTypes);

        Assert.Empty(typeof(HnswIndex).GetNestedTypes(BindingFlags.Public));
        Assert.DoesNotContain(
            typeof(HnswIndex).Assembly.GetTypes(),
            static type => type.IsPublic &&
                (type.Name.Contains("Debug", StringComparison.Ordinal) ||
                 type.Name.Contains("Storage", StringComparison.Ordinal) ||
                 type.Name.Contains("Graph", StringComparison.Ordinal) ||
                 type.Name.Contains("Priority", StringComparison.Ordinal)));
    }

    [Fact]
    public void PublicSearch_InvalidWorkspaceOrRequestedCountRejectsBeforeWritingResults()
    {
        var options = new HnswIndexOptions(4, 8, 2, 0x5EED_1151UL);
        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options);
        index.Add(10, [0f, 0f]);
        index.Add(20, [2f, 0f]);

        SearchResult[] results =
        [
            new(ulong.MaxValue, -1f),
            new(ulong.MaxValue - 1, -2f),
            new(ulong.MaxValue - 2, -3f)
        ];
        SearchResult[] original = results.ToArray();

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], results.AsSpan(0, 1), new HnswSearchWorkspace(maxElements: 1, maxEf: options.EfSearch)));
        Assert.Equal(original, results);

        Assert.Throws<ArgumentException>(
            () => index.Search([0f, 0f], results.AsSpan(0, 1), new HnswSearchWorkspace(index.Count, maxEf: options.EfSearch - 1)));
        Assert.Equal(original, results);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => index.Search([0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch)));
        Assert.Equal(original, results);

        Assert.Throws<ArgumentNullException>(
            () => index.Search([0f, 0f], results.AsSpan(0, 1), workspace: null!));
        Assert.Equal(original, results);
    }

    [Fact]
    public void PublicOpenedReadOnlyIndex_FailedAddsPreserveSearchableState()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(4, 16, 8, 0x5EED_1152UL);
        var source = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);
        source.Add(100, [0f, 0f, 0f]);
        source.Add(200, [2f, 0f, 0f]);
        source.Add(300, [0f, 3f, 0f]);

        SearchResult[] expected = Search(source, [0.25f, 0f, 0f], topK: 3);
        source.Save(temp.Path);

        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);
        Assert.Equal(3, opened.Count);

        Assert.Throws<InvalidOperationException>(() => opened.Add(400, [1f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(400, [float.NaN, 0f, 0f]));
        Assert.Throws<InvalidOperationException>(() => opened.Add(100, [9f, 9f, 9f]));

        Assert.Equal(3, opened.Count);
        Assert.Equal(expected, Search(opened, [0.25f, 0f, 0f], topK: 3));
    }

    [Fact]
    public void PublicConstruction_UnsupportedMetricsFailForDefaultAndExplicitOptions()
    {
        foreach (VectorMetric metric in new[] { VectorMetric.InnerProduct, VectorMetric.Cosine })
        {
            NotSupportedException defaultException = Assert.Throws<NotSupportedException>(
                () => new HnswIndex(3, metric));
            NotSupportedException explicitException = Assert.Throws<NotSupportedException>(
                () => new HnswIndex(3, metric, HnswIndexOptions.Default));

            Assert.Contains("squared Euclidean", defaultException.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("squared Euclidean", explicitException.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void GeneratedXmlDocumentation_IncludesEveryPublicHnswMember()
    {
        string xmlPath = Path.ChangeExtension(typeof(HnswIndex).Assembly.Location, ".xml");
        Assert.True(File.Exists(xmlPath), $"Expected generated XML docs beside VecNet assembly: {xmlPath}");

        XDocument document = XDocument.Load(xmlPath);
        HashSet<string> members = document
            .Descendants("member")
            .Select(static element => (string?)element.Attribute("name"))
            .Where(static name => name is not null)
            .Select(static name => name!)
            .ToHashSet(StringComparer.Ordinal);

        string[] expectedMembers =
        [
            "T:VecNet.HnswIndex",
            "M:VecNet.HnswIndex.#ctor(System.Int32,VecNet.VectorMetric)",
            "M:VecNet.HnswIndex.#ctor(System.Int32,VecNet.VectorMetric,VecNet.HnswIndexOptions)",
            "P:VecNet.HnswIndex.Dimension",
            "P:VecNet.HnswIndex.Metric",
            "P:VecNet.HnswIndex.Count",
            "P:VecNet.HnswIndex.Options",
            "M:VecNet.HnswIndex.Add(System.UInt64,System.ReadOnlySpan{System.Single})",
            "M:VecNet.HnswIndex.Search(System.ReadOnlySpan{System.Single},System.Span{VecNet.SearchResult},VecNet.HnswSearchWorkspace)",
            "M:VecNet.HnswIndex.Search(System.ReadOnlySpan{System.Single},System.ReadOnlySpan{System.UInt64},System.Span{VecNet.SearchResult},VecNet.HnswSearchWorkspace)",
            "M:VecNet.HnswIndex.Save(System.String)",
            "M:VecNet.HnswIndex.OpenReadOnly(System.String)",
            "T:VecNet.HnswIndexOptions",
            "M:VecNet.HnswIndexOptions.#ctor(System.Int32,System.Int32,System.Int32,System.UInt64)",
            "P:VecNet.HnswIndexOptions.M",
            "P:VecNet.HnswIndexOptions.EfConstruction",
            "P:VecNet.HnswIndexOptions.EfSearch",
            "P:VecNet.HnswIndexOptions.RandomSeed",
            "P:VecNet.HnswIndexOptions.Default",
            "T:VecNet.HnswSearchWorkspace",
            "M:VecNet.HnswSearchWorkspace.#ctor(System.Int32,System.Int32)",
            "P:VecNet.HnswSearchWorkspace.MaxElements",
            "P:VecNet.HnswSearchWorkspace.MaxEf",
            "T:VecNet.HnswMutableIndex",
            "M:VecNet.HnswMutableIndex.#ctor(VecNet.HnswIndex)",
            "P:VecNet.HnswMutableIndex.Dimension",
            "P:VecNet.HnswMutableIndex.Metric",
            "P:VecNet.HnswMutableIndex.Options",
            "P:VecNet.HnswMutableIndex.Count",
            "P:VecNet.HnswMutableIndex.BasePhysicalVectorCount",
            "P:VecNet.HnswMutableIndex.BaseLiveVectorCount",
            "P:VecNet.HnswMutableIndex.DeltaPhysicalVectorCount",
            "P:VecNet.HnswMutableIndex.DeltaLiveVectorCount",
            "P:VecNet.HnswMutableIndex.LiveVectorCount",
            "P:VecNet.HnswMutableIndex.BaseTombstoneCount",
            "P:VecNet.HnswMutableIndex.DeltaTombstoneCount",
            "P:VecNet.HnswMutableIndex.TombstoneCount",
            "P:VecNet.HnswMutableIndex.DeletedReservedIdCount",
            "P:VecNet.HnswMutableIndex.Generation",
            "M:VecNet.HnswMutableIndex.TryAdd(System.UInt64,System.ReadOnlySpan{System.Single})",
            "M:VecNet.HnswMutableIndex.TryDelete(System.UInt64)",
            "M:VecNet.HnswMutableIndex.Search(System.ReadOnlySpan{System.Single},System.Span{VecNet.SearchResult},VecNet.HnswMutableSearchWorkspace)",
            "M:VecNet.HnswMutableIndex.Search(System.ReadOnlySpan{System.Single},System.ReadOnlySpan{System.UInt64},System.Span{VecNet.SearchResult},VecNet.HnswMutableSearchWorkspace)",
            "M:VecNet.HnswMutableIndex.Checkpoint(System.String)",
            "T:VecNet.HnswMutableSearchWorkspace",
            "M:VecNet.HnswMutableSearchWorkspace.#ctor(VecNet.HnswMutableIndex,System.Int32)",
            "P:VecNet.HnswMutableSearchWorkspace.Generation",
            "P:VecNet.HnswMutableSearchWorkspace.MaxBaseElements",
            "P:VecNet.HnswMutableSearchWorkspace.MaxEfSearch",
            "P:VecNet.HnswMutableSearchWorkspace.MaxBaseCandidates",
            "P:VecNet.HnswMutableSearchWorkspace.MaxDeltaCandidates",
            "P:VecNet.HnswMutableSearchWorkspace.MaxDeltaFilterElements",
            "T:VecNet.HnswMutableCheckpointResult",
            "M:VecNet.HnswMutableCheckpointResult.#ctor(VecNet.HnswMutableCheckpointStatus,System.Int64,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32)",
            "P:VecNet.HnswMutableCheckpointResult.Status",
            "P:VecNet.HnswMutableCheckpointResult.Generation",
            "P:VecNet.HnswMutableCheckpointResult.RebuiltBaseVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.LiveVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.BasePhysicalVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.BaseLiveVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.DeltaPhysicalVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.DeltaLiveVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.BaseTombstoneCount",
            "P:VecNet.HnswMutableCheckpointResult.DeltaTombstoneCount",
            "P:VecNet.HnswMutableCheckpointResult.TombstoneCount",
            "P:VecNet.HnswMutableCheckpointResult.DeletedReservedIdCount",
            "P:VecNet.HnswMutableCheckpointResult.FoldedDeltaVectorCount",
            "P:VecNet.HnswMutableCheckpointResult.FoldedBaseTombstoneCount",
            "P:VecNet.HnswMutableCheckpointResult.FoldedDeltaTombstoneCount",
            "T:VecNet.HnswMutableCheckpointStatus",
            "F:VecNet.HnswMutableCheckpointStatus.Published",
            "F:VecNet.HnswMutableCheckpointStatus.NoChanges"
        ];

        foreach (string expectedMember in expectedMembers)
        {
            Assert.Contains(expectedMember, members);
        }
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
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
