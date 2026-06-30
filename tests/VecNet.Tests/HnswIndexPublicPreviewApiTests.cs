using System.Reflection;

namespace VecNet.Tests;

public sealed class HnswIndexPublicPreviewApiTests
{
    [Fact]
    public void PublicSurface_ExposesOnlyPreviewBuildSearchAndDurableMembers()
    {
        Assert.True(typeof(HnswIndex).IsPublic);
        Assert.True(typeof(HnswIndexOptions).IsPublic);
        Assert.True(typeof(HnswSearchWorkspace).IsPublic);

        Assert.Equal(
            ["Add", "OpenReadOnly", "Save", "Search"],
            typeof(HnswIndex)
                .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Where(static method => !method.IsSpecialName)
                .Select(static method => method.Name)
                .Order()
                .ToArray());

        Assert.Equal(
            ["Count", "Dimension", "Metric", "Options"],
            typeof(HnswIndex)
                .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Select(static property => property.Name)
                .Order()
                .ToArray());

        string[] expectedConstructors =
        [
            "Void .ctor(Int32, VecNet.VectorMetric)",
            "Void .ctor(Int32, VecNet.VectorMetric, VecNet.HnswIndexOptions)"
        ];
        Array.Sort(expectedConstructors, StringComparer.Ordinal);
        Assert.Equal(
            expectedConstructors,
            typeof(HnswIndex)
                .GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Select(static constructor => constructor.ToString()!)
                .OrderBy(static constructor => constructor, StringComparer.Ordinal)
                .ToArray());

        Assert.DoesNotContain(
            typeof(HnswIndex).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static),
            static method => method.Name.StartsWith("Debug", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicOptionsAndWorkspace_ExposePreviewSizingState()
    {
        HnswIndexOptions options = HnswIndexOptions.Default;

        Assert.Equal(new HnswIndexOptions(16, 200, 50, 0x564543_034UL), options);

        var workspace = new HnswSearchWorkspace(maxElements: 12, maxEf: options.EfSearch);

        Assert.Equal(12, workspace.MaxElements);
        Assert.Equal(options.EfSearch, workspace.MaxEf);
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswSearchWorkspace(-1, options.EfSearch));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswSearchWorkspace(1, 0));
    }

    [Fact]
    public void PublicConstruction_RemainsSquaredL2OnlyAndValidatesOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(0, VectorMetric.SquaredEuclidean));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswIndex(2, (VectorMetric)999));
        Assert.Throws<NotSupportedException>(() => new HnswIndex(2, VectorMetric.InnerProduct));
        Assert.Throws<NotSupportedException>(() => new HnswIndex(2, VectorMetric.Cosine));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(1, 2, 2, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(2, 1, 2, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new HnswIndex(2, VectorMetric.SquaredEuclidean, new HnswIndexOptions(2, 2, 0, 1)));

        var index = new HnswIndex(2, VectorMetric.SquaredEuclidean);

        Assert.Equal(2, index.Dimension);
        Assert.Equal(VectorMetric.SquaredEuclidean, index.Metric);
        Assert.Equal(0, index.Count);
        Assert.Equal(HnswIndexOptions.Default, index.Options);
    }

    [Fact]
    public void PublicPreviewWorkflow_AddSearchSaveOpenAndReadOnlyMutationRejection()
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        var options = new HnswIndexOptions(4, 16, 16, 0x5EED_0115UL);
        var index = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);

        index.Add(101, [0f, 0f, 0f]);
        index.Add(102, [2f, 0f, 0f]);
        index.Add(103, [0f, 3f, 0f]);

        var results = new SearchResult[2];
        int written = index.Search([0.5f, 0f, 0f], results, new HnswSearchWorkspace(index.Count, options.EfSearch));

        Assert.Equal(2, written);
        Assert.Equal([101UL, 102UL], results.Select(static result => result.Id));

        index.Save(temp.Path);
        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(index.Dimension, opened.Dimension);
        Assert.Equal(index.Metric, opened.Metric);
        Assert.Equal(index.Count, opened.Count);
        Assert.Equal(index.Options, opened.Options);

        var openedResults = new SearchResult[2];
        int openedWritten = opened.Search([0.5f, 0f, 0f], openedResults, new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch));

        Assert.Equal(written, openedWritten);
        Assert.Equal(results, openedResults);
        Assert.Throws<InvalidOperationException>(() => opened.Add(104, [1f, 1f, 1f]));
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
