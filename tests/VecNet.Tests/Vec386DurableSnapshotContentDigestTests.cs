using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class Vec386DurableSnapshotContentDigestTests
{
    [Fact]
    public void ExactFlatSave_WritesDeterministicDigestThatIgnoresCreatedUtcAndPreservesSearchParity()
    {
        using TempIndexDirectory first = TempIndexDirectory.Create();
        using TempIndexDirectory second = TempIndexDirectory.Create();
        using TempIndexDirectory different = TempIndexDirectory.Create();
        ExactFlatIndex source = CreateExactFlatIndex(lastComponent: 3f);
        ExactFlatIndex changed = CreateExactFlatIndex(lastComponent: 4f);
        float[] query = [1f, 0.5f, -0.25f];

        source.Save(first.Path);
        source.Save(second.Path);
        changed.Save(different.Path);

        string digest = ReadContentDigest(first.Path, ExactFlatIndexStorage.ManifestFileName);
        Assert.Equal(digest, ReadContentDigest(second.Path, ExactFlatIndexStorage.ManifestFileName));
        Assert.NotEqual(digest, ReadContentDigest(different.Path, ExactFlatIndexStorage.ManifestFileName));

        MutateManifest(first.Path, ExactFlatIndexStorage.ManifestFileName, root =>
            root["createdUtc"] = "2026-09-01T00:00:00.0000000Z");

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(first.Path);
        Assert.Equal(Search(source, query, topK: 3), Search(opened, query, topK: 3));
    }

    [Fact]
    public void ExactFlatOpenReadOnly_AcceptsOldManifestWithoutDigestAndRejectsMalformedOrMismatchedDigest()
    {
        using (TempIndexDirectory oldManifest = TempIndexDirectory.Create())
        {
            CreateExactFlatIndex(lastComponent: 3f).Save(oldManifest.Path);
            MutateManifest(oldManifest.Path, ExactFlatIndexStorage.ManifestFileName, root => root.Remove("contentDigest"));

            ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(oldManifest.Path);

            Assert.Equal(3, opened.VectorCount);
        }

        using (TempIndexDirectory malformed = TempIndexDirectory.Create())
        {
            CreateExactFlatIndex(lastComponent: 3f).Save(malformed.Path);
            MutateManifest(malformed.Path, ExactFlatIndexStorage.ManifestFileName, root => root["contentDigest"] = "not-a-digest");

            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(malformed.Path));
        }

        using (TempIndexDirectory wrongType = TempIndexDirectory.Create())
        {
            CreateExactFlatIndex(lastComponent: 3f).Save(wrongType.Path);
            MutateManifest(wrongType.Path, ExactFlatIndexStorage.ManifestFileName, root => root["contentDigest"] = 123);

            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(wrongType.Path));
        }

        using (TempIndexDirectory mismatched = TempIndexDirectory.Create())
        {
            CreateExactFlatIndex(lastComponent: 3f).Save(mismatched.Path);
            MutateManifest(mismatched.Path, ExactFlatIndexStorage.ManifestFileName, root => root["contentDigest"] = new string('0', 64));

            Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(mismatched.Path));
        }
    }

    [Fact]
    public void HnswSave_WritesDeterministicDigestThatIgnoresVolatileFieldsAndPreservesSearchParity()
    {
        using TempIndexDirectory first = TempIndexDirectory.Create();
        using TempIndexDirectory second = TempIndexDirectory.Create();
        using TempIndexDirectory different = TempIndexDirectory.Create();
        HnswIndex source = CreateHnswIndex(lastComponent: 3f);
        HnswIndex changed = CreateHnswIndex(lastComponent: 4f);
        float[] query = [1f, 0.25f];

        source.Save(first.Path);
        source.Save(second.Path);
        changed.Save(different.Path);

        string digest = ReadContentDigest(first.Path, HnswIndexStorage.ManifestFileName);
        Assert.Equal(digest, ReadContentDigest(second.Path, HnswIndexStorage.ManifestFileName));
        Assert.NotEqual(digest, ReadContentDigest(different.Path, HnswIndexStorage.ManifestFileName));

        MutateManifest(first.Path, HnswIndexStorage.ManifestFileName, root =>
        {
            root["snapshotId"] = Guid.NewGuid().ToString("D");
            root["createdUtc"] = "2026-09-01T00:00:00.0000000Z";
        });

        HnswIndex opened = HnswIndex.OpenReadOnly(first.Path);
        Assert.Equal(Search(source, query, topK: 3, efSearch: 8), Search(opened, query, topK: 3, efSearch: 8));
    }

    [Fact]
    public void HnswOpenReadOnly_AcceptsOldManifestWithoutDigestAndRejectsMalformedOrMismatchedDigest()
    {
        using (TempIndexDirectory oldManifest = TempIndexDirectory.Create())
        {
            CreateHnswIndex(lastComponent: 3f).Save(oldManifest.Path);
            MutateManifest(oldManifest.Path, HnswIndexStorage.ManifestFileName, root => root.Remove("contentDigest"));

            HnswIndex opened = HnswIndex.OpenReadOnly(oldManifest.Path);

            Assert.Equal(4, opened.Count);
        }

        using (TempIndexDirectory malformed = TempIndexDirectory.Create())
        {
            CreateHnswIndex(lastComponent: 3f).Save(malformed.Path);
            MutateManifest(malformed.Path, HnswIndexStorage.ManifestFileName, root => root["contentDigest"] = "not-a-digest");

            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(malformed.Path));
        }

        using (TempIndexDirectory wrongType = TempIndexDirectory.Create())
        {
            CreateHnswIndex(lastComponent: 3f).Save(wrongType.Path);
            MutateManifest(wrongType.Path, HnswIndexStorage.ManifestFileName, root => root["contentDigest"] = 123);

            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(wrongType.Path));
        }

        using (TempIndexDirectory mismatched = TempIndexDirectory.Create())
        {
            CreateHnswIndex(lastComponent: 3f).Save(mismatched.Path);
            MutateManifest(mismatched.Path, HnswIndexStorage.ManifestFileName, root => root["contentDigest"] = new string('0', 64));

            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(mismatched.Path));
        }
    }

    [Fact]
    public void MutableHnswCheckpoint_WritesDigestForPublishedDurableGenerationAndReopensWithSearchParity()
    {
        using TempIndexDirectory checkpoint = TempIndexDirectory.Create();
        HnswIndex baseIndex = CreateHnswIndex(lastComponent: 3f);
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(15, [0.25f, 0.25f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(20).Status);
        float[] query = [0f, 0f];
        SearchResult[] before = Search(mutable, query, topK: 4);

        HnswMutableCheckpointResult result = mutable.Checkpoint(checkpoint.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, result.Status);
        Assert.Equal(4, result.RebuiltBaseVectorCount);
        string digest = ReadContentDigest(checkpoint.Path, HnswIndexStorage.ManifestFileName);
        Assert.Equal(64, digest.Length);

        HnswIndex opened = HnswIndex.OpenReadOnly(checkpoint.Path);
        Assert.Equal(before, Search(opened, query, topK: 4, efSearch: 8));
    }

    private static ExactFlatIndex CreateExactIndex(VectorMetric metric, params (ulong Id, float[] Vector)[] rows)
    {
        var index = new ExactFlatIndex(rows[0].Vector.Length, metric);
        foreach ((ulong id, float[] vector) in rows)
        {
            index.Add(id, vector);
        }

        return index;
    }

    private static ExactFlatIndex CreateExactFlatIndex(float lastComponent) =>
        CreateExactIndex(
            VectorMetric.SquaredEuclidean,
            (10, [0f, 0f, 0f]),
            (20, [1f, 0f, 0f]),
            (30, [1f, 2f, lastComponent]));

    private static HnswIndex CreateHnswIndex(float lastComponent)
    {
        int[] levels = [1, 0, 0, 0];
        int nextLevel = 0;
        var index = new HnswIndex(
            2,
            VectorMetric.SquaredEuclidean,
            new HnswIndexOptions(2, 8, 8, 0x386),
            () => levels[nextLevel++]);
        index.Add(10, [0f, 0f]);
        index.Add(20, [1f, 0f]);
        index.Add(30, [2f, 0f]);
        index.Add(40, [1f, lastComponent]);
        return index;
    }

    private static string ReadContentDigest(string directory, string manifestFileName)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, manifestFileName)));
        string digest = document.RootElement.GetProperty("contentDigest").GetString()!;
        Assert.Equal(64, digest.Length);
        Assert.All(digest, c => Assert.True(c is >= '0' and <= '9' or >= 'a' and <= 'f'));
        return digest;
    }

    private static void MutateManifest(string directory, string manifestFileName, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, manifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString());
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK, int efSearch)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, efSearch), efSearch);
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK));
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
