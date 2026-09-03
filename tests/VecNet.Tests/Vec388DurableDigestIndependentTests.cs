using System.Text.Json;
using System.Text.Json.Nodes;

namespace VecNet.Tests;

public sealed class Vec388DurableDigestIndependentTests
{
    [Fact]
    public void ExactFlatDigest_IsLogicalAcrossCapacityPlanningAndVolatileManifestFields()
    {
        using TempIndexDirectory organic = TempIndexDirectory.Create();
        using TempIndexDirectory reserved = TempIndexDirectory.Create();
        using TempIndexDirectory changed = TempIndexDirectory.Create();
        ExactFlatIndex organicIndex = CreateExactFlat(capacity: null);
        ExactFlatIndex reservedIndex = CreateExactFlat(capacity: 32);
        ExactFlatIndex changedIndex = CreateExactFlat(capacity: 32, changedLastLane: true);
        float[][] queries =
        [
            [0f, 0f, 0f, 0f, 0f],
            [0.25f, -0.25f, 0.5f, -0.5f, 1f]
        ];

        organicIndex.Save(organic.Path);
        reservedIndex.Save(reserved.Path);
        changedIndex.Save(changed.Path);

        string organicDigest = AssertDigestPresent(organic.Path, ExactFlatIndexStorage.ManifestFileName);
        Assert.Equal(organicDigest, AssertDigestPresent(reserved.Path, ExactFlatIndexStorage.ManifestFileName));
        Assert.NotEqual(organicDigest, AssertDigestPresent(changed.Path, ExactFlatIndexStorage.ManifestFileName));

        PatchManifest(organic.Path, ExactFlatIndexStorage.ManifestFileName, root =>
            root["createdUtc"] = "2026-09-01T01:02:03.0000000Z");

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(organic.Path);
        foreach (float[] query in queries)
        {
            Assert.Equal(Search(organicIndex, query, topK: 5), Search(opened, query, topK: 5));
        }
    }

    [Fact]
    public void ExactFlatDigest_FailsClosedWhenPresentButMalformedWrongTypeOrMismatchedAndAllowsNoDigest()
    {
        AssertExactFlatManifestMutationOpens(root => root.Remove("contentDigest"));
        AssertExactFlatManifestMutationRejected(root => root["contentDigest"] = "ABCDEF" + new string('0', 58));
        AssertExactFlatManifestMutationRejected(root => root["contentDigest"] = JsonValue.Create(true));
        AssertExactFlatManifestMutationRejected(root =>
        {
            string current = root["contentDigest"]!.GetValue<string>();
            root["contentDigest"] = current[0] == '0'
                ? "1" + current[1..]
                : "0" + current[1..];
        });
    }

    [Fact]
    public void ImmutableHnswDigest_IsLogicalAcrossCapacityPlanningAndVolatileManifestFields()
    {
        using TempIndexDirectory organic = TempIndexDirectory.Create();
        using TempIndexDirectory reserved = TempIndexDirectory.Create();
        using TempIndexDirectory changed = TempIndexDirectory.Create();
        HnswIndex organicIndex = CreateHnsw(capacity: null);
        HnswIndex reservedIndex = CreateHnsw(capacity: 40);
        HnswIndex changedIndex = CreateHnsw(capacity: 40, changedLastLane: true);
        float[][] queries =
        [
            [-0.2f, 0.15f, 0.5f, -0.75f],
            [1.5f, -1f, 0.25f, 0f]
        ];

        organicIndex.Save(organic.Path);
        reservedIndex.Save(reserved.Path);
        changedIndex.Save(changed.Path);

        string organicDigest = AssertDigestPresent(organic.Path, HnswIndexStorage.ManifestFileName);
        Assert.Equal(organicDigest, AssertDigestPresent(reserved.Path, HnswIndexStorage.ManifestFileName));
        Assert.NotEqual(organicDigest, AssertDigestPresent(changed.Path, HnswIndexStorage.ManifestFileName));

        PatchManifest(organic.Path, HnswIndexStorage.ManifestFileName, root =>
        {
            root["createdUtc"] = "2026-09-01T04:05:06.0000000Z";
            root["snapshotId"] = Guid.Parse("11111111-2222-3333-4444-555555555555").ToString("D");
        });

        HnswIndex opened = HnswIndex.OpenReadOnly(organic.Path);
        foreach (float[] query in queries)
        {
            Assert.Equal(Search(organicIndex, query, topK: 6), Search(opened, query, topK: 6));
        }
    }

    [Fact]
    public void ImmutableHnswDigest_FailsClosedWhenPresentButMalformedWrongTypeOrMismatchedAndAllowsNoDigest()
    {
        AssertHnswManifestMutationOpens(root => root.Remove("contentDigest"));
        AssertHnswManifestMutationRejected(root => root["contentDigest"] = new string('a', 63));
        AssertHnswManifestMutationRejected(root => root["contentDigest"] = JsonValue.Create(17));
        AssertHnswManifestMutationRejected(root =>
        {
            string current = root["contentDigest"]!.GetValue<string>();
            root["contentDigest"] = current[0] == 'f'
                ? "e" + current[1..]
                : "f" + current[1..];
        });
    }

    [Fact]
    public void MutableHnswCheckpointDigest_EmitsLogicalDigestAndReopensWithSearchParity()
    {
        using TempIndexDirectory first = TempIndexDirectory.CreateMissing();
        using TempIndexDirectory second = TempIndexDirectory.CreateMissing();
        HnswMutableIndex fromOrganicBase = CreateMutatedHnsw(capacity: null);
        HnswMutableIndex fromReservedBase = CreateMutatedHnsw(capacity: 64);
        float[][] queries =
        [
            [0f, 0f, 0f, 0f],
            [0.75f, -0.5f, 0.125f, -0.25f]
        ];
        SearchResult[][] expected = queries
            .Select(query => Search(fromOrganicBase, query, topK: 6))
            .ToArray();

        HnswMutableCheckpointResult firstResult = fromOrganicBase.Checkpoint(first.Path);
        HnswMutableCheckpointResult secondResult = fromReservedBase.Checkpoint(second.Path);

        Assert.Equal(HnswMutableCheckpointStatus.Published, firstResult.Status);
        Assert.Equal(HnswMutableCheckpointStatus.Published, secondResult.Status);
        Assert.Equal(8, firstResult.RebuiltBaseVectorCount);
        Assert.Equal(firstResult.RebuiltBaseVectorCount, secondResult.RebuiltBaseVectorCount);
        string firstDigest = AssertDigestPresent(first.Path, HnswIndexStorage.ManifestFileName);
        Assert.Equal(firstDigest, AssertDigestPresent(second.Path, HnswIndexStorage.ManifestFileName));

        HnswIndex opened = HnswIndex.OpenReadOnly(first.Path);
        for (int i = 0; i < queries.Length; i++)
        {
            Assert.Equal(expected[i], Search(opened, queries[i], topK: 6));
        }
    }

    [Fact]
    public void MutableHnswCheckpointDigest_NoDigestStillOpensButBadPresentDigestFailsClosed()
    {
        using (TempIndexDirectory legacy = TempIndexDirectory.CreateMissing())
        {
            HnswMutableIndex mutable = CreateMutatedHnsw(capacity: null);
            SearchResult[] before = Search(mutable, [0f, 0f, 0f, 0f], topK: 6);
            Assert.Equal(HnswMutableCheckpointStatus.Published, mutable.Checkpoint(legacy.Path).Status);
            PatchManifest(legacy.Path, HnswIndexStorage.ManifestFileName, root => root.Remove("contentDigest"));

            HnswIndex opened = HnswIndex.OpenReadOnly(legacy.Path);

            Assert.Equal(before, Search(opened, [0f, 0f, 0f, 0f], topK: 6));
        }

        using (TempIndexDirectory malformed = TempIndexDirectory.CreateMissing())
        {
            HnswMutableIndex mutable = CreateMutatedHnsw(capacity: null);
            Assert.Equal(HnswMutableCheckpointStatus.Published, mutable.Checkpoint(malformed.Path).Status);
            PatchManifest(malformed.Path, HnswIndexStorage.ManifestFileName, root => root["contentDigest"] = new JsonObject());

            Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(malformed.Path));
        }
    }

    private static ExactFlatIndex CreateExactFlat(int? capacity, bool changedLastLane = false)
    {
        var index = capacity.HasValue
            ? new ExactFlatIndex(5, VectorMetric.InnerProduct, capacity.Value)
            : new ExactFlatIndex(5, VectorMetric.InnerProduct);

        AddExactRows(index, changedLastLane);
        if (capacity.HasValue)
        {
            index.EnsureCapacity(capacity.Value + 11);
        }

        return index;
    }

    private static void AddExactRows(ExactFlatIndex index, bool changedLastLane)
    {
        index.Add(900, [1f, 0f, -1f, 0.5f, changedLastLane ? 3.25f : 3f]);
        index.Add(100, [0f, 2f, 0.25f, -1f, 0.5f]);
        index.Add(500, [-1f, -1f, 1f, 1f, -0.75f]);
        index.Add(ulong.MaxValue - 4, [4f, 0f, 0f, 0f, -2f]);
    }

    private static HnswIndex CreateHnsw(int? capacity, bool changedLastLane = false)
    {
        int[] levels = [2, 0, 1, 0, 1, 0, 2, 0];
        int nextLevel = 0;
        var options = new HnswIndexOptions(4, 32, 32, 0x3880_0100UL);
        var index = capacity.HasValue
            ? new HnswIndex(4, VectorMetric.SquaredEuclidean, options, capacity.Value, () => levels[nextLevel++])
            : new HnswIndex(4, VectorMetric.SquaredEuclidean, options, () => levels[nextLevel++]);

        AddHnswRows(index, changedLastLane);
        if (capacity.HasValue)
        {
            index.EnsureCapacity(capacity.Value + 9);
        }

        return index;
    }

    private static void AddHnswRows(HnswIndex index, bool changedLastLane)
    {
        index.Add(11, [-2f, 0f, 1f, 0.25f]);
        index.Add(22, [-1f, 0.5f, 1.5f, -0.25f]);
        index.Add(33, [0f, 1f, -0.5f, 0.75f]);
        index.Add(44, [1f, -1f, 0.25f, -0.5f]);
        index.Add(55, [2f, 0.25f, -1.25f, changedLastLane ? 1.5f : 1.25f]);
        index.Add(66, [3f, -0.75f, 0.5f, -1f]);
        index.Add(77, [4f, 1.25f, -0.25f, 0f]);
        index.Add(88, [5f, -1.5f, 1.75f, 0.5f]);
    }

    private static HnswMutableIndex CreateMutatedHnsw(int? capacity)
    {
        HnswIndex baseIndex = CreateHnsw(capacity);
        var mutable = new HnswMutableIndex(baseIndex);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(101, [-1.5f, 0.25f, 0.25f, 0f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryAdd(202, [1.25f, -1.25f, 0.75f, 0.5f]).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(22).Status);
        Assert.Equal(VectorMutationStatus.Committed, mutable.TryDelete(77).Status);
        return mutable;
    }

    private static void AssertExactFlatManifestMutationOpens(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        ExactFlatIndex expected = CreateExactFlat(capacity: null);
        expected.Save(temp.Path);
        PatchManifest(temp.Path, ExactFlatIndexStorage.ManifestFileName, mutate);

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(temp.Path);

        Assert.Equal(Search(expected, [0.25f, -0.25f, 0.5f, -0.5f, 1f], topK: 5), Search(opened, [0.25f, -0.25f, 0.5f, -0.5f, 1f], topK: 5));
    }

    private static void AssertExactFlatManifestMutationRejected(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateExactFlat(capacity: null).Save(temp.Path);
        PatchManifest(temp.Path, ExactFlatIndexStorage.ManifestFileName, mutate);

        Assert.Throws<InvalidDataException>(() => ExactFlatIndex.OpenReadOnly(temp.Path));
    }

    private static void AssertHnswManifestMutationOpens(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        HnswIndex expected = CreateHnsw(capacity: null);
        expected.Save(temp.Path);
        PatchManifest(temp.Path, HnswIndexStorage.ManifestFileName, mutate);

        HnswIndex opened = HnswIndex.OpenReadOnly(temp.Path);

        Assert.Equal(Search(expected, [1.5f, -1f, 0.25f, 0f], topK: 6), Search(opened, [1.5f, -1f, 0.25f, 0f], topK: 6));
    }

    private static void AssertHnswManifestMutationRejected(Action<JsonObject> mutate)
    {
        using TempIndexDirectory temp = TempIndexDirectory.Create();
        CreateHnsw(capacity: null).Save(temp.Path);
        PatchManifest(temp.Path, HnswIndexStorage.ManifestFileName, mutate);

        Assert.Throws<InvalidDataException>(() => HnswIndex.OpenReadOnly(temp.Path));
    }

    private static string AssertDigestPresent(string directory, string manifestFileName)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, manifestFileName)));
        JsonElement value = document.RootElement.GetProperty("contentDigest");
        Assert.Equal(JsonValueKind.String, value.ValueKind);
        string digest = value.GetString()!;
        Assert.Equal(64, digest.Length);
        Assert.All(digest, static character => Assert.True(character is >= '0' and <= '9' or >= 'a' and <= 'f'));
        return digest;
    }

    private static void PatchManifest(string directory, string manifestFileName, Action<JsonObject> mutate)
    {
        string manifestPath = Path.Combine(directory, manifestFileName);
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(manifestPath))!;
        mutate(root);
        File.WriteAllText(manifestPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
    }

    private static SearchResult[] Search(ExactFlatIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results);
        return results[..written];
    }

    private static SearchResult[] Search(HnswIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
        return results[..written];
    }

    private static SearchResult[] Search(HnswMutableIndex index, float[] query, int topK)
    {
        var results = new SearchResult[topK];
        int written = index.Search(query, results, new HnswMutableSearchWorkspace(index, topK, index.Options.EfSearch));
        return results[..written];
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
        }

        private static string CreatePath() =>
            System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                "Vec388-" + Guid.NewGuid().ToString("N"));
    }
}
