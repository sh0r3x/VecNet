using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;
using VecNet;

string mode = GetOption(args, "--mode") ?? "jit";
string artifactRoot = Path.GetFullPath(GetOption(args, "--artifact-root") ?? Path.Combine(Path.GetTempPath(), "vec159 arm64 smoke"));
string runnerLabel = GetOption(args, "--runner-label") ?? Environment.GetEnvironmentVariable("RUNNER_NAME") ?? "unknown";
bool requireArm64 = args.Contains("--require-arm64", StringComparer.Ordinal);

Directory.CreateDirectory(artifactRoot);
PrintMetadata(mode, artifactRoot, runnerLabel);

if (requireArm64)
{
    Require(RuntimeInformation.OSArchitecture == Architecture.Arm64, "OS architecture must be Arm64 for Arm64 support evidence.");
    Require(RuntimeInformation.ProcessArchitecture == Architecture.Arm64, "Process architecture must be Arm64 for Arm64 support evidence.");
}

RunExactSmoke(artifactRoot);
RunHnswSmoke(artifactRoot);
RunMutableHnswSmoke(artifactRoot);
AssertNoTempFiles(artifactRoot);
AssertNoEmbeddedArtifactRoot(artifactRoot);

Console.WriteLine($"VEC159_SMOKE_PASSED mode={mode}");

static void RunExactSmoke(string artifactRoot)
{
    var exact = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
    exact.Add(10, [0f, 0f]);
    exact.Add(11, [1f, 0f]);
    exact.Add(12, [0f, 2f]);

    Span<SearchResult> results = stackalloc SearchResult[3];
    int written = exact.Search([0f, 0f], results);
    Require(written == 3, "Exact squared-L2 search should return three rows.");
    Require(results[0].Id == 10 && results[1].Id == 11 && results[2].Id == 12, "Exact squared-L2 result order is wrong.");

    var filterWorkspace = new ExactFlatSearchFilterWorkspace(exact.PhysicalVectorCount);
    written = exact.Search([0f, 0f], [12UL, 10UL, 999UL, 10UL], results[..2], filterWorkspace);
    Require(written == 2 && results[0].Id == 10 && results[1].Id == 12, "Exact raw allowlist filtering failed.");

    ExactFlatCandidateSet candidates = exact.CreateCandidateSet([12UL, 10UL, 999UL, 12UL]);
    written = exact.Search([0f, 0f], candidates, results[..2]);
    Require(written == 2 && results[0].Id == 10 && results[1].Id == 12, "Exact candidate-set filtering failed.");

    var inner = new ExactFlatIndex(2, VectorMetric.InnerProduct);
    inner.Add(20, [1f, 0f]);
    inner.Add(21, [0f, 2f]);
    written = inner.Search([0f, 1f], results[..2]);
    Require(written == 2 && results[0].Id == 21, "Exact inner-product canonical ordering failed.");

    var cosine = new ExactFlatIndex(2, VectorMetric.Cosine);
    cosine.Add(30, [2f, 0f]);
    cosine.Add(31, [0f, 3f]);
    written = cosine.Search([1f, 0f], results[..2]);
    Require(written == 2 && results[0].Id == 30 && results[0].Distance <= 1e-6f, "Exact cosine normalized ordering failed.");

    VectorMutationResult addResult = exact.TryAdd(13, [3f, 0f]);
    Require(addResult.Status == VectorMutationStatus.Committed, "Exact TryAdd should commit.");
    VectorMutationResult deleteResult = exact.TryDelete(11);
    Require(deleteResult.Status == VectorMutationStatus.Committed, "Exact TryDelete should commit.");
    Require(exact.TryAdd(11, [4f, 0f]).Status == VectorMutationStatus.DuplicateId, "Deleted exact ID should remain reserved.");

    string savePath = Path.Combine(artifactRoot, "exact save path with spaces");
    exact.Save(savePath);
    Require(File.Exists(Path.Combine(savePath, "exact-flat.manifest.json")), "Exact save manifest was not written.");
    ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(savePath);
    written = opened.Search([0f, 0f], results);
    Require(written == 3 && results[0].Id == 10 && results[1].Id == 12 && results[2].Id == 13, "Opened exact search failed.");
    Require(opened.TryAdd(99, [9f, 9f]).Status == VectorMutationStatus.ReadOnly, "Opened exact TryAdd should be read-only.");
    ExpectThrows<InvalidOperationException>(() => opened.Add(100, [10f, 10f]), "Opened exact Add should throw.");

    string checkpointPath = Path.Combine(artifactRoot, "exact checkpoint path with spaces");
    ExactFlatCheckpointResult checkpoint = exact.Checkpoint(checkpointPath);
    Require(checkpoint.Status == ExactFlatCheckpointStatus.Published, "Exact checkpoint should publish after delta/delete.");
    ExactFlatIndex openedCheckpoint = ExactFlatIndex.OpenReadOnly(checkpointPath);
    written = openedCheckpoint.Search([0f, 0f], results);
    Require(written == 3 && results[0].Id == 10 && results[1].Id == 12 && results[2].Id == 13, "Opened exact checkpoint search failed.");
}

static void RunHnswSmoke(string artifactRoot)
{
    HnswIndex hnsw = BuildHnsw();
    Span<SearchResult> results = stackalloc SearchResult[3];
    var workspace = new HnswSearchWorkspace(hnsw.Count, hnsw.Options.EfSearch);

    int written = hnsw.Search([0f, 0f], results, workspace);
    Require(written == 3 && Contains(results[..written], 100), "HNSW search did not return the nearest base ID.");

    written = hnsw.Search([0f, 0f], [104UL, 100UL, 999UL, 100UL], results[..2], workspace);
    Require(written == 2 && results[0].Id == 100 && results[1].Id == 104, "HNSW allowlist search failed.");

    string savePath = Path.Combine(artifactRoot, "hnsw save path with spaces");
    hnsw.Save(savePath);
    Require(File.Exists(Path.Combine(savePath, "hnsw.manifest.json")), "HNSW save manifest was not written.");
    HnswIndex opened = HnswIndex.OpenReadOnly(savePath);
    var openedWorkspace = new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch);
    written = opened.Search([0f, 0f], results, openedWorkspace);
    Require(written == 3 && Contains(results[..written], 100), "Opened HNSW search failed.");
    ExpectThrows<InvalidOperationException>(() => opened.Add(999, [9f, 9f]), "Opened HNSW Add should throw.");
}

static void RunMutableHnswSmoke(string artifactRoot)
{
    HnswIndex baseIndex = BuildHnsw();
    var mutable = new HnswMutableIndex(baseIndex);

    VectorMutationResult addResult = mutable.TryAdd(900, [0.05f, 0.05f]);
    Require(addResult.Status == VectorMutationStatus.Committed, "Mutable HNSW TryAdd should commit.");
    VectorMutationResult deleteResult = mutable.TryDelete(100);
    Require(deleteResult.Status == VectorMutationStatus.Committed, "Mutable HNSW base TryDelete should commit.");

    Span<SearchResult> results = stackalloc SearchResult[4];
    var workspace = new HnswMutableSearchWorkspace(mutable, results.Length);
    int written = mutable.Search([0f, 0f], results, workspace);
    Require(written == 4 && results[0].Id == 900 && !Contains(results[..written], 100), "Mutable HNSW search failed.");

    written = mutable.Search([0f, 0f], [900UL, 101UL, 999UL, 900UL], results[..2], workspace);
    Require(written == 2 && results[0].Id == 900 && results[1].Id == 101, "Mutable HNSW allowlist search failed.");

    string checkpointPath = Path.Combine(artifactRoot, "mutable hnsw checkpoint path with spaces");
    HnswMutableCheckpointResult checkpoint = mutable.Checkpoint(checkpointPath);
    Require(checkpoint.Status == HnswMutableCheckpointStatus.Published, "Mutable HNSW checkpoint should publish.");
    Require(mutable.TryAdd(100, [10f, 10f]).Status == VectorMutationStatus.DuplicateId, "Deleted mutable HNSW ID should remain reserved.");

    HnswIndex openedCheckpoint = HnswIndex.OpenReadOnly(checkpointPath);
    var openedWorkspace = new HnswSearchWorkspace(openedCheckpoint.Count, openedCheckpoint.Options.EfSearch);
    written = openedCheckpoint.Search([0f, 0f], results, openedWorkspace);
    Require(written == 4 && Contains(results[..written], 900) && !Contains(results[..written], 100), "Opened mutable HNSW checkpoint search failed.");
}

static HnswIndex BuildHnsw()
{
    var options = new HnswIndexOptions(M: 4, EfConstruction: 16, EfSearch: 16, RandomSeed: 0x159UL);
    var index = new HnswIndex(2, VectorMetric.SquaredEuclidean, options);
    index.Add(100, [0f, 0f]);
    index.Add(101, [1f, 0f]);
    index.Add(102, [0f, 1f]);
    index.Add(103, [1f, 1f]);
    index.Add(104, [2f, 0f]);
    index.Add(105, [0f, 2f]);
    index.Add(106, [2f, 2f]);
    index.Add(107, [3f, 0f]);
    return index;
}

static void PrintMetadata(string mode, string artifactRoot, string runnerLabel)
{
    Console.WriteLine("VEC159_METADATA_BEGIN");
    Console.WriteLine($"mode={mode}");
    Console.WriteLine($"artifactRoot={artifactRoot}");
    Console.WriteLine($"runnerLabel={runnerLabel}");
    Console.WriteLine($"osDescription={RuntimeInformation.OSDescription}");
    Console.WriteLine($"frameworkDescription={RuntimeInformation.FrameworkDescription}");
    Console.WriteLine($"runtimeIdentifier={RuntimeInformation.RuntimeIdentifier}");
    Console.WriteLine($"osArchitecture={RuntimeInformation.OSArchitecture}");
    Console.WriteLine($"processArchitecture={RuntimeInformation.ProcessArchitecture}");
    Console.WriteLine($"vectorHardwareAccelerated={Vector.IsHardwareAccelerated}");
    Console.WriteLine($"vectorFloatCount={Vector<float>.Count}");
    Console.WriteLine($"serverGc={GCSettings.IsServerGC}");
    Console.WriteLine($"processorCount={Environment.ProcessorCount.ToString(CultureInfo.InvariantCulture)}");
    Console.WriteLine("VEC159_METADATA_END");
}

static string? GetOption(string[] args, string name)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], name, StringComparison.Ordinal))
        {
            return args[i + 1];
        }
    }

    return null;
}

static bool Contains(ReadOnlySpan<SearchResult> results, ulong id)
{
    foreach (SearchResult result in results)
    {
        if (result.Id == id)
        {
            return true;
        }
    }

    return false;
}

static void AssertNoTempFiles(string artifactRoot)
{
    string? tempFile = Directory
        .EnumerateFileSystemEntries(artifactRoot, "*", SearchOption.AllDirectories)
        .FirstOrDefault(static path => Path.GetFileName(path).Contains(".tmp-", StringComparison.Ordinal));
    Require(tempFile is null, $"Temporary durable file was left behind: {tempFile}");
}

static void AssertNoEmbeddedArtifactRoot(string artifactRoot)
{
    string normalizedRoot = artifactRoot.Replace('\\', '/');
    foreach (string jsonPath in Directory.EnumerateFiles(artifactRoot, "*.json", SearchOption.AllDirectories))
    {
        string text = File.ReadAllText(jsonPath).Replace('\\', '/');
        Require(!text.Contains(normalizedRoot, StringComparison.Ordinal), $"Manifest embeds artifact root path: {jsonPath}");
    }
}

static void ExpectThrows<TException>(Action action, string message)
    where TException : Exception
{
    try
    {
        action();
    }
    catch (TException)
    {
        return;
    }

    throw new InvalidOperationException(message);
}

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
