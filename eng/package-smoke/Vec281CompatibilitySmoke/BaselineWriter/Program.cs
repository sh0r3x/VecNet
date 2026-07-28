using System.Globalization;
using VecNet;

if (args.Length != 2)
{
    throw new ArgumentException("Usage: Vec281BaselineWriter <artifact-root> <baseline-version>");
}

string artifactRoot = Path.GetFullPath(args[0]);
string baselineVersion = args[1];
string baselineRoot = Path.Combine(artifactRoot, "baselines", baselineVersion);

ResetDirectory(baselineRoot);
Directory.CreateDirectory(baselineRoot);

var exclusions = new List<string>();
WriteExactFlatSnapshot(baselineRoot, "exact-squared-l2", VectorMetric.SquaredEuclidean, [0.9f, 0.1f, 0f]);
WriteExactFlatSnapshot(baselineRoot, "exact-inner-product", VectorMetric.InnerProduct, [1f, 0.5f, 0f]);
WriteExactFlatSnapshot(baselineRoot, "exact-cosine", VectorMetric.Cosine, [2f, 1f, 0f]);
WriteHnswSnapshot(baselineRoot, "hnsw-squared-l2", VectorMetric.SquaredEuclidean, [0.9f, 0.1f, 0f]);

if (IsAtLeast(baselineVersion, 1, 2, 0))
{
    WriteHnswSnapshot(baselineRoot, "hnsw-cosine", VectorMetric.Cosine, [2f, 1f, 0f]);
}
else
{
    exclusions.Add("hnsw-cosine: VecNet 1.0.0 HNSW supports squared L2 only; cosine HNSW was added in the 1.2 package line.");
}

File.WriteAllLines(Path.Combine(baselineRoot, "excluded-scenarios.txt"), exclusions);
Console.WriteLine($"VEC281_BASELINE_WRITER_PASSED version={baselineVersion}");

static void WriteExactFlatSnapshot(string baselineRoot, string scenarioName, VectorMetric metric, float[] query)
{
    string scenarioRoot = Path.Combine(baselineRoot, scenarioName);
    Directory.CreateDirectory(scenarioRoot);

    var index = new ExactFlatIndex(3, metric);
    AddExactRows(index);

    SearchResult[] expected = SearchExact(index, query, top: 4);
    index.Save(scenarioRoot);
    WriteExpected(Path.Combine(baselineRoot, scenarioName + ".expected.tsv"), "exact", metric, query, expected);
}

static void WriteHnswSnapshot(string baselineRoot, string scenarioName, VectorMetric metric, float[] query)
{
    string scenarioRoot = Path.Combine(baselineRoot, scenarioName);
    Directory.CreateDirectory(scenarioRoot);

    var options = new HnswIndexOptions(M: 8, EfConstruction: 24, EfSearch: 24, RandomSeed: 0x564543_281UL);
    var index = new HnswIndex(3, metric, options);
    AddHnswRows(index);

    SearchResult[] expected = SearchHnsw(index, query, top: 4);
    index.Save(scenarioRoot);
    WriteExpected(Path.Combine(baselineRoot, scenarioName + ".expected.tsv"), "hnsw", metric, query, expected);
}

static void AddExactRows(ExactFlatIndex index)
{
    index.Add(40, [10f, 0f, 0f]);
    index.Add(10, [1f, 1f, 0f]);
    index.Add(30, [0f, 2f, 0f]);
    index.Add(20, [-1f, 0f, 0f]);
    index.Add(50, [0f, 0f, 5f]);
    index.Add(60, [1f, 1f, 1f]);
}

static void AddHnswRows(HnswIndex index)
{
    index.Add(40, [10f, 0f, 0f]);
    index.Add(10, [1f, 1f, 0f]);
    index.Add(30, [0f, 2f, 0f]);
    index.Add(20, [-1f, 0f, 0f]);
    index.Add(50, [0f, 0f, 5f]);
    index.Add(60, [1f, 1f, 1f]);
}

static SearchResult[] SearchExact(ExactFlatIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results);
    return results[..written];
}

static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results, new HnswSearchWorkspace(index.Count, index.Options.EfSearch));
    return results[..written];
}

static void WriteExpected(string path, string kind, VectorMetric metric, float[] query, SearchResult[] expected)
{
    var lines = new List<string>
    {
        "kind\t" + kind,
        "metric\t" + metric,
        "query\t" + string.Join(",", query.Select(static value => value.ToString("R", CultureInfo.InvariantCulture))),
        "count\t" + expected.Length.ToString(CultureInfo.InvariantCulture)
    };

    foreach (SearchResult result in expected)
    {
        lines.Add(result.Id.ToString(CultureInfo.InvariantCulture) + "\t" + result.Distance.ToString("R", CultureInfo.InvariantCulture));
    }

    File.WriteAllLines(path, lines);
}

static bool IsAtLeast(string version, int major, int minor, int patch)
{
    Version parsed = Version.Parse(version);
    return parsed.Major > major ||
        parsed.Major == major && (parsed.Minor > minor ||
        parsed.Minor == minor && parsed.Build >= patch);
}

static void ResetDirectory(string path)
{
    if (Directory.Exists(path))
    {
        Directory.Delete(path, recursive: true);
    }
}
