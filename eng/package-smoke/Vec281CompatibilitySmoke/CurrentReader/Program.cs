using System.Globalization;
using VecNet;

if (args.Length != 1)
{
    throw new ArgumentException("Usage: Vec281CurrentReader <artifact-root>");
}

string artifactRoot = Path.GetFullPath(args[0]);
string baselinesRoot = Path.Combine(artifactRoot, "baselines");
if (!Directory.Exists(baselinesRoot))
{
    throw new DirectoryNotFoundException("Baseline snapshot root was not found: " + baselinesRoot);
}

int scenarioCount = 0;
foreach (string expectedPath in Directory.EnumerateFiles(baselinesRoot, "*.expected.tsv", SearchOption.AllDirectories).OrderBy(static path => path, StringComparer.Ordinal))
{
    ExpectedScenario scenario = ReadExpected(expectedPath);
    string scenarioRoot = Path.Combine(Path.GetDirectoryName(expectedPath)!, Path.GetFileNameWithoutExtension(expectedPath).Replace(".expected", string.Empty));

    SearchResult[] actual = scenario.Kind switch
    {
        "exact" => SearchExact(ExactFlatIndex.OpenReadOnly(scenarioRoot), scenario.Query, scenario.Expected.Length),
        "hnsw" => SearchHnsw(HnswIndex.OpenReadOnly(scenarioRoot), scenario.Query, scenario.Expected.Length),
        _ => throw new InvalidDataException("Unknown scenario kind: " + scenario.Kind)
    };

    AssertResults(expectedPath, scenario.Expected, actual);
    scenarioCount++;
}

if (scenarioCount == 0)
{
    throw new InvalidOperationException("No baseline scenarios were found.");
}

Console.WriteLine($"VEC281_CURRENT_READER_PASSED scenarios={scenarioCount}");

static SearchResult[] SearchExact(ExactFlatIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results);
    return results[..written];
}

static SearchResult[] SearchHnsw(HnswIndex index, float[] query, int top)
{
    var results = new SearchResult[top];
    int written = index.Search(query, results, index.CreateSearchWorkspace());
    return results[..written];
}

static ExpectedScenario ReadExpected(string path)
{
    string[] lines = File.ReadAllLines(path);
    if (lines.Length < 4)
    {
        throw new InvalidDataException("Expected scenario file is incomplete: " + path);
    }

    string kind = ReadHeader(lines[0], "kind");
    VectorMetric metric = Enum.Parse<VectorMetric>(ReadHeader(lines[1], "metric"));
    float[] query = ReadHeader(lines[2], "query")
        .Split(',', StringSplitOptions.RemoveEmptyEntries)
        .Select(static value => float.Parse(value, CultureInfo.InvariantCulture))
        .ToArray();
    int count = int.Parse(ReadHeader(lines[3], "count"), CultureInfo.InvariantCulture);
    if (lines.Length != count + 4)
    {
        throw new InvalidDataException("Expected scenario result count does not match file length: " + path);
    }

    var expected = new SearchResult[count];
    for (int i = 0; i < count; i++)
    {
        string[] parts = lines[i + 4].Split('\t');
        if (parts.Length != 2)
        {
            throw new InvalidDataException("Expected scenario result line is invalid: " + path);
        }

        expected[i] = new SearchResult(
            ulong.Parse(parts[0], CultureInfo.InvariantCulture),
            float.Parse(parts[1], CultureInfo.InvariantCulture));
    }

    return new ExpectedScenario(kind, metric, query, expected);
}

static string ReadHeader(string line, string name)
{
    string prefix = name + "\t";
    if (!line.StartsWith(prefix, StringComparison.Ordinal))
    {
        throw new InvalidDataException("Expected header '" + name + "' was not found.");
    }

    return line[prefix.Length..];
}

static void AssertResults(string scenarioPath, SearchResult[] expected, SearchResult[] actual)
{
    if (actual.Length != expected.Length)
    {
        throw new InvalidOperationException($"{scenarioPath}: expected {expected.Length} results, got {actual.Length}.");
    }

    for (int i = 0; i < expected.Length; i++)
    {
        if (actual[i].Id != expected[i].Id)
        {
            throw new InvalidOperationException($"{scenarioPath}: result {i} expected ID {expected[i].Id}, got {actual[i].Id}.");
        }

        float delta = Math.Abs(actual[i].Distance - expected[i].Distance);
        if (delta > 0.00001f)
        {
            throw new InvalidOperationException($"{scenarioPath}: result {i} expected distance {expected[i].Distance:R}, got {actual[i].Distance:R}.");
        }
    }
}

internal sealed record ExpectedScenario(string Kind, VectorMetric Metric, float[] Query, SearchResult[] Expected);
