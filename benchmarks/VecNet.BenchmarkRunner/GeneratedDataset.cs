using System.Globalization;

namespace VecNet.BenchmarkRunner;

public enum GeneratedVectorProfile
{
    Uniform,
    NormSkewed,
    ZeroVector
}

public sealed class GeneratedDataset
{
    public const string Kind = "generated-uniform";
    public const string Distribution = "uniform[-1,1)";

    public GeneratedDataset(
        int dimension,
        int vectorCount,
        int queryCount,
        uint seed,
        float[] vectors,
        float[] queries,
        GeneratedVectorProfile vectorProfile = GeneratedVectorProfile.Uniform,
        string? datasetKind = null,
        string? distribution = null)
    {
        Dimension = dimension;
        VectorCount = vectorCount;
        QueryCount = queryCount;
        Seed = seed;
        Vectors = vectors;
        Queries = queries;
        VectorProfile = vectorProfile;
        DatasetKind = datasetKind ?? Kind;
        ProfileDistribution = distribution ?? Distribution;
    }

    public int Dimension { get; }

    public int VectorCount { get; }

    public int QueryCount { get; }

    public uint Seed { get; }

    public float[] Vectors { get; }

    public float[] Queries { get; }

    public GeneratedVectorProfile VectorProfile { get; }

    public string DatasetKind { get; }

    public string ProfileDistribution { get; }

    public string SeedText => string.Create(CultureInfo.InvariantCulture, $"0x{Seed:X8}");

    public ReadOnlySpan<float> GetVector(int row) => Vectors.AsSpan(row * Dimension, Dimension);

    public ReadOnlySpan<float> GetQuery(int row) => Queries.AsSpan(row * Dimension, Dimension);
}

public static class GeneratedDatasetFactory
{
    public static GeneratedDataset Create(
        GeneratedExactSearchOptions options,
        GeneratedVectorProfile vectorProfile = GeneratedVectorProfile.Uniform)
    {
        var random = new Random(unchecked((int)options.Seed));
        float[] vectors = new float[checked(options.VectorCount * options.Dimension)];
        float[] queries = new float[checked(options.QueryCount * options.Dimension)];

        Fill(random, vectors);
        Fill(random, queries);
        ApplyProfile(vectors, options.Dimension, vectorProfile, isQuery: false);
        ApplyProfile(queries, options.Dimension, vectorProfile, isQuery: true);

        return new GeneratedDataset(
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.Seed,
            vectors,
            queries,
            vectorProfile,
            GetDatasetKind(vectorProfile),
            GetDistribution(vectorProfile));
    }

    public static GeneratedVectorProfile NormalizeVectorProfile(string value)
    {
        string normalized = value.Trim().ToLowerInvariant();
        return normalized switch
        {
            "uniform" or "generated-uniform" => GeneratedVectorProfile.Uniform,
            "norm-skewed" or "normskewed" or "generated-norm-skewed" => GeneratedVectorProfile.NormSkewed,
            "zero-vector" or "zerovector" or "generated-zero-vector" => GeneratedVectorProfile.ZeroVector,
            _ => throw new ArgumentException($"Unsupported generated vector profile '{value}'.")
        };
    }

    public static string GetOptionValue(GeneratedVectorProfile vectorProfile) =>
        vectorProfile switch
        {
            GeneratedVectorProfile.Uniform => "uniform",
            GeneratedVectorProfile.NormSkewed => "norm-skewed",
            GeneratedVectorProfile.ZeroVector => "zero-vector",
            _ => throw new ArgumentOutOfRangeException(nameof(vectorProfile), vectorProfile, "Unsupported generated vector profile.")
        };

    public static string GetDatasetKind(GeneratedVectorProfile vectorProfile) =>
        vectorProfile switch
        {
            GeneratedVectorProfile.Uniform => GeneratedDataset.Kind,
            GeneratedVectorProfile.NormSkewed => "generated-norm-skewed",
            GeneratedVectorProfile.ZeroVector => "generated-zero-vector",
            _ => throw new ArgumentOutOfRangeException(nameof(vectorProfile), vectorProfile, "Unsupported generated vector profile.")
        };

    public static string GetDistribution(GeneratedVectorProfile vectorProfile) =>
        vectorProfile switch
        {
            GeneratedVectorProfile.Uniform => GeneratedDataset.Distribution,
            GeneratedVectorProfile.NormSkewed => "uniform[-1,1) scaled by deterministic row norm factors [0.125,8]",
            GeneratedVectorProfile.ZeroVector => "uniform[-1,1) with deterministic explicit all-zero vector and query rows",
            _ => throw new ArgumentOutOfRangeException(nameof(vectorProfile), vectorProfile, "Unsupported generated vector profile.")
        };

    private static void Fill(Random random, Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextSingle() * 2f - 1f;
        }
    }

    private static void ApplyProfile(Span<float> values, int dimension, GeneratedVectorProfile vectorProfile, bool isQuery)
    {
        switch (vectorProfile)
        {
            case GeneratedVectorProfile.Uniform:
                EnsureNonZeroRows(values, dimension);
                break;
            case GeneratedVectorProfile.NormSkewed:
                ApplyNormSkew(values, dimension, isQuery);
                EnsureNonZeroRows(values, dimension);
                break;
            case GeneratedVectorProfile.ZeroVector:
                ApplyZeroVectors(values, dimension, isQuery);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(vectorProfile), vectorProfile, "Unsupported generated vector profile.");
        }
    }

    private static void ApplyNormSkew(Span<float> values, int dimension, bool isQuery)
    {
        ReadOnlySpan<float> vectorFactors = [0.125f, 0.25f, 0.5f, 1f, 2f, 4f, 8f];
        ReadOnlySpan<float> queryFactors = [8f, 2f, 0.5f, 0.125f, 4f, 1f, 0.25f];
        ReadOnlySpan<float> factors = isQuery ? queryFactors : vectorFactors;
        for (int rowOffset = 0, row = 0; rowOffset < values.Length; rowOffset += dimension, row++)
        {
            float factor = factors[row % factors.Length];
            for (int i = 0; i < dimension; i++)
            {
                values[rowOffset + i] *= factor;
            }
        }
    }

    private static void ApplyZeroVectors(Span<float> values, int dimension, bool isQuery)
    {
        int zeroPeriod = isQuery ? 3 : 5;
        for (int rowOffset = 0, row = 0; rowOffset < values.Length; rowOffset += dimension, row++)
        {
            if (row % zeroPeriod != 0)
            {
                continue;
            }

            values.Slice(rowOffset, dimension).Clear();
        }
    }

    private static void EnsureNonZeroRows(Span<float> values, int dimension)
    {
        for (int rowOffset = 0; rowOffset < values.Length; rowOffset += dimension)
        {
            bool anyNonZero = false;
            for (int i = 0; i < dimension; i++)
            {
                anyNonZero |= values[rowOffset + i] != 0f;
            }

            if (!anyNonZero)
            {
                values[rowOffset] = 1e-6f;
            }
        }
    }
}
