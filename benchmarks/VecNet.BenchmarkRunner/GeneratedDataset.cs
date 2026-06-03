using System.Globalization;

namespace VecNet.BenchmarkRunner;

public sealed class GeneratedDataset
{
    public const string Kind = "generated-uniform";
    public const string Distribution = "uniform[-1,1)";

    public GeneratedDataset(int dimension, int vectorCount, int queryCount, uint seed, float[] vectors, float[] queries)
    {
        Dimension = dimension;
        VectorCount = vectorCount;
        QueryCount = queryCount;
        Seed = seed;
        Vectors = vectors;
        Queries = queries;
    }

    public int Dimension { get; }

    public int VectorCount { get; }

    public int QueryCount { get; }

    public uint Seed { get; }

    public float[] Vectors { get; }

    public float[] Queries { get; }

    public string SeedText => string.Create(CultureInfo.InvariantCulture, $"0x{Seed:X8}");

    public ReadOnlySpan<float> GetVector(int row) => Vectors.AsSpan(row * Dimension, Dimension);

    public ReadOnlySpan<float> GetQuery(int row) => Queries.AsSpan(row * Dimension, Dimension);
}

public static class GeneratedDatasetFactory
{
    public static GeneratedDataset Create(GeneratedExactSearchOptions options)
    {
        var random = new Random(unchecked((int)options.Seed));
        float[] vectors = new float[checked(options.VectorCount * options.Dimension)];
        float[] queries = new float[checked(options.QueryCount * options.Dimension)];

        Fill(random, vectors);
        Fill(random, queries);
        EnsureNonZeroRows(vectors, options.Dimension);
        EnsureNonZeroRows(queries, options.Dimension);

        return new GeneratedDataset(
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.Seed,
            vectors,
            queries);
    }

    private static void Fill(Random random, Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextSingle() * 2f - 1f;
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
