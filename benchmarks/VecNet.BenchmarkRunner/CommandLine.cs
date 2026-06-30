using System.Globalization;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner;

public static class CommandLine
{
    public static GeneratedExactSearchOptions Parse(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactSearchOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactSearchOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2009);
        string outputPath = values.TryGetValue("output", out string? output)
            ? output
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        string? baselineReportId = GetOptionalNonWhiteSpace(values, "baseline-report-id");

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        return new GeneratedExactSearchOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            baselineReportId,
            runs,
            warmupQueries);
    }

    public static GeneratedExactMatrixOptions ParseMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedMatrixOption);
        string presetName = GeneratedExactMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 8);
        int runs = GetPositiveInt(values, "runs", 1);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2014);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "matrix-manifest.json");

        int maxTopK = GeneratedExactMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new GeneratedExactMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            outputDirectory,
            manifestPath);
    }

    public static BenchmarkComparisonOptions ParseComparison(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? BenchmarkComparisonOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, BenchmarkComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedComparisonOption);
        string baselinePath = GetRequiredNonWhiteSpace(values, "baseline");
        string currentPath = GetRequiredNonWhiteSpace(values, "current");
        string outputPath = values.TryGetValue("output", out string? output)
            ? output
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                "comparisons",
                $"benchmark-comparison-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");

        return new BenchmarkComparisonOptions(baselinePath, currentPath, outputPath);
    }

    public static GeneratedExactFilteredOptions ParseGeneratedExactFiltered(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactFilteredOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactFilteredOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactFilteredOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2046);
        string filterKind = GeneratedExactFilteredOptions.NormalizeFilterKind(
            GetOptionalNonWhiteSpace(values, "filter") ?? GeneratedExactFilteredOptions.DefaultFilterKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-filtered-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (filterKind == "very-selective" && topK <= 1)
        {
            throw new ArgumentException("Option --filter very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactFilteredOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            filterKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            runs,
            warmupQueries);
    }

    public static GeneratedExactFilteredMatrixOptions ParseGeneratedExactFilteredMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactFilteredMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactFilteredMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactFilteredMatrixOption);
        string presetName = GeneratedExactFilteredMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactFilteredMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 4);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2047);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"exact-generated-filtered-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "exact-filtered-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        int maxTopK = GeneratedExactFilteredMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum generated exact-filter matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new GeneratedExactFilteredMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputDirectory,
            manifestPath);
    }

    public static GeneratedExactCandidateSetOptions ParseGeneratedExactCandidateSet(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactCandidateSetOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactCandidateSetOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactCandidateSetOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2053);
        string candidateSetKind = GeneratedExactCandidateSetOptions.NormalizeCandidateSetKind(
            GetOptionalNonWhiteSpace(values, "candidate-set") ?? GeneratedExactCandidateSetOptions.DefaultCandidateSetKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-candidate-set-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (candidateSetKind == "very-selective" && topK <= 1)
        {
            throw new ArgumentException("Option --candidate-set very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactCandidateSetOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            candidateSetKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            runs,
            warmupQueries);
    }

    public static GeneratedExactCandidateSetMatrixOptions ParseGeneratedExactCandidateSetMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactCandidateSetMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactCandidateSetMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactCandidateSetMatrixOption);
        string presetName = GeneratedExactCandidateSetMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactCandidateSetMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 4);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2054);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-candidate-set-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "exact-candidate-set-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        int maxTopK = GeneratedExactCandidateSetMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum generated exact candidate-set matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new GeneratedExactCandidateSetMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputDirectory,
            manifestPath);
    }

    public static GeneratedExactUpdateOptions ParseGeneratedExactUpdate(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactUpdateOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactUpdateOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactUpdateOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int baseVectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2061);
        int insertedDeltaCount = GetPositiveInt(values, "insertions", Math.Max(1, baseVectorCount / 10));
        int deletedBaseCount = GetPositiveInt(values, "deletes", Math.Max(1, baseVectorCount / 10));
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        string allowlistKind = GeneratedExactUpdateOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "allowlist") ?? GeneratedExactUpdateOptions.DefaultAllowlistKind);
        string candidateSetKind = GeneratedExactUpdateOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "candidate-set") ?? GeneratedExactUpdateOptions.DefaultCandidateSetKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-update-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (deletedBaseCount > baseVectorCount)
        {
            throw new ArgumentException("Option --deletes must be less than or equal to --vectors.");
        }

        int liveVectorCount = checked(baseVectorCount + insertedDeltaCount - deletedBaseCount);
        if (topK > liveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-mutation live vector count.");
        }

        if ((allowlistKind == "very-selective" || candidateSetKind == "very-selective") && topK <= 1)
        {
            throw new ArgumentException("Option --allowlist/--candidate-set very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactUpdateOptions(
            metric,
            dimension,
            baseVectorCount,
            queryCount,
            topK,
            seed,
            insertedDeltaCount,
            deletedBaseCount,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            allowlistKind,
            candidateSetKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            runs,
            warmupQueries);
    }

    public static GeneratedExactCheckpointOptions ParseGeneratedExactCheckpoint(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactCheckpointOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactCheckpointOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactCheckpointOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int baseVectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2067);
        int insertedDeltaCount = GetPositiveInt(values, "insertions", Math.Max(1, baseVectorCount / 10));
        int deletedBaseCount = GetPositiveInt(values, "deletes", Math.Max(1, baseVectorCount / 10));
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        string allowlistKind = GeneratedExactCheckpointOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "allowlist") ?? GeneratedExactCheckpointOptions.DefaultAllowlistKind);
        string candidateSetKind = GeneratedExactCheckpointOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "candidate-set") ?? GeneratedExactCheckpointOptions.DefaultCandidateSetKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-checkpoint-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (deletedBaseCount > baseVectorCount)
        {
            throw new ArgumentException("Option --deletes must be less than or equal to --vectors.");
        }

        int liveVectorCount = checked(baseVectorCount + insertedDeltaCount - deletedBaseCount);
        if (topK > liveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the pre-checkpoint live vector count.");
        }

        if ((allowlistKind == "very-selective" || candidateSetKind == "very-selective") && topK <= 1)
        {
            throw new ArgumentException("Option --allowlist/--candidate-set very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactCheckpointOptions(
            metric,
            dimension,
            baseVectorCount,
            queryCount,
            topK,
            seed,
            insertedDeltaCount,
            deletedBaseCount,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            allowlistKind,
            candidateSetKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            runs,
            warmupQueries);
    }

    public static GeneratedExactPracticalUpdateOptions ParseGeneratedExactPracticalUpdate(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactPracticalUpdateOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactPracticalUpdateOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactPracticalUpdateOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int baseVectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2079);
        int insertedDeltaCount = GetPositiveInt(values, "insertions", Math.Max(1, baseVectorCount / 10));
        int deletedBaseCount = GetPositiveInt(values, "deletes", Math.Max(1, baseVectorCount / 10));
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        string allowlistKind = GeneratedExactPracticalUpdateOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "allowlist") ?? GeneratedExactPracticalUpdateOptions.DefaultAllowlistKind);
        string candidateSetKind = GeneratedExactPracticalUpdateOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "candidate-set") ?? GeneratedExactPracticalUpdateOptions.DefaultCandidateSetKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-practical-update-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string checkpointDirectory = values.TryGetValue("checkpoint-directory", out string? checkpointDirectoryValue)
            ? checkpointDirectoryValue
            : Path.Combine(
                Path.GetDirectoryName(outputPath) ?? "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-practical-update-checkpoint-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(checkpointDirectory))
        {
            throw new ArgumentException("Option --checkpoint-directory must not be empty.");
        }

        if (deletedBaseCount > baseVectorCount)
        {
            throw new ArgumentException("Option --deletes must be less than or equal to --vectors.");
        }

        int liveVectorCount = checked(baseVectorCount + insertedDeltaCount - deletedBaseCount);
        if (topK > liveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-mutation live vector count.");
        }

        if ((allowlistKind == "very-selective" || candidateSetKind == "very-selective") && topK <= 1)
        {
            throw new ArgumentException("Option --allowlist/--candidate-set very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactPracticalUpdateOptions(
            metric,
            dimension,
            baseVectorCount,
            queryCount,
            topK,
            seed,
            insertedDeltaCount,
            deletedBaseCount,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            allowlistKind,
            candidateSetKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            checkpointDirectory,
            runs,
            warmupQueries);
    }

    public static GeneratedExactOpenedSearchOptions ParseGeneratedExactOpenedSearch(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactOpenedSearchOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactOpenedSearchOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactOpenedSearchOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int vectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2092);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-opened-search-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string indexDirectory = values.TryGetValue("index-directory", out string? indexDirectoryValue)
            ? indexDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-opened-search-index-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(indexDirectory))
        {
            throw new ArgumentException("Option --index-directory must not be empty.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        return new GeneratedExactOpenedSearchOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            indexDirectory,
            runs,
            warmupQueries);
    }

    public static GeneratedExactMemorySmokeOptions ParseGeneratedExactMemorySmoke(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactMemorySmokeOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactMemorySmokeOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactMemorySmokeOption);

        VectorMetric metric = GetEnum(values, "metric", VectorMetric.SquaredEuclidean);
        int dimension = GetPositiveInt(values, "dimension", 128);
        int baseVectorCount = GetPositiveInt(values, "vectors", 10_000);
        int queryCount = GetPositiveInt(values, "queries", 100);
        int topK = GetPositiveInt(values, "top-k", 10);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 1);
        uint seed = GetSeed(values, "seed", 0x5EED2094);
        int insertedDeltaCount = GetPositiveInt(values, "insertions", Math.Max(1, baseVectorCount / 10));
        int deletedBaseCount = GetPositiveInt(values, "deletes", Math.Max(1, baseVectorCount / 10));
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        string allowlistKind = GeneratedExactMemorySmokeOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "allowlist") ?? GeneratedExactMemorySmokeOptions.DefaultAllowlistKind);
        string candidateSetKind = GeneratedExactMemorySmokeOptions.NormalizeSelectivityKind(
            GetOptionalNonWhiteSpace(values, "candidate-set") ?? GeneratedExactMemorySmokeOptions.DefaultCandidateSetKind);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-memory-smoke-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string saveDirectory = values.TryGetValue("save-directory", out string? saveDirectoryValue)
            ? saveDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-memory-smoke-save-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(saveDirectory))
        {
            throw new ArgumentException("Option --save-directory must not be empty.");
        }

        string checkpointDirectory = values.TryGetValue("checkpoint-directory", out string? checkpointDirectoryValue)
            ? checkpointDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-memory-smoke-checkpoint-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(checkpointDirectory))
        {
            throw new ArgumentException("Option --checkpoint-directory must not be empty.");
        }

        if (deletedBaseCount > baseVectorCount)
        {
            throw new ArgumentException("Option --deletes must be less than or equal to --vectors.");
        }

        int liveVectorCount = checked(baseVectorCount + insertedDeltaCount - deletedBaseCount);
        if (topK > liveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-mutation live vector count.");
        }

        if ((allowlistKind == "very-selective" || candidateSetKind == "very-selective") && topK <= 1)
        {
            throw new ArgumentException("Option --allowlist/--candidate-set very-selective requires --top-k greater than 1.");
        }

        return new GeneratedExactMemorySmokeOptions(
            metric,
            dimension,
            baseVectorCount,
            queryCount,
            topK,
            seed,
            insertedDeltaCount,
            deletedBaseCount,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            allowlistKind,
            candidateSetKind,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputPath,
            saveDirectory,
            checkpointDirectory,
            warmupQueries);
    }

    public static GeneratedExactUpdateMatrixOptions ParseGeneratedExactUpdateMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactUpdateMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactUpdateMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactUpdateMatrixOption);
        string presetName = GeneratedExactUpdateMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactUpdateMatrixOptions.DefaultPresetName);

        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2062);
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-update-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "exact-update-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        return new GeneratedExactUpdateMatrixOptions(
            presetName,
            runs,
            warmupQueries,
            seed,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputDirectory,
            manifestPath);
    }

    public static GeneratedExactCheckpointMatrixOptions ParseGeneratedExactCheckpointMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? GeneratedExactCheckpointMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, GeneratedExactCheckpointMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedGeneratedExactCheckpointMatrixOption);
        string presetName = GeneratedExactCheckpointMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? GeneratedExactCheckpointMatrixOptions.DefaultPresetName);

        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2069);
        int duplicateInsertAttempts = GetNonNegativeInt(values, "duplicate-inserts", 1);
        int unknownDeleteAttempts = GetNonNegativeInt(values, "unknown-deletes", 1);
        int repeatedDeleteAttempts = GetNonNegativeInt(values, "repeated-deletes", 1);
        int duplicateIdsPerQuery = GetNonNegativeInt(values, "duplicate-ids", 0);
        int unknownIdsPerQuery = GetNonNegativeInt(values, "unknown-ids", 0);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-exact-checkpoint-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "exact-checkpoint-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        return new GeneratedExactCheckpointMatrixOptions(
            presetName,
            runs,
            warmupQueries,
            seed,
            duplicateInsertAttempts,
            unknownDeleteAttempts,
            repeatedDeleteAttempts,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            outputDirectory,
            manifestPath);
    }

    public static DurableHnswGeneratedOptions ParseDurableHnswGenerated(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? DurableHnswGeneratedOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, DurableHnswGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedDurableHnswGeneratedOption);
        DurableHnswGeneratedOptions defaults = DurableHnswGeneratedOptions.Default;

        VectorMetric metric = GetEnum(values, "metric", defaults.Metric);
        int dimension = GetPositiveInt(values, "dimension", defaults.Dimension);
        int vectorCount = GetPositiveInt(values, "vectors", defaults.VectorCount);
        int queryCount = GetPositiveInt(values, "queries", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        uint seed = GetSeed(values, "seed", defaults.Seed);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-durable-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string snapshotDirectory = values.TryGetValue("snapshot-directory", out string? snapshotDirectoryValue)
            ? snapshotDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-durable-snapshot-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(snapshotDirectory))
        {
            throw new ArgumentException("Option --snapshot-directory must not be empty.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("hnsw-generated-durable supports only SquaredEuclidean.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK || efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be at least --top-k and no more than 4096.");
        }

        return new DurableHnswGeneratedOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            snapshotDirectory,
            runs,
            warmupQueries,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    public static HnswMemorySmokeOptions ParseHnswMemorySmoke(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswMemorySmokeOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswMemorySmokeOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswMemorySmokeOption);
        HnswMemorySmokeOptions defaults = HnswMemorySmokeOptions.Default;

        VectorMetric metric = GetEnum(values, "metric", defaults.Metric);
        int dimension = GetPositiveInt(values, "dimension", defaults.Dimension);
        int vectorCount = GetPositiveInt(values, "vectors", defaults.VectorCount);
        int queryCount = GetPositiveInt(values, "queries", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        uint seed = GetSeed(values, "seed", defaults.Seed);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);
        int sampleIntervalMilliseconds = GetPositiveInt(values, "sample-interval-ms", defaults.SampleIntervalMilliseconds);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-hnsw-memory-smoke-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string snapshotDirectory = values.TryGetValue("snapshot-directory", out string? snapshotDirectoryValue)
            ? snapshotDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"generated-hnsw-memory-smoke-snapshot-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(snapshotDirectory))
        {
            throw new ArgumentException("Option --snapshot-directory must not be empty.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("generated-hnsw-memory-smoke supports only SquaredEuclidean.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK || efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be at least --top-k and no more than 4096.");
        }

        if (sampleIntervalMilliseconds > 1000)
        {
            throw new ArgumentException("Option --sample-interval-ms must be in the range 1..1000.");
        }

        return new HnswMemorySmokeOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            snapshotDirectory,
            warmupQueries,
            m,
            efConstruction,
            efSearch,
            hnswSeed,
            sampleIntervalMilliseconds);
    }

    public static HnswEstablishedComparisonOptions ParseHnswEstablishedComparison(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswEstablishedComparisonOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswEstablishedComparisonOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswEstablishedComparisonOption);
        HnswEstablishedComparisonOptions defaults = HnswEstablishedComparisonOptions.Default;

        VectorMetric metric = GetEnum(values, "metric", defaults.Metric);
        int dimension = GetPositiveInt(values, "dimension", defaults.Dimension);
        int vectorCount = GetPositiveInt(values, "vectors", defaults.VectorCount);
        int queryCount = GetPositiveInt(values, "queries", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        uint seed = GetSeed(values, "seed", defaults.Seed);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnswlib-generated-comparison-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
        string workDirectory = values.TryGetValue("work-directory", out string? workDirectoryValue)
            ? workDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnswlib-generated-comparison-work-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        string vecNetSnapshotDirectory = values.TryGetValue("vecnet-snapshot-directory", out string? vecNetSnapshotDirectoryValue)
            ? vecNetSnapshotDirectoryValue
            : Path.Combine(workDirectory, "vecnet-snapshot");
        string hnswlibIndexPath = values.TryGetValue("hnswlib-index", out string? hnswlibIndexValue)
            ? hnswlibIndexValue
            : Path.Combine(workDirectory, "hnswlib-index.bin");
        string hnswlibPythonPath = values.TryGetValue("hnswlib-python", out string? hnswlibPythonValue)
            ? hnswlibPythonValue
            : defaults.HnswlibPythonPath;

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(workDirectory))
        {
            throw new ArgumentException("Option --work-directory must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(vecNetSnapshotDirectory))
        {
            throw new ArgumentException("Option --vecnet-snapshot-directory must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(hnswlibIndexPath))
        {
            throw new ArgumentException("Option --hnswlib-index must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(hnswlibPythonPath))
        {
            throw new ArgumentException("Option --hnswlib-python must not be empty.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("hnswlib-generated-comparison supports only SquaredEuclidean.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK || efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be at least --top-k and no more than 4096.");
        }

        return new HnswEstablishedComparisonOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            workDirectory,
            vecNetSnapshotDirectory,
            hnswlibIndexPath,
            hnswlibPythonPath,
            runs,
            warmupQueries,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    public static DurableHnswGeneratedMatrixOptions ParseDurableHnswGeneratedMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? DurableHnswGeneratedMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, DurableHnswGeneratedMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedDurableHnswGeneratedMatrixOption);
        string presetName = DurableHnswGeneratedMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? DurableHnswGeneratedMatrixOptions.DefaultPresetName);
        uint seed = GetSeed(values, "seed", 0x5EED0750);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine("VecNet.BenchmarkRunner.Artifacts", "hnsw-generated-durable-matrix");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "durable-hnsw-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        return new DurableHnswGeneratedMatrixOptions(
            presetName,
            seed,
            outputDirectory,
            manifestPath);
    }

    public static HnswGeneratedMatrixOptions ParseHnswGeneratedMatrix(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswGeneratedMatrixOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswGeneratedMatrixOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswGeneratedMatrixOption);
        string presetName = HnswGeneratedMatrixOptions.NormalizePresetName(
            GetOptionalNonWhiteSpace(values, "preset") ?? HnswGeneratedMatrixOptions.DefaultPresetName);

        int vectorCount = GetPositiveInt(values, "vectors", 128);
        int queryCount = GetPositiveInt(values, "queries", 4);
        int runs = GetPositiveInt(values, "runs", 1);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", 0);
        uint seed = GetSeed(values, "seed", 0x5EED2037);
        string outputDirectory = values.TryGetValue("output-dir", out string? outputDirectoryValue)
            ? outputDirectoryValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-matrix-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Option --output-dir must not be empty.");
        }

        string manifestPath = values.TryGetValue("manifest", out string? manifestValue)
            ? manifestValue
            : Path.Combine(outputDirectory, "hnsw-matrix-manifest.json");
        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            throw new ArgumentException("Option --manifest must not be empty.");
        }

        int maxTopK = HnswGeneratedMatrixScenario.GetMaxTopK(presetName);
        if (vectorCount < maxTopK)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"vectors must be greater than or equal to the maximum HNSW matrix top-k ({maxTopK}) for preset '{presetName}'."));
        }

        return new HnswGeneratedMatrixOptions(
            presetName,
            vectorCount,
            queryCount,
            runs,
            warmupQueries,
            seed,
            outputDirectory,
            manifestPath);
    }

    public static HnswGeneratedOptions ParseHnswGenerated(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? HnswGeneratedOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, HnswGeneratedOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedHnswGeneratedOption);
        HnswGeneratedOptions defaults = HnswGeneratedOptions.Default;
        VectorMetric metric = GetEnum(values, "metric", defaults.Metric);
        int dimension = GetPositiveInt(values, "dimension", defaults.Dimension);
        int vectorCount = GetPositiveInt(values, "vectors", defaults.VectorCount);
        int queryCount = GetPositiveInt(values, "queries", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        uint seed = GetSeed(values, "seed", defaults.Seed);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);
        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : Path.Combine(
                "VecNet.BenchmarkRunner.Artifacts",
                $"hnsw-generated-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        if (metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("hnsw-generated supports only SquaredEuclidean.");
        }

        if (topK > vectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.");
        }

        if (efSearch < topK)
        {
            throw new ArgumentException("ef-search must be greater than or equal to top-k.");
        }

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be in the range 1..4096.");
        }

        return new HnswGeneratedOptions(
            metric,
            dimension,
            vectorCount,
            queryCount,
            topK,
            seed,
            outputPath,
            runs,
            warmupQueries,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    public static FashionMnistExternalDatasetOptions ParseExternalFashionMnist(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalDatasetOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalDatasetOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedExternalFashionMnistOption);
        FashionMnistExternalDatasetOptions defaults = FashionMnistExternalDatasetOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int truthDepth = GetPositiveInt(values, "truth-depth", defaults.TruthDepth);
        bool download = GetBoolean(values, "download", defaults.DownloadRawFiles);

        return new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, download);
    }

    public static FashionMnistExternalExactBenchmarkOptions ParseExternalFashionMnistExact(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalExactBenchmarkOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalExactBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, 1, IsSupportedExternalFashionMnistExactOption);
        FashionMnistExternalExactBenchmarkOptions defaults = FashionMnistExternalExactBenchmarkOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : defaults.OutputPath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        VectorMetric metric = GetExternalFashionMnistMetric(values, "metric", defaults.Metric);

        return new FashionMnistExternalExactBenchmarkOptions(cacheRoot, outputPath, queryCount, topK, runs, warmupQueries, metric);
    }

    public static FashionMnistExternalHnswBenchmarkOptions ParseExternalFashionMnistHnsw(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalHnswBenchmarkOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalHnswBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedExternalFashionMnistHnswOption);
        FashionMnistExternalHnswBenchmarkOptions defaults = FashionMnistExternalHnswBenchmarkOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : defaults.OutputPath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        VectorMetric metric = GetExternalFashionMnistMetric(values, "metric", defaults.Metric);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK)
        {
            throw new ArgumentException("Option --ef-search must be greater than or equal to --top-k.");
        }

        if (efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be in the range 1..4096.");
        }

        return new FashionMnistExternalHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            queryCount,
            topK,
            runs,
            warmupQueries,
            metric,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    public static FashionMnistExternalDurableHnswBenchmarkOptions ParseExternalFashionMnistDurableHnsw(IReadOnlyList<string> args)
    {
        string scenario = args.Count == 0 ? FashionMnistExternalDurableHnswBenchmarkOptions.ScenarioName : args[0];
        if (!string.Equals(scenario, FashionMnistExternalDurableHnswBenchmarkOptions.ScenarioName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported scenario '{scenario}'.");
        }

        Dictionary<string, string> values = ParseOptionValues(args, args.Count == 0 ? 0 : 1, IsSupportedExternalFashionMnistDurableHnswOption);
        FashionMnistExternalDurableHnswBenchmarkOptions defaults = FashionMnistExternalDurableHnswBenchmarkOptions.Default;
        string cacheRoot = values.TryGetValue("cache-root", out string? cacheRootValue)
            ? cacheRootValue
            : defaults.CacheRoot;
        if (string.IsNullOrWhiteSpace(cacheRoot))
        {
            throw new ArgumentException("Option --cache-root must not be empty.");
        }

        string outputPath = values.TryGetValue("output", out string? outputValue)
            ? outputValue
            : defaults.OutputPath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Option --output must not be empty.");
        }

        string snapshotDirectory = values.TryGetValue("snapshot-directory", out string? snapshotDirectoryValue)
            ? snapshotDirectoryValue
            : defaults.SnapshotDirectory;
        if (string.IsNullOrWhiteSpace(snapshotDirectory))
        {
            throw new ArgumentException("Option --snapshot-directory must not be empty.");
        }

        int queryCount = GetPositiveInt(values, "query-count", defaults.QueryCount);
        int topK = GetPositiveInt(values, "top-k", defaults.TopK);
        int runs = GetPositiveInt(values, "runs", defaults.Runs);
        if (runs > 5)
        {
            throw new ArgumentException("Option --runs must be in the range 1..5.");
        }

        int warmupQueries = GetNonNegativeInt(values, "warmup-queries", defaults.WarmupQueries);
        VectorMetric metric = GetExternalFashionMnistMetric(values, "metric", defaults.Metric);
        int m = GetPositiveInt(values, "m", defaults.M);
        int efConstruction = GetPositiveInt(values, "ef-construction", defaults.EfConstruction);
        int efSearch = GetPositiveInt(values, "ef-search", defaults.EfSearch);
        ulong hnswSeed = GetUInt64Seed(values, "hnsw-seed", defaults.HnswSeed);

        if (m is < 2 or > 64)
        {
            throw new ArgumentException("Option --m must be in the range 2..64.");
        }

        if (efConstruction < m || efConstruction > 4096)
        {
            throw new ArgumentException("Option --ef-construction must be at least --m and no more than 4096.");
        }

        if (efSearch < topK)
        {
            throw new ArgumentException("Option --ef-search must be greater than or equal to --top-k.");
        }

        if (efSearch > 4096)
        {
            throw new ArgumentException("Option --ef-search must be in the range 1..4096.");
        }

        return new FashionMnistExternalDurableHnswBenchmarkOptions(
            cacheRoot,
            outputPath,
            snapshotDirectory,
            queryCount,
            topK,
            runs,
            warmupQueries,
            metric,
            m,
            efConstruction,
            efSearch,
            hnswSeed);
    }

    private static Dictionary<string, string> ParseOptionValues(
        IReadOnlyList<string> args,
        int startIndex,
        Func<string, bool> isSupportedOption)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = startIndex; i < args.Count; i += 2)
        {
            string name = args[i];
            if (!name.StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Count)
            {
                throw new ArgumentException($"Expected an option/value pair at '{name}'.");
            }

            string optionName = name[2..];
            if (!isSupportedOption(optionName))
            {
                throw new ArgumentException($"Unsupported option '{name}'.");
            }

            string value = args[i + 1];
            if (value.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Option '{name}' requires a value.");
            }

            values[optionName] = value;
        }

        return values;
    }

    private static TEnum GetEnum<TEnum>(Dictionary<string, string> values, string name, TEnum defaultValue)
        where TEnum : struct, Enum
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!Enum.TryParse(value, ignoreCase: true, out TEnum parsed) || !Enum.IsDefined(parsed))
        {
            throw new ArgumentException($"Option --{name} has unsupported value '{value}'.");
        }

        return parsed;
    }

    private static int GetPositiveInt(Dictionary<string, string> values, string name, int defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new ArgumentException($"Option --{name} must be a positive integer.");
        }

        return parsed;
    }

    private static int GetNonNegativeInt(Dictionary<string, string> values, string name, int defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed < 0)
        {
            throw new ArgumentException($"Option --{name} must be a non-negative integer.");
        }

        return parsed;
    }

    private static bool IsSupportedGeneratedOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "baseline-report-id", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedComparisonOption(string name) =>
        string.Equals(name, "baseline", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "current", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactFilteredOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "filter", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactFilteredMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactCandidateSetOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "candidate-set", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactCandidateSetMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactUpdateOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "insertions", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "allowlist", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "candidate-set", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactCheckpointOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "insertions", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "allowlist", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "candidate-set", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactPracticalUpdateOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "insertions", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "allowlist", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "candidate-set", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "checkpoint-directory", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactOpenedSearchOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "index-directory", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactMemorySmokeOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "insertions", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "allowlist", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "candidate-set", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "save-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "checkpoint-directory", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactUpdateMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedGeneratedExactCheckpointMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-inserts", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "repeated-deletes", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "duplicate-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "unknown-ids", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "truth-depth", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "download", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistExactOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistHnswOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedExternalFashionMnistDurableHnswOption(string name) =>
        string.Equals(name, "cache-root", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "snapshot-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "query-count", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedDurableHnswGeneratedOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "snapshot-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswMemorySmokeOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "snapshot-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "sample-interval-ms", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswEstablishedComparisonOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "work-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vecnet-snapshot-directory", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnswlib-index", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnswlib-python", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedDurableHnswGeneratedMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswGeneratedOption(string name) =>
        string.Equals(name, "metric", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "dimension", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "top-k", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "m", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-construction", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "ef-search", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "hnsw-seed", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedHnswGeneratedMatrixOption(string name) =>
        string.Equals(name, "preset", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "vectors", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "runs", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "warmup-queries", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "seed", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "output-dir", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(name, "manifest", StringComparison.OrdinalIgnoreCase);

    private static VectorMetric GetExternalFashionMnistMetric(Dictionary<string, string> values, string name, VectorMetric defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (string.Equals(value, "squared-euclidean", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, nameof(VectorMetric.SquaredEuclidean), StringComparison.OrdinalIgnoreCase))
        {
            return VectorMetric.SquaredEuclidean;
        }

        throw new ArgumentException($"Option --{name} has unsupported value '{value}'.");
    }

    private static uint GetSeed(Dictionary<string, string> values, string name, uint defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        NumberStyles styles = value.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? NumberStyles.AllowHexSpecifier
            : NumberStyles.None;
        string normalized = styles == NumberStyles.AllowHexSpecifier ? value[2..] : value;

        if (!uint.TryParse(normalized, styles, CultureInfo.InvariantCulture, out uint parsed))
        {
            throw new ArgumentException($"Option --{name} must be an unsigned integer or hexadecimal value.");
        }

        return parsed;
    }

    private static string? GetOptionalNonWhiteSpace(Dictionary<string, string> values, string name)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return null;
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"Option --{name} must not be empty.");
        }

        return value;
    }

    private static string GetRequiredNonWhiteSpace(Dictionary<string, string> values, string name)
    {
        string? value = GetOptionalNonWhiteSpace(values, name);
        if (value is null)
        {
            throw new ArgumentException($"Option --{name} is required.");
        }

        return value;
    }

    private static bool GetBoolean(Dictionary<string, string> values, string name, bool defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        if (!bool.TryParse(value, out bool parsed))
        {
            throw new ArgumentException($"Option --{name} must be true or false.");
        }

        return parsed;
    }

    private static ulong GetUInt64Seed(Dictionary<string, string> values, string name, ulong defaultValue)
    {
        if (!values.TryGetValue(name, out string? value))
        {
            return defaultValue;
        }

        NumberStyles styles = value.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? NumberStyles.AllowHexSpecifier
            : NumberStyles.None;
        string normalized = styles == NumberStyles.AllowHexSpecifier ? value[2..] : value;

        if (!ulong.TryParse(normalized, styles, CultureInfo.InvariantCulture, out ulong parsed))
        {
            throw new ArgumentException($"Option --{name} must be an unsigned integer or hexadecimal value.");
        }

        return parsed;
    }
}
