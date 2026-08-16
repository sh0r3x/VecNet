using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class InnerProductHotPathScenario
{
    private const string SchemaName = "VecNet.InnerProductHotPathReport";
    private const string SchemaVersion = "0.1";
    private const double ExactFiniteTolerance = 0;

    public static InnerProductHotPathReport Run(InnerProductHotPathOptions options)
    {
        ValidateOptions(options);

        string[] operationShapes = InnerProductHotPathOptions.ExpandOperationShapes(options.OperationShape);
        var cases = new List<InnerProductHotPathCaseInfo>(checked(options.Dimensions.Length * operationShapes.Length));
        foreach (int dimension in options.Dimensions)
        {
            GeneratedDataset dataset = GeneratedDatasetFactory.Create(
                new GeneratedExactSearchOptions(
                    VectorMetric.InnerProduct,
                    dimension,
                    options.VectorCount,
                    options.QueryCount,
                    1,
                    options.Seed,
                    options.OutputPath,
                    BaselineReportId: null),
                GeneratedVectorProfile.Uniform);

            foreach (string operationShape in operationShapes)
            {
                cases.Add(RunCase(options, dataset, operationShape));
            }
        }

        InnerProductHotPathValidationSummaryInfo validation = AggregateValidation(cases);
        RepositoryInfo repository = RepositoryInfo.Create();

        return new InnerProductHotPathReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options, operationShapes),
            DateTimeOffset.UtcNow,
            InnerProductHotPathOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            new InnerProductHotPathEvidenceInfo(
                "smoke",
                "inner-product-hot-path-measurement",
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private harness output is for a later measured optimization decision and is not a public benchmark claim.",
                [
                    "Generated data only; no external dataset source, license, version or checksum applies.",
                    "This report measures benchmark-only distance-call loops, not public API end-to-end product performance.",
                    "No production source path is changed or selected by this report.",
                    "Managed allocations are measured around benchmark-only distance-call loops.",
                    "No external implementation comparison is included."
                ]),
            new InnerProductHotPathSourceInfo(repository.Commit, repository.Dirty),
            new InnerProductHotPathRunnerInfo(
                "VecNet.BenchmarkRunner",
                "0.1",
                InnerProductHotPathOptions.ScenarioName),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new InnerProductHotPathOptionsInfo(
                VectorMetric.InnerProduct.ToString(),
                options.Dimensions.ToArray(),
                operationShapes,
                options.VectorCount,
                options.QueryCount,
                options.Runs,
                options.WarmupIterations,
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.Seed)),
            cases.ToArray(),
            validation,
            [
                "Private inner-product hot-path harness output only; not a public benchmark claim.",
                "Current scalar distance mirrors VecNet inner-product semantics: distance is negative dot product over supplied vectors.",
                "Candidate shared-dot distance is benchmark-only and does not change production search behavior.",
                "Operation shapes model exact-flat full-scan distance calls and HNSW build/search distance-call patterns for measurement triage.",
                "Validation compares current and candidate distance categories and finite values; finite, positive-infinity, negative-infinity and NaN categories are tracked."
            ]);
    }

    public static void Write(InnerProductHotPathReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    public static InnerProductHotPathCaseValidationInfo ValidateDistances(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        int dimension,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> currentDistance,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> candidateDistance)
    {
        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive.");
        }

        if (left.Length % dimension != 0 || right.Length % dimension != 0)
        {
            throw new ArgumentException("Validation inputs must contain complete rows.");
        }

        int pairCount = Math.Min(left.Length / dimension, right.Length / dimension);
        var counters = new ValidationCounters();
        for (int row = 0; row < pairCount; row++)
        {
            ReadOnlySpan<float> leftRow = left.Slice(row * dimension, dimension);
            ReadOnlySpan<float> rightRow = right.Slice(row * dimension, dimension);
            CompareDistance(row, leftRow, rightRow, currentDistance, candidateDistance, counters);
        }

        return counters.ToInfo();
    }

    public static InnerProductHotPathCaseInfo[] ExpandCases(InnerProductHotPathOptions options)
    {
        ValidateOptions(options);
        string[] operationShapes = InnerProductHotPathOptions.ExpandOperationShapes(options.OperationShape);
        var cases = new InnerProductHotPathCaseInfo[checked(options.Dimensions.Length * operationShapes.Length)];
        int index = 0;
        foreach (int dimension in options.Dimensions)
        {
            foreach (string operationShape in operationShapes)
            {
                cases[index++] = new InnerProductHotPathCaseInfo(
                    CreateCaseId(dimension, operationShape),
                    VectorMetric.InnerProduct.ToString(),
                    dimension,
                    GetDimensionClass(dimension),
                    operationShape,
                    CreateWorkloadInfo(options, dimension, operationShape),
                    EmptyMeasurement(InnerProductHotPathPrimitives.CurrentScalarName),
                    EmptyMeasurement(InnerProductHotPathPrimitives.CandidateSharedDotName),
                    new InnerProductHotPathCaseValidationInfo("notRun", 0, 0, 0, 0, 0, 0, 0, 0, []));
            }
        }

        return cases;
    }

    private static InnerProductHotPathCaseInfo RunCase(
        InnerProductHotPathOptions options,
        GeneratedDataset dataset,
        string operationShape)
    {
        InnerProductHotPathWorkloadInfo workload = CreateWorkloadInfo(options, dataset.Dimension, operationShape);
        Warmup(options, dataset, operationShape, InnerProductHotPathPrimitives.CurrentScalarDistance);
        Warmup(options, dataset, operationShape, InnerProductHotPathPrimitives.CandidateSharedDotDistance);

        InnerProductHotPathImplementationMeasurementInfo current = Measure(
            options,
            dataset,
            operationShape,
            InnerProductHotPathPrimitives.CurrentScalarName,
            InnerProductHotPathPrimitives.CurrentScalarDistance);
        InnerProductHotPathImplementationMeasurementInfo candidate = Measure(
            options,
            dataset,
            operationShape,
            InnerProductHotPathPrimitives.CandidateSharedDotName,
            InnerProductHotPathPrimitives.CandidateSharedDotDistance);
        InnerProductHotPathCaseValidationInfo validation = ValidateCase(dataset, operationShape, options);

        return new InnerProductHotPathCaseInfo(
            CreateCaseId(dataset.Dimension, operationShape),
            VectorMetric.InnerProduct.ToString(),
            dataset.Dimension,
            GetDimensionClass(dataset.Dimension),
            operationShape,
            workload,
            current,
            candidate,
            validation);
    }

    private static InnerProductHotPathImplementationMeasurementInfo Measure(
        InnerProductHotPathOptions options,
        GeneratedDataset dataset,
        string operationShape,
        string implementation,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> distance)
    {
        long distanceCallCount = GetDistanceCallCount(options, operationShape);
        long totalTicks = 0;
        long totalAllocatedBytes = 0;
        float checksum = 0;

        for (int run = 0; run < options.Runs; run++)
        {
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            checksum = ExecuteDistanceCalls(options, dataset, operationShape, distance);
            long elapsed = Stopwatch.GetTimestamp() - start;
            long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;
            totalTicks += elapsed;
            totalAllocatedBytes += allocatedBytes;
        }

        double elapsedSeconds = (double)totalTicks / Stopwatch.Frequency;
        double elapsedMilliseconds = elapsedSeconds * 1000;
        long totalDistanceCalls = checked(distanceCallCount * options.Runs);
        return new InnerProductHotPathImplementationMeasurementInfo(
            implementation,
            implementation,
            totalDistanceCalls,
            elapsedMilliseconds,
            elapsedSeconds == 0 ? double.PositiveInfinity : totalDistanceCalls / elapsedSeconds,
            totalAllocatedBytes,
            totalDistanceCalls == 0 ? 0 : (double)totalAllocatedBytes / totalDistanceCalls,
            InnerProductHotPathPrimitives.Category(checksum),
            checksum);
    }

    private static void Warmup(
        InnerProductHotPathOptions options,
        GeneratedDataset dataset,
        string operationShape,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> distance)
    {
        for (int i = 0; i < options.WarmupIterations; i++)
        {
            _ = ExecuteDistanceCalls(options, dataset, operationShape, distance);
        }
    }

    private static float ExecuteDistanceCalls(
        InnerProductHotPathOptions options,
        GeneratedDataset dataset,
        string operationShape,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> distance)
    {
        double checksum = 0;
        switch (operationShape)
        {
            case InnerProductHotPathOptions.ExactFlatSearchShape:
                for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
                {
                    ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
                    for (int vectorRow = 0; vectorRow < dataset.VectorCount; vectorRow++)
                    {
                        checksum += distance(query, dataset.GetVector(vectorRow));
                    }
                }

                break;
            case InnerProductHotPathOptions.HnswBuildDistanceCallsShape:
                for (int vectorRow = 1; vectorRow < dataset.VectorCount; vectorRow++)
                {
                    ReadOnlySpan<float> inserted = dataset.GetVector(vectorRow);
                    int comparisons = Math.Min(vectorRow, options.EfConstruction);
                    int start = vectorRow - comparisons;
                    for (int candidateRow = start; candidateRow < vectorRow; candidateRow++)
                    {
                        checksum += distance(inserted, dataset.GetVector(candidateRow));
                    }
                }

                break;
            case InnerProductHotPathOptions.HnswSearchDistanceCallsShape:
                int searchComparisons = Math.Min(dataset.VectorCount, options.EfSearch);
                for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
                {
                    ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
                    int start = queryRow % dataset.VectorCount;
                    for (int offset = 0; offset < searchComparisons; offset++)
                    {
                        checksum += distance(query, dataset.GetVector((start + offset) % dataset.VectorCount));
                    }
                }

                break;
            default:
                throw new ArgumentException($"Unsupported operation shape '{operationShape}'.", nameof(operationShape));
        }

        return (float)checksum;
    }

    private static InnerProductHotPathCaseValidationInfo ValidateCase(
        GeneratedDataset dataset,
        string operationShape,
        InnerProductHotPathOptions options)
    {
        var counters = new ValidationCounters();
        int validationCount = Math.Min(32, (int)Math.Min(int.MaxValue, GetDistanceCallCount(options, operationShape)));
        int compared = 0;

        switch (operationShape)
        {
            case InnerProductHotPathOptions.ExactFlatSearchShape:
                for (int queryRow = 0; queryRow < dataset.QueryCount && compared < validationCount; queryRow++)
                {
                    ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
                    for (int vectorRow = 0; vectorRow < dataset.VectorCount && compared < validationCount; vectorRow++)
                    {
                        CompareDistance(
                            compared,
                            query,
                            dataset.GetVector(vectorRow),
                            InnerProductHotPathPrimitives.CurrentScalarDistance,
                            InnerProductHotPathPrimitives.CandidateSharedDotDistance,
                            counters);
                        compared++;
                    }
                }

                break;
            case InnerProductHotPathOptions.HnswBuildDistanceCallsShape:
                for (int vectorRow = 1; vectorRow < dataset.VectorCount && compared < validationCount; vectorRow++)
                {
                    ReadOnlySpan<float> inserted = dataset.GetVector(vectorRow);
                    int comparisons = Math.Min(vectorRow, options.EfConstruction);
                    int start = vectorRow - comparisons;
                    for (int candidateRow = start; candidateRow < vectorRow && compared < validationCount; candidateRow++)
                    {
                        CompareDistance(
                            compared,
                            inserted,
                            dataset.GetVector(candidateRow),
                            InnerProductHotPathPrimitives.CurrentScalarDistance,
                            InnerProductHotPathPrimitives.CandidateSharedDotDistance,
                            counters);
                        compared++;
                    }
                }

                break;
            case InnerProductHotPathOptions.HnswSearchDistanceCallsShape:
                int searchComparisons = Math.Min(dataset.VectorCount, options.EfSearch);
                for (int queryRow = 0; queryRow < dataset.QueryCount && compared < validationCount; queryRow++)
                {
                    ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
                    int start = queryRow % dataset.VectorCount;
                    for (int offset = 0; offset < searchComparisons && compared < validationCount; offset++)
                    {
                        CompareDistance(
                            compared,
                            query,
                            dataset.GetVector((start + offset) % dataset.VectorCount),
                            InnerProductHotPathPrimitives.CurrentScalarDistance,
                            InnerProductHotPathPrimitives.CandidateSharedDotDistance,
                            counters);
                        compared++;
                    }
                }

                break;
        }

        ValidateBoundaryCategories(counters);
        return counters.ToInfo();
    }

    private static void ValidateBoundaryCategories(ValidationCounters counters)
    {
        ReadOnlySpan<float> finiteLeft = [1.25f, -2.5f, 4f, -8f];
        ReadOnlySpan<float> finiteRight = [-3f, -0.5f, 0.25f, 2f];
        ReadOnlySpan<float> positiveInfinityLeft = [float.MaxValue, float.MaxValue];
        ReadOnlySpan<float> positiveInfinityRight = [-float.MaxValue, -float.MaxValue];
        ReadOnlySpan<float> negativeInfinityLeft = [float.MaxValue, float.MaxValue];
        ReadOnlySpan<float> negativeInfinityRight = [float.MaxValue, float.MaxValue];
        ReadOnlySpan<float> nanLeft = [float.NaN, 1f];
        ReadOnlySpan<float> nanRight = [2f, 3f];

        CompareDistance(10_000, finiteLeft, finiteRight, InnerProductHotPathPrimitives.CurrentScalarDistance, InnerProductHotPathPrimitives.CandidateSharedDotDistance, counters);
        CompareDistance(10_001, positiveInfinityLeft, positiveInfinityRight, InnerProductHotPathPrimitives.CurrentScalarDistance, InnerProductHotPathPrimitives.CandidateSharedDotDistance, counters);
        CompareDistance(10_002, negativeInfinityLeft, negativeInfinityRight, InnerProductHotPathPrimitives.CurrentScalarDistance, InnerProductHotPathPrimitives.CandidateSharedDotDistance, counters);
        CompareDistance(10_003, nanLeft, nanRight, InnerProductHotPathPrimitives.CurrentScalarDistance, InnerProductHotPathPrimitives.CandidateSharedDotDistance, counters);
    }

    private static void CompareDistance(
        int pairIndex,
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> currentDistance,
        Func<ReadOnlySpan<float>, ReadOnlySpan<float>, float> candidateDistance,
        ValidationCounters counters)
    {
        float current = currentDistance(left, right);
        float candidate = candidateDistance(left, right);
        string currentCategory = InnerProductHotPathPrimitives.Category(current);
        string candidateCategory = InnerProductHotPathPrimitives.Category(candidate);
        counters.ComparedDistanceCount++;

        if (!string.Equals(currentCategory, candidateCategory, StringComparison.Ordinal))
        {
            counters.CategoryMismatchCount++;
            counters.AddExample(pairIndex, currentCategory, candidateCategory, current, candidate);
            return;
        }

        switch (currentCategory)
        {
            case "finite":
                if (Math.Abs((double)current - candidate) <= ExactFiniteTolerance)
                {
                    counters.FiniteMatchCount++;
                }
                else
                {
                    counters.FiniteDistanceMismatchCount++;
                    counters.MaxFiniteAbsoluteDelta = Math.Max(counters.MaxFiniteAbsoluteDelta, Math.Abs((double)current - candidate));
                    counters.AddExample(pairIndex, currentCategory, candidateCategory, current, candidate);
                }

                break;
            case "positiveInfinity":
                counters.PositiveInfinityMatchCount++;
                break;
            case "negativeInfinity":
                counters.NegativeInfinityMatchCount++;
                break;
            case "nan":
                counters.NaNMatchCount++;
                break;
        }
    }

    private static InnerProductHotPathValidationSummaryInfo AggregateValidation(List<InnerProductHotPathCaseInfo> cases)
    {
        int failedCases = cases.Count(item => item.Validation.Status != "passed");
        int comparedDistanceCount = cases.Sum(item => item.Validation.ComparedDistanceCount);
        int categoryMismatchCount = cases.Sum(item => item.Validation.CategoryMismatchCount);
        int finiteDistanceMismatchCount = cases.Sum(item => item.Validation.FiniteDistanceMismatchCount);
        int positiveInfinityComparisons = cases.Sum(item => item.Validation.PositiveInfinityMatchCount);
        int negativeInfinityComparisons = cases.Sum(item => item.Validation.NegativeInfinityMatchCount);
        int nanComparisons = cases.Sum(item => item.Validation.NaNMatchCount);

        return new InnerProductHotPathValidationSummaryInfo(
            failedCases == 0 ? "passed" : "failed",
            cases.Count,
            cases.Count - failedCases,
            failedCases,
            comparedDistanceCount,
            categoryMismatchCount,
            finiteDistanceMismatchCount,
            positiveInfinityComparisons,
            negativeInfinityComparisons,
            nanComparisons,
            "Candidate distances must preserve current scalar negative-dot finite values exactly for validation samples and must not change finite, positive-infinity, negative-infinity or NaN categories.");
    }

    private static InnerProductHotPathWorkloadInfo CreateWorkloadInfo(
        InnerProductHotPathOptions options,
        int dimension,
        string operationShape)
    {
        long distanceCallCount = GetDistanceCallCount(options, operationShape);
        string callShape = operationShape switch
        {
            InnerProductHotPathOptions.ExactFlatSearchShape => "query vectors compared against every stored vector",
            InnerProductHotPathOptions.HnswBuildDistanceCallsShape => "inserted vectors compared against a bounded preceding candidate window",
            InnerProductHotPathOptions.HnswSearchDistanceCallsShape => "query vectors compared against a bounded efSearch candidate window",
            _ => throw new ArgumentException($"Unsupported operation shape '{operationShape}'.", nameof(operationShape))
        };

        return new InnerProductHotPathWorkloadInfo(
            options.VectorCount,
            options.QueryCount,
            distanceCallCount,
            callShape,
            string.Create(
                CultureInfo.InvariantCulture,
                $"benchmark-only inner-product distance calls over dimension {dimension} for {operationShape}"),
            "dataset generation, report writing, production ExactFlatIndex.Search and production HnswIndex.Add/Search are excluded");
    }

    private static long GetDistanceCallCount(InnerProductHotPathOptions options, string operationShape) =>
        operationShape switch
        {
            InnerProductHotPathOptions.ExactFlatSearchShape => checked((long)options.QueryCount * options.VectorCount),
            InnerProductHotPathOptions.HnswBuildDistanceCallsShape => GetHnswBuildDistanceCallCount(options.VectorCount, options.EfConstruction),
            InnerProductHotPathOptions.HnswSearchDistanceCallsShape => checked((long)options.QueryCount * Math.Min(options.VectorCount, options.EfSearch)),
            _ => throw new ArgumentException($"Unsupported operation shape '{operationShape}'.", nameof(operationShape))
        };

    private static long GetHnswBuildDistanceCallCount(int vectorCount, int efConstruction)
    {
        long calls = 0;
        for (int vectorRow = 1; vectorRow < vectorCount; vectorRow++)
        {
            calls += Math.Min(vectorRow, efConstruction);
        }

        return calls;
    }

    private static InnerProductHotPathImplementationMeasurementInfo EmptyMeasurement(string implementation) =>
        new(implementation, implementation, 0, 0, 0, 0, 0, "finite", 0);

    private static string GetDimensionClass(int dimension) =>
        dimension is 128 or 384 or 768 or 1536 ? "representative" : "awkward";

    private static string CreateCaseId(int dimension, string operationShape) =>
        string.Create(CultureInfo.InvariantCulture, $"{operationShape}-{dimension}d");

    private static string CreateReportId(string? commit, InnerProductHotPathOptions options, string[] operationShapes)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        string dimensions = string.Join("-", options.Dimensions);
        string shapes = string.Join("-", operationShapes.Select(static item => item.Replace("-distance-calls", string.Empty, StringComparison.Ordinal)));
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{InnerProductHotPathOptions.ScenarioName}-{commitPart}-{dimensions}d-{shapes}-{options.VectorCount}v-{options.QueryCount}q-{options.Runs}r-{options.WarmupIterations}w-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}");
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static void ValidateOptions(InnerProductHotPathOptions options)
    {
        if (options.Dimensions.Length == 0)
        {
            throw new ArgumentException("At least one dimension is required.", nameof(options));
        }

        foreach (int dimension in options.Dimensions)
        {
            if (dimension <= 0)
            {
                throw new ArgumentException("All dimensions must be positive.", nameof(options));
            }
        }

        if (options.VectorCount <= 0)
        {
            throw new ArgumentException("Vector count must be positive.", nameof(options));
        }

        if (options.QueryCount <= 0)
        {
            throw new ArgumentException("Query count must be positive.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("Runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupIterations < 0)
        {
            throw new ArgumentException("Warmup iterations must be non-negative.", nameof(options));
        }

        if (options.EfConstruction <= 0)
        {
            throw new ArgumentException("ef-construction must be positive.", nameof(options));
        }

        if (options.EfSearch <= 0)
        {
            throw new ArgumentException("ef-search must be positive.", nameof(options));
        }

        _ = InnerProductHotPathOptions.ExpandOperationShapes(options.OperationShape);
    }

    private sealed class ValidationCounters
    {
        private readonly List<string> _driftExamples = [];

        public int ComparedDistanceCount { get; set; }

        public int FiniteMatchCount { get; set; }

        public int PositiveInfinityMatchCount { get; set; }

        public int NegativeInfinityMatchCount { get; set; }

        public int NaNMatchCount { get; set; }

        public int CategoryMismatchCount { get; set; }

        public int FiniteDistanceMismatchCount { get; set; }

        public double MaxFiniteAbsoluteDelta { get; set; }

        public void AddExample(int pairIndex, string currentCategory, string candidateCategory, float current, float candidate)
        {
            if (_driftExamples.Count >= 8)
            {
                return;
            }

            _driftExamples.Add(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"pair={pairIndex}; currentCategory={currentCategory}; candidateCategory={candidateCategory}; current={current:R}; candidate={candidate:R}"));
        }

        public InnerProductHotPathCaseValidationInfo ToInfo() =>
            new(
                CategoryMismatchCount == 0 && FiniteDistanceMismatchCount == 0 ? "passed" : "failed",
                ComparedDistanceCount,
                FiniteMatchCount,
                PositiveInfinityMatchCount,
                NegativeInfinityMatchCount,
                NaNMatchCount,
                CategoryMismatchCount,
                FiniteDistanceMismatchCount,
                MaxFiniteAbsoluteDelta,
                _driftExamples.ToArray());
    }
}
