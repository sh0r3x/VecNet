using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec036HnswGeneratedTests
{
    [Fact]
    public void ParseHnswGenerated_UsesPrivateDefaults()
    {
        HnswGeneratedOptions options = CommandLine.ParseHnswGenerated(["hnsw-generated"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(10_000, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2036u, options.Seed);
        Assert.Equal(16, options.M);
        Assert.Equal(200, options.EfConstruction);
        Assert.Equal(50, options.EfSearch);
        Assert.Equal(0x564543_034UL, options.HnswSeed);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));
        Assert.EndsWith(".json", options.OutputPath);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnsw-generated", "--metric", "InnerProduct")]
    [InlineData("hnsw-generated", "--metric", "Unknown")]
    [InlineData("hnsw-generated", "--dimension", "0")]
    [InlineData("hnsw-generated", "--vectors", "0")]
    [InlineData("hnsw-generated", "--queries", "0")]
    [InlineData("hnsw-generated", "--top-k", "3", "--vectors", "2")]
    [InlineData("hnsw-generated", "--runs", "0")]
    [InlineData("hnsw-generated", "--runs", "6")]
    [InlineData("hnsw-generated", "--warmup-queries", "-1")]
    [InlineData("hnsw-generated", "--m", "1")]
    [InlineData("hnsw-generated", "--m", "65")]
    [InlineData("hnsw-generated", "--m", "8", "--ef-construction", "7")]
    [InlineData("hnsw-generated", "--ef-construction", "4097")]
    [InlineData("hnsw-generated", "--top-k", "10", "--ef-search", "9")]
    [InlineData("hnsw-generated", "--ef-search", "4097")]
    [InlineData("hnsw-generated", "--hnsw-seed", "0xNOTHEX")]
    [InlineData("hnsw-generated", "--unknown-option", "1")]
    [InlineData("hnsw-generated", "--output", "")]
    public void ParseHnswGenerated_RejectsInvalidCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ParseHnswGenerated_AcceptsCosineAndRejectsInnerProduct()
    {
        HnswGeneratedOptions cosine = CommandLine.ParseHnswGenerated(
            [
                "hnsw-generated",
                "--metric", "Cosine",
                "--dimension", "5",
                "--vectors", "12",
                "--queries", "2",
                "--top-k", "3",
                "--ef-search", "3"
            ]);

        Assert.Equal(VectorMetric.Cosine, cosine.Metric);

        ArgumentException rejected = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswGenerated(
                [
                    "hnsw-generated",
                    "--metric", "InnerProduct",
                    "--dimension", "5",
                    "--vectors", "12",
                    "--queries", "2",
                    "--top-k", "3",
                    "--ef-search", "3"
                ]));
        Assert.Contains("Cosine", rejected.Message, StringComparison.Ordinal);
        Assert.Contains("SquaredEuclidean", rejected.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Run_CosineProducesPassingPrivateReportAndMetricFields()
    {
        string outputPath = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec232-hnsw-generated-cosine-" + Guid.NewGuid().ToString("N") + ".json");
        string[] arguments =
        [
            "hnsw-generated",
            "--metric", "Cosine",
            "--dimension", "7",
            "--vectors", "32",
            "--queries", "3",
            "--top-k", "4",
            "--runs", "1",
            "--warmup-queries", "1",
            "--seed", "0x5EED2320",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000002320",
            "--output", outputPath
        ];
        HnswGeneratedOptions options = CommandLine.ParseHnswGenerated(arguments);

        HnswBenchmarkReport report = HnswGeneratedScenario.Run(options, arguments);
        HnswGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal(VectorMetric.Cosine.ToString(), report.Dataset.Metric);
        Assert.Equal(VectorMetric.Cosine.ToString(), report.Index.Metric);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.False(report.Evidence.PublicClaimEligible);
    }

    [Fact]
    public void Run_ProducesPrivateHnswReportWithBuildRecallAllocationAndMemoryEstimates()
    {
        string outputPath = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec036-direct-" + Guid.NewGuid().ToString("N") + ".json");
        string[] arguments =
        [
            "hnsw-generated",
            "--dimension", "12",
            "--vectors", "64",
            "--queries", "6",
            "--top-k", "5",
            "--runs", "3",
            "--warmup-queries", "4",
            "--seed", "0x5EED036A",
            "--m", "4",
            "--ef-construction", "16",
            "--ef-search", "8",
            "--hnsw-seed", "0x000000000000036A",
            "--output", outputPath
        ];
        HnswGeneratedOptions options = CommandLine.ParseHnswGenerated(arguments);

        HnswBenchmarkReport report = HnswGeneratedScenario.Run(options, arguments);
        HnswGeneratedScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.Equal("VecNet.HnswBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-036", report.TaskId);
        Assert.Equal("hnsw-generated", report.Command.Scenario);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("generated-hnsw-smoke", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.RegressionGateEligible);
        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Dataset.Metric);
        Assert.Equal("scalar-reference-generated", report.Truth.Kind);
        Assert.Equal(5, report.Truth.Depth);
        Assert.Equal("HnswIndex", report.Index.Type);
        Assert.Equal(4, report.Hnsw.M);
        Assert.Equal(4, report.Hnsw.MMax);
        Assert.Equal(8, report.Hnsw.MMax0);
        Assert.Equal(16, report.Hnsw.EfConstruction);
        Assert.Equal(8, report.Hnsw.EfSearch);
        Assert.Equal("0x000000000000036A", report.Hnsw.RandomSeed);
        Assert.Contains("row order", report.Hnsw.InsertionOrder, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Build.Status);
        Assert.True(report.Build.ElapsedMilliseconds >= 0);
        Assert.Equal("measured", report.Build.ManagedAllocations.Status);
        Assert.Equal("bytes", report.Build.ManagedAllocations.Unit);
        Assert.True(long.Parse(report.Build.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Contains("Add calls", report.Build.IncludedOperations, StringComparison.Ordinal);
        Assert.Equal(6, report.Search.MeasuredQueryCount);
        Assert.Equal(3, report.Search.Runs.Length);
        Assert.Equal<int>([1, 2, 3], report.Search.Runs.Select(run => run.RunNumber).ToArray());
        Assert.Equal(3, report.Search.Aggregate.RunCount);
        Assert.Equal(6, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.All(report.Search.Runs, run =>
        {
            Assert.Equal(6, run.MeasuredQueryCount);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.LatencyP50Milliseconds >= 0);
            Assert.True(run.LatencyP95Milliseconds >= run.LatencyP50Milliseconds);
            Assert.True(run.LatencyP99Milliseconds >= run.LatencyP95Milliseconds);
            Assert.True(run.Qps > 0);
            Assert.True(run.ManagedAllocatedBytes >= 0);
            Assert.True(run.ManagedAllocatedBytesPerQuery >= 0);
        });
        Assert.Equal("measured", report.Measurement.Latency.Status);
        Assert.Equal("internal HnswIndex.Search(query, results, workspace)", report.Measurement.Latency.TimedOperation);
        Assert.Contains("HNSW build", report.Measurement.Latency.ExcludedOperations, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.True(double.Parse(report.Measurement.ManagedAllocations.Value, CultureInfo.InvariantCulture) >= 0);
        Assert.Contains("caller-owned SearchResult[] and HnswSearchWorkspace", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Equal("absent", report.Measurement.Memory.Value);
        Assert.Equal("measured", report.Measurement.RepeatedRuns.Status);
        Assert.Equal("measured", report.Measurement.RunToRunNoise.Status);
        Assert.Equal("executed", report.Measurement.Warmup.Status);
        Assert.Equal(4, report.Measurement.Warmup.WarmupCount);
        Assert.Equal("estimated", report.MemoryEstimate.Status);
        Assert.Contains("layout-derived", report.MemoryEstimate.EstimateKind, StringComparison.OrdinalIgnoreCase);
        Assert.True(report.MemoryEstimate.TotalEstimatedBytes > 0);
        Assert.True(report.MemoryEstimate.VectorBytes > 0);
        Assert.True(report.MemoryEstimate.GraphAdjacencyBytes > 0);
        Assert.True(report.MemoryEstimate.SearchWorkspaceBytes > 0);
        Assert.Contains("not a resident/process/GC-heap measurement", report.MemoryEstimate.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(report.MemoryEstimate.Exclusions, item => item.Contains("Dictionary", StringComparison.OrdinalIgnoreCase));
        Assert.InRange(report.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Metrics.OrderedAgreement, 0, 1);
        Assert.Equal("passed", report.Metrics.DistanceToleranceStatus);
        Assert.Equal(0, report.Metrics.DistanceMismatchCount);
        Assert.Equal(0, report.Metrics.MissingResultCount);
        Assert.Equal(0, report.Metrics.ExtraResultCount);
        Assert.Equal("passed", report.Metrics.ReturnedResultIntegrity.Status);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.QueryCountMismatchCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.ResultCountViolationCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.NonFiniteDistanceCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, report.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Contains("Every returned", report.Metrics.ReturnedResultIntegrity.Policy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("set recall@k", report.Metrics.RecallDefinition, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Every returned HNSW result", report.Metrics.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not required", report.Metrics.DistanceValidationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.AllowsApproximateRecallBelowOne);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("estimated", root.GetProperty("memoryEstimate").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("latencyTicks", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Run_AllowsPassingApproximateReportWhenRecallIsBelowOne()
    {
        HnswBenchmarkReport? belowPerfect = null;
        for (uint seed = 0x5EED3600; seed < 0x5EED3650 && belowPerfect is null; seed++)
        {
            var options = new HnswGeneratedOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 24,
                VectorCount: 96,
                QueryCount: 8,
                TopK: 10,
                Seed: seed,
                OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec036-approx.json",
                Runs: 1,
                WarmupQueries: 0,
                M: 2,
                EfConstruction: 2,
                EfSearch: 10,
                HnswSeed: 0x36UL);
            HnswBenchmarkReport report = HnswGeneratedScenario.Run(options, ["hnsw-generated"]);
            if (report.Metrics.RecallAtK < 1)
            {
                belowPerfect = report;
            }
        }

        Assert.NotNull(belowPerfect);
        Assert.Equal("passed", belowPerfect.Validation.Status);
        Assert.True(belowPerfect.Validation.AllowsApproximateRecallBelowOne);
        Assert.InRange(belowPerfect.Metrics.RecallAtK, 0, 0.999999);
        Assert.Equal("passed", belowPerfect.Metrics.DistanceToleranceStatus);
        Assert.Equal("passed", belowPerfect.Metrics.ReturnedResultIntegrity.Status);
        Assert.True(belowPerfect.Metrics.ReturnedResultIntegrity.CheckedResultCount > 0);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DuplicateIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.UnknownIdCount);
        Assert.Equal(0, belowPerfect.Metrics.ReturnedResultIntegrity.DistanceMismatchCount);
        Assert.Equal(0, belowPerfect.Metrics.MissingResultCount);
        Assert.False(belowPerfect.Evidence.PublicClaimEligible);
        Assert.False(belowPerfect.Eligibility.BaselineCandidateEligible);
        Assert.False(belowPerfect.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void ValidateReturnedResults_PassesForWellFormedApproximateResultsOutsideExactTopK()
    {
        GeneratedDataset dataset = CreateGeneratedDataset(dimension: 5, vectorCount: 8, queryCount: 2, seed: 0x5EED3601);
        SearchResult[][] actual =
        [
            [
                ResultFor(dataset, queryRow: 0, id: 6),
                ResultFor(dataset, queryRow: 0, id: 2)
            ],
            [
                ResultFor(dataset, queryRow: 1, id: 7),
                ResultFor(dataset, queryRow: 1, id: 1)
            ]
        ];

        HnswReturnedResultIntegrityInfo integrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, actual, topK: 2);

        Assert.Equal("passed", integrity.Status);
        Assert.Equal(4, integrity.CheckedResultCount);
        Assert.Equal(0, integrity.QueryCountMismatchCount);
        Assert.Equal(0, integrity.ResultCountViolationCount);
        Assert.Equal(0, integrity.NonFiniteDistanceCount);
        Assert.Equal(0, integrity.DuplicateIdCount);
        Assert.Equal(0, integrity.UnknownIdCount);
        Assert.Equal(0, integrity.DistanceMismatchCount);
    }

    [Fact]
    public void ValidateReturnedResults_FailsMalformedApproximateResultsWithSpecificIntegrityCounts()
    {
        GeneratedDataset dataset = CreateGeneratedDataset(dimension: 4, vectorCount: 8, queryCount: 4, seed: 0x5EED3602);
        SearchResult[][] actual =
        [
            [
                ResultFor(dataset, queryRow: 0, id: 0),
                ResultFor(dataset, queryRow: 0, id: 1),
                ResultFor(dataset, queryRow: 0, id: 2)
            ],
            [
                ResultFor(dataset, queryRow: 1, id: 3),
                ResultFor(dataset, queryRow: 1, id: 3)
            ],
            [
                new SearchResult(99, 1)
            ],
            [
                new SearchResult(4, float.NaN),
                new SearchResult(5, ResultFor(dataset, queryRow: 3, id: 5).Distance + 1)
            ]
        ];

        HnswReturnedResultIntegrityInfo integrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, actual, topK: 2);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(8, integrity.CheckedResultCount);
        Assert.Equal(0, integrity.QueryCountMismatchCount);
        Assert.Equal(1, integrity.ResultCountViolationCount);
        Assert.Equal(1, integrity.NonFiniteDistanceCount);
        Assert.Equal(1, integrity.DuplicateIdCount);
        Assert.Equal(1, integrity.UnknownIdCount);
        Assert.Equal(2, integrity.DistanceMismatchCount);
        Assert.Contains("failed", integrity.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateReturnedResults_FailsWhenResultQueryCountDoesNotMatchDataset()
    {
        GeneratedDataset dataset = CreateGeneratedDataset(dimension: 4, vectorCount: 8, queryCount: 2, seed: 0x5EED3603);
        SearchResult[][] actual =
        [
            [
                ResultFor(dataset, queryRow: 0, id: 0)
            ]
        ];

        HnswReturnedResultIntegrityInfo integrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, actual, topK: 1);

        Assert.Equal("failed", integrity.Status);
        Assert.Equal(1, integrity.QueryCountMismatchCount);
        Assert.Equal(1, integrity.CheckedResultCount);
    }

    private static GeneratedDataset CreateGeneratedDataset(int dimension, int vectorCount, int queryCount, uint seed)
    {
        var options = new GeneratedExactSearchOptions(
            VectorMetric.SquaredEuclidean,
            dimension,
            vectorCount,
            queryCount,
            TopK: 1,
            Seed: seed,
            OutputPath: "VecNet.BenchmarkRunner.Artifacts/vec036-integrity-test.json",
            BaselineReportId: null);
        return GeneratedDatasetFactory.Create(options);
    }

    private static SearchResult ResultFor(GeneratedDataset dataset, int queryRow, ulong id) =>
        new(id, SquaredEuclidean(dataset.GetQuery(queryRow), dataset.GetVector(checked((int)id))));

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }
}
