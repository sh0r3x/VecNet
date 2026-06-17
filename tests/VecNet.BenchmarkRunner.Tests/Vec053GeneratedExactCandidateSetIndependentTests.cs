using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec053GeneratedExactCandidateSetIndependentTests
{
    public static IEnumerable<object[]> CandidateInputShapes()
    {
        VectorMetric[] metrics =
        [
            VectorMetric.SquaredEuclidean,
            VectorMetric.InnerProduct,
            VectorMetric.Cosine
        ];

        (string Label, string Kind, int DuplicateIds, int UnknownIds, int ExpectedKnown)[] shapes =
        [
            ("empty", "empty", 11, 17, 0),
            ("all", "all", 3, 5, 37),
            ("broad", "broad", 4, 7, 19),
            ("selective", "selective", 6, 9, 4),
            ("very-selective", "very-selective", 8, 10, 7),
            ("duplicate-heavy", "selective", 25, 2, 4),
            ("unknown-heavy", "broad", 2, 31, 19)
        ];

        uint seed = 0x5EED_5300;
        foreach (VectorMetric metric in metrics)
        {
            foreach ((string label, string kind, int duplicateIds, int unknownIds, int expectedKnown) in shapes)
            {
                yield return [metric, label, kind, duplicateIds, unknownIds, expectedKnown, seed++];
            }
        }
    }

    [Theory]
    [MemberData(nameof(CandidateInputShapes))]
    public void CandidateSetReport_MatchesRawAllowlistFilteredTruthForAdversarialShapes(
        VectorMetric metric,
        string shapeLabel,
        string candidateSetKind,
        int duplicateIds,
        int unknownIds,
        int expectedKnown,
        uint seed)
    {
        const int vectorCount = 37;
        const int queryCount = 5;
        const int topK = 8;
        int expectedCheckedResults = queryCount * Math.Min(topK, expectedKnown);

        var candidateOptions = new GeneratedExactCandidateSetOptions(
            metric,
            Dimension: 17,
            VectorCount: vectorCount,
            QueryCount: queryCount,
            TopK: topK,
            Seed: seed,
            CandidateSetKind: candidateSetKind,
            DuplicateIdsPerQuery: duplicateIds,
            UnknownIdsPerQuery: unknownIds,
            OutputPath: NewArtifactPath($"{shapeLabel}-candidate-set.json"),
            Runs: 2,
            WarmupQueries: 3);
        var filteredOptions = new GeneratedExactFilteredOptions(
            metric,
            Dimension: candidateOptions.Dimension,
            VectorCount: candidateOptions.VectorCount,
            QueryCount: candidateOptions.QueryCount,
            TopK: candidateOptions.TopK,
            Seed: candidateOptions.Seed,
            FilterKind: candidateSetKind,
            DuplicateIdsPerQuery: duplicateIds,
            UnknownIdsPerQuery: unknownIds,
            OutputPath: NewArtifactPath($"{shapeLabel}-raw-allowlist.json"),
            Runs: candidateOptions.Runs,
            WarmupQueries: candidateOptions.WarmupQueries);

        GeneratedExactCandidateSetBenchmarkReport candidateReport = GeneratedExactCandidateSetScenario.Run(
            candidateOptions,
            ["generated-exact-candidate-set", "--candidate-set", candidateSetKind]);
        GeneratedExactFilteredBenchmarkReport rawAllowlistReport = GeneratedExactFilteredScenario.Run(
            filteredOptions,
            ["exact-generated-filtered", "--filter", candidateSetKind]);

        Assert.Equal("passed", candidateReport.Validation.Status);
        Assert.Equal("passed", rawAllowlistReport.Validation.Status);
        Assert.Equal("passed", candidateReport.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal("passed", rawAllowlistReport.Metrics.FilteredResultIntegrity.Status);
        Assert.Equal(1.0, candidateReport.Metrics.RecallAtK);
        Assert.Equal(rawAllowlistReport.Metrics.RecallAtK, candidateReport.Metrics.RecallAtK);
        Assert.Equal(rawAllowlistReport.Metrics.OrderedAgreement, candidateReport.Metrics.OrderedAgreement);
        Assert.Equal(rawAllowlistReport.Metrics.DistanceMismatchCount, candidateReport.Metrics.DistanceMismatchCount);
        Assert.Equal(rawAllowlistReport.Metrics.MissingResultCount, candidateReport.Metrics.MissingResultCount);
        Assert.Equal(rawAllowlistReport.Metrics.ExtraResultCount, candidateReport.Metrics.ExtraResultCount);
        Assert.Equal(rawAllowlistReport.Metrics.FilteredResultIntegrity.CheckedResultCount, candidateReport.Metrics.FilteredResultIntegrity.CheckedResultCount);
        Assert.Equal(rawAllowlistReport.Metrics.FilteredResultIntegrity.WrongIdCount, candidateReport.Metrics.FilteredResultIntegrity.WrongIdCount);
        Assert.Equal(rawAllowlistReport.Metrics.FilteredResultIntegrity.UnresolvedWrongIdCount, candidateReport.Metrics.FilteredResultIntegrity.UnresolvedWrongIdCount);
        Assert.Equal(rawAllowlistReport.Metrics.FilteredResultIntegrity.DistanceMismatchCount, candidateReport.Metrics.FilteredResultIntegrity.DistanceMismatchCount);
        Assert.Equal(expectedCheckedResults, candidateReport.Metrics.FilteredResultIntegrity.CheckedResultCount);

        Assert.Equal(candidateSetKind, candidateReport.CandidateInput.Kind);
        Assert.Equal(candidateSetKind, rawAllowlistReport.Filter.Kind);
        Assert.Equal(expectedKnown, candidateReport.CandidateInput.KnownIdCountPerQuery);
        Assert.Equal(expectedKnown, rawAllowlistReport.Filter.KnownIdCountPerQuery);
        Assert.Equal(expectedKnown, candidateReport.CandidateSet.CountPerQuery);
        Assert.Equal(expectedKnown, candidateReport.CandidateSet.MinCount);
        Assert.Equal(expectedKnown, candidateReport.CandidateSet.MaxCount);
        Assert.Equal(queryCount * expectedKnown, candidateReport.CandidateSet.TotalCandidateCount);
        Assert.Equal(duplicateIds, candidateReport.CandidateInput.DuplicateIdCountPerQuery);
        Assert.Equal(unknownIds, candidateReport.CandidateInput.UnknownIdCountPerQuery);
        Assert.Equal(expectedKnown + duplicateIds + unknownIds, candidateReport.CandidateInput.InputIdCountPerQuery);
        Assert.Contains("authorization", candidateReport.CandidateInput.ApplicationScope, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("constructedOutsideMeasuredSearch", candidateReport.CandidateSet.ConstructionStatus);
        Assert.True(candidateReport.CandidateSet.ConstructedBeforeMeasuredSearch);
        Assert.Contains("coalesced", candidateReport.CandidateSet.DuplicateHandling, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ignored", candidateReport.CandidateSet.UnknownIdHandling, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("public ExactFlatIndex.Search(query, candidateSet, results)", candidateReport.Measurement.Latency.TimedOperation);
        Assert.Equal("public ExactFlatIndex.Search(query, allowedIds, results, workspace)", rawAllowlistReport.Measurement.Latency.TimedOperation);
        Assert.DoesNotContain("allowedIds", candidateReport.Measurement.Latency.TimedOperation, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("workspace", candidateReport.Measurement.Latency.TimedOperation, StringComparison.OrdinalIgnoreCase);
        Assert.False(candidateReport.Evidence.PublicClaimEligible);
        Assert.False(candidateReport.Evidence.BaselineCandidateEligible);
        Assert.False(candidateReport.Evidence.RegressionGateEligible);
        Assert.False(candidateReport.Eligibility.PublicClaimEligible);
        Assert.False(candidateReport.Eligibility.BaselineCandidateEligible);
        Assert.False(candidateReport.Eligibility.RegressionGateEligible);
    }

    [Fact]
    public void SingleRunReport_MarksWarmupAndRunToRunNoiseAbsentWhileAllocationRemainsSearchOnly()
    {
        GeneratedExactCandidateSetBenchmarkReport report = GeneratedExactCandidateSetScenario.Run(
            new GeneratedExactCandidateSetOptions(
                VectorMetric.Cosine,
                Dimension: 11,
                VectorCount: 29,
                QueryCount: 4,
                TopK: 6,
                Seed: 0x5EED_5351,
                CandidateSetKind: "broad",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 23,
                OutputPath: NewArtifactPath("single-run.json"),
                Runs: 1,
                WarmupQueries: 0),
            ["generated-exact-candidate-set", "--candidate-set", "broad"]);
        string json = ReportWriter.Serialize(report);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement measurement = root.GetProperty("measurement");

        Assert.Equal("singleRun", report.Measurement.RepeatedRuns.Status);
        Assert.False(report.Measurement.RepeatedRuns.VarianceMeasured);
        Assert.Equal("notMeasured", report.Measurement.RunToRunNoise.Status);
        Assert.False(report.Measurement.RunToRunNoise.NoiseMeasured);
        Assert.Equal("absent", report.Measurement.Warmup.Status);
        Assert.Equal(0, report.Measurement.Warmup.WarmupCount);
        Assert.Single(report.Search.Runs);
        Assert.Equal(1, report.Search.Aggregate.RunCount);
        Assert.Equal(report.Search.MeasuredQueryCount, report.Search.Aggregate.MeasuredQueryCountPerRun);
        Assert.Equal("measured", report.Measurement.ManagedAllocations.Status);
        Assert.Equal("bytesPerQuery", report.Measurement.ManagedAllocations.Unit);
        Assert.Contains("public ExactFlatIndex.Search(query, candidateSet, results)", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Contains("prebuilt ExactFlatCandidateSet", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Contains("candidate-set construction", report.Measurement.ManagedAllocations.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("ExactFlatIndex.CreateCandidateSet", report.Measurement.ManagedAllocations.Reason, StringComparison.Ordinal);
        Assert.Equal("notMeasured", report.Measurement.Memory.Status);
        Assert.Contains("candidate-set retained memory", report.Measurement.Memory.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("singleRun", measurement.GetProperty("repeatedRuns").GetProperty("status").GetString());
        Assert.False(measurement.GetProperty("repeatedRuns").GetProperty("varianceMeasured").GetBoolean());
        Assert.Equal("notMeasured", measurement.GetProperty("runToRunNoise").GetProperty("status").GetString());
        Assert.Equal("absent", measurement.GetProperty("warmup").GetProperty("status").GetString());
        Assert.DoesNotContain("latencyTicks", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("baselineCandidateEligible\":true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("regressionGateEligible\":true", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("publicClaimEligible\":true", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CandidateSetParser_IsolatedFromRawAllowlistMatrixAndHnswOptions()
    {
        GeneratedExactCandidateSetOptions options = CommandLine.ParseGeneratedExactCandidateSet(
            [
                "generated-exact-candidate-set",
                "--candidate-set", "verySelective",
                "--candidate-set", "ALL",
                "--vectors", "16",
                "--queries", "2",
                "--top-k", "4"
            ]);

        Assert.Equal("all", options.CandidateSetKind);
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--filter", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--preset", "smoke"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--output-dir", "VecNet.BenchmarkRunner.Artifacts/matrix"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--manifest", "manifest.json"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--baseline-report-id", "baseline"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--ef-search", "10"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactCandidateSet(["generated-exact-candidate-set", "--hnsw-seed", "0x1234"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFiltered(["exact-generated-filtered", "--candidate-set", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseGeneratedExactFilteredMatrix(["exact-generated-filtered-matrix", "--candidate-set", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseMatrix(["exact-generated-matrix", "--candidate-set", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--candidate-set", "all"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGeneratedMatrix(["hnsw-generated-matrix", "--candidate-set", "all"]));
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec053-independent-" + Path.GetFileNameWithoutExtension(fileName) + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
