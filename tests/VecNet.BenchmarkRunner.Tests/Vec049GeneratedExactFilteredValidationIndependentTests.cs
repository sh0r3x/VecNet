using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec049GeneratedExactFilteredValidationIndependentTests
{
    [Fact]
    public void ValidateFilteredResults_AcceptsMultiPositionSquaredL2NearTiePermutation()
    {
        const int dimension = 386;
        const float first = 1000f;
        float second = first + 0.05f;
        float third = first + 0.10f;
        var truth = new TruthSet(
            [[new TruthItem(1, first), new TruthItem(2, second), new TruthItem(3, third)]],
            depth: 3);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(3, third), new SearchResult(1, first), new SearchResult(2, second)]],
            topK: 3,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("passed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(3, comparison.Integrity.WrongIdCount);
        Assert.Equal(3, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(3, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(0, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.DistanceMismatchCount);
        Assert.Equal("acceptedNearTie", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("accepted D-026 near-tie/order-equivalence case", comparison.Integrity.Classification);
    }

    [Fact]
    public void ValidateFilteredResults_DoesNotHideUnknownIdInsideOtherwiseNearTiePermutation()
    {
        const int dimension = 386;
        const float first = 1000f;
        float second = first + 0.05f;
        float third = first + 0.10f;
        var truth = new TruthSet(
            [[new TruthItem(1, first), new TruthItem(2, second), new TruthItem(3, third)]],
            depth: 3);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, second), new SearchResult(99, second), new SearchResult(1, first)]],
            topK: 3,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(2.0 / 3.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(3, comparison.Integrity.WrongIdCount);
        Assert.Equal(3, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(2, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal("unresolved", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
    }

    [Fact]
    public void ValidateFilteredResults_RejectsSquaredL2SwapJustOutsideNearTieTolerance()
    {
        const int dimension = 386;
        const float first = 1000f;
        float second = first + 0.75f;
        Assert.True(second - first > SquaredEuclideanTolerance(dimension, first) + SquaredEuclideanTolerance(dimension, second));
        var truth = new TruthSet([[new TruthItem(1, first), new TruthItem(2, second)]], depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, second), new SearchResult(1, first)]],
            topK: 2,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(2, comparison.Integrity.WrongIdCount);
        Assert.Equal(2, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(0, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(2, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(2, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal("notApplicable", comparison.Integrity.OrderEquivalenceStatus);
    }

    [Fact]
    public void ValidateFilteredResults_RejectsNearTiePermutationWhenReturnedDistanceBelongsToWrongId()
    {
        const int dimension = 386;
        const float first = 1000f;
        float second = first + 0.05f;
        float wrongSecondDistance = second + (SquaredEuclideanTolerance(dimension, second) * 3f);
        var truth = new TruthSet([[new TruthItem(1, first), new TruthItem(2, second)]], depth: 2);

        GeneratedExactFilteredResultComparison comparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            truth,
            [[new SearchResult(2, wrongSecondDistance), new SearchResult(1, first)]],
            topK: 2,
            dimension,
            VectorMetric.SquaredEuclidean);

        Assert.Equal("failed", comparison.Integrity.Status);
        Assert.Equal(1.0, comparison.RecallAtK);
        Assert.Equal(0.0, comparison.OrderedAgreement);
        Assert.Equal(2, comparison.Integrity.WrongIdCount);
        Assert.Equal(2, comparison.Integrity.OrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.ToleratedNearTieOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedWrongIdCount);
        Assert.Equal(1, comparison.Integrity.UnresolvedOrderMismatchCount);
        Assert.Equal(1, comparison.Integrity.DistanceMismatchCount);
        Assert.Equal("unresolved", comparison.Integrity.OrderEquivalenceStatus);
        Assert.Equal("filtered result validation failure", comparison.Integrity.Classification);
    }

    [Fact]
    public void Run_Vec049ReportFieldsPreserveVec046EligibilityPosture()
    {
        string reportPath = NewArtifactPath("accepted-near-tie-report.json");

        GeneratedExactFilteredBenchmarkReport report = GeneratedExactFilteredScenario.Run(
            new GeneratedExactFilteredOptions(
                VectorMetric.SquaredEuclidean,
                Dimension: 386,
                VectorCount: 1000,
                QueryCount: 16,
                TopK: 100,
                Seed: 0x5EED049B,
                FilterKind: "selective",
                DuplicateIdsPerQuery: 1,
                UnknownIdsPerQuery: 1,
                OutputPath: reportPath,
                Runs: 3,
                WarmupQueries: 8),
            ["exact-generated-filtered", "--filter", "selective"]);
        GeneratedExactFilteredScenario.Write(report, reportPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(reportPath));
        JsonElement root = document.RootElement;
        JsonElement integrity = root.GetProperty("metrics").GetProperty("filteredResultIntegrity");

        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-046", root.GetProperty("taskId").GetString());
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("acceptedNearTie", integrity.GetProperty("orderEquivalenceStatus").GetString());
        Assert.Equal(2, integrity.GetProperty("toleratedNearTieOrderMismatchCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("unresolvedWrongIdCount").GetInt32());
        Assert.Equal(0, integrity.GetProperty("unresolvedOrderMismatchCount").GetInt32());
        Assert.Equal("accepted D-026 near-tie/order-equivalence case", integrity.GetProperty("classification").GetString());
        Assert.False(root.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        AssertNoPropertyNamed(root, "comparisonArtifact", "regressionDecision", "regressionGateStatus");
    }

    [Fact]
    public void MatrixSmokeLinkedReportsCarryHardenedIntegrityFieldsWithoutChangingMatrixEligibility()
    {
        string directory = NewArtifactDirectory("matrix-smoke");
        string manifestPath = Path.Combine(directory, "exact-filtered-matrix-manifest.json");
        var options = new GeneratedExactFilteredMatrixOptions(
            GeneratedExactFilteredMatrixOptions.SmokePresetName,
            VectorCount: 100,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED0490,
            DuplicateIdsPerQuery: 1,
            UnknownIdsPerQuery: 1,
            OutputDirectory: directory,
            ManifestPath: manifestPath);

        GeneratedExactFilteredMatrixManifest manifest = GeneratedExactFilteredMatrixScenario.Run(
            options,
            ["exact-generated-filtered-matrix", "--preset", "smoke"]);
        GeneratedExactFilteredMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal(8, manifest.CaseCount);
        Assert.Equal(8, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);

        GeneratedExactFilteredMatrixCaseManifest firstCase = manifest.Cases[0];
        Assert.Equal("passed", firstCase.Status);
        Assert.Equal("passed", firstCase.ValidationStatus);

        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(firstCase.ReportPath));
        JsonElement reportRoot = reportDocument.RootElement;
        JsonElement integrity = reportRoot.GetProperty("metrics").GetProperty("filteredResultIntegrity");

        Assert.Equal("VecNet.ExactFilteredBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
        Assert.Equal("VEC-046", reportRoot.GetProperty("taskId").GetString());
        Assert.True(integrity.TryGetProperty("toleratedNearTieOrderMismatchCount", out _));
        Assert.True(integrity.TryGetProperty("unresolvedWrongIdCount", out _));
        Assert.True(integrity.TryGetProperty("unresolvedOrderMismatchCount", out _));
        Assert.True(integrity.TryGetProperty("orderEquivalenceStatus", out _));
        Assert.True(integrity.TryGetProperty("classification", out _));
        Assert.False(reportRoot.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
    }

    private static float SquaredEuclideanTolerance(int dimension, float scalarReference)
    {
        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(scalarReference));
        return (float)Math.Max(2e-4, relative);
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            "vec049-independent-" + prefix + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string NewArtifactPath(string fileName) =>
        Path.Combine(NewArtifactDirectory(Path.GetFileNameWithoutExtension(fileName)), fileName);

    private static void AssertNoPropertyNamed(JsonElement element, params string[] disallowedNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                Assert.DoesNotContain(disallowedNames, name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase));
                AssertNoPropertyNamed(property.Value, disallowedNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoPropertyNamed(item, disallowedNames);
            }
        }
    }
}
