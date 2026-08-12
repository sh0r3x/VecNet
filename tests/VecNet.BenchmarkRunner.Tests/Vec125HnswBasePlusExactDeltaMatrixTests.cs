using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec125HnswBasePlusExactDeltaMatrixTests
{
    [Fact]
    public void ParseHnswBasePlusExactDeltaMatrix_UsesPrivateSmokeDefaults()
    {
        HnswBasePlusExactDeltaMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaMatrix(["generated-hnsw-base-plus-exact-delta-matrix"]);

        Assert.Equal("smoke", options.PresetName);
        Assert.Equal(64, options.BaseVectorCount);
        Assert.Equal(4, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(0, options.WarmupQueries);
        Assert.Equal(0x5EED2125u, options.Seed);
        Assert.Equal(1, options.DuplicateInsertAttempts);
        Assert.Equal(1, options.UnknownDeleteAttempts);
        Assert.Equal(1, options.RepeatedDeleteAttempts);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputDirectory, StringComparison.OrdinalIgnoreCase);
        Assert.False(Path.IsPathRooted(options.OutputDirectory));
        Assert.EndsWith("hnsw-base-plus-exact-delta-matrix-manifest.json", options.ManifestPath, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--preset", "large")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--preset", " ")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--vectors", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--preset", "standard", "--vectors", "73")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--queries", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--runs", "0")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--runs", "6")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--warmup-queries", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--duplicate-inserts", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--unknown-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--repeated-deletes", "-1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--output-dir", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--manifest", "")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--metric", "SquaredEuclidean")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--dimension", "128")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--top-k", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--insertions", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--deletes", "10")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--delta-deletes", "1")]
    [InlineData("generated-hnsw-base-plus-exact-delta-matrix", "--output", "single.json")]
    [InlineData("generated-hnsw-base-plus-exact-delta", "--output-dir", "matrix")]
    public void ParseHnswBasePlusExactDeltaMatrix_RejectsInvalidOrOutOfScopeOptions(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaMatrix(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void ExpandCases_StandardPresetCoversDimensionsTopKAndUpdateProfilesDeterministically()
    {
        string outputDirectory = NewArtifactDirectory("expand-standard");
        var options = new HnswBasePlusExactDeltaMatrixOptions(
            "standard",
            BaseVectorCount: 128,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0xFFFF_FFFC,
            DuplicateInsertAttempts: 2,
            UnknownDeleteAttempts: 3,
            RepeatedDeleteAttempts: 4,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswBasePlusExactDeltaMatrixScenario.MatrixCase[] cases =
            HnswBasePlusExactDeltaMatrixScenario.ExpandCases(options);

        Assert.Equal(24, cases.Length);
        Assert.Equal([VectorMetric.SquaredEuclidean, VectorMetric.InnerProduct, VectorMetric.Cosine], cases.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128], cases.Select(item => item.Options.Dimension).Distinct().OrderBy(item => item).ToArray());
        Assert.Equal([10, 50], cases.Select(item => item.Options.TopK).Distinct().OrderBy(item => item).ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], cases.Select(item => item.UpdateProfileName).Distinct().Order().ToArray());
        Assert.All(cases, matrixCase =>
        {
            Assert.Equal(128, matrixCase.Options.BaseVectorCount);
            Assert.Equal(2, matrixCase.Options.QueryCount);
            Assert.Equal(2, matrixCase.Options.DuplicateInsertAttempts);
            Assert.Equal(3, matrixCase.Options.UnknownDeleteAttempts);
            Assert.Equal(4, matrixCase.Options.RepeatedDeleteAttempts);
            Assert.Equal("balanced-m8", matrixCase.HnswProfileName);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
            Assert.False(Path.IsPathRooted(matrixCase.RelativeReportPath));
            Assert.StartsWith(outputDirectory, matrixCase.Options.OutputPath, StringComparison.OrdinalIgnoreCase);
        });

        Assert.Equal(0xFFFF_FFFCu, cases[0].Options.Seed);
        Assert.Equal(19u, cases[^1].Options.Seed);
        Assert.Equal("0x484EACA8FFFD2501", FormatHex(cases[0].Options.HnswSeed));
        Assert.Equal("0x484EACA8FFFD2518", FormatHex(cases[^1].Options.HnswSeed));
        Assert.Equal("case-001-SquaredEuclidean-balanced-m8-low-churn-32d-10k.json", cases[0].RelativeReportPath);
        Assert.Equal("case-024-Cosine-balanced-m8-tombstone-heavy-128d-50k.json", cases[^1].RelativeReportPath);
    }

    [Fact]
    public void Run_SmokeManifestLinksVec124ReportsWithSummariesAndFalseEligibility()
    {
        string outputDirectory = NewArtifactDirectory("smoke-manifest");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        string[] arguments =
        [
            "generated-hnsw-base-plus-exact-delta-matrix",
            "--preset", "smoke",
            "--vectors", "24",
            "--queries", "2",
            "--runs", "2",
            "--warmup-queries", "1",
            "--seed", "0x5EED1250",
            "--duplicate-inserts", "2",
            "--unknown-deletes", "3",
            "--repeated-deletes", "4",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];
        HnswBasePlusExactDeltaMatrixOptions options =
            CommandLine.ParseHnswBasePlusExactDeltaMatrix(arguments);

        HnswBasePlusExactDeltaMatrixManifest manifest =
            HnswBasePlusExactDeltaMatrixScenario.Run(options, arguments);
        HnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal("VecNet.HnswBasePlusExactDeltaMatrixManifest", manifest.SchemaName);
        Assert.Equal("0.1", manifest.SchemaVersion);
        Assert.Equal("VEC-125", manifest.TaskId);
        Assert.Equal("generated-hnsw-base-plus-exact-delta-matrix", manifest.ScenarioName);
        Assert.Equal("smoke", manifest.PresetName);
        Assert.Equal(6, manifest.CaseCount);
        Assert.Equal(6, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.False(manifest.Eligibility.PublicClaimEligible);
        Assert.False(manifest.Eligibility.BaselineCandidateEligible);
        Assert.False(manifest.Eligibility.RegressionGateEligible);
        Assert.Equal("private-raw", manifest.Eligibility.PrivacyClass);
        Assert.Equal("smoke", manifest.Eligibility.EvidenceStatus);
        Assert.Equal([16], manifest.Design.Dimensions);
        Assert.Equal([1, 10], manifest.Design.TopKValues);

        foreach (HnswBasePlusExactDeltaMatrixCaseManifest matrixCase in manifest.Cases)
        {
            Assert.Equal("passed", matrixCase.Status);
            Assert.Equal("passed", matrixCase.ValidationStatus);
            Assert.False(Path.IsPathRooted(matrixCase.LinkedReportPath));
            Assert.NotNull(matrixCase.LinkedReportId);
            Assert.Equal("recorded", matrixCase.RecallSummary.Status);
            Assert.InRange(matrixCase.RecallSummary.RecallAtK.GetValueOrDefault(), 0, 1);
            Assert.InRange(matrixCase.RecallSummary.OrderedAgreement.GetValueOrDefault(), 0, 1);
            Assert.Equal("passed", matrixCase.RecallSummary.ReturnedResultIntegrityStatus);
            Assert.Equal("recorded", matrixCase.UnderfillSummary.Status);
            Assert.Equal(matrixCase.QueryCount * matrixCase.TopK, matrixCase.UnderfillSummary.TotalRequestedResultSlots);
            Assert.True(matrixCase.UnderfillSummary.TotalReturnedResults >= 0);
            Assert.Equal("recorded", matrixCase.MutationSummary.Status);
            Assert.Equal(matrixCase.InsertedDeltaVectorCount + matrixCase.DeletedBaseVectorCount + matrixCase.DeletedDeltaVectorCount, matrixCase.MutationSummary.CommittedMutationCount);
            Assert.Equal(2, matrixCase.MutationSummary.StatusDuplicateId);
            Assert.Equal(3, matrixCase.MutationSummary.StatusUnknownId);
            Assert.Equal(4, matrixCase.MutationSummary.StatusAlreadyDeleted);
            Assert.Equal("recorded", matrixCase.CountSummary.Status);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, matrixCase.CountSummary.LiveVectorCount);
            Assert.Equal(matrixCase.DeletedBaseVectorCount + matrixCase.DeletedDeltaVectorCount, matrixCase.CountSummary.TombstoneCount);

            string linkedReportPath = Path.Combine(outputDirectory, matrixCase.LinkedReportPath);
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(linkedReportPath));
            JsonElement reportRoot = reportDocument.RootElement;
            Assert.Equal("VecNet.HnswBasePlusExactDeltaBenchmarkReport", reportRoot.GetProperty("schemaName").GetString());
            Assert.Equal("VEC-124", reportRoot.GetProperty("taskId").GetString());
            Assert.Equal("generated-hnsw-base-plus-exact-delta", reportRoot.GetProperty("scenarioName").GetString());
            Assert.Equal(matrixCase.LinkedReportId, reportRoot.GetProperty("reportId").GetString());
            Assert.Equal(matrixCase.Dimension, reportRoot.GetProperty("dataset").GetProperty("dimension").GetInt32());
            Assert.Equal(matrixCase.PhysicalVectorCount, reportRoot.GetProperty("dataset").GetProperty("vectorCount").GetInt32());
            Assert.Equal(matrixCase.TopK, reportRoot.GetProperty("scenario").GetProperty("topK").GetInt32());
            Assert.Equal("private-raw", reportRoot.GetProperty("privacyClass").GetString());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(reportRoot.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        }

        using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifestDocument.RootElement;
        Assert.Equal("VecNet.HnswBasePlusExactDeltaMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("generated-hnsw-base-plus-exact-delta-matrix", root.GetProperty("command").GetProperty("scenario").GetString());
        Assert.Equal(6, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible");
    }

    [Fact]
    public void Run_WhenReportPathIsBlocked_RecordsBlockedCaseAndContinues()
    {
        string outputDirectory = NewArtifactDirectory("blocked-case");
        var options = new HnswBasePlusExactDeltaMatrixOptions(
            "smoke",
            BaseVectorCount: 24,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED1251,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));
        Directory.CreateDirectory(Path.Combine(outputDirectory, "case-002-SquaredEuclidean-balanced-m4-low-churn-16d-10k.json"));

        HnswBasePlusExactDeltaMatrixManifest manifest =
            HnswBasePlusExactDeltaMatrixScenario.Run(options, ["generated-hnsw-base-plus-exact-delta-matrix"]);

        Assert.Equal(6, manifest.CaseCount);
        Assert.Equal(5, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(0, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(1, manifest.Aggregate.BlockedCaseCount);

        HnswBasePlusExactDeltaMatrixCaseManifest blocked = Assert.Single(manifest.Cases, item => item.Status == "blocked");
        Assert.Equal(2, blocked.CaseNumber);
        Assert.Equal("blocked", blocked.ValidationStatus);
        Assert.Null(blocked.LinkedReportId);
        Assert.Equal("notAvailable", blocked.RecallSummary.Status);
        Assert.Equal("notAvailable", blocked.UnderfillSummary.Status);
        Assert.Equal("notAvailable", blocked.MutationSummary.Status);
        Assert.Equal("notAvailable", blocked.CountSummary.Status);
        Assert.False(string.IsNullOrWhiteSpace(blocked.ErrorMessage));
    }

    [Fact]
    public void Run_WhenCaseOptionsAreInvalid_RecordsFailedCases()
    {
        string outputDirectory = NewArtifactDirectory("failed-case");
        var options = new HnswBasePlusExactDeltaMatrixOptions(
            "smoke",
            BaseVectorCount: 1,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x5EED1252,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswBasePlusExactDeltaMatrixManifest manifest =
            HnswBasePlusExactDeltaMatrixScenario.Run(options, ["generated-hnsw-base-plus-exact-delta-matrix"]);

        Assert.Equal(6, manifest.CaseCount);
        Assert.Equal(0, manifest.Aggregate.PassedCaseCount);
        Assert.Equal(6, manifest.Aggregate.FailedCaseCount);
        Assert.Equal(0, manifest.Aggregate.BlockedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.Equal("notAvailable", matrixCase.RecallSummary.Status);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
        });
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec125-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static void AssertNoBooleanPropertyTrueForNames(JsonElement element, params string[] propertyNames)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.True &&
                    propertyNames.Any(name => string.Equals(name, property.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    Assert.Fail($"Property '{property.Name}' must not be true.");
                }

                AssertNoBooleanPropertyTrueForNames(property.Value, propertyNames);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                AssertNoBooleanPropertyTrueForNames(item, propertyNames);
            }
        }
    }
}
