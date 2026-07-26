using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec125HnswBasePlusExactDeltaMatrixIndependentTests
{
    [Theory]
    [InlineData("--m", "8")]
    [InlineData("--ef-construction", "64")]
    [InlineData("--ef-search", "64")]
    [InlineData("--hnsw-seed", "0x1234")]
    public void MatrixParserRejectsSingleCaseHnswTuningControls(string optionName, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseHnswBasePlusExactDeltaMatrix(
                [
                    "generated-hnsw-base-plus-exact-delta-matrix",
                    "--preset", "smoke",
                    optionName, value
                ]));

        Assert.Contains($"Unsupported option '{optionName}'", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StandardPresetHasStableCaseOrderCoverageAndLinkedReportNames()
    {
        HnswBasePlusExactDeltaMatrixScenario.MatrixCase[] first =
            HnswBasePlusExactDeltaMatrixScenario.ExpandCases(StandardOptions(NewArtifactDirectory("standard-a")));
        HnswBasePlusExactDeltaMatrixScenario.MatrixCase[] second =
            HnswBasePlusExactDeltaMatrixScenario.ExpandCases(StandardOptions(NewArtifactDirectory("standard-b")));

        string[] expectedPaths =
        [
            "case-001-SquaredEuclidean-balanced-m8-low-churn-32d-10k.json",
            "case-002-SquaredEuclidean-balanced-m8-tombstone-heavy-32d-10k.json",
            "case-003-SquaredEuclidean-balanced-m8-low-churn-32d-50k.json",
            "case-004-SquaredEuclidean-balanced-m8-tombstone-heavy-32d-50k.json",
            "case-005-SquaredEuclidean-balanced-m8-low-churn-128d-10k.json",
            "case-006-SquaredEuclidean-balanced-m8-tombstone-heavy-128d-10k.json",
            "case-007-SquaredEuclidean-balanced-m8-low-churn-128d-50k.json",
            "case-008-SquaredEuclidean-balanced-m8-tombstone-heavy-128d-50k.json",
            "case-009-Cosine-balanced-m8-low-churn-32d-10k.json",
            "case-010-Cosine-balanced-m8-tombstone-heavy-32d-10k.json",
            "case-011-Cosine-balanced-m8-low-churn-32d-50k.json",
            "case-012-Cosine-balanced-m8-tombstone-heavy-32d-50k.json",
            "case-013-Cosine-balanced-m8-low-churn-128d-10k.json",
            "case-014-Cosine-balanced-m8-tombstone-heavy-128d-10k.json",
            "case-015-Cosine-balanced-m8-low-churn-128d-50k.json",
            "case-016-Cosine-balanced-m8-tombstone-heavy-128d-50k.json"
        ];

        Assert.Equal(expectedPaths, first.Select(item => item.RelativeReportPath).ToArray());
        Assert.Equal(expectedPaths, second.Select(item => item.RelativeReportPath).ToArray());
        Assert.Equal([VectorMetric.SquaredEuclidean, VectorMetric.Cosine], first.Select(item => item.Options.Metric).Distinct().ToArray());
        Assert.Equal([32, 128], first.Select(item => item.Options.Dimension).Distinct().ToArray());
        Assert.Equal([10, 50], first.Select(item => item.Options.TopK).Distinct().ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], first.Select(item => item.UpdateProfileName).Distinct().ToArray());
        Assert.All(first, matrixCase =>
        {
            Assert.Equal(matrixCase.RelativeReportPath, Path.GetFileName(matrixCase.RelativeReportPath));
            Assert.False(Path.IsPathFullyQualified(matrixCase.RelativeReportPath));
            Assert.Equal("balanced-m8", matrixCase.HnswProfileName);
            Assert.Equal(64, matrixCase.Options.EfSearch);
            Assert.True(matrixCase.Options.EfSearch >= matrixCase.Options.TopK);
        });
    }

    [Fact]
    public void SuccessfulMatrixSummariesAreCopiedFromLinkedVec124Reports()
    {
        string outputDirectory = NewArtifactDirectory("linked-summary");
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        string[] args =
        [
            "generated-hnsw-base-plus-exact-delta-matrix",
            "--vectors", "24",
            "--queries", "2",
            "--runs", "1",
            "--seed", "0x12500001",
            "--duplicate-inserts", "1",
            "--unknown-deletes", "2",
            "--repeated-deletes", "3",
            "--output-dir", outputDirectory,
            "--manifest", manifestPath
        ];

        HnswBasePlusExactDeltaMatrixOptions options = CommandLine.ParseHnswBasePlusExactDeltaMatrix(args);
        HnswBasePlusExactDeltaMatrixManifest manifest = HnswBasePlusExactDeltaMatrixScenario.Run(options, args);
        HnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, manifestPath);

        Assert.Equal(4, manifest.Cases.Length);
        Assert.All(manifest.Cases, matrixCase =>
        {
            string reportPath = Path.Combine(outputDirectory, matrixCase.LinkedReportPath);
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(reportPath));
            JsonElement report = document.RootElement;

            Assert.Equal("VecNet.HnswBasePlusExactDeltaBenchmarkReport", report.GetProperty("schemaName").GetString());
            Assert.Equal("0.1", report.GetProperty("schemaVersion").GetString());
            Assert.Equal("VEC-124", report.GetProperty("taskId").GetString());
            Assert.False(report.GetProperty("evidence").GetProperty("publicClaimEligible").GetBoolean());
            Assert.False(report.GetProperty("validation").GetProperty("baselineCandidateEligible").GetBoolean());
            Assert.False(report.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());

            JsonElement metrics = report.GetProperty("metrics");
            Assert.Equal(metrics.GetProperty("recallAtK").GetDouble(), matrixCase.RecallSummary.RecallAtK);
            Assert.Equal(metrics.GetProperty("orderedAgreement").GetDouble(), matrixCase.RecallSummary.OrderedAgreement);
            Assert.Equal(metrics.GetProperty("returnedResultIntegrity").GetProperty("status").GetString(), matrixCase.RecallSummary.ReturnedResultIntegrityStatus);

            JsonElement underfill = report.GetProperty("underfill");
            Assert.Equal(underfill.GetProperty("totalReturnedResults").GetInt32(), matrixCase.UnderfillSummary.TotalReturnedResults);
            Assert.Equal(underfill.GetProperty("underfilledSlotCount").GetInt32(), matrixCase.UnderfillSummary.UnderfilledSlotCount);

            JsonElement mutations = report.GetProperty("mutations");
            Assert.Equal(mutations.GetProperty("committedMutationCount").GetInt32(), matrixCase.MutationSummary.CommittedMutationCount);
            Assert.Equal(mutations.GetProperty("statusCounts").GetProperty("unknownId").GetInt32(), matrixCase.MutationSummary.StatusUnknownId);

            JsonElement counts = report.GetProperty("counts");
            Assert.Equal(counts.GetProperty("liveVectorCount").GetInt32(), matrixCase.CountSummary.LiveVectorCount);
            Assert.Equal(counts.GetProperty("deletedReservedIdCount").GetInt32(), matrixCase.CountSummary.DeletedReservedIdCount);
        });
    }

    [Fact]
    public void BlockedReportWriteIsSerializedAsRepresentedUnavailableCase()
    {
        string outputDirectory = NewArtifactDirectory("blocked-json");
        Directory.CreateDirectory(Path.Combine(outputDirectory, "case-002-SquaredEuclidean-balanced-m4-low-churn-16d-10k.json"));
        string manifestPath = Path.Combine(outputDirectory, "manifest.json");
        var options = new HnswBasePlusExactDeltaMatrixOptions(
            "smoke",
            BaseVectorCount: 24,
            QueryCount: 1,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x12500002,
            DuplicateInsertAttempts: 0,
            UnknownDeleteAttempts: 0,
            RepeatedDeleteAttempts: 0,
            OutputDirectory: outputDirectory,
            ManifestPath: manifestPath);

        HnswBasePlusExactDeltaMatrixManifest manifest =
            HnswBasePlusExactDeltaMatrixScenario.Run(options, [HnswBasePlusExactDeltaMatrixOptions.ScenarioName]);
        HnswBasePlusExactDeltaMatrixScenario.WriteManifest(manifest, manifestPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;

        Assert.Equal(4, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(3, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(1, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());

        JsonElement blocked = root.GetProperty("cases").EnumerateArray().Single(item => item.GetProperty("status").GetString() == "blocked");
        Assert.Equal("blocked", blocked.GetProperty("validationStatus").GetString());
        Assert.Equal(JsonValueKind.Null, blocked.GetProperty("linkedReportId").ValueKind);
        Assert.Equal("notAvailable", blocked.GetProperty("recallSummary").GetProperty("status").GetString());
        Assert.Equal("notAvailable", blocked.GetProperty("underfillSummary").GetProperty("status").GetString());
        Assert.Equal("notAvailable", blocked.GetProperty("mutationSummary").GetProperty("status").GetString());
        Assert.Equal("notAvailable", blocked.GetProperty("countSummary").GetProperty("status").GetString());
        Assert.False(string.IsNullOrWhiteSpace(blocked.GetProperty("errorMessage").GetString()));
    }

    [Fact]
    public void FailedCasesKeepConfiguredSummaryContextWithoutLinkedReports()
    {
        string outputDirectory = NewArtifactDirectory("failed-context");
        var options = new HnswBasePlusExactDeltaMatrixOptions(
            "smoke",
            BaseVectorCount: 1,
            QueryCount: 3,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x12500003,
            DuplicateInsertAttempts: 5,
            UnknownDeleteAttempts: 6,
            RepeatedDeleteAttempts: 7,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

        HnswBasePlusExactDeltaMatrixManifest manifest =
            HnswBasePlusExactDeltaMatrixScenario.Run(options, [HnswBasePlusExactDeltaMatrixOptions.ScenarioName]);

        Assert.Equal(manifest.CaseCount, manifest.Aggregate.FailedCaseCount);
        Assert.All(manifest.Cases, matrixCase =>
        {
            Assert.Equal("failed", matrixCase.Status);
            Assert.Equal("failed", matrixCase.ValidationStatus);
            Assert.Null(matrixCase.LinkedReportId);
            Assert.False(File.Exists(Path.Combine(outputDirectory, matrixCase.LinkedReportPath)));
            Assert.Equal("notAvailable", matrixCase.RecallSummary.Status);
            Assert.Equal("notAvailable", matrixCase.UnderfillSummary.Status);
            Assert.Equal(matrixCase.QueryCount, matrixCase.UnderfillSummary.QueryCount);
            Assert.Equal(matrixCase.TopK, matrixCase.UnderfillSummary.RequestedResultCountPerQuery);
            Assert.Equal(matrixCase.QueryCount * matrixCase.TopK, matrixCase.UnderfillSummary.TotalRequestedResultSlots);
            Assert.Equal("notAvailable", matrixCase.MutationSummary.Status);
            Assert.Equal(5, matrixCase.MutationSummary.DuplicateInsertAttempts);
            Assert.Equal(6, matrixCase.MutationSummary.UnknownDeleteAttempts);
            Assert.Equal(7, matrixCase.MutationSummary.RepeatedDeleteAttempts);
            Assert.Equal("notAvailable", matrixCase.CountSummary.Status);
            Assert.Equal(matrixCase.BaseVectorCount + matrixCase.InsertedDeltaVectorCount, matrixCase.CountSummary.PhysicalVectorCount);
            Assert.Equal(matrixCase.ExpectedLiveVectorCount, matrixCase.CountSummary.ExpectedLiveVectorCount);
            Assert.False(string.IsNullOrWhiteSpace(matrixCase.ErrorMessage));
        });
    }

    private static HnswBasePlusExactDeltaMatrixOptions StandardOptions(string outputDirectory) =>
        new(
            "standard",
            BaseVectorCount: 128,
            QueryCount: 2,
            Runs: 1,
            WarmupQueries: 0,
            Seed: 0x12501250,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            OutputDirectory: outputDirectory,
            ManifestPath: Path.Combine(outputDirectory, "manifest.json"));

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"test-agent-vec125-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
