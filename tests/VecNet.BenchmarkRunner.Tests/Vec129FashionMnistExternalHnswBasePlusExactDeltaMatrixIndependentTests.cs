using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using VecNet.BenchmarkRunner;
using VecNet.BenchmarkRunner.ExternalDatasets;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec129FashionMnistExternalHnswBasePlusExactDeltaMatrixIndependentTests
{
    [Fact]
    public void ParserAndExpansion_KeepSmokeAsFirstAcceptedCaseAndStandardSeedsFromVec128()
    {
        string outputDirectory = NewArtifactDirectory("expand");
        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions standardOptions =
            CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(
                [
                    "EXTERNAL-FASHION-MNIST-HNSW-BASE-PLUS-EXACT-DELTA-MATRIX",
                    "--PRESET", "STANDARD",
                    "--CACHE-ROOT", "VecNet.DatasetCache",
                    "--QUERY-COUNT", "50",
                    "--RUNS", "1",
                    "--WARMUP-QUERIES", "3",
                    "--METRIC", "SQUARED-EUCLIDEAN",
                    "--SEED", "0x5EED2128",
                    "--DUPLICATE-INSERTS", "1",
                    "--UNKNOWN-DELETES", "1",
                    "--REPEATED-DELETES", "1",
                    "--OUTPUT-DIR", outputDirectory,
                    "--MANIFEST", Path.Combine(outputDirectory, "manifest.json")
                ]);

        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase[] standardCases =
            FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExpandCases(standardOptions);

        Assert.Equal("standard", standardOptions.PresetName);
        Assert.Equal(8, standardCases.Length);
        Assert.Equal(
            [
                "case-001-10k-low-churn-wide-m16-ef192",
                "case-002-10k-low-churn-wide-m16-ef384",
                "case-003-10k-tombstone-heavy-wide-m16-ef192",
                "case-004-10k-tombstone-heavy-wide-m16-ef384",
                "case-005-100k-low-churn-wide-m16-ef192",
                "case-006-100k-low-churn-wide-m16-ef384",
                "case-007-100k-tombstone-heavy-wide-m16-ef192",
                "case-008-100k-tombstone-heavy-wide-m16-ef384"
            ],
            standardCases.Select(matrixCase => matrixCase.CaseId).ToArray());
        Assert.Equal(
            [
                "0x5EED2128",
                "0x5EED2129",
                "0x5EED212A",
                "0x5EED212B",
                "0x5EED212C",
                "0x5EED212D",
                "0x5EED212E",
                "0x5EED212F"
            ],
            standardCases.Select(matrixCase => FormatHex(matrixCase.Options.Seed)).ToArray());
        Assert.Equal(
            [
                "0x484E535700012801",
                "0x484E535700012802",
                "0x484E535700012803",
                "0x484E535700012804",
                "0x484E535700012805",
                "0x484E535700012806",
                "0x484E535700012807",
                "0x484E535700012808"
            ],
            standardCases.Select(matrixCase => FormatHex(matrixCase.Options.HnswSeed)).ToArray());
        Assert.Equal([10, 100], standardCases.Select(matrixCase => matrixCase.Options.TopK).Distinct().ToArray());
        Assert.Equal(["low-churn", "tombstone-heavy"], standardCases.Select(matrixCase => matrixCase.UpdateProfileName).Distinct().ToArray());
        Assert.Equal(["wide-m16-ef192", "wide-m16-ef384"], standardCases.Select(matrixCase => matrixCase.HnswProfileName).Distinct().ToArray());

        FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions smokeOptions =
            standardOptions with { PresetName = "smoke" };
        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase smokeCase =
            Assert.Single(FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExpandCases(smokeOptions));

        Assert.Equal(standardCases[0].CaseId, smokeCase.CaseId);
        Assert.Equal(10, smokeCase.Options.TopK);
        Assert.Equal("low-churn", smokeCase.UpdateProfileName);
        Assert.Equal("wide-m16-ef192", smokeCase.HnswProfileName);
        Assert.Equal(59_000, smokeCase.Options.BaseVectorCount);
        Assert.Equal(500, smokeCase.Options.InsertedDeltaCount);
        Assert.Equal(100, smokeCase.Options.DeletedBaseCount);
        Assert.Equal(0, smokeCase.Options.DeletedDeltaCount);
        Assert.Equal(59_400, smokeCase.Options.LiveVectorCount);
        Assert.Equal(192, smokeCase.Options.EfSearch);
        Assert.False(Path.IsPathRooted(smokeCase.RelativeReportPath));
        Assert.DoesNotContain(Path.DirectorySeparatorChar, smokeCase.RelativeReportPath);
    }

    [Theory]
    [InlineData("--top-k", "10")]
    [InlineData("--base-vectors", "59000")]
    [InlineData("--insertions", "500")]
    [InlineData("--deletes", "100")]
    [InlineData("--delta-deletes", "0")]
    [InlineData("--output", "case.json")]
    [InlineData("--m", "16")]
    [InlineData("--ef-construction", "128")]
    [InlineData("--ef-search", "192")]
    [InlineData("--hnsw-seed", "0x484E535700012801")]
    [InlineData("--snapshot-directory", "snapshot")]
    [InlineData("--checkpoint-directory", "checkpoint")]
    [InlineData("--checkpoint", "true")]
    [InlineData("--filter", "all")]
    [InlineData("--allowlist", "broad")]
    [InlineData("--candidate-set", "selective")]
    [InlineData("--hnswlib-python", "python.exe")]
    [InlineData("--work-directory", "work")]
    [InlineData("--vecnet-snapshot-directory", "snapshot")]
    [InlineData("--hnswlib-index", "index.bin")]
    [InlineData("--baseline", "baseline.json")]
    [InlineData("--current", "current.json")]
    [InlineData("--baseline-report-id", "report-id")]
    [InlineData("--download", "false")]
    [InlineData("--download-raw-files", "false")]
    [InlineData("--truth-refresh", "true")]
    [InlineData("--truth-depth", "100")]
    [InlineData("--public-claim", "true")]
    [InlineData("--public-claim-eligible", "true")]
    [InlineData("--regression-gate", "true")]
    [InlineData("--comparison-publication", "true")]
    public void Parser_RejectsSingleCaseDurableFilterComparisonRefreshAndClaimOptionFamilies(string option, string value)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => CommandLine.ParseExternalFashionMnistHnswBasePlusExactDeltaMatrix(
                [FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName, option, value]));

        Assert.Contains("Unsupported option", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MissingCache_CommandWritesBlockedStandardManifestAndReturnsFailure()
    {
        string directory = NewArtifactDirectory("missing-cache-command");
        string manifestPath = Path.Combine(directory, "manifest.json");
        string[] args =
        [
            FashionMnistExternalHnswBasePlusExactDeltaMatrixOptions.ScenarioName,
            "--preset", "standard",
            "--cache-root", Path.Combine(directory, "missing-cache"),
            "--query-count", "50",
            "--runs", "1",
            "--warmup-queries", "3",
            "--seed", "0x5EED2128",
            "--output-dir", directory,
            "--manifest", manifestPath
        ];

        int exitCode = BenchmarkRunnerProgram.Run(args);

        Assert.Equal(1, exitCode);
        Assert.True(File.Exists(manifestPath));

        string json = File.ReadAllText(manifestPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaMatrixManifest", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("VEC-129", root.GetProperty("taskId").GetString());
        Assert.Equal("standard", root.GetProperty("presetName").GetString());
        Assert.Equal("unavailable", root.GetProperty("cacheTruth").GetProperty("status").GetString());
        Assert.Contains("must not download", root.GetProperty("cacheTruth").GetProperty("cachePolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("must not refresh", root.GetProperty("cacheTruth").GetProperty("truthPolicy").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(8, root.GetProperty("caseCount").GetInt32());
        AssertStatusCountsMatchAggregate(root);
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(0, root.GetProperty("aggregate").GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(8, root.GetProperty("aggregate").GetProperty("blockedCaseCount").GetInt32());

        foreach (JsonElement matrixCase in root.GetProperty("cases").EnumerateArray())
        {
            Assert.Equal("blocked", matrixCase.GetProperty("status").GetString());
            Assert.Equal("blocked", matrixCase.GetProperty("validationStatus").GetString());
            Assert.Equal(JsonValueKind.Null, matrixCase.GetProperty("linkedReportId").ValueKind);
            Assert.Equal("notAvailable", matrixCase.GetProperty("recallOrderSummary").GetProperty("status").GetString());
            Assert.Equal("notAvailable", matrixCase.GetProperty("allocationSummary").GetProperty("status").GetString());
            Assert.False(Path.IsPathRooted(matrixCase.GetProperty("linkedReportPath").GetString()!));
            Assert.False(File.Exists(Path.Combine(directory, matrixCase.GetProperty("linkedReportPath").GetString()!)));
            Assert.Contains(FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, matrixCase.GetProperty("commandArguments").EnumerateArray().Select(item => item.GetString()));
        }

        AssertNoBooleanPropertyTrueForNames(root, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible", "comparisonPublicationEligible");
        AssertNoPropertyNamed(root, "downloadRawFiles", "truthRefresh", "snapshotDirectory", "checkpointDirectory", "hnswlibPython", "packageMetadata", "nugetPublication");
        Assert.DoesNotContain("\"taskId\": \"VEC-127\"", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AggregateSummaries_UseRecordedCasesAndKeepFailedBlockedCasesVisible()
    {
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest passed = MatrixCase(
            caseNumber: 1,
            status: "passed",
            validationStatus: "passed",
            topK: 10,
            updateProfile: "low-churn",
            recall: 0.95,
            orderedAgreement: 0.90,
            integrityStatus: "passed",
            distanceToleranceStatus: "passed",
            underfilledQueries: 0,
            underfilledSlots: 0,
            allocationBytes: 0,
            generationMatches: true,
            committedMutations: 12,
            committedStatusCount: 12,
            duplicateStatusCount: 1,
            unknownStatusCount: 1,
            alreadyDeletedStatusCount: 1,
            liveCount: 59_400,
            tombstoneCount: 100,
            tombstoneRatio: 100.0 / 59_500,
            deltaInsertRatio: 500.0 / 59_000,
            nonFalseEligibility: false);
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest failed = MatrixCase(
            caseNumber: 2,
            status: "failed",
            validationStatus: "failed",
            topK: 100,
            updateProfile: "tombstone-heavy",
            recall: 0.80,
            orderedAgreement: 0.40,
            integrityStatus: "failed",
            distanceToleranceStatus: "failed",
            underfilledQueries: 2,
            underfilledSlots: 7,
            allocationBytes: 16,
            generationMatches: false,
            committedMutations: 25,
            committedStatusCount: 24,
            duplicateStatusCount: 2,
            unknownStatusCount: 3,
            alreadyDeletedStatusCount: 4,
            liveCount: 52_500,
            tombstoneCount: 5_500,
            tombstoneRatio: 5_500.0 / 58_000,
            deltaInsertRatio: 2_000.0 / 56_000,
            nonFalseEligibility: true);
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest blocked = MatrixCase(
            caseNumber: 3,
            status: "blocked",
            validationStatus: "blocked",
            topK: 100,
            updateProfile: "low-churn",
            recall: null,
            orderedAgreement: null,
            integrityStatus: "notAvailable",
            distanceToleranceStatus: null,
            underfilledQueries: null,
            underfilledSlots: null,
            allocationBytes: null,
            generationMatches: null,
            committedMutations: null,
            committedStatusCount: null,
            duplicateStatusCount: null,
            unknownStatusCount: null,
            alreadyDeletedStatusCount: null,
            liveCount: null,
            tombstoneCount: null,
            tombstoneRatio: null,
            deltaInsertRatio: null,
            nonFalseEligibility: false);

        ExternalHnswBasePlusExactDeltaMatrixAggregate aggregate =
            InvokeCreateAggregate([passed, failed, blocked], passed: 1, failed: 1, blocked: 1);

        Assert.Equal(1, aggregate.PassedCaseCount);
        Assert.Equal(1, aggregate.FailedCaseCount);
        Assert.Equal(1, aggregate.BlockedCaseCount);
        Assert.Equal(1, aggregate.ReturnedResultIntegrityNotPassedCaseCount);
        Assert.Equal(1, aggregate.DistanceToleranceNotPassedCaseCount);
        Assert.Equal(0.80, aggregate.Recall.MinimumRecallAtK);
        Assert.Equal(0.95, aggregate.Recall.MaximumRecallAtK);
        Assert.Equal(0.40, aggregate.Order.MinimumOrderedAgreement);
        Assert.Equal(0.90, aggregate.Order.MaximumOrderedAgreement);
        Assert.Equal(1, aggregate.Underfill.CaseCountWithAnyUnderfill);
        Assert.Equal(2, aggregate.Underfill.TotalUnderfilledQueryCount);
        Assert.Equal(7, aggregate.Underfill.TotalUnderfilledSlotCount);
        Assert.Contains(aggregate.Underfill.WorstByTopKAndUpdateProfile, item => item.Group == "100k-tombstone-heavy" && item.WorstUnderfilledSlotCount == 7);
        Assert.Equal(16, aggregate.Allocation.MaximumMeanManagedAllocatedBytesPerSearchCall);
        Assert.Equal(1, aggregate.Allocation.CaseCountWithAllocationGreaterThanZero);
        Assert.Equal(1, aggregate.Mutations.CaseCountWithMutationOrGenerationMismatch);
        Assert.Equal(37, aggregate.Mutations.TotalCommittedMutationCount);
        Assert.Equal(3, aggregate.Mutations.TotalDuplicateIdStatusCount);
        Assert.Equal(4, aggregate.Mutations.TotalUnknownIdStatusCount);
        Assert.Equal(5, aggregate.Mutations.TotalAlreadyDeletedStatusCount);
        Assert.Equal(52_500, aggregate.Counts.MinimumLiveVectorCount);
        Assert.Equal(59_400, aggregate.Counts.MaximumLiveVectorCount);
        Assert.Equal(5_500, aggregate.Counts.MaximumTombstoneCount);
        Assert.Equal(1, aggregate.Eligibility.LinkedReportNonFalseEligibilityCount);
        Assert.False(aggregate.Eligibility.ManifestPublicClaimEligible);
        Assert.False(aggregate.Eligibility.ManifestBaselineCandidateEligible);
        Assert.False(aggregate.Eligibility.ManifestRegressionGateEligible);
        Assert.False(aggregate.Eligibility.ComparisonPublicationEligible);
    }

    [Fact]
    public void LinkedReportCaseSummary_ReusesVec127IdentityAndCarriesNotMeasuredMemory()
    {
        FashionMnistAdmissionResult admission = CreateSyntheticAdmission("linked-vec127", baseCount: 30, queryCount: 4, truthDepth: 4);
        string cacheRoot = CacheRootFromManifest(admission.ManifestPath);
        string directory = Path.Combine(Directory.GetParent(cacheRoot)!.FullName, "matrix");
        Directory.CreateDirectory(directory);
        var caseOptions = new FashionMnistExternalHnswBasePlusExactDeltaOptions(
            cacheRoot,
            Path.Combine(directory, "case-001-10k-low-churn-wide-m16-ef192.json"),
            QueryCount: 3,
            TopK: 3,
            BaseVectorCount: 18,
            InsertedDeltaCount: 4,
            DeletedBaseCount: 2,
            DeletedDeltaCount: 1,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            Runs: 1,
            WarmupQueries: 1,
            VectorMetric.SquaredEuclidean,
            Seed: 0x5EED2128,
            M: 2,
            EfConstruction: 8,
            EfSearch: 8,
            HnswSeed: 0x484E535700012801);

        ExternalHnswBasePlusExactDeltaBenchmarkReport report =
            FashionMnistExternalHnswBasePlusExactDeltaScenario.Run(
                caseOptions,
                FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.CreateCaseArguments(caseOptions));
        FashionMnistExternalHnswBasePlusExactDeltaScenario.Write(report, caseOptions.OutputPath);

        var matrixCase = new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase(
            "case-001-10k-low-churn-wide-m16-ef192",
            "low-churn",
            "wide-m16-ef192",
            "case-001-10k-low-churn-wide-m16-ef192.json",
            new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExternalDeltaUpdateProfile(
                "low-churn",
                BaseRowCount: 18,
                DeltaRowCount: 4,
                DeletedBaseCount: 2,
                DeletedDeltaCount: 1,
                ExpectedLiveCount: 19,
                "test profile"),
            new FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.ExternalDeltaHnswProfile(
                "wide-m16-ef192",
                M: 2,
                EfConstruction: 8,
                EfSearch: 8),
            caseOptions);

        ExternalHnswBasePlusExactDeltaMatrixCaseManifest manifestCase =
            InvokeCreateCaseManifest(matrixCase, report, "passed", "passed", null);

        Assert.Equal("VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-127", report.TaskId);
        Assert.Equal(report.ReportId, manifestCase.LinkedReportId);
        Assert.Equal("passed", manifestCase.Status);
        Assert.Equal("recorded", manifestCase.AllocationSummary.Status);
        Assert.Equal("notMeasured", manifestCase.AllocationSummary.MemoryStatus);
        Assert.Equal(report.Measurement.Memory.Status, manifestCase.AllocationSummary.MemoryStatus);
        Assert.Equal(report.Mutations.StatusCounts.Committed, manifestCase.MutationSummary.StatusCommitted);
        Assert.Equal(report.Counts.LiveVectorCount, manifestCase.CountSummary.LiveVectorCount);
        Assert.False(manifestCase.EligibilitySummary.PublicClaimEligible);
        Assert.False(manifestCase.EligibilitySummary.BaselineCandidateEligible);
        Assert.False(manifestCase.EligibilitySummary.RegressionGateEligible);
        Assert.False(manifestCase.EligibilitySummary.ValidationPublicClaimEligible);
        Assert.False(manifestCase.EligibilitySummary.ValidationBaselineCandidateEligible);
        Assert.False(manifestCase.EligibilitySummary.ValidationRegressionGateEligible);

        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(caseOptions.OutputPath));
        JsonElement reportRoot = reportDocument.RootElement;
        Assert.Equal("VEC-127", reportRoot.GetProperty("taskId").GetString());
        Assert.Equal("notMeasured", reportRoot.GetProperty("measurement").GetProperty("memory").GetProperty("status").GetString());
        AssertNoBooleanPropertyTrueForNames(reportRoot, "publicClaimEligible", "baselineCandidateEligible", "regressionGateEligible");
    }

    private static ExternalHnswBasePlusExactDeltaMatrixCaseManifest MatrixCase(
        int caseNumber,
        string status,
        string validationStatus,
        int topK,
        string updateProfile,
        double? recall,
        double? orderedAgreement,
        string integrityStatus,
        string? distanceToleranceStatus,
        int? underfilledQueries,
        int? underfilledSlots,
        double? allocationBytes,
        bool? generationMatches,
        int? committedMutations,
        int? committedStatusCount,
        int? duplicateStatusCount,
        int? unknownStatusCount,
        int? alreadyDeletedStatusCount,
        int? liveCount,
        int? tombstoneCount,
        double? tombstoneRatio,
        double? deltaInsertRatio,
        bool nonFalseEligibility)
    {
        string caseId = string.Create(CultureInfo.InvariantCulture, $"case-{caseNumber:D3}-{topK}k-{updateProfile}-wide-m16-ef192");
        return new ExternalHnswBasePlusExactDeltaMatrixCaseManifest(
            caseNumber,
            caseId,
            updateProfile,
            "wide-m16-ef192",
            "fashion-mnist-784-euclidean",
            VectorMetric.SquaredEuclidean.ToString(),
            Dimension: 784,
            QueryCount: 50,
            topK,
            Runs: 1,
            WarmupQueries: 3,
            WorkloadSeed: "0x5EED2128",
            HnswSeed: "0x484E535700012801",
            M: 16,
            EfConstruction: 128,
            EfSearch: 192,
            ImmutableBaseStartRow: 0,
            ImmutableBaseEndRowInclusive: 58_999,
            ImmutableBaseRowCount: 59_000,
            DeltaStartRow: 59_000,
            DeltaEndRowInclusive: 59_499,
            DeltaRowCount: 500,
            UnusedCandidateRowCount: 500,
            DeletedBaseVectorCount: 100,
            DeletedDeltaVectorCount: 0,
            DuplicateInsertAttempts: 1,
            UnknownDeleteAttempts: 1,
            RepeatedDeleteAttempts: 1,
            ExpectedPhysicalVectorCount: 59_500,
            ExpectedLiveVectorCount: 59_400,
            LinkedReportPath: $"{caseId}.json",
            CommandArguments: [FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName],
            LinkedReportId: status == "blocked" ? null : $"{caseId}-report",
            status,
            validationStatus,
            new ExternalHnswBasePlusExactDeltaMatrixRecallOrderSummary(
                recall is null ? "notAvailable" : "recorded",
                recall,
                orderedAgreement,
                distanceToleranceStatus,
                distanceToleranceStatus == "failed" ? 1 : 0,
                recall is null ? null : 0,
                recall is null ? null : 0),
            new ExternalHnswBasePlusExactDeltaMatrixIntegritySummary(
                integrityStatus,
                integrityStatus == "notAvailable" ? null : 50 * topK,
                0,
                0,
                0,
                0,
                0,
                0,
                integrityStatus == "failed" ? 1 : 0),
            new ExternalHnswBasePlusExactDeltaMatrixUnderfillSummary(
                underfilledSlots is null ? "notAvailable" : "recorded",
                QueryCount: 50,
                RequestedResultCountPerQuery: topK,
                TotalRequestedResultSlots: 50 * topK,
                underfilledSlots is null ? null : (50 * topK) - underfilledSlots.Value,
                underfilledQueries,
                underfilledSlots),
            new ExternalHnswBasePlusExactDeltaMatrixAllocationSummary(
                allocationBytes is null ? "notAvailable" : "recorded",
                MeanElapsedMilliseconds: allocationBytes is null ? null : 1,
                LatencyP50Milliseconds: allocationBytes is null ? null : 0.1,
                LatencyP95Milliseconds: allocationBytes is null ? null : 0.2,
                LatencyP99Milliseconds: allocationBytes is null ? null : 0.3,
                Qps: allocationBytes is null ? null : 1000,
                MeanManagedAllocatedBytesPerSearchCall: allocationBytes,
                ManagedAllocationStatus: allocationBytes is null ? null : "measured",
                MemoryStatus: allocationBytes is null ? null : "notMeasured"),
            new ExternalHnswBasePlusExactDeltaMatrixMutationSummary(
                committedMutations is null ? "notAvailable" : "recorded",
                InsertedDeltaVectorCount: 500,
                DeletedBaseVectorCount: 100,
                DeletedDeltaVectorCount: 0,
                DuplicateInsertAttempts: 1,
                UnknownDeleteAttempts: 1,
                RepeatedDeleteAttempts: 1,
                committedMutations,
                committedStatusCount,
                duplicateStatusCount,
                unknownStatusCount,
                alreadyDeletedStatusCount,
                generationMatches,
                committedMutations,
                committedMutations),
            new ExternalHnswBasePlusExactDeltaMatrixCountSummary(
                liveCount is null ? "notAvailable" : "recorded",
                ExpectedBasePhysicalVectorCount: 59_000,
                ExpectedDeltaPhysicalVectorCount: 500,
                ExpectedPhysicalVectorCount: 59_500,
                ExpectedLiveVectorCount: 59_400,
                BasePhysicalVectorCount: liveCount is null ? null : 59_000,
                BaseLiveVectorCount: liveCount is null ? null : liveCount - 500,
                DeltaPhysicalVectorCount: liveCount is null ? null : 500,
                DeltaLiveVectorCount: liveCount is null ? null : 500,
                BaseTombstoneCount: tombstoneCount,
                DeltaTombstoneCount: 0,
                TombstoneCount: tombstoneCount,
                LiveVectorCount: liveCount,
                DeletedReservedIdCount: tombstoneCount,
                Generation: committedMutations,
                TombstoneRatio: tombstoneRatio,
                DeltaInsertRatio: deltaInsertRatio),
            new ExternalHnswBasePlusExactDeltaMatrixEligibilitySummary(
                PublicClaimEligible: nonFalseEligibility,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ValidationPublicClaimEligible: false,
                ValidationBaselineCandidateEligible: false,
                ValidationRegressionGateEligible: false),
            status == "blocked" ? "blocked by test" : null);
    }

    private static ExternalHnswBasePlusExactDeltaMatrixCaseManifest InvokeCreateCaseManifest(
        FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.MatrixCase matrixCase,
        ExternalHnswBasePlusExactDeltaBenchmarkReport report,
        string status,
        string validationStatus,
        string? errorMessage)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario).GetMethod(
            "CreateCaseManifest",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaMatrixCaseManifest>(
            method.Invoke(
                null,
                [
                    1,
                    matrixCase,
                    FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario.CreateCaseArguments(matrixCase.Options),
                    report,
                    status,
                    validationStatus,
                    errorMessage
                ]));
    }

    private static ExternalHnswBasePlusExactDeltaMatrixAggregate InvokeCreateAggregate(
        ExternalHnswBasePlusExactDeltaMatrixCaseManifest[] cases,
        int passed,
        int failed,
        int blocked)
    {
        MethodInfo? method = typeof(FashionMnistExternalHnswBasePlusExactDeltaMatrixScenario).GetMethod(
            "CreateAggregate",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        return Assert.IsType<ExternalHnswBasePlusExactDeltaMatrixAggregate>(
            method.Invoke(null, [cases, passed, failed, blocked]));
    }

    private static FashionMnistAdmissionResult CreateSyntheticAdmission(string prefix, int baseCount, int queryCount, int truthDepth)
    {
        string cacheRoot = NewArtifactDirectory(prefix);
        FashionMnistDatasetSpecification spec = WriteSyntheticRawFiles(cacheRoot, baseCount, queryCount, rows: 3, columns: 5);
        return FashionMnistExternalDatasetScenario.Run(
            new FashionMnistExternalDatasetOptions(cacheRoot, queryCount, truthDepth, DownloadRawFiles: false),
            ["external-fashion-mnist", "--download", "false"],
            spec);
    }

    private static FashionMnistDatasetSpecification WriteSyntheticRawFiles(
        string cacheRoot,
        int baseCount,
        int queryCount,
        int rows,
        int columns)
    {
        const string datasetId = "fashion-mnist-784-euclidean";
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        string rawDirectory = Path.Combine(cacheRoot, "raw", datasetId);
        Directory.CreateDirectory(rawDirectory);

        string trainImages = Path.Combine(rawDirectory, "train-images-idx3-ubyte.gz");
        string trainLabels = Path.Combine(rawDirectory, "train-labels-idx1-ubyte.gz");
        string queryImages = Path.Combine(rawDirectory, "t10k-images-idx3-ubyte.gz");
        string queryLabels = Path.Combine(rawDirectory, "t10k-labels-idx1-ubyte.gz");

        File.WriteAllBytes(trainImages, CreateImageIdxGzip(baseCount, rows, columns, CreatePixels(baseCount, rows * columns, offset: 19)).ToArray());
        File.WriteAllBytes(trainLabels, CreateLabelIdxGzip(baseCount, CreateLabels(baseCount)).ToArray());
        File.WriteAllBytes(queryImages, CreateImageIdxGzip(queryCount, rows, columns, CreatePixels(queryCount, rows * columns, offset: 97)).ToArray());
        File.WriteAllBytes(queryLabels, CreateLabelIdxGzip(queryCount, CreateLabels(queryCount)).ToArray());

        static FashionMnistRawFileSpec Spec(string path, string fileName, string role, int expectedCount) =>
            new(fileName, role, expectedCount, FileChecksum.ComputeMd5(path), downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            datasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: baseCount,
            QueryCount: queryCount,
            ImageRows: rows,
            ImageColumns: columns,
            Dimension: checked(rows * columns),
            TrainImages: Spec(trainImages, "train-images-idx3-ubyte.gz", "base-images", baseCount),
            TrainLabels: Spec(trainLabels, "train-labels-idx1-ubyte.gz", "base-labels", baseCount),
            QueryImages: Spec(queryImages, "t10k-images-idx3-ubyte.gz", "query-images", queryCount),
            QueryLabels: Spec(queryLabels, "t10k-labels-idx1-ubyte.gz", "query-labels", queryCount));
    }

    private static byte[] CreatePixels(int count, int dimension, int offset)
    {
        var payload = new byte[checked(count * dimension)];
        for (int row = 0; row < count; row++)
        {
            for (int column = 0; column < dimension; column++)
            {
                payload[(row * dimension) + column] = (byte)((row * 37 + column * 11 + offset) % 251);
            }
        }

        return payload;
    }

    private static byte[] CreateLabels(int count)
    {
        var labels = new byte[count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = (byte)(i % 10);
        }

        return labels;
    }

    private static MemoryStream CreateImageIdxGzip(int count, int rows, int columns, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2051);
        WriteInt32BigEndian(decoded, count);
        WriteInt32BigEndian(decoded, rows);
        WriteInt32BigEndian(decoded, columns);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream CreateLabelIdxGzip(int count, byte[] payload)
    {
        using var decoded = new MemoryStream();
        WriteInt32BigEndian(decoded, 2049);
        WriteInt32BigEndian(decoded, count);
        decoded.Write(payload);
        return Gzip(decoded.ToArray());
    }

    private static MemoryStream Gzip(byte[] decoded)
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            gzip.Write(decoded);
        }

        compressed.Position = 0;
        return compressed;
    }

    private static void WriteInt32BigEndian(Stream stream, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, value);
        stream.Write(buffer);
    }

    private static string CacheRootFromManifest(string manifestPath) =>
        Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(manifestPath)!)!.FullName)!.FullName;

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec129-independent-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string FormatHex(uint value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X8}");

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static void AssertStatusCountsMatchAggregate(JsonElement root)
    {
        JsonElement cases = root.GetProperty("cases");
        int passed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "passed");
        int failed = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "failed");
        int skipped = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "skipped");
        int blocked = cases.EnumerateArray().Count(matrixCase => matrixCase.GetProperty("status").GetString() == "blocked");
        JsonElement aggregate = root.GetProperty("aggregate");

        Assert.Equal(root.GetProperty("caseCount").GetInt32(), passed + failed + skipped + blocked);
        Assert.Equal(passed, aggregate.GetProperty("passedCaseCount").GetInt32());
        Assert.Equal(failed, aggregate.GetProperty("failedCaseCount").GetInt32());
        Assert.Equal(skipped, aggregate.GetProperty("skippedCaseCount").GetInt32());
        Assert.Equal(blocked, aggregate.GetProperty("blockedCaseCount").GetInt32());
    }

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
