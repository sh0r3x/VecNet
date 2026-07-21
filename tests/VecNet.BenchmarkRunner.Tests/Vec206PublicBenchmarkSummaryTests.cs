using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec206PublicBenchmarkSummaryTests
{
    [Fact]
    public void Serialize_UsesCanonicalCamelCasePublicSummaryShape()
    {
        PublicBenchmarkSummary summary = CreateValidSummary();

        string json = PublicBenchmarkSummaryWriter.Serialize(summary);

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.PublicBenchmarkSummary", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("public-exact-flat-summary-vec206-smoke", root.GetProperty("summaryId").GetString());
        Assert.True(root.GetProperty("publicClaimEligible").GetBoolean());
        Assert.Equal(
            PublicBenchmarkSummaryGate.AcceptedPublicClaimStatus,
            root.GetProperty("publicClaimStatus").GetString());
        Assert.Equal("ExactFlatIndex", root.GetProperty("scope").GetProperty("algorithm").GetString());
        Assert.Equal("VecNet", root.GetProperty("source").GetProperty("packageId").GetString());
        Assert.Equal("exact-generated", root.GetProperty("commands")[0].GetProperty("name").GetString());
        Assert.Equal("generated", root.GetProperty("dataset").GetProperty("kind").GetString());
        Assert.Equal("reported", root.GetProperty("measurements").GetProperty("latency").GetProperty("status").GetString());
        Assert.Equal("absent", root.GetProperty("measurements").GetProperty("processMemory").GetProperty("status").GetString());
        Assert.Equal("absent", root.GetProperty("measurements").GetProperty("gcHeap").GetProperty("status").GetString());
        Assert.Equal("reported", root.GetProperty("measurements").GetProperty("persistedBytes").GetProperty("status").GetString());
        Assert.Equal("redacted", root.GetProperty("privacy").GetProperty("redactionStatus").GetString());
        Assert.Equal("reviewed", root.GetProperty("review").GetProperty("status").GetString());

        Assert.DoesNotContain("\"SchemaName\"", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"PublicClaimEligible\"", json, StringComparison.Ordinal);
        Assert.Contains("\"publicClaimEligible\": true", json, StringComparison.Ordinal);
    }

    [Fact]
    public void Validate_MinimalValidExactFlatStyleSummaryPasses()
    {
        PublicBenchmarkSummaryValidationResult validation =
            PublicBenchmarkSummaryWriter.Validate(CreateValidSummary());

        Assert.True(validation.IsValid, string.Join(Environment.NewLine, validation.Errors));
        Assert.Empty(validation.Errors);
    }

    [Fact]
    public void Validate_MissingClaimClassTextScopeOrReviewStatusFails()
    {
        PublicBenchmarkSummary missingClaimClass = CreateValidSummary() with { ClaimClass = "" };
        PublicBenchmarkSummary missingClaimText = CreateValidSummary() with { ClaimText = " " };
        PublicBenchmarkSummary missingScope = CreateValidSummary() with { Scope = null! };
        PublicBenchmarkSummary missingReviewStatus = CreateValidSummary() with
        {
            Review = CreateValidSummary().Review with { Status = "" }
        };

        AssertInvalidContains(missingClaimClass, "claimClass");
        AssertInvalidContains(missingClaimText, "claimText");
        AssertInvalidContains(missingScope, "scope");
        AssertInvalidContains(missingReviewStatus, "review.status");
    }

    [Theory]
    [InlineData(@"C:\Users\owner\private\runner-report.json")]
    [InlineData("/home/owner/.cache/vecnet/runner-report.json")]
    [InlineData("VecNet.BenchmarkRunner.Artifacts/exact-generated.json")]
    [InlineData("https://pkgs.dev.azure.com/private-org/_packaging/feed/nuget/v3/index.json")]
    public void Validate_LocalAbsolutePathsAndPrivateArtifactRootsFail(string publicFieldValue)
    {
        PublicBenchmarkSummary summary = CreateValidSummary() with
        {
            Commands =
            [
                new PublicBenchmarkCommand(
                    "exact-generated",
                    ["dotnet", "run", "--", "--output", publicFieldValue],
                    "repository root")
            ]
        };

        AssertInvalidContains(summary, "local path");
    }

    [Fact]
    public void Validate_EmbeddedWindowsAbsolutePathInCommandArgumentFails()
    {
        PublicBenchmarkSummary summary = CreateValidSummary() with
        {
            Commands =
            [
                new PublicBenchmarkCommand(
                    "exact-generated",
                    ["dotnet", "run", "--", @"--output=C:\Users\owner\private\runner-report.json"],
                    "repository root")
            ]
        };

        AssertInvalidContains(summary, "local path");
    }

    [Fact]
    public void Validate_EmbeddedUnixPrivateRootPathInDisclosureFails()
    {
        PublicBenchmarkSummary summary = CreateValidSummary() with
        {
            Environment = CreateValidSummary().Environment with
            {
                Disclosure = "Private raw report was written under /home/owner/.cache/vecnet/runner-report.json."
            }
        };

        AssertInvalidContains(summary, "local path");
    }

    [Fact]
    public void Validate_AmbiguousMemoryWordingFails()
    {
        PublicBenchmarkSummary summary = CreateValidSummary() with
        {
            Measurements = CreateValidSummary().Measurements with
            {
                ProcessMemory = new PublicBenchmarkMeasurementCategory(
                    "reported",
                    "bytes",
                    "single memory value covers process memory, managed allocation, GC heap and persisted bytes",
                    ["memory=1000000"],
                    [])
            }
        };

        AssertInvalidContains(summary, "ambiguous memory");
    }

    [Fact]
    public void Validate_PrivateRawReportIdentifiersCannotBeMarkedAsPublicSummaries()
    {
        PublicBenchmarkSummary rawReportIdAsSummary = CreateValidSummary() with
        {
            SummaryId = "exact-generated-SquaredEuclidean-384d-10000v-100q-10k-5r-10w-5EED2009-abc123"
        };
        PublicBenchmarkSummary privateRawArtifactPublished = CreateValidSummary() with
        {
            Privacy = CreateValidSummary().Privacy with
            {
                SourceArtifacts =
                [
                    new PublicBenchmarkSourceArtifact(
                        "private-raw-runner-report",
                        "VecNet.BenchmarkReport",
                        "exact-generated-SquaredEuclidean-384d-10000v-100q-10k-5r-10w-5EED2009-abc123",
                        PubliclyIncluded: true)
                ]
            }
        };

        AssertInvalidContains(rawReportIdAsSummary, "private raw runner report identifier");
        AssertInvalidContains(privateRawArtifactPublished, "private raw runner artifacts");
    }

    [Fact]
    public void Serialize_InvalidSummaryIsRejectedBeforeWritingPublicJson()
    {
        PublicBenchmarkSummary summary = CreateValidSummary() with
        {
            PublicClaimEligible = false
        };

        InvalidOperationException exception =
            Assert.Throws<InvalidOperationException>(() => PublicBenchmarkSummaryWriter.Serialize(summary));

        Assert.Contains("publicClaimEligible", exception.Message, StringComparison.Ordinal);
    }

    private static void AssertInvalidContains(PublicBenchmarkSummary summary, string expectedMessageFragment)
    {
        PublicBenchmarkSummaryValidationResult validation = PublicBenchmarkSummaryWriter.Validate(summary);

        Assert.False(validation.IsValid);
        Assert.Contains(
            validation.Errors,
            error => error.Contains(expectedMessageFragment, StringComparison.OrdinalIgnoreCase));
    }

    private static PublicBenchmarkSummary CreateValidSummary() =>
        new(
            SchemaName: PublicBenchmarkSummaryGate.SchemaName,
            SchemaVersion: PublicBenchmarkSummaryGate.SchemaVersion,
            SummaryId: "public-exact-flat-summary-vec206-smoke",
            GeneratedAtUtc: DateTimeOffset.Parse("2026-07-21T00:00:00Z"),
            ClaimClass: "exact-flat-generated-latency-recall-storage",
            ClaimText: "Exact-flat generated-data summary candidate for VecNet package identity.",
            Scope: new PublicBenchmarkClaimScope(
                "VecNet core package",
                "ExactFlatIndex",
                "SquaredEuclidean",
                "exact-generated public summary candidate",
                "E4",
                "Generated exact-flat queries with disclosed dimensions, metric, top-k and run counts."),
            PublicClaimEligible: true,
            PublicClaimStatus: PublicBenchmarkSummaryGate.AcceptedPublicClaimStatus,
            Source: new PublicBenchmarkSourceIdentity(
                "VecNet",
                "1.0.0",
                "abcdef0123456789abcdef0123456789abcdef01",
                "v1.0.0",
                "clean"),
            Commands:
            [
                new PublicBenchmarkCommand(
                    "exact-generated",
                    ["dotnet", "run", "--project", "benchmarks/VecNet.BenchmarkRunner", "--", "exact-generated"],
                    "repository root")
            ],
            Environment: new PublicBenchmarkEnvironment(
                "Windows 11",
                "X64",
                ".NET 10",
                "disclosed CPU model",
                "32 GB RAM",
                "SSD",
                "Public summary includes reproducibility-relevant hardware and runtime details only."),
            Dataset: new PublicBenchmarkDatasetIdentity(
                "generated",
                "VecNet generated uniform float32 vectors",
                "schema 0.1",
                "generated by VecNet.BenchmarkRunner",
                "not applicable",
                "seed:0x5EED2009",
                "0x5EED2009",
                VectorCount: 10_000,
                QueryCount: 100,
                Dimension: 384,
                Metric: "SquaredEuclidean",
                TruthMethod: "scalar exact top-k with VecNet tie policy"),
            Measurements: new PublicBenchmarkMeasurementCategories(
                Category("reported", "milliseconds", "pooled per-query latency samples", ["p50=0.10", "p95=0.20"]),
                Category("reported", "queriesPerSecond", "measured query throughput", ["qps=1000"]),
                Category("reported", "ratio", "recall at requested top-k", ["recallAtK=1.0"]),
                Category("reported", "bytesPerQuery", "managed allocations per query", ["bytesPerQuery=0"]),
                Category("absent", "bytes", "process private bytes", ["notReported"]),
                Category("absent", "bytes", "GC heap bytes", ["notReported"]),
                Category("absent", "bytes", "sampled peak private bytes", ["notReported"]),
                Category("absent", "bytes", "layout lower-bound estimate", ["notReported"]),
                Category("reported", "bytes", "persisted file bytes", ["bytes=0"]),
                Category("reported", "milliseconds", "index build duration", ["milliseconds=1.0"]),
                Category("absent", "milliseconds", "index open duration", ["notApplicable"]),
                Category("absent", "milliseconds", "checkpoint duration", ["notApplicable"])),
            Privacy: new PublicBenchmarkPrivacyStatus(
                "redacted",
                "redacted-public-summary",
                "Private raw artifacts are not published.",
                ["local absolute paths", "private cache roots", "raw artifact identifiers"],
                [
                    new PublicBenchmarkSourceArtifact(
                        "redacted-derived-evidence",
                        "VecNet.PublicBenchmarkSummary.Source",
                        "source artifact identifiers are redacted",
                        PubliclyIncluded: false)
                ]),
            Limitations:
            [
                "Generated vectors do not imply semantic-search relevance.",
                "No database or service replacement claim is made."
            ],
            Review: new PublicBenchmarkReviewStatus(
                "reviewed",
                "Review Agent",
                "2026-07-21T00:00:00Z",
                "Schema-shape test fixture only; not public benchmark evidence."));

    private static PublicBenchmarkMeasurementCategory Category(
        string status,
        string unit,
        string semantics,
        string[] values) =>
        new(status, unit, semantics, values, []);
}
