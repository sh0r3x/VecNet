using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec118HnswEstablishedComparisonTests
{
    [Fact]
    public void ParseHnswEstablishedComparison_UsesPrivatePinnedDefaults()
    {
        HnswEstablishedComparisonOptions options = CommandLine.ParseHnswEstablishedComparison(["hnswlib-generated-comparison"]);

        Assert.Equal(VectorMetric.SquaredEuclidean, options.Metric);
        Assert.Equal(128, options.Dimension);
        Assert.Equal(4096, options.VectorCount);
        Assert.Equal(100, options.QueryCount);
        Assert.Equal(10, options.TopK);
        Assert.Equal(1, options.Runs);
        Assert.Equal(3, options.WarmupQueries);
        Assert.Equal(0x5EED2118u, options.Seed);
        Assert.Equal(8, options.M);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(128, options.EfSearch);
        Assert.Equal(0x484E535700011818UL, options.HnswSeed);
        Assert.Contains("vec-118-tools", options.HnswlibPythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith(Path.Combine("Scripts", "python.exe"), options.HnswlibPythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.WorkDirectory);
        Assert.Equal([128, 384, 768], HnswEstablishedComparisonOptions.RepresentativeDimensions);
        Assert.Equal([386], HnswEstablishedComparisonOptions.OptionalAdversarialDimensions);
        Assert.Equal("0.8.0", HnswEstablishedComparisonOptions.HnswlibVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", HnswEstablishedComparisonOptions.HnswlibSourceDistributionSha256);
        Assert.Equal("Apache-2.0", HnswEstablishedComparisonOptions.HnswlibLicense);
    }

    [Theory]
    [InlineData("unexpected")]
    [InlineData("hnswlib-generated-comparison", "--metric", "Cosine")]
    [InlineData("hnswlib-generated-comparison", "--metric", "InnerProduct")]
    [InlineData("hnswlib-generated-comparison", "--dimension", "0")]
    [InlineData("hnswlib-generated-comparison", "--vectors", "0")]
    [InlineData("hnswlib-generated-comparison", "--queries", "0")]
    [InlineData("hnswlib-generated-comparison", "--top-k", "3", "--vectors", "2")]
    [InlineData("hnswlib-generated-comparison", "--runs", "0")]
    [InlineData("hnswlib-generated-comparison", "--runs", "6")]
    [InlineData("hnswlib-generated-comparison", "--warmup-queries", "-1")]
    [InlineData("hnswlib-generated-comparison", "--m", "1")]
    [InlineData("hnswlib-generated-comparison", "--m", "65")]
    [InlineData("hnswlib-generated-comparison", "--m", "8", "--ef-construction", "7")]
    [InlineData("hnswlib-generated-comparison", "--ef-construction", "4097")]
    [InlineData("hnswlib-generated-comparison", "--top-k", "10", "--ef-search", "9")]
    [InlineData("hnswlib-generated-comparison", "--ef-search", "4097")]
    [InlineData("hnswlib-generated-comparison", "--hnsw-seed", "0xNOTHEX")]
    [InlineData("hnswlib-generated-comparison", "--output", "")]
    [InlineData("hnswlib-generated-comparison", "--work-directory", "")]
    [InlineData("hnswlib-generated-comparison", "--vecnet-snapshot-directory", "")]
    [InlineData("hnswlib-generated-comparison", "--hnswlib-index", "")]
    [InlineData("hnswlib-generated-comparison", "--hnswlib-python", "")]
    [InlineData("hnswlib-generated-comparison", "--cache-root", "VecNet.DatasetCache")]
    [InlineData("hnswlib-generated-comparison", "--preset", "smoke")]
    [InlineData("hnswlib-generated-comparison", "--sample-interval-ms", "1")]
    public void ParseHnswEstablishedComparison_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void Run_WithMissingHnswlibPython_FailsBeforeWritingFakeEvidence()
    {
        string directory = NewArtifactDirectory("missing-tool");
        string outputPath = Path.Combine(directory, "missing-tool.json");
        var options = new HnswEstablishedComparisonOptions(
            VectorMetric.SquaredEuclidean,
            Dimension: 8,
            VectorCount: 16,
            QueryCount: 2,
            TopK: 3,
            Seed: 0x5EED1180,
            OutputPath: outputPath,
            WorkDirectory: Path.Combine(directory, "work"),
            VecNetSnapshotDirectory: Path.Combine(directory, "vecnet-snapshot"),
            HnswlibIndexPath: Path.Combine(directory, "hnswlib-index.bin"),
            HnswlibPythonPath: Path.Combine(directory, "missing-python.exe"),
            Runs: 1,
            WarmupQueries: 0,
            M: 4,
            EfConstruction: 8,
            EfSearch: 4,
            HnswSeed: 0x1180UL);

        FileNotFoundException exception = Assert.Throws<FileNotFoundException>(() =>
            HnswEstablishedComparisonScenario.Run(options, ["hnswlib-generated-comparison"]));

        Assert.Contains("unavailable", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(outputPath));
    }

    [Fact]
    public void Run_WhenPinnedHnswlibExists_ProducesPrivateComparisonReportWithIntegrityAndFalseEligibility()
    {
        string pythonPath = HnswEstablishedComparisonOptions.Default.HnswlibPythonPath;
        if (!File.Exists(pythonPath))
        {
            return;
        }

        string directory = NewArtifactDirectory("report");
        string outputPath = Path.Combine(directory, "hnswlib-generated-comparison.json");
        string[] arguments =
        [
            "hnswlib-generated-comparison",
            "--metric", "SquaredEuclidean",
            "--dimension", "12",
            "--vectors", "48",
            "--queries", "4",
            "--top-k", "5",
            "--runs", "2",
            "--warmup-queries", "1",
            "--seed", "0x5EED1181",
            "--m", "4",
            "--ef-construction", "12",
            "--ef-search", "8",
            "--hnsw-seed", "0x0000000000001181",
            "--hnswlib-python", pythonPath,
            "--output", outputPath,
            "--work-directory", Path.Combine(directory, "work"),
            "--vecnet-snapshot-directory", Path.Combine(directory, "vecnet-snapshot"),
            "--hnswlib-index", Path.Combine(directory, "hnswlib-index.bin")
        ];
        HnswEstablishedComparisonOptions options = CommandLine.ParseHnswEstablishedComparison(arguments);

        HnswEstablishedComparisonReport report = HnswEstablishedComparisonScenario.Run(options, arguments);
        HnswEstablishedComparisonScenario.Write(report, outputPath);

        Assert.True(File.Exists(outputPath));
        Assert.True(File.Exists(options.HnswlibIndexPath));
        Assert.True(File.Exists(Path.Combine(options.WorkDirectory, "run-hnswlib.py")));
        Assert.Equal("VecNet.HnswEstablishedComparisonReport", report.SchemaName);
        Assert.Equal("0.1", report.SchemaVersion);
        Assert.Equal("VEC-118", report.TaskId);
        Assert.Equal("hnswlib-generated-comparison", report.ScenarioName);
        Assert.Equal("local-evidence", report.ClaimClass);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.Equal("smoke", report.Evidence.Status);
        Assert.Equal("private-hnswlib-generated-comparison", report.Evidence.Scope);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.False(report.Evidence.BaselineCandidateEligible);
        Assert.False(report.Evidence.ComparisonPublicationEligible);
        Assert.False(report.Evidence.RegressionGateEligible);

        Assert.Equal("hnswlib", report.SourcePinning.PackageName);
        Assert.Equal("PyPI", report.SourcePinning.PackageSource);
        Assert.Equal("0.8.0", report.SourcePinning.PackageVersion);
        Assert.Equal("cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c", report.SourcePinning.SourceDistributionSha256);
        Assert.Equal("Apache-2.0", report.SourcePinning.License);
        Assert.Contains("non-shipping", report.SourcePinning.LicensePosture, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("native", report.SourcePinning.NativeBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("No hnswlib", report.SourcePinning.ProductDependencyPosture, StringComparison.Ordinal);

        Assert.Equal([128, 384, 768], report.Design.RepresentativeGeneratedDimensions);
        Assert.Equal([386], report.Design.OptionalAdversarialTailDimensions);
        Assert.Equal(12, report.Design.CurrentDimension);
        Assert.Equal("custom-smoke", report.Design.CurrentDimensionRole);
        Assert.Contains("384", report.Design.WorkloadPolicy, StringComparison.Ordinal);
        Assert.Contains("386", report.Design.TailDimensionPolicy, StringComparison.Ordinal);
        Assert.Contains("must not replace", report.Design.TailDimensionPolicy, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("generated-uniform", report.Dataset.Kind);
        Assert.Equal("generated-no-external-source", report.Dataset.SourceVerificationStatus);
        Assert.Equal(VectorMetric.SquaredEuclidean.ToString(), report.Parameters.Metric);
        Assert.Equal(12, report.Parameters.Dimension);
        Assert.Equal(48, report.Parameters.VectorCount);
        Assert.Equal(4, report.Parameters.QueryCount);
        Assert.Equal(5, report.Parameters.TopK);
        Assert.Equal(2, report.Parameters.Runs);
        Assert.Equal(1, report.Parameters.WarmupQueries);
        Assert.Equal(4, report.Parameters.M);
        Assert.Equal(12, report.Parameters.EfConstruction);
        Assert.Equal(12, report.Parameters.HnswlibEfConstruction);
        Assert.Equal(8, report.Parameters.EfSearch);
        Assert.Equal(8, report.Parameters.HnswlibEf);
        Assert.Equal(1, report.Parameters.ThreadCount);

        Assert.Contains("identical", report.Methodology.IdenticalInputsPolicy, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Python", report.Methodology.PythonBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("num_threads=1", report.Methodology.ThreadingPolicy, StringComparison.Ordinal);
        Assert.Contains("Nearest-rank", report.Methodology.LatencyPercentileEstimator, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("VecNet", report.VecNet.Name);
        Assert.Contains(".NET", report.VecNet.ImplementationType, StringComparison.Ordinal);
        Assert.Equal("measured", report.VecNet.Build.Status);
        Assert.True(report.VecNet.Build.ElapsedMilliseconds >= 0);
        Assert.Equal("measured", report.VecNet.Build.ManagedAllocations.Status);
        Assert.Equal("measured", report.VecNet.Search.Status);
        AssertSearch(report.VecNet.Search, expectedRuns: 2, expectedQueries: 4);
        Assert.Equal("measured", report.VecNet.Search.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.VecNet.Memory.Status);
        Assert.Equal("fileFacts", report.VecNet.PersistedBytes.Status);
        Assert.True(long.Parse(report.VecNet.PersistedBytes.Value, CultureInfo.InvariantCulture) > 0);
        Assert.Equal("passed", report.VecNet.Metrics.ReturnedResultIntegrity.Status);
        Assert.InRange(report.VecNet.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.VecNet.Metrics.OrderedAgreement, 0, 1);

        Assert.Equal("hnswlib", report.Hnswlib.Name);
        Assert.Contains("native", report.Hnswlib.ImplementationType, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("0.8.0", report.Hnswlib.Version);
        Assert.Contains("Python", report.Hnswlib.RuntimeBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("measured", report.Hnswlib.Build.Status);
        Assert.Equal("notMeasured", report.Hnswlib.Build.ManagedAllocations.Status);
        Assert.Equal("measured", report.Hnswlib.Search.Status);
        AssertSearch(report.Hnswlib.Search, expectedRuns: 2, expectedQueries: 4);
        Assert.Equal("notMeasured", report.Hnswlib.Search.ManagedAllocations.Status);
        Assert.Equal("notMeasured", report.Hnswlib.Memory.Status);
        Assert.Equal("fileFacts", report.Hnswlib.PersistedBytes.Status);
        Assert.True(long.Parse(report.Hnswlib.PersistedBytes.Value, CultureInfo.InvariantCulture) > 0);
        Assert.Equal("passed", report.Hnswlib.Metrics.ReturnedResultIntegrity.Status);
        Assert.InRange(report.Hnswlib.Metrics.RecallAtK, 0, 1);
        Assert.InRange(report.Hnswlib.Metrics.OrderedAgreement, 0, 1);

        Assert.Equal("passed", report.Validation.Status);
        Assert.True(report.Validation.FiniteVectors);
        Assert.True(report.Validation.TruthGenerated);
        Assert.True(report.Validation.IdenticalVectorsQueriesIdsAndParameters);
        Assert.True(report.Validation.VecNetComparedToTruth);
        Assert.True(report.Validation.HnswlibComparedToTruth);
        Assert.True(report.Validation.VecNetReturnedResultIntegrityPassed);
        Assert.True(report.Validation.HnswlibReturnedResultIntegrityPassed);
        Assert.False(report.Validation.PublicClaimEligible);
        Assert.False(report.Validation.BaselineCandidateEligible);
        Assert.False(report.Validation.ComparisonPublicationEligible);
        Assert.False(report.Validation.RegressionGateEligible);
        Assert.True(report.Validation.ReportIsPrivateRaw);
        Assert.False(report.Eligibility.PublicClaimEligible);
        Assert.False(report.Eligibility.BaselineCandidateEligible);
        Assert.False(report.Eligibility.ComparisonPublicationEligible);
        Assert.False(report.Eligibility.RegressionGateEligible);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.HnswEstablishedComparisonReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("0.8.0", root.GetProperty("sourcePinning").GetProperty("packageVersion").GetString());
        Assert.Equal("Apache-2.0", root.GetProperty("sourcePinning").GetProperty("license").GetString());
        Assert.True(root.TryGetProperty("methodology", out JsonElement methodology));
        Assert.Contains("Python", methodology.GetProperty("pythonBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("vecNet").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("passed", root.GetProperty("hnswlib").GetProperty("metrics").GetProperty("returnedResultIntegrity").GetProperty("status").GetString());
        Assert.Equal("notMeasured", root.GetProperty("hnswlib").GetProperty("memory").GetProperty("status").GetString());
        Assert.Equal("fileFacts", root.GetProperty("hnswlib").GetProperty("persistedBytes").GetProperty("status").GetString());
        Assert.False(root.GetProperty("eligibility").GetProperty("publicClaimEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("baselineCandidateEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("comparisonPublicationEligible").GetBoolean());
        Assert.False(root.GetProperty("eligibility").GetProperty("regressionGateEligible").GetBoolean());
        Assert.DoesNotContain("\"publicClaimEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"comparisonPublicationEligible\": true", File.ReadAllText(outputPath), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExistingHnswScenarioParsersRejectHnswlibComparisonOptions()
    {
        _ = CommandLine.ParseHnswEstablishedComparison(["hnswlib-generated-comparison", "--vectors", "16", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseHnswGenerated(["hnsw-generated", "--vectors", "16", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);
        _ = CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke", "--vectors", "16", "--queries", "1", "--top-k", "3", "--ef-search", "3"]);

        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswGenerated(["hnsw-generated", "--hnswlib-python", "python"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswMemorySmoke(["generated-hnsw-memory-smoke", "--hnswlib-index", "index.bin"]));
        Assert.Throws<ArgumentException>(() => CommandLine.ParseHnswEstablishedComparison(["hnswlib-generated-comparison", "--cache-root", "VecNet.DatasetCache"]));
    }

    private static void AssertSearch(HnswEstablishedComparisonSearchInfo search, int expectedRuns, int expectedQueries)
    {
        Assert.Equal(expectedQueries, search.MeasuredQueryCount);
        Assert.Equal(expectedRuns, search.Runs.Length);
        Assert.Equal(expectedRuns, search.Aggregate.RunCount);
        Assert.Equal(expectedQueries, search.Aggregate.MeasuredQueryCountPerRun);
        Assert.True(search.ElapsedMilliseconds >= 0);
        Assert.True(search.LatencyP50Milliseconds >= 0);
        Assert.True(search.LatencyP95Milliseconds >= search.LatencyP50Milliseconds);
        Assert.True(search.LatencyP99Milliseconds >= search.LatencyP95Milliseconds);
        Assert.True(search.Qps > 0);
        Assert.All(search.Runs, run =>
        {
            Assert.Equal(expectedQueries, run.MeasuredQueryCount);
            Assert.True(run.ElapsedMilliseconds >= 0);
            Assert.True(run.LatencyP50Milliseconds >= 0);
            Assert.True(run.LatencyP95Milliseconds >= run.LatencyP50Milliseconds);
            Assert.True(run.LatencyP99Milliseconds >= run.LatencyP95Milliseconds);
            Assert.True(run.Qps > 0);
        });
    }

    private static string NewArtifactDirectory(string prefix)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec118-{prefix}-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
