using System.Globalization;
using System.Text.Json;
using VecNet.BenchmarkRunner;

namespace VecNet.BenchmarkRunner.Tests;

public sealed class Vec360InnerProductHotPathBenchmarkTests
{
    [Fact]
    public void ParseInnerProductHotPath_UsesApprovedDimensionsAndAllOperationShapesByDefault()
    {
        InnerProductHotPathOptions options = CommandLine.ParseInnerProductHotPath([InnerProductHotPathOptions.ScenarioName]);

        Assert.Equal([31, 33, 127, 128, 129, 384, 386, 768, 769, 1536], options.Dimensions);
        Assert.Equal(512, options.VectorCount);
        Assert.Equal(16, options.QueryCount);
        Assert.Equal(1, options.Runs);
        Assert.Equal(1, options.WarmupIterations);
        Assert.Equal(0x5EED2360u, options.Seed);
        Assert.Equal(InnerProductHotPathOptions.AllOperationShapes, options.OperationShape);
        Assert.Equal(64, options.EfConstruction);
        Assert.Equal(64, options.EfSearch);
        Assert.StartsWith("VecNet.BenchmarkRunner.Artifacts", options.OutputPath);
        Assert.False(Path.IsPathRooted(options.OutputPath));

        InnerProductHotPathCaseInfo[] cases = InnerProductHotPathScenario.ExpandCases(options);

        Assert.Equal(30, cases.Length);
        Assert.Contains(cases, item => item.Dimension == 128 && item.DimensionClass == "representative");
        Assert.Contains(cases, item => item.Dimension == 384 && item.DimensionClass == "representative");
        Assert.Contains(cases, item => item.Dimension == 768 && item.DimensionClass == "representative");
        Assert.Contains(cases, item => item.Dimension == 1536 && item.DimensionClass == "representative");
        Assert.Contains(cases, item => item.Dimension == 31 && item.DimensionClass == "awkward");
        Assert.Contains(cases, item => item.Dimension == 33 && item.DimensionClass == "awkward");
        Assert.Contains(cases, item => item.Dimension == 127 && item.DimensionClass == "awkward");
        Assert.Contains(cases, item => item.Dimension == 129 && item.DimensionClass == "awkward");
        Assert.Contains(cases, item => item.Dimension == 386 && item.DimensionClass == "awkward");
        Assert.Contains(cases, item => item.Dimension == 769 && item.DimensionClass == "awkward");
        Assert.Equal(
            [
                InnerProductHotPathOptions.ExactFlatSearchShape,
                InnerProductHotPathOptions.HnswBuildDistanceCallsShape,
                InnerProductHotPathOptions.HnswSearchDistanceCallsShape
            ],
            cases.Select(item => item.OperationShape).Distinct().Order().ToArray());
    }

    [Theory]
    [InlineData("inner-product-hot-path", "--dimensions", "31,31")]
    [InlineData("inner-product-hot-path", "--dimensions", "31,0")]
    [InlineData("inner-product-hot-path", "--dimensions", "31,nope")]
    [InlineData("inner-product-hot-path", "--operation-shape", "mutable-update")]
    [InlineData("inner-product-hot-path", "--runs", "6")]
    [InlineData("inner-product-hot-path", "--warmup-iterations", "-1")]
    [InlineData("inner-product-hot-path", "--metric", "InnerProduct")]
    [InlineData("inner-product-hot-path", "--output", "")]
    public void ParseInnerProductHotPath_RejectsInvalidOrOutOfScopeCommandLines(params string[] args)
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(() => CommandLine.ParseInnerProductHotPath(args));

        Assert.NotEmpty(exception.Message);
    }

    [Fact]
    public void RunInnerProductHotPath_ReportsMeasurementsValidationAndNoRawPathOrTaskWording()
    {
        string outputPath = NewArtifactPath("inner-product-hot-path.json");
        string[] arguments =
        [
            InnerProductHotPathOptions.ScenarioName,
            "--dimensions", "31,128",
            "--vectors", "8",
            "--queries", "2",
            "--runs", "1",
            "--warmup-iterations", "0",
            "--operation-shape", "all",
            "--ef-construction", "4",
            "--ef-search", "4",
            "--seed", "0x5EED2362",
            "--output", outputPath
        ];

        InnerProductHotPathOptions options = CommandLine.ParseInnerProductHotPath(arguments);
        InnerProductHotPathReport report = InnerProductHotPathScenario.Run(options);
        InnerProductHotPathScenario.Write(report, outputPath);
        string json = File.ReadAllText(outputPath);

        Assert.Equal("VecNet.InnerProductHotPathReport", report.SchemaName);
        Assert.Equal(InnerProductHotPathOptions.ScenarioName, report.ScenarioName);
        Assert.Equal("private-raw", report.PrivacyClass);
        Assert.False(report.Evidence.PublicClaimEligible);
        Assert.Equal("InnerProduct", report.Options.Metric);
        Assert.Equal([31, 128], report.Options.Dimensions);
        Assert.Equal("passed", report.Validation.Status);
        Assert.Equal(6, report.Validation.CaseCount);
        Assert.Equal(0, report.Validation.CategoryMismatchCount);
        Assert.Equal(0, report.Validation.FiniteDistanceMismatchCount);
        Assert.True(report.Validation.PositiveInfinityComparisons > 0);
        Assert.True(report.Validation.NegativeInfinityComparisons > 0);
        Assert.True(report.Validation.NaNComparisons > 0);
        Assert.All(report.Cases, item =>
        {
            Assert.Equal("InnerProduct", item.Metric);
            Assert.Equal("passed", item.Validation.Status);
            Assert.True(item.CurrentScalar.DistanceCallCount > 0);
            Assert.Equal(item.CurrentScalar.DistanceCallCount, item.CandidateSharedDot.DistanceCallCount);
            Assert.Equal(0, item.CurrentScalar.ManagedAllocatedBytes);
            Assert.Equal(0, item.CandidateSharedDot.ManagedAllocatedBytes);
        });

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        Assert.Equal("VecNet.InnerProductHotPathReport", root.GetProperty("schemaName").GetString());
        Assert.Equal("InnerProduct", root.GetProperty("options").GetProperty("metric").GetString());
        Assert.Equal("exact-flat-search", root.GetProperty("cases")[0].GetProperty("operationShape").GetString());
        Assert.True(root.GetProperty("cases")[0].GetProperty("currentScalar").GetProperty("distanceCallCount").GetInt64() > 0);
        Assert.True(root.GetProperty("cases")[0].GetProperty("candidateSharedDot").GetProperty("distanceCallsPerSecond").GetDouble() > 0);
        Assert.Equal("passed", root.GetProperty("validation").GetProperty("status").GetString());
        Assert.DoesNotContain(outputPath, json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("VEC-360", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Agent", json, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateDistances_DetectsFiniteAndCategoryDrift()
    {
        float[] left = [1f, 2f, float.MaxValue, float.MaxValue];
        float[] right = [3f, 4f, float.MaxValue, -float.MaxValue];

        InnerProductHotPathCaseValidationInfo finiteDrift = InnerProductHotPathScenario.ValidateDistances(
            left.AsSpan(0, 2),
            right.AsSpan(0, 2),
            dimension: 2,
            InnerProductHotPathPrimitives.CurrentScalarDistance,
            static (left, right) => InnerProductHotPathPrimitives.CurrentScalarDistance(left, right) + 1);
        InnerProductHotPathCaseValidationInfo categoryDrift = InnerProductHotPathScenario.ValidateDistances(
            left.AsSpan(2, 2),
            right.AsSpan(2, 2),
            dimension: 2,
            InnerProductHotPathPrimitives.CurrentScalarDistance,
            static (_, _) => float.NaN);

        Assert.Equal("failed", finiteDrift.Status);
        Assert.Equal(1, finiteDrift.FiniteDistanceMismatchCount);
        Assert.Equal(0, finiteDrift.CategoryMismatchCount);
        Assert.NotEmpty(finiteDrift.DriftExamples);
        Assert.Equal("failed", categoryDrift.Status);
        Assert.Equal(1, categoryDrift.CategoryMismatchCount);
        Assert.Equal(0, categoryDrift.FiniteDistanceMismatchCount);
        Assert.NotEmpty(categoryDrift.DriftExamples);
    }

    private static string NewArtifactPath(string fileName)
    {
        string directory = Path.Combine(
            "VecNet.BenchmarkRunner.Artifacts",
            string.Create(CultureInfo.InvariantCulture, $"vec360-{Guid.NewGuid():N}"));
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, fileName);
    }
}
