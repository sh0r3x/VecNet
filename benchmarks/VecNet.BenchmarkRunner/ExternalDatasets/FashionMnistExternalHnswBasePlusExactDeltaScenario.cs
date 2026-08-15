using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public static class FashionMnistExternalHnswBasePlusExactDeltaScenario
{
    private const string TaskId = "VEC-127";
    private const string SchemaName = "VecNet.ExternalHnswBasePlusExactDeltaBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static ExternalHnswBasePlusExactDeltaBenchmarkReport Run(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset = LoadAndValidateDataset(options);

        BuildMeasurement build = BuildBaseIndex(options, dataset);
        var composite = new HnswBasePlusExactDeltaIndex(build.Index);
        long generationBeforeMutations = composite.Generation;
        MutationExecution mutationExecution = ExecuteMutations(options, dataset, composite);
        ulong[] liveIds = BuildLiveIds(options);
        TruthSet truth = GenerateLiveTruth(dataset, options, liveIds);

        WarmupSearch(options, dataset, composite);
        SearchMeasurement measurement = MeasureSearch(options, dataset, composite);

        ResultComparison comparison = ResultComparer.Compare(
            truth,
            measurement.Results,
            options.TopK,
            dataset.Dimension,
            options.Metric);
        HnswBasePlusExactDeltaReturnedResultIntegrityInfo returnedIntegrity =
            ValidateReturnedResults(dataset, measurement.Results, options, liveIds);
        HnswBasePlusExactDeltaUnderfillInfo underfill = CreateUnderfill(options, measurement.Results);
        int extraResultCount = CountExtraResults(truth, measurement.Results, options.TopK);

        bool statusCountsMatched = mutationExecution.StatusCounts.Committed ==
                mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount &&
            mutationExecution.StatusCounts.DuplicateId == options.DuplicateInsertAttempts &&
            mutationExecution.StatusCounts.UnknownId == options.UnknownDeleteAttempts &&
            mutationExecution.StatusCounts.AlreadyDeleted == options.RepeatedDeleteAttempts &&
            mutationExecution.StatusCounts.ReadOnly == 0 &&
            mutationExecution.StatusCounts.Unsupported == 0;
        bool generationMatched = mutationExecution.GenerationAfterMutations - generationBeforeMutations ==
            mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount;
        bool validationPassed = statusCountsMatched &&
            generationMatched &&
            returnedIntegrity.Status == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        HnswBasePlusExactDeltaCountInfo counts = CreateCountInfo(options, composite);

        return new ExternalHnswBasePlusExactDeltaBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new ExternalBenchmarkDatasetInfo(
                dataset.Manifest.DatasetId,
                dataset.Manifest.Source,
                dataset.Manifest.License,
                dataset.Manifest.Privacy,
                dataset.Manifest.Shape,
                dataset.Manifest.Metric,
                new ExternalBenchmarkAdmissionManifestInfo(
                    dataset.Manifest.SchemaName,
                    dataset.Manifest.SchemaVersion,
                    dataset.Paths.RelativeManifestPath,
                    dataset.ManifestSha256),
                dataset.Manifest.RawFiles,
                dataset.Manifest.Conversion.OutputFiles,
                dataset.Manifest.Conversion,
                dataset.Manifest.Labels),
            new ExternalHnswBasePlusExactDeltaWorkloadInfo(
                dataset.Manifest.DatasetId,
                dataset.BaseCount,
                dataset.QueryMatrixCount,
                options.QueryCount,
                options.TopK,
                dataset.Dimension,
                dataset.Manifest.Shape.SourceDataType,
                dataset.Manifest.Shape.ConvertedDataType,
                "first N query vectors from the admitted Fashion-MNIST query matrix; query rows are never candidate vectors",
                ImmutableBaseStartRow: 0,
                ImmutableBaseEndRowInclusive: options.BaseVectorCount - 1,
                options.BaseVectorCount,
                DeltaStartRow: options.BaseVectorCount,
                DeltaEndRowInclusive: options.PhysicalCandidateVectorCount - 1,
                options.InsertedDeltaCount,
                dataset.BaseCount - options.PhysicalCandidateVectorCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                "contiguous admitted base-matrix rows: immutable base first, exact delta immediately after base, remaining admitted rows unused",
                "candidate external IDs are original Fashion-MNIST base row ordinals; base IDs are 0..baseVectorCount-1 and delta IDs are baseVectorCount..baseVectorCount+insertions-1",
                "build immutable HNSW base, add exact delta rows, delete base rows from the start, delete delta rows from the start, attempt duplicate/reserved insert, unknown delete and repeated delete",
                string.Create(CultureInfo.InvariantCulture, $"0x{options.Seed:X8}")),
            new ExternalBenchmarkTruthInfo(
                dataset.Truth.SchemaName,
                dataset.Truth.SchemaVersion,
                dataset.Manifest.Truth.Kind,
                dataset.Manifest.Truth.RelativePath,
                dataset.TruthSha256,
                "first N query vectors from the admitted query matrix",
                dataset.Truth.QuerySubsetCount,
                dataset.Truth.TruthDepth,
                options.TopK,
                dataset.Truth.TiePolicy,
                "Existing admitted Fashion-MNIST exact truth is loaded only as cache/readiness guard; final recall/order use updated live-view truth.",
                dataset.Truth.SourceRawSha256),
            new ExternalHnswBasePlusExactDeltaUpdatedTruthInfo(
                "scalar-reference-external-live-hnsw-base-plus-exact-delta",
                "computed in memory during the scenario from the post-update live view: selected immutable base rows plus committed delta rows minus base and delta tombstones",
                Persisted: false,
                options.QueryCount,
                truth.Depth,
                liveIds.Length,
                FashionMnistExactTruth.TiePolicy(options.Metric),
                FashionMnistExactTruth.DistanceSemantics(options.Metric),
                "existing admitted truth artifact validates cache/truth readiness only and is not final updated truth",
                "live candidate IDs are selected immutable base rows and committed exact delta rows after tombstone suppression"),
            new ScenarioInfo(
                FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "cache checks, checksum validation, matrix/truth load, immutable HNSW base build, composite construction, update application, exact updated truth construction, warmup, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "InternalHnswBasePlusExactDelta",
                nameof(HnswBasePlusExactDeltaIndex),
                options.Metric.ToString(),
                dataset.Dimension,
                counts.LiveVectorCount,
                "internal HnswBasePlusExactDeltaIndex over admitted Fashion-MNIST immutable HnswIndex base, exact in-memory delta and tombstone overlay; no public mutable HNSW API, durable overlay persistence, checkpoint/rebuild, filtering, comparison tooling or matrix mode"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "admitted Fashion-MNIST base matrix row order, immutable base rows first, original row ordinals as external IDs",
                $"{options.Metric} only"),
            new HnswBuildInfo(
                "measured",
                build.ElapsedMilliseconds,
                new MeasurementStatusInfo(
                    "measured",
                    build.ManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                    "bytes",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around immutable HnswIndex construction and Add calls for selected Fashion-MNIST base rows only; cache checks, matrix/truth loading, composite construction, mutation application and exact updated truth generation are excluded."),
                options.BaseVectorCount,
                dataset.Dimension,
                "internal HnswIndex construction and selected admitted base-vector Add calls",
                "cache checks, checksum validation, matrix load, truth load, composite construction, update application, exact updated truth generation, warmup, measured composite search, result comparison and report writing"),
            counts,
            new HnswBasePlusExactDeltaMutationInfo(
                mutationExecution.InsertedCount,
                mutationExecution.DeletedBaseCount,
                mutationExecution.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                mutationExecution.InsertedCount + mutationExecution.DeletedBaseCount + mutationExecution.DeletedDeltaCount,
                generationBeforeMutations,
                mutationExecution.GenerationAfterMutations,
                mutationExecution.GenerationAfterMutations - generationBeforeMutations,
                generationMatched,
                mutationExecution.StatusCounts),
            new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            CreateMeasurement(options, measurement),
            new HnswBasePlusExactDeltaMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                returnedIntegrity.DistanceMismatchCount == 0 ? "passed" : "failed",
                returnedIntegrity.DistanceMismatchCount,
                comparison.MissingResultCount,
                extraResultCount,
                returnedIntegrity,
                "set recall@k = returned live ids intersect exact updated top-k live ids divided by min(k, post-update live vector count), summed across measured queries",
                "Every returned external composite result is checked for finite distance, no duplicate ID within its query, post-update live ID membership, no tombstoned ID, and selected-metric distance matching recomputation for that returned ID/query within the accepted ResultComparer tolerance. HNSW base search is approximate and recall/order are recorded, not required."),
            underfill,
            new ExternalHnswBasePlusExactDeltaValidationInfo(
                validationPassed ? "passed" : "failed",
                "external-fashion-mnist-hnsw-base-plus-exact-delta-smoke",
                LoadedExistingTruthGuard: true,
                UpdatedTruthGeneratedFromLiveView: true,
                HnswBaseBuilt: true,
                MutationsApplied: true,
                MutationStatusCountsMatched: statusCountsMatched,
                GenerationMovementMatchedCommittedMutations: generationMatched,
                FinalRunComparedToUpdatedTruth: true,
                ReturnedResultsAreLiveAndNotTombstoned: returnedIntegrity.Status == "passed",
                AllowsApproximateRecallBelowOne: true,
                AllowsUnderfill: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new ExternalBenchmarkEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "External Fashion-MNIST HNSW base-plus-exact-delta reports are private local evidence only until a reviewed public summary policy and public mutable-HNSW admission exist.",
                "No external mutable/update HNSW baseline-candidate policy is accepted.",
                "No external mutable/update HNSW regression-gate policy, threshold, comparison artifact or hard gate is accepted."),
            [
                "Private external Fashion-MNIST HNSW base-plus-exact-delta smoke evidence only; not a public benchmark claim.",
                "This command validates an already admitted Fashion-MNIST cache and existing truth artifact; it does not download, convert or regenerate dataset artifacts.",
                "Existing admitted truth is a cache/readiness guard only; exact updated truth is computed in memory from the post-update live view.",
                "This report exercises an internal composite type and does not add or imply a public mutable/update HNSW API.",
                "Durable mutable overlay persistence, checkpoint/rebuild, direct graph mutation, filtering, external matrix orchestration and hnswlib/FAISS comparison are out of scope.",
                "Latency/QPS/allocation time only internal composite Search calls with caller-owned result buffers and workspace.",
                "Immutable HNSW base build, update application and exact updated truth generation are setup work and excluded from measured search timing.",
                "Approximate recall below 1.0 and underfill are allowed and recorded.",
                "Memory is explicitly not measured in VEC-127.",
                "Public claims, baseline candidates, comparison artifacts and regression gates are not created by this report."
            ]);
    }

    public static void Write(ExternalHnswBasePlusExactDeltaBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    internal static HnswBasePlusExactDeltaReturnedResultIntegrityInfo ValidateReturnedResults(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        SearchResult[][] actual,
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        IReadOnlyCollection<ulong> liveIds)
    {
        var live = new HashSet<ulong>(liveIds);
        int checkedResultCount = 0;
        int queryCountMismatchCount = actual.Length == options.QueryCount ? 0 : 1;
        int resultCountViolationCount = 0;
        int nonFiniteDistanceCount = 0;
        int duplicateIdCount = 0;
        int unknownIdCount = 0;
        int tombstonedIdCount = 0;
        int distanceMismatchCount = 0;
        int queryCount = Math.Min(options.QueryCount, actual.Length);
        int maxExpectedResults = Math.Min(options.TopK, live.Count);

        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            SearchResult[] returned = actual[queryRow];
            if (returned.Length > maxExpectedResults)
            {
                resultCountViolationCount++;
            }

            var seen = new HashSet<ulong>();
            for (int i = 0; i < returned.Length; i++)
            {
                SearchResult result = returned[i];
                checkedResultCount++;

                if (!float.IsFinite(result.Distance))
                {
                    nonFiniteDistanceCount++;
                }

                if (!seen.Add(result.Id))
                {
                    duplicateIdCount++;
                }

                if (result.Id >= (ulong)dataset.BaseCount ||
                    result.Id >= (ulong)options.PhysicalCandidateVectorCount)
                {
                    unknownIdCount++;
                    continue;
                }

                if (!live.Contains(result.Id))
                {
                    tombstonedIdCount++;
                    continue;
                }

                float expectedDistance = ScalarGroundTruth.CalculateDistance(
                    dataset.GetQueryVector(queryRow),
                    dataset.GetBaseVector(checked((int)result.Id)),
                    options.Metric);
                if (!ResultComparer.DistanceMatches(expectedDistance, result.Distance, dataset.Dimension, options.Metric))
                {
                    distanceMismatchCount++;
                }
            }
        }

        bool passed = queryCountMismatchCount == 0 &&
            resultCountViolationCount == 0 &&
            nonFiniteDistanceCount == 0 &&
            duplicateIdCount == 0 &&
            unknownIdCount == 0 &&
            tombstonedIdCount == 0 &&
            distanceMismatchCount == 0;

        return new HnswBasePlusExactDeltaReturnedResultIntegrityInfo(
            passed ? "passed" : "failed",
            checkedResultCount,
            queryCountMismatchCount,
            resultCountViolationCount,
            nonFiniteDistanceCount,
            duplicateIdCount,
            unknownIdCount,
            tombstonedIdCount,
            distanceMismatchCount,
            "For every returned external composite result: distance must be finite; IDs must be unique within a query; ID must be one of the selected post-update live Fashion-MNIST base-row IDs; tombstoned IDs must not be returned; and reported distance must match recomputed selected-metric distance for that query and returned ID within the accepted ResultComparer tolerance.",
            passed
                ? "All returned external composite results are live, not tombstoned, well formed and distance-integrity checked."
                : "One or more returned external composite results failed live-ID, tombstone, well-formedness or distance-integrity checks.");
    }

    private static FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset LoadAndValidateDataset(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options)
    {
        var guardOptions = new FashionMnistExternalHnswBenchmarkOptions(
            options.CacheRoot,
            options.OutputPath,
            options.QueryCount,
            options.TopK,
            Runs: 1,
            WarmupQueries: options.WarmupQueries,
            options.Metric,
            options.M,
            options.EfConstruction,
            options.EfSearch,
            options.HnswSeed);
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset =
            FashionMnistExternalHnswBenchmarkScenario.LoadAndValidateDataset(guardOptions);

        Require(options.PhysicalCandidateVectorCount <= dataset.BaseCount, "base vectors plus insertions must fit the admitted Fashion-MNIST base matrix.");
        Require(options.TopK <= options.LiveVectorCount, "top-k must not exceed the post-update live vector count.");
        Require(options.WarmupQueries == 0 || options.WarmupQueries <= dataset.QueryMatrixCount, "warmup query count must not exceed admitted query matrix count.");
        return dataset;
    }

    private static BuildMeasurement BuildBaseIndex(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var index = new HnswIndex(dataset.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetBaseVector(row));
        }

        long elapsed = Stopwatch.GetTimestamp() - start;
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;
        return new BuildMeasurement(index, (double)elapsed / Stopwatch.Frequency * 1000, allocatedBytes);
    }

    private static MutationExecution ExecuteMutations(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var counts = new MutableMutationStatusCounts();
        VectorMutationResult lastResult = default;
        int inserted = 0;
        int deletedBase = 0;
        int deletedDelta = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            ulong id = (ulong)(options.BaseVectorCount + i);
            lastResult = composite.TryAdd(id, dataset.GetBaseVector(checked(options.BaseVectorCount + i)));
            counts.Add(lastResult.Status);
            if (lastResult.Status == VectorMutationStatus.Committed)
            {
                inserted++;
            }
        }

        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            lastResult = composite.TryDelete((ulong)i);
            counts.Add(lastResult.Status);
            if (lastResult.Status == VectorMutationStatus.Committed)
            {
                deletedBase++;
            }
        }

        for (int i = 0; i < options.DeletedDeltaCount; i++)
        {
            lastResult = composite.TryDelete((ulong)(options.BaseVectorCount + i));
            counts.Add(lastResult.Status);
            if (lastResult.Status == VectorMutationStatus.Committed)
            {
                deletedDelta++;
            }
        }

        for (int i = 0; i < options.DuplicateInsertAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(i % options.BaseVectorCount);
            lastResult = composite.TryAdd(id, dataset.GetBaseVector(options.BaseVectorCount + (i % options.InsertedDeltaCount)));
            counts.Add(lastResult.Status);
        }

        ulong firstUnknownId = (ulong)dataset.BaseCount;
        for (int i = 0; i < options.UnknownDeleteAttempts; i++)
        {
            lastResult = composite.TryDelete(firstUnknownId + (ulong)i);
            counts.Add(lastResult.Status);
        }

        int committedDeleteCount = options.DeletedBaseCount + options.DeletedDeltaCount;
        for (int i = 0; i < options.RepeatedDeleteAttempts; i++)
        {
            ulong id = i % committedDeleteCount < options.DeletedBaseCount
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(options.BaseVectorCount + ((i - options.DeletedBaseCount) % options.DeletedDeltaCount));
            lastResult = composite.TryDelete(id);
            counts.Add(lastResult.Status);
        }

        return new MutationExecution(inserted, deletedBase, deletedDelta, lastResult.Generation, counts.ToInfo());
    }

    private static ulong[] BuildLiveIds(FashionMnistExternalHnswBasePlusExactDeltaOptions options)
    {
        var ids = new ulong[options.LiveVectorCount];
        int write = 0;
        for (int row = options.DeletedBaseCount; row < options.BaseVectorCount; row++)
        {
            ids[write++] = (ulong)row;
        }

        for (int row = options.DeletedDeltaCount; row < options.InsertedDeltaCount; row++)
        {
            ids[write++] = (ulong)(options.BaseVectorCount + row);
        }

        return ids;
    }

    private static TruthSet GenerateLiveTruth(
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        ulong[] liveIds)
    {
        var results = new TruthItem[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQueryVector(queryRow);
            var candidates = new TruthItem[liveIds.Length];
            for (int i = 0; i < liveIds.Length; i++)
            {
                ulong id = liveIds[i];
                candidates[i] = new TruthItem(id, ScalarGroundTruth.CalculateDistance(query, dataset.GetBaseVector(checked((int)id)), options.Metric));
            }

            Array.Sort(candidates, CompareTruthItems);
            var top = new TruthItem[options.TopK];
            Array.Copy(candidates, top, options.TopK);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    private static void WarmupSearch(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateWorkspace(options);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            composite.Search(dataset.GetQueryVector(i % dataset.QueryMatrixCount), results, workspace);
        }
    }

    private static SearchMeasurement MeasureSearch(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, composite, captureResults);
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureSingleRun(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        FashionMnistExternalHnswBenchmarkScenario.LoadedExternalDataset dataset,
        HnswBasePlusExactDeltaIndex composite,
        bool captureResults)
    {
        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateWorkspace(options);
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQueryVector(queryRow);
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = composite.Search(query, results, workspace);
            long elapsed = Stopwatch.GetTimestamp() - start;
            long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;

            latencyTicks[queryRow] = elapsed;
            totalTicks += elapsed;
            totalAllocatedBytes += allocatedBytes;

            if (captureResults)
            {
                var queryResults = new SearchResult[written];
                results.AsSpan(0, written).CopyTo(queryResults);
                allResults![queryRow] = queryResults;
            }
        }

        Array.Sort(latencyTicks);
        double elapsedSeconds = (double)totalTicks / Stopwatch.Frequency;
        return new SingleRunMeasurement(
            new SearchRunInfo(
                RunNumber: 0,
                options.QueryCount,
                elapsedSeconds * 1000,
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.50, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.95, Stopwatch.Frequency),
                LatencyPercentiles.NearestRankMilliseconds(latencyTicks, 0.99, Stopwatch.Frequency),
                elapsedSeconds == 0 ? double.PositiveInfinity : options.QueryCount / elapsedSeconds,
                totalAllocatedBytes,
                (double)totalAllocatedBytes / options.QueryCount),
            allResults);
    }

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(FashionMnistExternalHnswBasePlusExactDeltaOptions options) =>
        new(
            options.BaseVectorCount,
            options.EfSearch,
            Math.Min(options.BaseVectorCount, options.EfSearch),
            options.TopK);

    private static HnswBasePlusExactDeltaCountInfo CreateCountInfo(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        HnswBasePlusExactDeltaIndex composite)
    {
        int physicalCount = checked(composite.BasePhysicalVectorCount + composite.DeltaPhysicalVectorCount);
        return new HnswBasePlusExactDeltaCountInfo(
            composite.BasePhysicalVectorCount,
            composite.BaseLiveVectorCount,
            composite.DeltaPhysicalVectorCount,
            composite.DeltaLiveVectorCount,
            composite.BaseTombstoneCount,
            composite.DeltaTombstoneCount,
            composite.TombstoneCount,
            composite.LiveVectorCount,
            composite.DeletedReservedIdCount,
            composite.Generation,
            physicalCount == 0 ? 0 : (double)composite.TombstoneCount / physicalCount,
            options.BaseVectorCount == 0 ? 0 : (double)composite.DeltaPhysicalVectorCount / options.BaseVectorCount,
            "Base physical rows remain in the immutable HNSW graph after tombstone; delta physical rows remain after delta tombstone; live count is base live plus delta live; deleted/reserved IDs cannot be reused in this writable composite instance.");
    }

    private static HnswBasePlusExactDeltaUnderfillInfo CreateUnderfill(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        SearchResult[][] actual)
    {
        int totalReturned = 0;
        int underfilledQueries = 0;
        int underfilledSlots = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            int returned = actual[i].Length;
            totalReturned += returned;
            if (returned < options.TopK)
            {
                underfilledQueries++;
                underfilledSlots += options.TopK - returned;
            }
        }

        return new HnswBasePlusExactDeltaUnderfillInfo(
            options.QueryCount,
            options.TopK,
            checked(options.QueryCount * options.TopK),
            totalReturned,
            underfilledQueries,
            underfilledSlots,
            "Underfill is recorded when the composite returns fewer than requested top-k live results for a query. This can occur from tombstone-filtered HNSW overfetch limits or approximate base search and is measured against exact updated truth rather than treated as exact-search failure.");
    }

    private static MeasurementInfo CreateMeasurement(
        FashionMnistExternalHnswBasePlusExactDeltaOptions options,
        SearchMeasurement measurement) =>
        new(
            Latency: new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredQuery",
                "internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace)",
                "cache checks, checksum validation, matrix/truth load, immutable HNSW base build, composite construction, update application, exact updated truth construction, warmup queries, final-run result capture/comparison and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Top-level search latency percentile fields and search aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            ManagedAllocations: new MeasurementStatusInfo(
                "measured",
                measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                "bytesPerSearchCall",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswBasePlusExactDeltaSearchWorkspace; cache checks, matrix/truth load, HNSW build, mutation application, exact updated truth, warmup, result capture/comparison and report writing are excluded."),
            Memory: new MeasurementStatusInfo(
                "notMeasured",
                "absent",
                "bytes",
                "Process working set, resident memory, private bytes, managed heap size, layout estimates and peak memory are not measured in VEC-127."),
            RepeatedRuns: new RepeatedRunInfo(
                options.Runs > 1 ? "measured" : "singleRun",
                options.Runs,
                options.Runs > 1,
                options.Runs > 1
                    ? "Multiple measured external HNSW base-plus-exact-delta search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                    : "Only one measured run executed, so cross-run variance/noise is not measured."),
            RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
            Warmup: new WarmupInfo(
                options.WarmupQueries > 0 ? "executed" : "absent",
                options.WarmupQueries,
                options.WarmupQueries > 0
                    ? "Warmup queries executed after mutation application using caller-owned results/workspace and excluded from measured timing and allocation totals."
                    : "No warmup queries were requested."));

    private static AggregateTimingInfo AggregateRuns(SearchRunInfo[] runs, int measuredQueryCountPerRun) =>
        new(
            runs.Length,
            measuredQueryCountPerRun,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds),
            runs.Average(run => run.LatencyP50Milliseconds),
            runs.Average(run => run.LatencyP95Milliseconds),
            runs.Average(run => run.LatencyP99Milliseconds),
            runs.Average(run => run.Qps),
            runs.Min(run => run.Qps),
            runs.Max(run => run.Qps),
            runs.Average(run => run.ManagedAllocatedBytes),
            runs.Min(run => run.ManagedAllocatedBytes),
            runs.Max(run => run.ManagedAllocatedBytes),
            runs.Average(run => run.ManagedAllocatedBytesPerQuery),
            runs.Min(run => run.ManagedAllocatedBytesPerQuery),
            runs.Max(run => run.ManagedAllocatedBytesPerQuery));

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";
        return new RunToRunNoiseInfo(
            measured ? "measured" : "notMeasured",
            runs.Length,
            measured,
            "Across measured external HNSW base-plus-exact-delta runs for internal composite Search(query, results, workspace); warmup, setup, cache checks, matrix/truth loading, immutable HNSW build, update application, exact updated truth, result capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local external updated-HNSW noise inspection."
                : "Only one measured run executed, so run-to-run noise is unavailable and cannot be measured.",
            "Private local descriptive metadata only; not BenchmarkDotNet statistics, not confidence intervals, not baseline comparison math, not an acceptable-noise threshold and not a regression decision.",
            CreateMetricNoise(runs, "milliseconds", run => run.ElapsedMilliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "queriesPerSecond", run => run.Qps, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP50Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP95Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "milliseconds", run => run.LatencyP99Milliseconds, measured, unavailableReason),
            CreateMetricNoise(runs, "bytesPerSearchCall", run => run.ManagedAllocatedBytesPerQuery, measured, unavailableReason));
    }

    private static RunToRunMetricNoiseInfo CreateMetricNoise(
        SearchRunInfo[] runs,
        string unit,
        Func<SearchRunInfo, double> valueSelector,
        bool measured,
        string unavailableReason)
    {
        if (!measured)
        {
            return new RunToRunMetricNoiseInfo("notMeasured", unit, null, null, null, null, null, null, unavailableReason);
        }

        double[] values = runs.Select(valueSelector).ToArray();
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate(values);
        return new RunToRunMetricNoiseInfo(
            "measured",
            unit,
            FiniteOrNull(statistics.Mean),
            statistics.SampleStandardDeviation,
            statistics.CoefficientOfVariation,
            FiniteOrNull(statistics.Min),
            FiniteOrNull(statistics.Max),
            FiniteOrNull(statistics.Spread),
            "Computed across measured runs using the documented private descriptive-statistics formula.");
    }

    private static ExternalBenchmarkEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "external-fashion-mnist-hnsw-base-plus-exact-delta-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private external HNSW base-plus-exact-delta smoke output is not reviewed public evidence.",
            "No external mutable/update HNSW baseline-candidate policy is accepted.",
            "No external mutable/update HNSW regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "External Fashion-MNIST HNSW base-plus-exact-delta smoke evidence only; no external matrix or comparison claim applies.",
                "Cache checks, checksum validation, matrix/truth loading, immutable HNSW build, update application, exact updated truth generation, warmup, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                "Managed allocations are measured for the internal composite Search call boundary only; resident/process/GC/peak memory is explicitly not measured.",
                "Not eligible for public performance, recall, memory, allocation, mutable-HNSW, baseline, regression-gate, external-dataset matrix, hnswlib/FAISS comparison or concurrency claims."
            ]);

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
    }

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static float SquaredEuclideanDistance(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static bool DistanceMatches(float expected, float actual, int dimension)
    {
        if (!float.IsFinite(actual))
        {
            return false;
        }

        double relative =
            (8.0 * dimension / 16_777_216.0) *
            Math.Max(1.0, Math.Abs(expected));
        float tolerance = (float)Math.Max(2e-4, relative);
        return MathF.Abs(expected - actual) <= tolerance;
    }

    private static void ValidateOptions(FashionMnistExternalHnswBasePlusExactDeltaOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.CacheRoot))
        {
            throw new ArgumentException("Cache root must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(options));
        }

        if (options.QueryCount <= 0)
        {
            throw new ArgumentException("Query count must be positive.", nameof(options));
        }

        if (options.TopK <= 0)
        {
            throw new ArgumentException("top-k must be positive.", nameof(options));
        }

        if (options.BaseVectorCount <= 0)
        {
            throw new ArgumentException("base vector count must be positive.", nameof(options));
        }

        if (options.InsertedDeltaCount <= 0)
        {
            throw new ArgumentException("inserted delta count must be positive.", nameof(options));
        }

        if (options.DeletedBaseCount < 0 || options.DeletedBaseCount > options.BaseVectorCount)
        {
            throw new ArgumentException("deleted base count must be non-negative and no larger than base vector count.", nameof(options));
        }

        if (options.DeletedDeltaCount < 0 || options.DeletedDeltaCount > options.InsertedDeltaCount)
        {
            throw new ArgumentException("deleted delta count must be non-negative and no larger than inserted delta count.", nameof(options));
        }

        if (options.LiveVectorCount <= 0 || options.TopK > options.LiveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-update live vector count.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine))
        {
            throw new ArgumentException("external-fashion-mnist-hnsw-base-plus-exact-delta supports only SquaredEuclidean, InnerProduct and Cosine.", nameof(options));
        }

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.RepeatedDeleteAttempts > 0 && options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            throw new ArgumentException("repeated delete attempts require at least one committed delete.", nameof(options));
        }

        if (options.M is < 2 or > 64)
        {
            throw new ArgumentException("m must be in the range 2..64.", nameof(options));
        }

        if (options.EfConstruction < options.M || options.EfConstruction > 4096)
        {
            throw new ArgumentException("ef-construction must be at least m and no more than 4096.", nameof(options));
        }

        if (options.EfSearch < options.TopK || options.EfSearch > 4096)
        {
            throw new ArgumentException("ef-search must be at least top-k and no more than 4096.", nameof(options));
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidDataException(message);
        }
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static string CreateReportId(string? commit, FashionMnistExternalHnswBasePlusExactDeltaOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{FashionMnistExternalHnswBasePlusExactDeltaOptions.ScenarioName}-{commitPart}-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private sealed record BuildMeasurement(HnswIndex Index, double ElapsedMilliseconds, long ManagedAllocatedBytes);

    private sealed record MutationExecution(
        int InsertedCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        long GenerationAfterMutations,
        GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

    private sealed class MutableMutationStatusCounts
    {
        public int Committed { get; private set; }

        public int DuplicateId { get; private set; }

        public int UnknownId { get; private set; }

        public int AlreadyDeleted { get; private set; }

        public int ReadOnly { get; private set; }

        public int Unsupported { get; private set; }

        public void Add(VectorMutationStatus status)
        {
            switch (status)
            {
                case VectorMutationStatus.Committed:
                    Committed++;
                    break;
                case VectorMutationStatus.DuplicateId:
                    DuplicateId++;
                    break;
                case VectorMutationStatus.UnknownId:
                    UnknownId++;
                    break;
                case VectorMutationStatus.AlreadyDeleted:
                    AlreadyDeleted++;
                    break;
                case VectorMutationStatus.ReadOnly:
                    ReadOnly++;
                    break;
                case VectorMutationStatus.Unsupported:
                    Unsupported++;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(status), "Mutation status is not supported.");
            }
        }

        public GeneratedExactUpdateMutationStatusCountInfo ToInfo() =>
            new(Committed, DuplicateId, UnknownId, AlreadyDeleted, ReadOnly, Unsupported);
    }

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);
}
