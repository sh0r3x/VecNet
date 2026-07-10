using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class HnswBasePlusExactDeltaGeneratedScenario
{
    private const string TaskId = "VEC-124";
    private const string SchemaName = "VecNet.HnswBasePlusExactDeltaBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static HnswBasePlusExactDeltaBenchmarkReport Run(
        HnswBasePlusExactDeltaGeneratedOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);

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
            options.Dimension,
            VectorMetric.SquaredEuclidean);
        HnswBasePlusExactDeltaReturnedResultIntegrityInfo returnedIntegrity =
            ValidateReturnedResults(dataset, measurement.Results, options.TopK, liveIds);
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

        return new HnswBasePlusExactDeltaBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswBasePlusExactDeltaGeneratedOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswBasePlusExactDeltaGeneratedOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new DatasetInfo(
                GeneratedDataset.Kind,
                "generated-no-external-source",
                GeneratedDataset.Distribution,
                dataset.SeedText,
                VectorMetric.SquaredEuclidean.ToString(),
                options.Dimension,
                options.PhysicalVectorCount,
                options.QueryCount),
            new TruthInfo(
                "scalar-reference-generated-live-hnsw-base-plus-exact-delta",
                truth.Depth,
                "post-update live base plus live delta minus tombstones, ordered by ascending scalar-reference squared-L2 distance and ascending external ID"),
            new ScenarioInfo(
                HnswBasePlusExactDeltaGeneratedOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, immutable HNSW base build, update application, exact updated truth construction, warmup queries, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "InternalHnswBasePlusExactDelta",
                nameof(HnswBasePlusExactDeltaIndex),
                VectorMetric.SquaredEuclidean.ToString(),
                options.Dimension,
                counts.LiveVectorCount,
                "internal HnswBasePlusExactDeltaIndex over immutable HnswIndex base, exact in-memory delta and tombstone overlay; no public mutable HNSW API, persistence, checkpoint/rebuild, filtering, matrix preset or external dataset mode"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated base vector row order, external base ids 0..baseVectorCount-1; delta ids continue from baseVectorCount",
                "SquaredEuclidean only"),
            new HnswBuildInfo(
                "measured",
                build.ElapsedMilliseconds,
                new MeasurementStatusInfo(
                    "measured",
                    build.ManagedAllocatedBytes.ToString(CultureInfo.InvariantCulture),
                    "bytes",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around immutable HnswIndex construction and Add calls for generated base vectors only; generated data setup, composite construction, mutation application and exact updated truth generation are excluded."),
                options.BaseVectorCount,
                options.Dimension,
                "internal HnswIndex construction and generated base-vector Add calls",
                "generated data setup, composite construction, update application, exact updated truth generation, warmup, measured composite search, result comparison and report writing"),
            new HnswBasePlusExactDeltaWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                dataset.SeedText,
                "build immutable HNSW base, committed exact-delta inserts, committed base tombstone deletes, configured delta tombstone deletes, duplicate/reserved insert attempts, unknown delete attempts, repeated delete attempts",
                "Base IDs are 0..baseVectorCount-1; committed delta IDs are baseVectorCount..physicalVectorCount-1; deleted IDs remain reserved inside the writable composite instance; unknown IDs start above physicalVectorCount."),
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
            new MeasurementInfo(
                Latency: new LatencyMeasurementInfo(
                    "measured",
                    "milliseconds",
                    "perMeasuredQuery",
                    "internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace)",
                    "generated data setup, immutable HNSW base build, composite construction, update application, exact updated truth construction, warmup queries, final-run result capture/comparison and report writing",
                    "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                    "Top-level search latency percentile fields and search aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                    "Raw per-query latency samples are not emitted in report JSON."),
                ManagedAllocations: new MeasurementStatusInfo(
                    "measured",
                    measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                    "bytesPerSearchCall",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each internal HnswBasePlusExactDeltaIndex.Search(query, results, workspace) call using caller-owned SearchResult[] and HnswBasePlusExactDeltaSearchWorkspace; setup, HNSW build, mutation application, exact updated truth, warmup, result capture/comparison and report writing are excluded."),
                Memory: new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, private bytes, managed heap size and peak memory are not measured in VEC-124."),
                RepeatedRuns: new RepeatedRunInfo(
                    options.Runs > 1 ? "measured" : "singleRun",
                    options.Runs,
                    options.Runs > 1,
                    options.Runs > 1
                        ? "Multiple measured generated HNSW base-plus-exact-delta search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                        : "Only one measured run executed, so cross-run variance/noise is not measured."),
                RunToRunNoise: CreateRunToRunNoise(measurement.Runs),
                Warmup: new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed after mutation application using caller-owned results/workspace and excluded from measured timing and allocation totals."
                        : "No warmup queries were requested.")),
            new HnswBasePlusExactDeltaMetricsInfo(
                comparison.RecallAtK,
                comparison.OrderedAgreement,
                returnedIntegrity.DistanceMismatchCount == 0 ? "passed" : "failed",
                returnedIntegrity.DistanceMismatchCount,
                comparison.MissingResultCount,
                extraResultCount,
                returnedIntegrity,
                "set recall@k = returned live ids intersect exact updated top-k live ids divided by min(k, post-update live vector count), summed across measured queries",
                "Every returned composite result is checked for finite distance, no duplicate ID within its query, generated live ID membership, no tombstoned ID, and squared-L2 distance matching recomputation for that returned ID/query within the accepted D-026 tolerance. HNSW base search is approximate and recall/order are recorded, not required."),
            underfill,
            new HnswBasePlusExactDeltaValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-hnsw-base-plus-exact-delta-smoke",
                FiniteVectors: true,
                LiveTruthGenerated: true,
                HnswBaseBuilt: true,
                MutationsApplied: true,
                MutationStatusCountsMatched: statusCountsMatched,
                GenerationMovementMatchedCommittedMutations: generationMatched,
                FinalRunComparedToTruth: true,
                ReturnedResultsAreLiveAndNotTombstoned: returnedIntegrity.Status == "passed",
                AllowsApproximateRecallBelowOne: true,
                AllowsUnderfill: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            new HnswEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private generated HNSW base-plus-exact-delta smoke output is not reviewed public evidence and has no public reporting policy.",
                "No generated mutable/update HNSW baseline-candidate policy is accepted.",
                "No generated mutable/update HNSW regression-gate policy, threshold, comparison artifact or hard gate is accepted."),
            [
                "Private generated HNSW base-plus-exact-delta smoke evidence only; not a public benchmark claim.",
                "Generated finite squared-L2 data only; no external dataset source, license, version or checksum applies.",
                "This report exercises an internal composite type and does not add or imply a public mutable/update HNSW API.",
                "Durable mutable overlay persistence, checkpoint/rebuild, direct graph mutation, filtering, matrix presets and external datasets are out of scope.",
                "Latency/QPS/allocation time only internal composite Search calls with caller-owned result buffers and workspace.",
                "Immutable HNSW base build, update application and exact updated truth generation are setup work and excluded from measured search timing.",
                "Approximate recall below 1.0 and underfill are allowed and recorded.",
                "Public claims, baseline candidates, comparison artifacts and regression gates are not created by this report."
            ]);
    }

    public static void Write(HnswBasePlusExactDeltaBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    public static HnswBasePlusExactDeltaReturnedResultIntegrityInfo ValidateReturnedResults(
        GeneratedDataset dataset,
        SearchResult[][] actual,
        int topK,
        IReadOnlyCollection<ulong> liveIds)
    {
        var live = new HashSet<ulong>(liveIds);
        int checkedResultCount = 0;
        int queryCountMismatchCount = actual.Length == dataset.QueryCount ? 0 : 1;
        int resultCountViolationCount = 0;
        int nonFiniteDistanceCount = 0;
        int duplicateIdCount = 0;
        int unknownIdCount = 0;
        int tombstonedIdCount = 0;
        int distanceMismatchCount = 0;
        int queryCount = Math.Min(dataset.QueryCount, actual.Length);
        int maxExpectedResults = Math.Min(topK, live.Count);

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

                if (result.Id >= (ulong)dataset.VectorCount)
                {
                    unknownIdCount++;
                    continue;
                }

                if (!live.Contains(result.Id))
                {
                    tombstonedIdCount++;
                    continue;
                }

                float expectedDistance = SquaredEuclideanDistance(
                    dataset.GetQuery(queryRow),
                    dataset.GetVector(checked((int)result.Id)));
                if (!DistanceMatches(expectedDistance, result.Distance, dataset.Dimension))
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
            "For every returned composite result: distance must be finite; IDs must be unique within a query; ID must be one of the post-update live generated IDs; tombstoned IDs must not be returned; and reported distance must match recomputed squared-L2 for that query and returned ID within the accepted D-026 tolerance.",
            passed
                ? "All returned composite results are live, not tombstoned, well formed and distance-integrity checked."
                : "One or more returned composite results failed live-ID, tombstone, well-formedness or distance-integrity checks.");
    }

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswBasePlusExactDeltaGeneratedOptions options) =>
        new(
            VectorMetric.SquaredEuclidean,
            options.Dimension,
            options.PhysicalVectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            options.Runs,
            options.WarmupQueries);

    private static BuildMeasurement BuildBaseIndex(HnswBasePlusExactDeltaGeneratedOptions options, GeneratedDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var index = new HnswIndex(options.Dimension, VectorMetric.SquaredEuclidean, hnswOptions);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        long elapsed = Stopwatch.GetTimestamp() - start;
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocationStart;
        return new BuildMeasurement(index, (double)elapsed / Stopwatch.Frequency * 1000, allocatedBytes);
    }

    private static MutationExecution ExecuteMutations(
        HnswBasePlusExactDeltaGeneratedOptions options,
        GeneratedDataset dataset,
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
            lastResult = composite.TryAdd(id, dataset.GetVector(options.BaseVectorCount + i));
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
            lastResult = composite.TryAdd(id, dataset.GetVector(options.BaseVectorCount + (i % options.InsertedDeltaCount)));
            counts.Add(lastResult.Status);
        }

        ulong firstUnknownId = (ulong)options.PhysicalVectorCount + 1UL;
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

    private static ulong[] BuildLiveIds(HnswBasePlusExactDeltaGeneratedOptions options)
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
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaGeneratedOptions options,
        ulong[] liveIds)
    {
        var results = new TruthItem[dataset.QueryCount][];
        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            var candidates = new TruthItem[liveIds.Length];
            for (int i = 0; i < liveIds.Length; i++)
            {
                ulong id = liveIds[i];
                candidates[i] = new TruthItem(id, SquaredEuclideanDistance(query, dataset.GetVector(checked((int)id))));
            }

            Array.Sort(candidates, CompareTruthItems);
            var top = new TruthItem[options.TopK];
            Array.Copy(candidates, top, options.TopK);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    private static void WarmupSearch(
        HnswBasePlusExactDeltaGeneratedOptions options,
        GeneratedDataset dataset,
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
            composite.Search(dataset.GetQuery(i % dataset.QueryCount), results, workspace);
        }
    }

    private static SearchMeasurement MeasureSearch(
        HnswBasePlusExactDeltaGeneratedOptions options,
        GeneratedDataset dataset,
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
        HnswBasePlusExactDeltaGeneratedOptions options,
        GeneratedDataset dataset,
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
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
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

    private static HnswBasePlusExactDeltaSearchWorkspace CreateWorkspace(HnswBasePlusExactDeltaGeneratedOptions options) =>
        new(
            options.BaseVectorCount,
            options.EfSearch,
            Math.Min(options.BaseVectorCount, options.EfSearch),
            options.TopK);

    private static HnswBasePlusExactDeltaCountInfo CreateCountInfo(
        HnswBasePlusExactDeltaGeneratedOptions options,
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
        HnswBasePlusExactDeltaGeneratedOptions options,
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
            "Across measured generated HNSW base-plus-exact-delta runs for internal composite Search(query, results, workspace); warmup, setup, immutable HNSW build, update application, exact updated truth, result capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local updated-HNSW noise inspection."
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

    private static HnswEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-hnsw-base-plus-exact-delta-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW base-plus-exact-delta smoke output is not reviewed public evidence.",
            "No generated mutable/update HNSW baseline-candidate policy is accepted.",
            "No generated mutable/update HNSW regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated squared-L2 HNSW base-plus-exact-delta smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Immutable HNSW base build, update application, exact updated truth generation, warmup queries, final-run result capture/comparison and report writing are excluded from measured search latency and QPS.",
                "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                "Managed allocations are measured for the internal composite Search call boundary only; resident/process memory is explicitly not measured.",
                "Not eligible for public performance, recall, memory, allocation, mutable-HNSW, baseline, regression-gate, external-dataset, matrix or concurrency claims."
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

    private static void ValidateOptions(HnswBasePlusExactDeltaGeneratedOptions options)
    {
        if (options.Metric != VectorMetric.SquaredEuclidean)
        {
            throw new ArgumentException("generated-hnsw-base-plus-exact-delta supports only SquaredEuclidean.", nameof(options));
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

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.RepeatedDeleteAttempts > 0 && options.DeletedBaseCount + options.DeletedDeltaCount == 0)
        {
            throw new ArgumentException("repeated delete attempts require at least one committed delete.", nameof(options));
        }

        if (options.EfSearch < options.TopK)
        {
            throw new ArgumentException("ef-search must be greater than or equal to top-k.", nameof(options));
        }

        if (options.M is < 2 or > 64)
        {
            throw new ArgumentException("m must be in the range 2..64.", nameof(options));
        }

        if (options.EfConstruction < options.M || options.EfConstruction > 4096)
        {
            throw new ArgumentException("ef-construction must be at least m and no more than 4096.", nameof(options));
        }

        if (options.EfSearch > 4096)
        {
            throw new ArgumentException("ef-search must be in the range 1..4096.", nameof(options));
        }
    }

    private static void ValidateFinite(GeneratedDataset dataset)
    {
        foreach (float value in dataset.Vectors)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Generated vector data must be finite.");
            }
        }

        foreach (float value in dataset.Queries)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException("Generated query data must be finite.");
            }
        }
    }

    private static string CreateReportId(string? commit, HnswBasePlusExactDeltaGeneratedOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswBasePlusExactDeltaGeneratedOptions.ScenarioName}-{commitPart}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

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
