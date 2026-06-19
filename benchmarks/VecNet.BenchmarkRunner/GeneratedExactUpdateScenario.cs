using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactUpdateScenario
{
    private const string TaskId = "VEC-061";
    private const string SchemaName = "VecNet.ExactUpdateBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static GeneratedExactUpdateBenchmarkReport Run(
        GeneratedExactUpdateOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);

        ExactFlatIndex index = BuildBaseIndex(options, dataset);
        long generationBeforeMutations = index.Generation;
        MutationExecution mutationExecution = ExecuteMutations(options, dataset, index, generationBeforeMutations);
        int liveVectorCount = mutationExecution.LastResult.VectorCount;

        ulong[] liveIds = BuildLiveIds(options);
        GeneratedUpdateFilterInputSet rawAllowlists = GenerateFilterInputs(
            options,
            options.AllowlistKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);
        GeneratedUpdateFilterInputSet candidateInputs = GenerateFilterInputs(
            options,
            options.CandidateSetKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);
        ExactFlatCandidateSet[] candidateSets = BuildCandidateSets(index, candidateInputs);

        TruthSet unfilteredTruth = GenerateLiveTruth(dataset, options, liveIds, candidateFilter: null);
        TruthSet rawAllowlistTruth = GenerateLiveTruth(dataset, options, liveIds, rawAllowlists.InputIds);
        TruthSet candidateSetTruth = GenerateLiveTruth(dataset, options, liveIds, candidateInputs.InputIds);

        WarmupSearch(options, dataset, index, rawAllowlists, candidateSets);
        SearchMeasurement unfilteredMeasurement = MeasureUnfilteredSearch(options, dataset, index);
        SearchMeasurement rawAllowlistMeasurement = MeasureRawAllowlistSearch(options, dataset, index, rawAllowlists);
        SearchMeasurement candidateSetMeasurement = MeasureCandidateSetSearch(options, dataset, index, candidateSets);

        GeneratedExactFilteredResultComparison unfilteredComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            unfilteredMeasurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison rawAllowlistComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            rawAllowlistTruth,
            rawAllowlistMeasurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison candidateSetComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            candidateSetTruth,
            candidateSetMeasurement.Results,
            options.TopK,
            options.Dimension,
            options.Metric);

        bool statusCountsMatched = mutationExecution.StatusCounts.Committed == mutationExecution.InsertedCount + mutationExecution.DeletedCount &&
            mutationExecution.StatusCounts.DuplicateId == options.DuplicateInsertAttempts &&
            mutationExecution.StatusCounts.UnknownId == options.UnknownDeleteAttempts &&
            mutationExecution.StatusCounts.AlreadyDeleted == options.RepeatedDeleteAttempts &&
            mutationExecution.StatusCounts.ReadOnly == 0 &&
            mutationExecution.StatusCounts.Unsupported == 0;
        bool generationMatched = mutationExecution.GenerationAfterMutations - generationBeforeMutations ==
            mutationExecution.InsertedCount + mutationExecution.DeletedCount;
        bool validationPassed = statusCountsMatched &&
            generationMatched &&
            unfilteredComparison.Integrity.Status == "passed" &&
            rawAllowlistComparison.Integrity.Status == "passed" &&
            candidateSetComparison.Integrity.Status == "passed";

        RepositoryInfo repository = RepositoryInfo.Create();
        int tombstoneCount = mutationExecution.LastResult.TombstoneCount;
        int physicalVectorCount = index.VectorCount;
        int deltaVectorCount = mutationExecution.LastResult.DeltaCount;
        double tombstoneRatio = physicalVectorCount == 0 ? 0 : (double)tombstoneCount / physicalVectorCount;
        GeneratedExactUpdateCountInfo counts = new(
            physicalVectorCount,
            liveVectorCount,
            options.BaseVectorCount,
            deltaVectorCount,
            tombstoneCount,
            tombstoneRatio,
            "physicalVectorCount",
            DeletedOrReservedIdCount: tombstoneCount,
            "For the accepted VEC-059 pre-checkpoint exact-flat update model, deleted/reserved IDs are represented by tombstones and remain reserved against reuse in the writable index instance.",
            "ExactFlatIndex.VectorCount is physical stored-row count; VectorMutationResult.VectorCount is live visible count.");

        GeneratedExactUpdateCandidateSetInfo candidateSetInfo = CreateCandidateSetInfo(candidateSets, candidateInputs);

        return new GeneratedExactUpdateBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactUpdateOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactUpdateOptions.ScenarioName, commandArguments.ToArray()),
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
                options.Metric.ToString(),
                options.Dimension,
                physicalVectorCount,
                options.QueryCount),
            new TruthInfo(
                "scalar-reference-generated-live-update",
                options.TopK,
                "live base plus committed delta minus tombstones, ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                GeneratedExactUpdateOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, base index build, mutation execution, live truth construction, allowlist/candidate input generation, candidate-set construction after mutations, warmup queries, final-run result capture/comparison and report writing are excluded from search timing"),
            new IndexInfo(
                "ExactUpdate",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                physicalVectorCount,
                "public ExactFlatIndex constructor; VEC-059 TryAdd/TryDelete delta/tombstone mutation API; post-mutation public unfiltered, raw allowlist and candidate-set search; no checkpoint/rebuild, persistence timing, HNSW update, stored labels, VectorData or public claims"),
            new GeneratedExactUpdateWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                dataset.SeedText,
                "base build via Add, committed TryAdd delta inserts, committed TryDelete base tombstones, duplicate/reserved TryAdd attempts, unknown TryDelete attempts, repeated TryDelete attempts",
                "Generated base IDs are 0..baseVectorCount-1; committed delta IDs are baseVectorCount..physicalVectorCount-1; deleted base IDs remain reserved; unknown IDs start above physicalVectorCount."),
            counts,
            new GeneratedExactUpdateMutationInfo(
                mutationExecution.InsertedCount,
                mutationExecution.DeletedCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                mutationExecution.InsertedCount + mutationExecution.DeletedCount,
                generationBeforeMutations,
                mutationExecution.GenerationAfterMutations,
                mutationExecution.GenerationAfterMutations - generationBeforeMutations,
                generationMatched,
                mutationExecution.StatusCounts),
            rawAllowlists.Info,
            candidateInputs.Info,
            candidateSetInfo,
            new GeneratedExactUpdateSearchesInfo(
                CreateOperationSearchInfo("unfilteredSearch", "public ExactFlatIndex.Search(query, results)", unfilteredMeasurement),
                CreateOperationSearchInfo("rawAllowlistSearch", "public ExactFlatIndex.Search(query, allowedIds, results, workspace)", rawAllowlistMeasurement),
                CreateOperationSearchInfo("candidateSetSearch", "public ExactFlatIndex.Search(query, candidateSet, results)", candidateSetMeasurement)),
            new GeneratedExactUpdateMeasurementInfo(
                CreateOperationMeasurementInfo(
                    "unfilteredSearch",
                    "public ExactFlatIndex.Search(query, results)",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, results) call using caller-owned SearchResult[]; setup, mutation execution, truth, warmup and report writing are excluded.",
                    unfilteredMeasurement),
                CreateOperationMeasurementInfo(
                    "rawAllowlistSearch",
                    "public ExactFlatIndex.Search(query, allowedIds, results, workspace)",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, allowedIds, results, workspace) call using caller-owned SearchResult[] and ExactFlatSearchFilterWorkspace; allowlist generation, mutation execution, truth, warmup and report writing are excluded.",
                    rawAllowlistMeasurement),
                CreateOperationMeasurementInfo(
                    "candidateSetSearch",
                    "public ExactFlatIndex.Search(query, candidateSet, results)",
                    "Measured with GC.GetAllocatedBytesForCurrentThread around each public ExactFlatIndex.Search(query, candidateSet, results) call using caller-owned SearchResult[] and prebuilt post-mutation ExactFlatCandidateSet instances; candidate-set construction, mutation execution, truth, warmup and report writing are excluded.",
                    candidateSetMeasurement),
                new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "millisecondsAndBytes",
                    "Mutation TryAdd/TryDelete latency and allocation are not measured in VEC-061; mutation execution is setup for post-mutation search measurement and is not mixed into search latency/QPS/allocation fields."),
                new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "millisecondsAndBytes",
                    "live-view Save cost is deferred to a later checkpoint/save-cost task"),
                new MeasurementStatusInfo(
                    "notMeasured",
                    "absent",
                    "bytes",
                    "Process working set, resident memory, private bytes, managed heap size, GC committed memory and peak memory are not measured in VEC-061."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup queries executed after mutations and after candidate-set construction for all measured search modes, and excluded from measured timing/allocation totals."
                        : "No warmup queries were requested."),
                "Generated data setup, base index build, mutation execution, live truth construction, allowlist/candidate input generation, post-mutation candidate-set construction, warmup, final-run result capture/comparison and report writing are excluded from all measured search samples."),
            new GeneratedExactUpdateMetricsInfo(
                CreateOperationMetrics(unfilteredComparison),
                CreateOperationMetrics(rawAllowlistComparison),
                CreateOperationMetrics(candidateSetComparison)),
            new GeneratedExactUpdateValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-exact-update-smoke",
                FiniteVectors: true,
                LiveTruthGenerated: true,
                MutationStatusCountsMatched: statusCountsMatched,
                GenerationMovementMatchedCommittedMutations: generationMatched,
                CandidateSetsConstructedAfterMutations: true,
                FinalRunUnfilteredComparedToTruth: true,
                FinalRunRawAllowlistComparedToTruth: true,
                FinalRunCandidateSetComparedToTruth: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateMemoryEstimateInfo(options, counts, candidateSetInfo),
            new GeneratedExactUpdateEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                "Private generated exact-update smoke evidence is not reviewed public evidence and has no public reporting policy.",
                "No exact-update baseline-candidate policy is accepted yet.",
                "No exact-update regression-gate policy, threshold, comparison artifact or hard gate is accepted yet."),
            [
                "Private generated exact-update smoke evidence only; not a public benchmark claim.",
                "Generated data only; no external dataset source, license, version or checksum applies.",
                "Physical count and live count are distinct and both are reported; ExactFlatIndex.VectorCount is physical stored-row count.",
                "Tombstones and deleted/reserved IDs can degrade scan cost and retained memory until checkpoint/rebuild exists.",
                "Candidate-set construction occurs after mutations but outside measured candidate-set search latency/allocation samples.",
                "Setup allocation, mutation allocation and candidate-set construction allocation are excluded from search allocation samples.",
                "Resident/process memory is not measured.",
                "Live-view Save cost is deferred in this first smoke foundation.",
                "No checkpoint/rebuild, HNSW update/durability, VectorData, SQL/database, compression, SSD/DiskANN, public docs or public claims are included."
            ]);
    }

    public static void Write(GeneratedExactUpdateBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactUpdateOptions options) =>
        new(
            options.Metric,
            options.Dimension,
            options.PhysicalVectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            options.Runs,
            options.WarmupQueries);

    private static ExactFlatIndex BuildBaseIndex(GeneratedExactUpdateOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static MutationExecution ExecuteMutations(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        long generationBeforeMutations)
    {
        var counts = new MutableMutationStatusCounts();
        VectorMutationResult lastResult = default;
        int inserted = 0;
        int deleted = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            ulong id = (ulong)(options.BaseVectorCount + i);
            lastResult = index.TryAdd(id, dataset.GetVector(options.BaseVectorCount + i));
            counts.Add(lastResult.Status);
            if (lastResult.Status == VectorMutationStatus.Committed)
            {
                inserted++;
            }
        }

        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            lastResult = index.TryDelete((ulong)i);
            counts.Add(lastResult.Status);
            if (lastResult.Status == VectorMutationStatus.Committed)
            {
                deleted++;
            }
        }

        for (int i = 0; i < options.DuplicateInsertAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(i % options.BaseVectorCount);
            lastResult = index.TryAdd(id, dataset.GetVector(options.BaseVectorCount + (i % options.InsertedDeltaCount)));
            counts.Add(lastResult.Status);
        }

        ulong firstUnknownId = (ulong)options.PhysicalVectorCount + 1UL;
        for (int i = 0; i < options.UnknownDeleteAttempts; i++)
        {
            lastResult = index.TryDelete(firstUnknownId + (ulong)i);
            counts.Add(lastResult.Status);
        }

        for (int i = 0; i < options.RepeatedDeleteAttempts; i++)
        {
            lastResult = index.TryDelete((ulong)(i % options.DeletedBaseCount));
            counts.Add(lastResult.Status);
        }

        if (counts.Total == 0)
        {
            throw new ArgumentException("At least one generated exact update mutation attempt is required.", nameof(options));
        }

        return new MutationExecution(
            inserted,
            deleted,
            lastResult.Generation,
            lastResult,
            counts.ToInfo());
    }

    private static ulong[] BuildLiveIds(GeneratedExactUpdateOptions options)
    {
        int liveCount = checked(options.BaseVectorCount - options.DeletedBaseCount + options.InsertedDeltaCount);
        var ids = new ulong[liveCount];
        int write = 0;
        for (int row = options.DeletedBaseCount; row < options.BaseVectorCount; row++)
        {
            ids[write++] = (ulong)row;
        }

        for (int row = options.BaseVectorCount; row < options.PhysicalVectorCount; row++)
        {
            ids[write++] = (ulong)row;
        }

        return ids;
    }

    private static GeneratedUpdateFilterInputSet GenerateFilterInputs(
        GeneratedExactUpdateOptions options,
        string kind,
        ulong[] liveIds,
        int duplicateIdsPerQuery,
        int unknownIdsPerQuery)
    {
        int knownPerQuery = GetKnownCount(kind, liveIds.Length, options.TopK);
        int inputLength = checked(knownPerQuery + duplicateIdsPerQuery + unknownIdsPerQuery);
        var inputs = new ulong[options.QueryCount][];

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            var input = new ulong[inputLength];
            int write = 0;
            int start = liveIds.Length == 0
                ? 0
                : (int)(((ulong)options.Seed + ((ulong)queryRow * 2_654_435_761UL)) % (ulong)liveIds.Length);
            for (int i = 0; i < knownPerQuery; i++)
            {
                input[write++] = liveIds[(start + i) % liveIds.Length];
            }

            for (int i = 0; i < duplicateIdsPerQuery; i++)
            {
                input[write++] = knownPerQuery == 0
                    ? (ulong)options.PhysicalVectorCount + 1UL
                    : input[i % knownPerQuery];
            }

            ulong firstUnknown = (ulong)options.PhysicalVectorCount + 1UL + ((ulong)queryRow * (ulong)Math.Max(1, unknownIdsPerQuery));
            for (int i = 0; i < unknownIdsPerQuery; i++)
            {
                input[write++] = firstUnknown + (ulong)i;
            }

            inputs[queryRow] = input;
        }

        var info = new GeneratedExactUpdateFilterInputInfo(
            kind,
            GetSelectivityTarget(kind),
            liveIds.Length == 0 ? 0 : (double)knownPerQuery / liveIds.Length,
            knownPerQuery,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            inputLength,
            checked(knownPerQuery * options.QueryCount),
            checked(duplicateIdsPerQuery * options.QueryCount),
            checked(unknownIdsPerQuery * options.QueryCount),
            "deterministic query-rotated post-mutation live IDs followed by requested duplicate known IDs and requested unknown IDs",
            "knownCount = all: liveVectorCount; broad: ceiling(liveVectorCount * 0.50); selective: ceiling(liveVectorCount * 0.10); very-selective: min(liveVectorCount, topK - 1); empty: 0. For query q, known live IDs start at (seed + q * 2654435761) mod liveVectorCount and advance by one modulo the post-mutation live ID list.",
            "Duplicate input IDs repeat earlier known live IDs when knownCount is greater than zero; empty inputs duplicate unknown IDs so no indexed row becomes visible.",
            "Unknown input IDs are greater than physicalVectorCount and are deliberately ignored by raw allowlist search and candidate-set construction.",
            "Inputs are generated against the post-mutation live view, so tombstoned IDs are excluded and committed delta IDs are eligible.");

        return new GeneratedUpdateFilterInputSet(inputs, info);
    }

    private static ExactFlatCandidateSet[] BuildCandidateSets(
        ExactFlatIndex index,
        GeneratedUpdateFilterInputSet candidateInputs)
    {
        var candidateSets = new ExactFlatCandidateSet[candidateInputs.InputIds.Length];
        for (int queryRow = 0; queryRow < candidateInputs.InputIds.Length; queryRow++)
        {
            candidateSets[queryRow] = index.CreateCandidateSet(candidateInputs.InputIds[queryRow]);
        }

        return candidateSets;
    }

    private static GeneratedExactUpdateCandidateSetInfo CreateCandidateSetInfo(
        ExactFlatCandidateSet[] candidateSets,
        GeneratedUpdateFilterInputSet candidateInputs)
    {
        int minCount = candidateSets.Length == 0 ? 0 : candidateSets.Min(item => item.Count);
        int maxCount = candidateSets.Length == 0 ? 0 : candidateSets.Max(item => item.Count);
        double meanCount = candidateSets.Length == 0 ? 0 : candidateSets.Average(item => item.Count);
        int totalCount = candidateSets.Sum(item => item.Count);

        return new GeneratedExactUpdateCandidateSetInfo(
            "constructedAfterMutationsOutsideMeasuredSearch",
            "public ExactFlatIndex.CreateCandidateSet(allowedIds)",
            "Candidate-set construction is completed after all mutation attempts and before warmup/measured search; it is excluded from latency samples and QPS.",
            "Candidate-set construction may allocate and is excluded from measured search allocation samples.",
            ConstructedAfterMutations: true,
            ConstructedBeforeWarmupAndMeasuredSearch: true,
            candidateSets.Length,
            candidateInputs.Info.KnownLiveIdCountPerQuery,
            minCount,
            maxCount,
            meanCount,
            totalCount,
            "Candidate sets are opaque, exact-flat index-bound, generation-bound runtime objects built from the current post-mutation generation.",
            "Stale candidate-set failure cost is out of scope; measured candidate-set search uses only sets freshly built after the mutation workload.",
            "Candidate sets are transient setup artifacts, not persisted filters or public row-ordinal sidecars.");
    }

    private static TruthSet GenerateLiveTruth(
        GeneratedDataset dataset,
        GeneratedExactUpdateOptions options,
        ulong[] liveIds,
        ulong[][]? candidateFilter)
    {
        var results = new TruthItem[dataset.QueryCount][];
        double[]? vectorMagnitudes = options.Metric == VectorMetric.Cosine ? CalculateVectorMagnitudes(dataset) : null;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            ulong[] visibleIds = candidateFilter is null
                ? liveIds
                : IntersectLiveIds(liveIds, candidateFilter[queryRow], options.PhysicalVectorCount);
            if (visibleIds.Length == 0)
            {
                results[queryRow] = [];
                continue;
            }

            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            double queryMagnitude = options.Metric == VectorMetric.Cosine ? CalculateMagnitude(query) : 0;
            var candidates = new TruthItem[visibleIds.Length];
            for (int i = 0; i < visibleIds.Length; i++)
            {
                int row = checked((int)visibleIds[i]);
                float distance = CalculateDistance(
                    query,
                    dataset.GetVector(row),
                    options.Metric,
                    queryMagnitude,
                    vectorMagnitudes is null ? 0 : vectorMagnitudes[row]);
                candidates[i] = new TruthItem(visibleIds[i], distance);
            }

            Array.Sort(candidates, CompareTruthItems);
            int resultCount = Math.Min(options.TopK, candidates.Length);
            var top = new TruthItem[resultCount];
            Array.Copy(candidates, top, resultCount);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    private static ulong[] IntersectLiveIds(ulong[] liveIds, ulong[] inputIds, int physicalVectorCount)
    {
        if (inputIds.Length == 0 || liveIds.Length == 0)
        {
            return [];
        }

        var live = new HashSet<ulong>(liveIds);
        var selected = new SortedSet<ulong>();
        foreach (ulong id in inputIds)
        {
            if (id < (ulong)physicalVectorCount && live.Contains(id))
            {
                selected.Add(id);
            }
        }

        return selected.ToArray();
    }

    private static void WarmupSearch(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        GeneratedUpdateFilterInputSet rawAllowlists,
        ExactFlatCandidateSet[] candidateSets)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int queryRow = i % dataset.QueryCount;
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            index.Search(query, results);
            index.Search(query, rawAllowlists.InputIds[queryRow], results, workspace);
            index.Search(query, candidateSets[queryRow], results);
        }
    }

    private static SearchMeasurement MeasureUnfilteredSearch(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, captureResults, (queryRow, results) =>
                index.Search(dataset.GetQuery(queryRow), results));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SearchMeasurement MeasureRawAllowlistSearch(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        GeneratedUpdateFilterInputSet rawAllowlists)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, captureResults, (queryRow, results) =>
                index.Search(dataset.GetQuery(queryRow), rawAllowlists.InputIds[queryRow], results, workspace));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SearchMeasurement MeasureCandidateSetSearch(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ExactFlatCandidateSet[] candidateSets)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? capturedResults = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            bool captureResults = runIndex == options.Runs - 1;
            SingleRunMeasurement run = MeasureSingleRun(options, dataset, captureResults, (queryRow, results) =>
                index.Search(dataset.GetQuery(queryRow), candidateSets[queryRow], results));
            runs[runIndex] = run.Summary with { RunNumber = runIndex + 1 };
            if (captureResults)
            {
                capturedResults = run.Results;
            }
        }

        return new SearchMeasurement(capturedResults ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureSingleRun(
        GeneratedExactUpdateOptions options,
        GeneratedDataset dataset,
        bool captureResults,
        SearchOperation operation)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < dataset.QueryCount; queryRow++)
        {
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = operation(queryRow, results);
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

    private static GeneratedExactUpdateOperationSearchInfo CreateOperationSearchInfo(
        string name,
        string timedOperation,
        SearchMeasurement measurement) =>
        new(
            name,
            timedOperation,
            new SearchInfo(
                measurement.Aggregate.MeasuredQueryCountPerRun,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate));

    private static GeneratedExactUpdateOperationMeasurementInfo CreateOperationMeasurementInfo(
        string operationName,
        string timedOperation,
        string allocationReason,
        SearchMeasurement measurement) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredQuery",
                timedOperation,
                "generated data setup, base index build, mutation execution, live truth construction, allowlist/candidate input generation, post-mutation candidate-set construction, warmup queries, final-run result capture/comparison and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Top-level search latency percentile fields and search aggregate mean latency percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                measurement.Aggregate.MeanManagedAllocatedBytesPerQuery.ToString(CultureInfo.InvariantCulture),
                "bytesPerSearchCall",
                allocationReason),
            new RepeatedRunInfo(
                measurement.Runs.Length > 1 ? "measured" : "singleRun",
                measurement.Runs.Length,
                measurement.Runs.Length > 1,
                measurement.Runs.Length > 1
                    ? $"Multiple measured generated exact update {operationName} runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                    : $"Only one measured generated exact update {operationName} run executed, so cross-run variance/noise is not measured."),
            CreateRunToRunNoise(operationName, timedOperation, measurement.Runs));

    private static RunToRunNoiseInfo CreateRunToRunNoise(
        string operationName,
        string timedOperation,
        SearchRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        string unavailableReason = "Only one measured run exists; this field does not establish run-to-run variation.";
        return new RunToRunNoiseInfo(
            measured ? "measured" : "notMeasured",
            runs.Length,
            measured,
            $"Across measured generated exact update {operationName} runs for {timedOperation}; warmup, setup, mutation execution, truth, candidate-set construction, result capture/comparison and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local exact-update noise inspection."
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

    private static GeneratedExactUpdateOperationMetricsInfo CreateOperationMetrics(
        GeneratedExactFilteredResultComparison comparison) =>
        new(
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.Integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
            comparison.Integrity.DistanceMismatchCount,
            comparison.Integrity.MissingResultCount,
            comparison.Integrity.ExtraResultCount,
            comparison.Integrity);

    private static GeneratedExactUpdateMemoryEstimateInfo CreateMemoryEstimateInfo(
        GeneratedExactUpdateOptions options,
        GeneratedExactUpdateCountInfo counts,
        GeneratedExactUpdateCandidateSetInfo candidateSet) =>
        new(
            "estimatedPayloadLowerBounds",
            "Conservative payload lower-bound estimates only; not managed object overhead, array slack capacity, dictionary/hash-set capacity, GC heap size, working set, private bytes or peak memory.",
            checked((long)counts.PhysicalVectorCount * sizeof(ulong)),
            checked((long)counts.PhysicalVectorCount * options.Dimension * sizeof(float)),
            checked((long)counts.LiveVectorCount * options.Dimension * sizeof(float)),
            checked((long)candidateSet.TotalCandidateCount * sizeof(int)),
            new MeasurementStatusInfo(
                "notAvailable",
                "absent",
                "bytes",
                "Tombstone/deleted-reservation HashSet<ulong> retained capacity is not exposed; no defensible retained-memory byte estimate is reported."),
            new MeasurementStatusInfo(
                "notMeasured",
                "absent",
                "bytes",
                "Process working set, private bytes, resident memory, managed heap size, GC committed memory and peak memory are not measured."),
            "These estimates do not establish resident/process memory, managed heap size, object overhead, collection capacity, allocation, peak memory or preview-readiness evidence.");

    private static GeneratedExactUpdateEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-exact-update-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated exact-update smoke evidence is not reviewed public evidence and has no public reporting policy.",
            "No exact-update baseline-candidate policy is accepted yet.",
            "No exact-update regression-gate policy, threshold, comparison artifact or hard gate is accepted yet.",
            [
                "Generated exact-update smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Post-mutation unfiltered, raw allowlist and candidate-set search modes are measured separately.",
                "Mutation execution, candidate-set construction and live-view Save cost are not included in measured search latency/QPS/allocation samples.",
                "Latency percentiles are nearest-rank per-run query latency samples aggregated as per-run means, not BenchmarkDotNet statistics.",
                "Managed allocations are measured for public search calls only; resident/process memory is explicitly not measured.",
                "Not a public claim, baseline candidate, regression gate, preview-readiness result, Linux x64 validation or BenchmarkDotNet-grade evidence."
            ]);

    private static int GetKnownCount(string kind, int liveVectorCount, int topK) =>
        kind switch
        {
            "all" => liveVectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.50), 1, liveVectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.10), 1, liveVectorCount),
            "very-selective" => Math.Min(liveVectorCount, topK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact update selectivity kind.", nameof(kind))
        };

    private static string GetSelectivityTarget(string kind) =>
        kind switch
        {
            "all" => "100% of live visible rows",
            "broad" => "approximately 50% of live visible rows",
            "selective" => "approximately 10% of live visible rows",
            "very-selective" => "fewer than top-k live visible rows",
            "empty" => "0% of live visible rows",
            _ => "unknown"
        };

    private static double[] CalculateVectorMagnitudes(GeneratedDataset dataset)
    {
        var magnitudes = new double[dataset.VectorCount];
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            magnitudes[row] = CalculateMagnitude(dataset.GetVector(row));
        }

        return magnitudes;
    }

    private static double CalculateMagnitude(ReadOnlySpan<float> values)
    {
        double squaredMagnitude = 0;
        foreach (float value in values)
        {
            squaredMagnitude += (double)value * value;
        }

        return Math.Sqrt(squaredMagnitude);
    }

    private static float CalculateDistance(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        VectorMetric metric,
        double queryMagnitude,
        double vectorMagnitude) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclidean(query, vector),
            VectorMetric.InnerProduct => InnerProduct(query, vector),
            VectorMetric.Cosine => Cosine(query, vector, queryMagnitude, vectorMagnitude),
            _ => throw new ArgumentOutOfRangeException(nameof(metric), "Metric is not supported.")
        };

    private static float SquaredEuclidean(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double sum = 0;
        for (int i = 0; i < query.Length; i++)
        {
            double difference = query[i] - vector[i];
            sum += difference * difference;
        }

        return (float)sum;
    }

    private static float InnerProduct(ReadOnlySpan<float> query, ReadOnlySpan<float> vector)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (double)query[i] * vector[i];
        }

        return (float)-dotProduct;
    }

    private static float Cosine(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> vector,
        double queryMagnitude,
        double vectorMagnitude)
    {
        double dotProduct = 0;
        for (int i = 0; i < query.Length; i++)
        {
            dotProduct += (query[i] / queryMagnitude) * (vector[i] / vectorMagnitude);
        }

        return (float)(1 - dotProduct);
    }

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static void ValidateOptions(GeneratedExactUpdateOptions options)
    {
        if (options.TopK > options.BaseVectorCount + options.InsertedDeltaCount - options.DeletedBaseCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-mutation live vector count.", nameof(options));
        }

        if (options.Runs <= 0 || options.Runs > 5)
        {
            throw new ArgumentException("runs must be in the range 1..5.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
        }

        if (options.InsertedDeltaCount <= 0)
        {
            throw new ArgumentException("inserted delta count must be positive.", nameof(options));
        }

        if (options.DeletedBaseCount <= 0 || options.DeletedBaseCount > options.BaseVectorCount)
        {
            throw new ArgumentException("deleted base count must be positive and no larger than base vector count.", nameof(options));
        }

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.RepeatedDeleteAttempts > 0 && options.DeletedBaseCount == 0)
        {
            throw new ArgumentException("repeated delete attempts require at least one committed delete.", nameof(options));
        }

        if (options.DuplicateIdsPerQuery < 0 || options.UnknownIdsPerQuery < 0)
        {
            throw new ArgumentException("input duplicate and unknown ID counts must be non-negative.", nameof(options));
        }

        if ((options.AllowlistKind == "very-selective" || options.CandidateSetKind == "very-selective") && options.TopK <= 1)
        {
            throw new ArgumentException("very-selective update filters require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactUpdateOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactUpdateOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}d-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-{options.Seed:X8}");
    }

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private delegate int SearchOperation(int queryRow, SearchResult[] results);

    private sealed record GeneratedUpdateFilterInputSet(ulong[][] InputIds, GeneratedExactUpdateFilterInputInfo Info);

    private sealed record MutationExecution(
        int InsertedCount,
        int DeletedCount,
        long GenerationAfterMutations,
        VectorMutationResult LastResult,
        GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

    private sealed class MutableMutationStatusCounts
    {
        public int Committed { get; private set; }

        public int DuplicateId { get; private set; }

        public int UnknownId { get; private set; }

        public int AlreadyDeleted { get; private set; }

        public int ReadOnly { get; private set; }

        public int Unsupported { get; private set; }

        public int Total => Committed + DuplicateId + UnknownId + AlreadyDeleted + ReadOnly + Unsupported;

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
