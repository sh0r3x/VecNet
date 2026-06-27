using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactPracticalUpdateScenario
{
    private const string TaskId = "VEC-079";
    private const string SchemaName = "VecNet.ExactPracticalUpdateModeReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "exact-flat.manifest.json";
    private const string IdsFileName = "exact-flat.ids.u64";
    private const string VectorsFileName = "exact-flat.vectors.f32";

    public static GeneratedExactPracticalUpdateBenchmarkReport Run(
        GeneratedExactPracticalUpdateOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        ulong[] liveIds = BuildLiveIds(options);
        PracticalUpdateFilterInputSet rawAllowlists = GenerateFilterInputs(
            options,
            options.AllowlistKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);
        PracticalUpdateFilterInputSet candidateInputs = GenerateFilterInputs(
            options,
            options.CandidateSetKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);

        TruthSet unfilteredTruth = GenerateLiveTruth(dataset, options, liveIds, candidateFilter: null);
        TruthSet rawAllowlistTruth = GenerateLiveTruth(dataset, options, liveIds, rawAllowlists.InputIds);
        TruthSet candidateSetTruth = GenerateLiveTruth(dataset, options, liveIds, candidateInputs.InputIds);

        var mutationRuns = new GeneratedExactPracticalUpdateOperationRunInfo[options.Runs];
        var searchRuns = new GeneratedExactPracticalUpdateOperationRunInfo[options.Runs];
        var checkpointRuns = new GeneratedExactPracticalUpdateOperationRunInfo[options.Runs];
        var openRuns = new GeneratedExactPracticalUpdateOperationRunInfo[options.Runs];
        PracticalUpdateRunCapture? finalCapture = null;

        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            ExactFlatIndex index = BuildBaseIndex(options, dataset);
            ExactFlatCandidateSet staleCandidateSet = index.CreateCandidateSet(BuildStaleCandidateInput(options));
            long generationBeforeMutation = index.Generation;

            long mutationStart = Stopwatch.GetTimestamp();
            MutationExecution mutationExecution = ExecuteMutations(options, dataset, index, generationBeforeMutation);
            long mutationElapsed = Stopwatch.GetTimestamp() - mutationStart;
            mutationRuns[runIndex] = new GeneratedExactPracticalUpdateOperationRunInfo(
                runIndex + 1,
                TicksToMilliseconds(mutationElapsed),
                "completed",
                generationBeforeMutation,
                index.Generation,
                CountMutationAttempts(mutationExecution.StatusCounts),
                "Mutation batch timing includes deterministic per-attempt ID/vector lookup and status-count bookkeeping around public TryAdd/TryDelete calls; setup, truth, validation searches, checkpoint/open and report writing are excluded.");

            WarmupSearch(options, dataset, index);
            long searchGeneration = index.Generation;
            long searchElapsed = MeasureUnfilteredSearch(index, dataset, options, out SearchResult[][] searchResults);
            searchRuns[runIndex] = new GeneratedExactPracticalUpdateOperationRunInfo(
                runIndex + 1,
                TicksToMilliseconds(searchElapsed),
                "completed",
                searchGeneration,
                index.Generation,
                options.QueryCount,
                "Only public ExactFlatIndex.Search(query, results) calls are inside the summed per-query Stopwatch samples; query lookup, result allocation/capture/copying and validation construction are outside each measured interval.");

            bool staleCandidateSetRejected = ValidateStaleCandidateSetRejected(index, dataset, options, staleCandidateSet);
            SearchResult[][] rawAllowlistResults = CaptureRawAllowlistSearch(index, dataset, options, rawAllowlists);
            ExactFlatCandidateSet[] freshCandidateSets = BuildCandidateSets(index, candidateInputs);
            SearchResult[][] candidateSetResults = CaptureCandidateSetSearch(index, dataset, options, freshCandidateSets);

            string checkpointDirectory = Path.Combine(
                options.CheckpointDirectory,
                string.Create(CultureInfo.InvariantCulture, $"run-{runIndex + 1:000}"));
            long generationBeforeCheckpoint = index.Generation;
            long checkpointStart = Stopwatch.GetTimestamp();
            ExactFlatCheckpointResult checkpointResult = index.Checkpoint(checkpointDirectory);
            long checkpointElapsed = Stopwatch.GetTimestamp() - checkpointStart;
            checkpointRuns[runIndex] = new GeneratedExactPracticalUpdateOperationRunInfo(
                runIndex + 1,
                TicksToMilliseconds(checkpointElapsed),
                checkpointResult.Status.ToString(),
                generationBeforeCheckpoint,
                index.Generation,
                1,
                "Only public ExactFlatIndex.Checkpoint(directoryPath) is inside this elapsed Stopwatch boundary.");

            long openStart = Stopwatch.GetTimestamp();
            ExactFlatIndex reopened = ExactFlatIndex.OpenReadOnly(checkpointDirectory);
            long openElapsed = Stopwatch.GetTimestamp() - openStart;
            openRuns[runIndex] = new GeneratedExactPracticalUpdateOperationRunInfo(
                runIndex + 1,
                TicksToMilliseconds(openElapsed),
                "opened",
                0,
                reopened.Generation,
                1,
                "Only public ExactFlatIndex.OpenReadOnly(directoryPath) is inside this elapsed Stopwatch boundary.");

            if (runIndex == options.Runs - 1)
            {
                finalCapture = new PracticalUpdateRunCapture(
                    index,
                    reopened,
                    mutationExecution,
                    staleCandidateSetRejected,
                    freshCandidateSets,
                    checkpointResult,
                    generationBeforeCheckpoint,
                    checkpointDirectory,
                    searchResults,
                    rawAllowlistResults,
                    candidateSetResults,
                    CaptureUnfilteredSearch(reopened, dataset, options));
            }
        }

        if (finalCapture is null)
        {
            throw new InvalidOperationException("At least one practical-update run is required.");
        }

        GeneratedExactFilteredResultComparison postMutationComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            finalCapture.PostMutationSearchResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison rawAllowlistComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            rawAllowlistTruth,
            finalCapture.RawAllowlistResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison candidateSetComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            candidateSetTruth,
            finalCapture.CandidateSetResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison reopenedComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            finalCapture.ReopenedResults,
            options.TopK,
            options.Dimension,
            options.Metric);

        GeneratedExactPracticalUpdateOutputsInfo outputs = InspectCheckpointOutput(
            finalCapture.CheckpointDirectory,
            finalCapture.CheckpointResult);
        GeneratedExactPracticalUpdateCountsInfo counts = CreateCounts(options, finalCapture);
        GeneratedExactPracticalUpdateMutationInfo mutationInfo = CreateMutationInfo(options, finalCapture.MutationExecution);
        GeneratedExactPracticalUpdateGenerationInfo generations = CreateGenerationInfo(finalCapture);
        GeneratedExactPracticalUpdateCandidateSetInfo candidateSetInfo = CreateCandidateSetInfo(
            finalCapture.FreshCandidateSets,
            candidateInputs,
            finalCapture.StaleCandidateSetRejected);
        bool mutationCountsMatched =
            mutationInfo.InsertSuccessCount == options.InsertedDeltaCount &&
            mutationInfo.DeleteSuccessCount == options.DeletedBaseCount &&
            mutationInfo.DuplicateInsertFailures == options.DuplicateInsertAttempts &&
            mutationInfo.UnknownDeleteFailures == options.UnknownDeleteAttempts &&
            mutationInfo.RepeatedDeleteFailures == options.RepeatedDeleteAttempts &&
            mutationInfo.StatusCounts.ReadOnly == 0 &&
            mutationInfo.StatusCounts.Unsupported == 0;
        bool validationPassed =
            mutationCountsMatched &&
            generations.MutationDeltaMatchesCommittedMutations &&
            generations.CheckpointAdvancedExactlyOnce &&
            postMutationComparison.Integrity.Status == "passed" &&
            rawAllowlistComparison.Integrity.Status == "passed" &&
            candidateSetComparison.Integrity.Status == "passed" &&
            reopenedComparison.Integrity.Status == "passed" &&
            finalCapture.StaleCandidateSetRejected &&
            finalCapture.CheckpointResult.Status == ExactFlatCheckpointStatus.Published &&
            outputs.CheckpointOutputBytes > 0;

        RepositoryInfo repository = RepositoryInfo.Create();
        return new GeneratedExactPracticalUpdateBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactPracticalUpdateOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactPracticalUpdateOptions.ScenarioName, commandArguments.ToArray()),
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
                options.PhysicalVectorCount,
                options.QueryCount),
            new TruthInfo(
                "scalar-reference-generated-live-practical-update",
                options.TopK,
                "live base plus committed exact delta minus tombstones, ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                GeneratedExactPracticalUpdateOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data creation, base index build, truth construction, post-mutation search query lookup and result allocation/capture/copying, raw allowlist/candidate input construction, candidate-set construction, validation searches, output-byte scans and report writing are excluded from measured timing"),
            new IndexInfo(
                "ExactPracticalUpdate",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.PhysicalVectorCount,
                "Existing public ExactFlatIndex APIs only: TryAdd, TryDelete, Generation, Search, raw allowlist Search, CreateCandidateSet/Search, Checkpoint and OpenReadOnly; no src/VecNet changes, HNSW hybrid search, preview API or public claim."),
            new GeneratedExactPracticalUpdateWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount + options.DuplicateInsertAttempts,
                options.DeletedBaseCount + options.UnknownDeleteAttempts + options.RepeatedDeleteAttempts,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                dataset.SeedText,
                "committed TryAdd delta inserts, committed TryDelete base tombstones, duplicate/reserved TryAdd attempts, unknown TryDelete attempts, repeated TryDelete attempts, then explicit checkpoint/open",
                "Base IDs are 0..baseVectorCount-1; committed delta IDs are baseVectorCount..physicalVectorCount-1; deleted base IDs remain reserved; unknown IDs start above physicalVectorCount.",
                "new-or-empty-directory"),
            counts,
            mutationInfo,
            generations,
            rawAllowlists.Info,
            candidateInputs.Info,
            candidateSetInfo,
            new GeneratedExactPracticalUpdateOperationsInfo(
                new("mutations", "public ExactFlatIndex.TryAdd/TryDelete mutation batch with deterministic per-attempt bookkeeping", mutationRuns, Aggregate(mutationRuns)),
                new("postMutationExactSearch", "public ExactFlatIndex.Search(query, results)", searchRuns, Aggregate(searchRuns)),
                new("checkpoint", "public ExactFlatIndex.Checkpoint(directoryPath)", checkpointRuns, Aggregate(checkpointRuns)),
                new("open", "public ExactFlatIndex.OpenReadOnly(directoryPath)", openRuns, Aggregate(openRuns)),
                NotMeasured("milliseconds", "Raw allowlist searches are validation-only and executed outside measured timing in VEC-079."),
                NotMeasured("milliseconds", "Fresh candidate-set searches are validation-only and executed outside measured timing in VEC-079."),
                NotMeasured("milliseconds", "Stale candidate-set rejection is validation-only and executed outside measured timing in VEC-079.")),
            new GeneratedExactPracticalUpdateMeasurementInfo(
                CreateMeasurement("mutations", "public ExactFlatIndex.TryAdd/TryDelete mutation batch with deterministic per-attempt bookkeeping", mutationRuns, "perMutationBatch"),
                CreateMeasurement(
                    "postMutationExactSearch",
                    "public ExactFlatIndex.Search(query, results)",
                    searchRuns,
                    "perSearchBatch",
                    "sum of one elapsed Stopwatch sample per public ExactFlatIndex.Search(query, results) call in each operation run"),
                CreateMeasurement("checkpoint", "public ExactFlatIndex.Checkpoint(directoryPath)", checkpointRuns, "perCheckpointCall"),
                CreateMeasurement("open", "public ExactFlatIndex.OpenReadOnly(directoryPath)", openRuns, "perOpenCall"),
                NotMeasured("bytes", "Managed allocation for mutation calls is not measured in VEC-079."),
                NotMeasured("bytes", "Managed allocation for post-mutation search calls is not measured in VEC-079."),
                NotMeasured("bytes", "Managed allocation for checkpoint is not measured in VEC-079."),
                NotMeasured("bytes", "Managed allocation for open is not measured in VEC-079."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "executed" : "absent",
                    options.WarmupQueries,
                    options.WarmupQueries > 0
                        ? "Warmup unfiltered searches execute after mutations and before measured post-mutation exact search, and are excluded from timing."
                        : "No warmup queries were requested."),
                "Generated data creation, base index build, truth construction, post-mutation search query lookup and result allocation/capture/copying, raw allowlist/candidate input construction, candidate-set construction, validation searches, output-byte scans and report writing are excluded from measured timing boundaries."),
            outputs,
            new GeneratedExactPracticalUpdateMetricsInfo(
                CreateOperationMetrics(postMutationComparison),
                CreateOperationMetrics(rawAllowlistComparison),
                CreateOperationMetrics(candidateSetComparison),
                CreateOperationMetrics(reopenedComparison)),
            new GeneratedExactPracticalUpdateValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-exact-practical-update-smoke",
                FiniteVectors: true,
                LiveTruthGenerated: true,
                MutationCountsMatched: mutationCountsMatched,
                GenerationBeforeAfterMutationReported: true,
                PostMutationExactSearchComparedToTruth: postMutationComparison.Integrity.Status == "passed",
                RawAllowlistVisibleAfterMutation: rawAllowlistComparison.Integrity.Status == "passed",
                FreshCandidateSetVisibleAfterMutation: candidateSetComparison.Integrity.Status == "passed",
                StaleCandidateSetRejectedAfterMutation: finalCapture.StaleCandidateSetRejected,
                CheckpointPublished: finalCapture.CheckpointResult.Status == ExactFlatCheckpointStatus.Published,
                ReopenedOutputParity: reopenedComparison.Integrity.Status == "passed",
                CheckpointOutputBytesScannedOutsideTiming: true,
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateResources(),
            CreateEligibility(),
            [
                "Private generated exact practical-update smoke evidence only; not a public benchmark claim.",
                "This scenario uses existing exact-flat APIs only and does not modify src/VecNet.",
                "HNSW base-plus-exact-delta search, HNSW mutation/filtering/metrics and preview API admission are out of scope.",
                "Measured mutation, search, checkpoint and open boundaries are separate Stopwatch scopes.",
                "Setup, generated data creation, truth construction, post-mutation search query lookup and result allocation/capture/copying, filter/candidate inputs, validation searches, output-byte scans and report writing are excluded from measured timing.",
                "Actual resident/process/GC/private/peak memory and peak temporary disk are explicitly not measured.",
                "Public, preview, baseline, comparison and regression eligibility are false."
            ]);
    }

    public static void Write(GeneratedExactPracticalUpdateBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactPracticalUpdateOptions options) =>
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

    private static ExactFlatIndex BuildBaseIndex(GeneratedExactPracticalUpdateOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static MutationExecution ExecuteMutations(
        GeneratedExactPracticalUpdateOptions options,
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
            ulong id = (ulong)(i % options.DeletedBaseCount);
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

        return new MutationExecution(
            inserted,
            deleted,
            generationBeforeMutations,
            index.Generation,
            lastResult,
            counts.ToInfo());
    }

    private static ulong[] BuildLiveIds(GeneratedExactPracticalUpdateOptions options)
    {
        var ids = new ulong[options.LiveVectorCount];
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

    private static ulong[] BuildStaleCandidateInput(GeneratedExactPracticalUpdateOptions options) =>
        Enumerable.Range(0, Math.Min(options.BaseVectorCount, Math.Max(1, options.TopK)))
            .Select(static value => (ulong)value)
            .ToArray();

    private static PracticalUpdateFilterInputSet GenerateFilterInputs(
        GeneratedExactPracticalUpdateOptions options,
        string kind,
        ulong[] liveIds,
        int duplicateIdsPerQuery,
        int unknownIdsPerQuery)
    {
        int knownPerQuery = GetKnownCount(kind, liveIds.Length, options.TopK);
        int inputCount = checked(knownPerQuery + duplicateIdsPerQuery + unknownIdsPerQuery);
        var inputs = new ulong[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            var ids = new ulong[inputCount];
            int write = 0;
            int start = liveIds.Length == 0
                ? 0
                : (int)(((ulong)options.Seed + ((ulong)queryRow * 2_654_435_761UL)) % (ulong)liveIds.Length);
            for (int i = 0; i < knownPerQuery; i++)
            {
                ids[write++] = liveIds[(start + i) % liveIds.Length];
            }

            for (int i = 0; i < duplicateIdsPerQuery; i++)
            {
                ids[write++] = knownPerQuery == 0
                    ? (ulong)options.PhysicalVectorCount + 1UL
                    : ids[i % knownPerQuery];
            }

            ulong firstUnknown = (ulong)options.PhysicalVectorCount + 1UL + ((ulong)queryRow * (ulong)Math.Max(1, unknownIdsPerQuery));
            for (int i = 0; i < unknownIdsPerQuery; i++)
            {
                ids[write++] = firstUnknown + (ulong)i;
            }

            inputs[queryRow] = ids;
        }

        var info = new GeneratedExactUpdateFilterInputInfo(
            kind,
            GetSelectivityTarget(kind),
            liveIds.Length == 0 ? 0 : (double)knownPerQuery / liveIds.Length,
            knownPerQuery,
            duplicateIdsPerQuery,
            unknownIdsPerQuery,
            inputCount,
            checked(knownPerQuery * options.QueryCount),
            checked(duplicateIdsPerQuery * options.QueryCount),
            checked(unknownIdsPerQuery * options.QueryCount),
            "deterministic query-rotated live IDs followed by requested duplicate known IDs and requested unknown IDs",
            "knownPerQuery = all: liveCount; broad: ceiling(liveCount * 0.50); selective: ceiling(liveCount * 0.10); very-selective: min(liveCount, topK - 1); empty: 0. For query q, known IDs rotate through the post-mutation live ID set.",
            "Duplicate input IDs are deliberately admitted and coalesced by raw allowlist search or candidate-set construction.",
            "Unknown input IDs are deliberately admitted and ignored by raw allowlist search or candidate-set construction.",
            "Inputs are generated from the post-mutation live view and include live delta rows while excluding tombstoned base rows.");

        return new PracticalUpdateFilterInputSet(inputs, info);
    }

    private static ExactFlatCandidateSet[] BuildCandidateSets(
        ExactFlatIndex index,
        PracticalUpdateFilterInputSet inputs)
    {
        var candidateSets = new ExactFlatCandidateSet[inputs.InputIds.Length];
        for (int i = 0; i < candidateSets.Length; i++)
        {
            candidateSets[i] = index.CreateCandidateSet(inputs.InputIds[i]);
        }

        return candidateSets;
    }

    private static TruthSet GenerateLiveTruth(
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options,
        ulong[] liveIds,
        ulong[][]? candidateFilter)
    {
        var results = new TruthItem[options.QueryCount][];
        double[]? vectorMagnitudes = options.Metric == VectorMetric.Cosine ? CalculateVectorMagnitudes(dataset) : null;
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            HashSet<ulong>? allowed = candidateFilter is null ? null : new HashSet<ulong>(candidateFilter[queryRow]);
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            double queryMagnitude = options.Metric == VectorMetric.Cosine ? CalculateMagnitude(query) : 0;
            var candidates = new List<TruthItem>(liveIds.Length);
            foreach (ulong id in liveIds)
            {
                if (allowed is not null && !allowed.Contains(id))
                {
                    continue;
                }

                int row = checked((int)id);
                float distance = CalculateDistance(
                    query,
                    dataset.GetVector(row),
                    options.Metric,
                    queryMagnitude,
                    vectorMagnitudes is null ? 0 : vectorMagnitudes[row]);
                candidates.Add(new TruthItem(id, distance));
            }

            candidates.Sort(CompareTruthItems);
            int resultCount = Math.Min(options.TopK, candidates.Count);
            results[queryRow] = candidates.Take(resultCount).ToArray();
        }

        return new TruthSet(results, options.TopK);
    }

    private static void WarmupSearch(GeneratedExactPracticalUpdateOptions options, GeneratedDataset dataset, ExactFlatIndex index)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int queryRow = i % options.QueryCount;
            index.Search(dataset.GetQuery(queryRow), results);
        }
    }

    private static SearchResult[][] CaptureUnfilteredSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options)
    {
        var allResults = new SearchResult[options.QueryCount][];
        var results = new SearchResult[options.TopK];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), results);
            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            allResults[queryRow] = queryResults;
        }

        return allResults;
    }

    private static long MeasureUnfilteredSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options,
        out SearchResult[][] allResults)
    {
        allResults = new SearchResult[options.QueryCount][];
        var results = new SearchResult[options.TopK];
        long totalTicks = 0;
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);

            long start = Stopwatch.GetTimestamp();
            int written = index.Search(query, results);
            totalTicks += Stopwatch.GetTimestamp() - start;

            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            allResults[queryRow] = queryResults;
        }

        return totalTicks;
    }

    private static SearchResult[][] CaptureRawAllowlistSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options,
        PracticalUpdateFilterInputSet inputs)
    {
        var allResults = new SearchResult[options.QueryCount][];
        var results = new SearchResult[options.TopK];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), inputs.InputIds[queryRow], results, workspace);
            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            allResults[queryRow] = queryResults;
        }

        return allResults;
    }

    private static SearchResult[][] CaptureCandidateSetSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options,
        ExactFlatCandidateSet[] candidateSets)
    {
        var allResults = new SearchResult[options.QueryCount][];
        var results = new SearchResult[options.TopK];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), candidateSets[queryRow], results);
            var queryResults = new SearchResult[written];
            results.AsSpan(0, written).CopyTo(queryResults);
            allResults[queryRow] = queryResults;
        }

        return allResults;
    }

    private static bool ValidateStaleCandidateSetRejected(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactPracticalUpdateOptions options,
        ExactFlatCandidateSet staleCandidateSet)
    {
        var sentinel = new[] { new SearchResult(ulong.MaxValue, float.NaN) };
        try
        {
            _ = index.Search(dataset.GetQuery(0), staleCandidateSet, sentinel);
            return false;
        }
        catch (InvalidOperationException)
        {
            return sentinel[0].Id == ulong.MaxValue && float.IsNaN(sentinel[0].Distance);
        }
    }

    private static GeneratedExactPracticalUpdateOutputsInfo InspectCheckpointOutput(
        string checkpointDirectory,
        ExactFlatCheckpointResult checkpointResult)
    {
        string manifestPath = Path.Combine(checkpointDirectory, ManifestFileName);
        string idsPath = Path.Combine(checkpointDirectory, IdsFileName);
        string vectorsPath = Path.Combine(checkpointDirectory, VectorsFileName);
        long manifestBytes = new FileInfo(manifestPath).Length;
        long idsBytes = new FileInfo(idsPath).Length;
        long vectorsBytes = new FileInfo(vectorsPath).Length;
        return new GeneratedExactPracticalUpdateOutputsInfo(
            checkpointResult.Status.ToString(),
            checkpointDirectory,
            3,
            checked(manifestBytes + idsBytes + vectorsBytes),
            manifestBytes,
            idsBytes,
            vectorsBytes,
            checkpointResult.LiveVectorCount,
            "File byte lengths are scanned after measured Checkpoint and Open calls, outside all measured timing.");
    }

    private static GeneratedExactPracticalUpdateCountsInfo CreateCounts(
        GeneratedExactPracticalUpdateOptions options,
        PracticalUpdateRunCapture capture)
    {
        int physicalAfterMutation = options.PhysicalVectorCount;
        int liveBeforeCheckpoint = capture.MutationExecution.LastResult.VectorCount;
        int deltaBeforeCheckpoint = capture.MutationExecution.LastResult.DeltaCount;
        int tombstonesBeforeCheckpoint = capture.MutationExecution.LastResult.TombstoneCount;
        int liveBaseBeforeCheckpoint = liveBeforeCheckpoint - deltaBeforeCheckpoint;
        return new GeneratedExactPracticalUpdateCountsInfo(
            options.BaseVectorCount,
            physicalAfterMutation,
            liveBeforeCheckpoint,
            liveBaseBeforeCheckpoint,
            deltaBeforeCheckpoint,
            tombstonesBeforeCheckpoint,
            tombstonesBeforeCheckpoint,
            physicalAfterMutation == 0 ? 0 : (double)tombstonesBeforeCheckpoint / physicalAfterMutation,
            "physicalVectorCountAfterMutation",
            options.BaseVectorCount == 0 ? 0 : (double)deltaBeforeCheckpoint / options.BaseVectorCount,
            "initialBaseCount",
            capture.CheckpointResult.PhysicalVectorCount,
            capture.CheckpointResult.LiveVectorCount,
            capture.CheckpointResult.DeltaVectorCount,
            capture.CheckpointResult.TombstoneCount);
    }

    private static GeneratedExactPracticalUpdateMutationInfo CreateMutationInfo(
        GeneratedExactPracticalUpdateOptions options,
        MutationExecution execution) =>
        new(
            options.InsertedDeltaCount + options.DuplicateInsertAttempts,
            execution.InsertedCount,
            options.DeletedBaseCount + options.UnknownDeleteAttempts + options.RepeatedDeleteAttempts,
            execution.DeletedCount,
            options.DuplicateInsertAttempts,
            execution.StatusCounts.DuplicateId,
            options.UnknownDeleteAttempts,
            execution.StatusCounts.UnknownId,
            options.RepeatedDeleteAttempts,
            execution.StatusCounts.AlreadyDeleted,
            execution.InsertedCount + execution.DeletedCount,
            execution.StatusCounts);

    private static GeneratedExactPracticalUpdateGenerationInfo CreateGenerationInfo(PracticalUpdateRunCapture capture)
    {
        long mutationDelta = capture.MutationExecution.GenerationAfterMutations - capture.MutationExecution.GenerationBeforeMutations;
        return new GeneratedExactPracticalUpdateGenerationInfo(
            capture.MutationExecution.GenerationBeforeMutations,
            capture.MutationExecution.GenerationAfterMutations,
            capture.GenerationBeforeCheckpoint,
            capture.CheckpointResult.Generation,
            mutationDelta,
            capture.CheckpointResult.Generation - capture.GenerationBeforeCheckpoint,
            mutationDelta == capture.MutationExecution.InsertedCount + capture.MutationExecution.DeletedCount,
            capture.CheckpointResult.Generation == capture.GenerationBeforeCheckpoint + 1);
    }

    private static GeneratedExactPracticalUpdateCandidateSetInfo CreateCandidateSetInfo(
        ExactFlatCandidateSet[] freshCandidateSets,
        PracticalUpdateFilterInputSet candidateInputs,
        bool staleRejected)
    {
        int total = freshCandidateSets.Sum(static item => item.Count);
        return new GeneratedExactPracticalUpdateCandidateSetInfo(
            "public ExactFlatIndex.CreateCandidateSet(allowedIds)",
            "Stale and fresh candidate-set construction are validation setup and excluded from measured timing.",
            StaleCandidateSetConstructedBeforeMutation: true,
            staleRejected,
            FreshCandidateSetConstructedAfterMutation: true,
            freshCandidateSets.Length,
            candidateInputs.Info.KnownLiveIdCountPerQuery,
            total,
            "Candidate sets are opaque, exact-flat index-bound and generation-bound runtime objects.",
            "Committed mutation stales candidate sets created before mutation; stale search must fail clearly before writing results.");
    }

    private static GeneratedExactPracticalUpdateOperationMeasurementInfo CreateMeasurement(
        string operationName,
        string timedOperation,
        GeneratedExactPracticalUpdateOperationRunInfo[] runs,
        string sampleScope,
        string measurementMethod = "single elapsed Stopwatch sample per operation run") =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                sampleScope,
                timedOperation,
                "generated data creation, base index build, truth construction, post-mutation search query lookup and result allocation/capture/copying, filter/candidate input construction, candidate-set construction, validation searches, output-byte scans and report writing",
                measurementMethod,
                "Mean/min/max are private local descriptive metadata across equivalent generated practical-update runs, not BenchmarkDotNet statistics.",
                "Raw per-run elapsed milliseconds are emitted in operations."),
            new RepeatedRunInfo(
                runs.Length > 1 ? "measured" : "singleRun",
                runs.Length,
                runs.Length > 1,
                runs.Length > 1
                    ? $"Multiple measured {operationName} runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                    : $"Only one measured {operationName} run executed, so cross-run variance/noise is not measured."),
            CreateRunToRunNoise(runs));

    private static RunToRunMetricNoiseInfo CreateRunToRunNoise(GeneratedExactPracticalUpdateOperationRunInfo[] runs)
    {
        bool measured = runs.Length > 1;
        if (!measured)
        {
            return new RunToRunMetricNoiseInfo(
                "notMeasured",
                "milliseconds",
                null,
                null,
                null,
                null,
                null,
                null,
                "Only one measured operation run exists; this field does not establish run-to-run variation.");
        }

        double[] values = runs.Select(static run => run.ElapsedMilliseconds).ToArray();
        DescriptiveStatistics statistics = RunToRunNoiseStatistics.Calculate(values);
        return new RunToRunMetricNoiseInfo(
            "measured",
            "milliseconds",
            FiniteOrNull(statistics.Mean),
            statistics.SampleStandardDeviation,
            statistics.CoefficientOfVariation,
            FiniteOrNull(statistics.Min),
            FiniteOrNull(statistics.Max),
            FiniteOrNull(statistics.Spread),
            "Computed across measured operation runs using the documented private descriptive-statistics formula.");
    }

    private static GeneratedExactCheckpointOperationMetricsInfo CreateOperationMetrics(
        GeneratedExactFilteredResultComparison comparison) =>
        new(
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.Integrity.DistanceMismatchCount == 0 ? "passed" : "failed",
            comparison.Integrity.DistanceMismatchCount,
            comparison.Integrity.MissingResultCount,
            comparison.Integrity.ExtraResultCount,
            comparison.Integrity);

    private static GeneratedExactPracticalUpdateOperationAggregateInfo Aggregate(
        GeneratedExactPracticalUpdateOperationRunInfo[] runs) =>
        new(
            runs.Length,
            runs.Average(static run => run.ElapsedMilliseconds),
            runs.Min(static run => run.ElapsedMilliseconds),
            runs.Max(static run => run.ElapsedMilliseconds));

    private static int CountMutationAttempts(GeneratedExactUpdateMutationStatusCountInfo counts) =>
        counts.Committed +
        counts.DuplicateId +
        counts.UnknownId +
        counts.AlreadyDeleted +
        counts.ReadOnly +
        counts.Unsupported;

    private static GeneratedExactPracticalUpdateEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-exact-practical-update-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated exact practical-update smoke evidence is not reviewed public evidence and has no public reporting policy.",
            "One private smoke report does not establish Phase 6D preview readiness.",
            "No exact practical-update baseline-candidate policy is accepted yet.",
            "No exact practical-update comparison-artifact policy is accepted yet.",
            "No exact practical-update regression-gate policy, threshold or hard gate is accepted yet.",
            [
                "Generated exact practical-update smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Mutation, search, checkpoint and open timing boundaries are separate private local Stopwatch measurements.",
                "Actual resident/process/GC/private/peak memory and peak temporary disk are not measured.",
                "Not a public claim, preview-readiness result, baseline candidate, comparison artifact, regression gate, Linux validation or BenchmarkDotNet-grade evidence."
            ]);

    private static GeneratedExactPracticalUpdateResourceInfo CreateResources() =>
        new(
            NotMeasured("bytes", "Actual resident memory is not measured in VEC-079."),
            NotMeasured("bytes", "Actual process memory is not measured in VEC-079."),
            NotMeasured("bytes", "Actual GC heap/committed/fragmented memory is not measured in VEC-079."),
            NotMeasured("bytes", "Actual private memory/private bytes are not measured in VEC-079."),
            NotMeasured("bytes", "Actual peak memory is not measured in VEC-079."),
            NotMeasured("bytes", "Peak temporary disk is not sampled or measured in VEC-079; final checkpoint output bytes are reported separately."),
            "measuredFinalOutputBytesOnly",
            "Resource fields do not establish actual memory, peak memory, peak temporary disk, preview readiness or public performance evidence.");

    private static GeneratedExactPracticalUpdateEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated exact practical-update reports are not reviewed public benchmark summaries.",
            "Phase 6D preview API/packageability admission is separate and not satisfied by this smoke report.",
            "No exact practical-update baseline-candidate policy is accepted.",
            "No exact practical-update comparison-artifact policy is accepted.",
            "No exact practical-update regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static int GetKnownCount(string kind, int liveVectorCount, int topK) =>
        kind switch
        {
            "all" => liveVectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.50), 1, liveVectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.10), 1, liveVectorCount),
            "very-selective" => Math.Min(liveVectorCount, topK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact practical-update selectivity kind.", nameof(kind))
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

    private static void ValidateOptions(GeneratedExactPracticalUpdateOptions options)
    {
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

        if (options.TopK > options.LiveVectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the post-mutation live vector count.", nameof(options));
        }

        if (options.DuplicateInsertAttempts < 0 || options.UnknownDeleteAttempts < 0 || options.RepeatedDeleteAttempts < 0)
        {
            throw new ArgumentException("mutation failure-attempt counts must be non-negative.", nameof(options));
        }

        if (options.DuplicateIdsPerQuery < 0 || options.UnknownIdsPerQuery < 0)
        {
            throw new ArgumentException("input duplicate and unknown ID counts must be non-negative.", nameof(options));
        }

        if ((options.AllowlistKind == "very-selective" || options.CandidateSetKind == "very-selective") && options.TopK <= 1)
        {
            throw new ArgumentException("very-selective practical-update filters require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactPracticalUpdateOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactPracticalUpdateOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}d-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.Seed:X8}");
    }

    private static double TicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record PracticalUpdateFilterInputSet(ulong[][] InputIds, GeneratedExactUpdateFilterInputInfo Info);

    private sealed record PracticalUpdateRunCapture(
        ExactFlatIndex Index,
        ExactFlatIndex Reopened,
        MutationExecution MutationExecution,
        bool StaleCandidateSetRejected,
        ExactFlatCandidateSet[] FreshCandidateSets,
        ExactFlatCheckpointResult CheckpointResult,
        long GenerationBeforeCheckpoint,
        string CheckpointDirectory,
        SearchResult[][] PostMutationSearchResults,
        SearchResult[][] RawAllowlistResults,
        SearchResult[][] CandidateSetResults,
        SearchResult[][] ReopenedResults);

    private sealed record MutationExecution(
        int InsertedCount,
        int DeletedCount,
        long GenerationBeforeMutations,
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
}
