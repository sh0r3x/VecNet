using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactCheckpointScenario
{
    private const string TaskId = "VEC-067";
    private const string SchemaName = "VecNet.ExactCheckpointBenchmarkReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "exact-flat.manifest.json";
    private const string IdsFileName = "exact-flat.ids.u64";
    private const string VectorsFileName = "exact-flat.vectors.f32";

    public static GeneratedExactCheckpointBenchmarkReport Run(
        GeneratedExactCheckpointOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        ulong[] liveIds = BuildLiveIds(options);
        GeneratedCheckpointFilterInputSet rawAllowlists = GenerateFilterInputs(
            options,
            options.AllowlistKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);
        GeneratedCheckpointFilterInputSet candidateInputs = GenerateFilterInputs(
            options,
            options.CandidateSetKind,
            liveIds,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery);

        TruthSet unfilteredTruth = GenerateLiveTruth(dataset, options, liveIds, candidateFilter: null);
        TruthSet rawAllowlistTruth = GenerateLiveTruth(dataset, options, liveIds, rawAllowlists.InputIds);
        TruthSet candidateSetTruth = GenerateLiveTruth(dataset, options, liveIds, candidateInputs.InputIds);

        string checkpointRoot = CreateCheckpointRoot(options.OutputPath);
        var runs = new GeneratedExactCheckpointOperationRunInfo[options.Runs];
        CheckpointRunCapture? finalCapture = null;
        for (int runIndex = 0; runIndex < options.Runs; runIndex++)
        {
            ExactFlatIndex index = BuildBaseIndex(options, dataset);
            MutationExecution mutationExecution = ExecuteMutations(options, dataset, index);
            ExactFlatCandidateSet[] staleCandidateSets = BuildCandidateSets(index, candidateInputs);
            string outputDirectory = Path.Combine(checkpointRoot, string.Create(CultureInfo.InvariantCulture, $"run-{runIndex + 1:000}"));
            long generationBeforeCheckpoint = index.Generation;

            long start = Stopwatch.GetTimestamp();
            ExactFlatCheckpointResult checkpointResult = index.Checkpoint(outputDirectory);
            long elapsed = Stopwatch.GetTimestamp() - start;

            runs[runIndex] = new GeneratedExactCheckpointOperationRunInfo(
                runIndex + 1,
                StopwatchTicksToMilliseconds(elapsed),
                checkpointResult.Status.ToString(),
                generationBeforeCheckpoint,
                index.Generation,
                "per-run fresh ignored artifact directory; output-byte scan occurs after the timed public Checkpoint call");

            if (runIndex == options.Runs - 1)
            {
                finalCapture = new CheckpointRunCapture(
                    index,
                    mutationExecution,
                    staleCandidateSets,
                    checkpointResult,
                    generationBeforeCheckpoint,
                    outputDirectory);
            }
        }

        if (finalCapture is null)
        {
            throw new InvalidOperationException("At least one checkpoint run is required.");
        }

        SearchResult[][] preCheckpointResults = CaptureUnfilteredSearch(BuildPreCheckpointIndex(options, dataset), dataset, options);
        SearchResult[][] postCheckpointResults = CaptureUnfilteredSearch(finalCapture.Index, dataset, options);
        ExactFlatIndex reopened = ExactFlatIndex.OpenReadOnly(finalCapture.OutputDirectory);
        SearchResult[][] reopenedResults = CaptureUnfilteredSearch(reopened, dataset, options);
        SearchResult[][] rawAllowlistResults = CaptureRawAllowlistSearch(finalCapture.Index, dataset, options, rawAllowlists);
        ExactFlatCandidateSet[] postCheckpointCandidateSets = BuildCandidateSets(finalCapture.Index, candidateInputs);
        SearchResult[][] candidateSetResults = CaptureCandidateSetSearch(finalCapture.Index, dataset, options, postCheckpointCandidateSets);

        GeneratedExactFilteredResultComparison preComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            preCheckpointResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison postComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            postCheckpointResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison reopenedComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            unfilteredTruth,
            reopenedResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison rawAllowlistComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            rawAllowlistTruth,
            rawAllowlistResults,
            options.TopK,
            options.Dimension,
            options.Metric);
        GeneratedExactFilteredResultComparison candidateSetComparison = GeneratedExactFilteredScenario.ValidateFilteredResults(
            candidateSetTruth,
            candidateSetResults,
            options.TopK,
            options.Dimension,
            options.Metric);

        bool staleCandidateSetsRejected = ValidateStaleCandidateSets(finalCapture.Index, dataset, options, finalCapture.StaleCandidateSets);
        bool deletedReservedIdsRejected = ValidateDeletedReservedIds(options, dataset, finalCapture.Index);
        GeneratedExactCheckpointCountInfo preCounts = CreatePreCheckpointCounts(options, finalCapture.MutationExecution);
        GeneratedExactCheckpointResultInfo resultInfo = CreateCheckpointResultInfo(finalCapture.CheckpointResult);
        GeneratedExactCheckpointCountInfo postCounts = CreatePostCheckpointCounts(options, finalCapture.CheckpointResult);
        bool checkpointCountsMatched = CountsMatch(preCounts, finalCapture.CheckpointResult);
        bool postCountsMatched = PostCountsMatch(preCounts, postCounts, finalCapture.CheckpointResult);
        bool generationAdvanced = finalCapture.CheckpointResult.Generation == finalCapture.GenerationBeforeCheckpoint + 1;
        GeneratedExactCheckpointOutputInfo checkpointOutput = InspectCheckpointOutput(finalCapture.OutputDirectory, finalCapture.CheckpointResult.LiveVectorCount);
        GeneratedExactCheckpointCandidateSetInfo candidateSetInfo = CreateCandidateSetInfo(
            postCheckpointCandidateSets,
            candidateInputs,
            staleCandidateSetsRejected);

        bool validationPassed =
            finalCapture.CheckpointResult.Status == ExactFlatCheckpointStatus.Published &&
            checkpointCountsMatched &&
            postCountsMatched &&
            generationAdvanced &&
            preComparison.Integrity.Status == "passed" &&
            postComparison.Integrity.Status == "passed" &&
            reopenedComparison.Integrity.Status == "passed" &&
            rawAllowlistComparison.Integrity.Status == "passed" &&
            candidateSetComparison.Integrity.Status == "passed" &&
            staleCandidateSetsRejected &&
            deletedReservedIdsRejected &&
            string.Equals(checkpointOutput.ValidationOpenStatus, "passed", StringComparison.Ordinal);

        RepositoryInfo repository = RepositoryInfo.Create();
        GeneratedExactCheckpointOperationAggregateInfo aggregate = AggregateCheckpointRuns(runs);
        GeneratedExactCheckpointMutationInfo mutationInfo = CreateMutationInfo(finalCapture.MutationExecution);

        return new GeneratedExactCheckpointBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactCheckpointOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactCheckpointOptions.ScenarioName, commandArguments.ToArray()),
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
                "scalar-reference-generated-live-checkpoint",
                options.TopK,
                "live base plus committed delta minus visibility tombstones, ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                GeneratedExactCheckpointOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, base index build, mutation execution, live truth construction, allowlist/candidate input generation, candidate-set construction, validation searches, output-byte scan, cleanup outside the public call and report writing are excluded from checkpoint timing"),
            new IndexInfo(
                "ExactCheckpoint",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.PhysicalVectorCount,
                "public ExactFlatIndex constructor; VEC-059 TryAdd/TryDelete mutation setup; public ExactFlatIndex.Checkpoint(directoryPath) measured as the maintenance operation; no active durable-location replacement, matrix preset, HNSW durability, VectorData or public claim"),
            new GeneratedExactCheckpointWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                dataset.SeedText,
                "new-or-empty-directory",
                "per-run fresh ignored artifact directory under the report output directory",
                LiveViewSaveComparisonMeasured: false,
                RawAllowlistBehaviorIncluded: true,
                CandidateSetBehaviorIncluded: true,
                NoChangesBehaviorIncluded: false,
                FailurePathBehaviorIncluded: false,
                "base build via Add, committed TryAdd delta inserts, committed TryDelete base tombstones, duplicate/reserved TryAdd attempts, unknown TryDelete attempts, repeated TryDelete attempts",
                "Generated base IDs are 0..baseVectorCount-1; committed delta IDs are baseVectorCount..physicalVectorCount-1; deleted base IDs remain reserved; unknown IDs start above physicalVectorCount."),
            preCounts,
            resultInfo,
            postCounts,
            mutationInfo,
            rawAllowlists.Info,
            candidateInputs.Info,
            candidateSetInfo,
            new GeneratedExactCheckpointOperationsInfo(
                new GeneratedExactCheckpointOperationInfo(
                    "checkpoint",
                    "public ExactFlatIndex.Checkpoint(directoryPath)",
                    runs,
                    aggregate),
                NotMeasured("millisecondsAndBytes", "Live-view Save comparison was deferred for this checkpoint report."),
                NotMeasured("milliseconds", "Post-checkpoint unfiltered search is validation-only in VEC-067 and is not timed."),
                NotMeasured("milliseconds", "Post-checkpoint raw allowlist search is validation-only in VEC-067 and is not timed."),
                NotMeasured("milliseconds", "Post-checkpoint candidate-set search is validation-only in VEC-067 and is not timed."),
                NotIncluded("NoChanges checkpoint behavior is deferred; this smoke report measures successful Published checkpoint only."),
                NotIncluded("Failure-path checkpoint behavior is deferred; this smoke report measures successful Published checkpoint only."),
                NotMeasured("bytes", "Resident/process memory, working set, private bytes, managed heap and peak memory are not measured.")),
            new GeneratedExactCheckpointMeasurementInfo(
                new GeneratedExactCheckpointOperationMeasurementInfo(
                    new LatencyMeasurementInfo(
                        "measured",
                        "milliseconds",
                        "perCheckpointCall",
                        "public ExactFlatIndex.Checkpoint(directoryPath)",
                        "generated data creation, base index construction, mutation execution, truth construction, output directory naming, validation searches, reopened-output searches, output-byte scans, cleanup outside the public call and report serialization/writing",
                        "single elapsed Stopwatch sample per measured checkpoint run",
                        "Mean/min/max are private local descriptive metadata across independently rebuilt equivalent checkpoint runs, not BenchmarkDotNet statistics.",
                        "Raw per-run checkpoint elapsed milliseconds are emitted in operations.checkpoint.runs."),
                    new RepeatedRunInfo(
                        options.Runs > 1 ? "measured" : "singleRun",
                        options.Runs,
                        options.Runs > 1,
                        options.Runs > 1
                            ? "Multiple measured checkpoint runs executed on independently rebuilt equivalent pre-checkpoint index instances; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                            : "Only one measured checkpoint run executed, so cross-run variance/noise is not measured."),
                    CreateCheckpointRunToRunNoise(runs)),
                NotMeasured("bytesPerCheckpointCall", "Managed allocation for the public Checkpoint call is not measured in VEC-067."),
                NotMeasured("millisecondsAndBytes", "Live-view Save comparison was deferred for this checkpoint report."),
                NotMeasured("millisecondsAndBytes", "Post-checkpoint search timing/allocation is deferred; searches are executed only for validation outside checkpoint duration."),
                NotMeasured("bytes", "Process working set, resident memory, private bytes, managed heap size, GC committed memory and peak memory are not measured."),
                new WarmupInfo(
                    options.WarmupQueries > 0 ? "notApplicable" : "absent",
                    options.WarmupQueries,
                    "Checkpoint is a one-shot mutating operation in VEC-067; warmup queries are not used for checkpoint timing."),
                "Generated data setup, base index build, mutation execution, truth construction, raw allowlist input generation, candidate-set input generation, candidate-set construction, validation searches, reopened-output validation, output-byte scans, cleanup outside the public call and report writing are excluded from checkpoint duration."),
            new GeneratedExactCheckpointOutputsInfo(
                checkpointOutput,
                NotMeasured("bytes", "Save output bytes are absent because live-view Save comparison is not measured in VEC-067.")),
            new GeneratedExactCheckpointMetricsInfo(
                CreateOperationMetrics(preComparison),
                CreateOperationMetrics(postComparison),
                CreateOperationMetrics(reopenedComparison),
                CreateOperationMetrics(rawAllowlistComparison),
                CreateOperationMetrics(candidateSetComparison)),
            new GeneratedExactCheckpointValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-exact-checkpoint-smoke",
                FiniteVectors: true,
                LiveTruthGenerated: true,
                PreCheckpointInMemoryComparedToTruth: true,
                CheckpointResultStatusPublished: finalCapture.CheckpointResult.Status == ExactFlatCheckpointStatus.Published,
                CheckpointResultCountsMatched: checkpointCountsMatched,
                PostCheckpointCountsMatched: postCountsMatched,
                GenerationAdvancedExactlyOnce: generationAdvanced,
                PostCheckpointInMemoryComparedToTruth: true,
                ReopenedCheckpointOutputComparedToTruth: true,
                RawAllowlistComparedToTruth: true,
                CandidateSetComparedToTruth: true,
                PreCheckpointCandidateSetsRejectedAsStale: staleCandidateSetsRejected,
                DeletedReservedIdsRejectedAfterCheckpoint: deletedReservedIdsRejected,
                OutputBytesScannedOutsideCheckpointDuration: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                PreviewReadinessEligible: false,
                ReportIsPrivateRaw: true),
            CreateMemoryEstimateInfo(options, preCounts, postCounts, candidateSetInfo),
            new GeneratedExactCheckpointEligibilityInfo(
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                RegressionGateEligible: false,
                PreviewReadinessEligible: false,
                "Private generated exact-checkpoint smoke evidence is not reviewed public evidence and has no public reporting policy.",
                "No exact-checkpoint baseline-candidate policy is accepted yet.",
                "No exact-checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted yet.",
                "One local generated checkpoint smoke report does not establish resource, durability, crash, concurrency, Linux or release-package readiness."),
            [
                "Private generated exact-checkpoint smoke evidence only; not a public benchmark claim.",
                "Generated data only; no external dataset source, license, version or checksum applies.",
                "Checkpoint duration measures the public ExactFlatIndex.Checkpoint(directoryPath) call including durable write, read-only validation, in-memory publication and result creation.",
                "Setup, data generation, mutation execution, truth construction, validation, output-byte scan and report writing are excluded from checkpoint duration.",
                "Physical and live counts are distinct before checkpoint.",
                "Published checkpoint compacts physical/base count to the pre-checkpoint live count and clears delta/tombstone visibility state.",
                "Deleted/reserved IDs remain retained in the writable instance but are not persisted in checkpoint output.",
                "Output bytes are private local artifact measurements, not public persisted-size claims.",
                "Memory fields are lower-bound estimates or explicit notMeasured/notAvailable values, not resident/process/GC/peak memory evidence.",
                "Live-view Save comparison is not measured in VEC-067.",
                "Post-checkpoint search timing is not measured; post-checkpoint searches are validation only.",
                "No baseline, comparison, regression gate, preview-readiness result, public docs or public claims are included."
            ]);
    }

    public static void Write(GeneratedExactCheckpointBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactCheckpointOptions options) =>
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

    private static ExactFlatIndex BuildPreCheckpointIndex(GeneratedExactCheckpointOptions options, GeneratedDataset dataset)
    {
        ExactFlatIndex index = BuildBaseIndex(options, dataset);
        _ = ExecuteMutations(options, dataset, index);
        return index;
    }

    private static ExactFlatIndex BuildBaseIndex(GeneratedExactCheckpointOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static MutationExecution ExecuteMutations(
        GeneratedExactCheckpointOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        long generationBeforeMutations = index.Generation;
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
            lastResult.Generation,
            lastResult,
            counts.ToInfo());
    }

    private static ulong[] BuildLiveIds(GeneratedExactCheckpointOptions options)
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

    private static GeneratedCheckpointFilterInputSet GenerateFilterInputs(
        GeneratedExactCheckpointOptions options,
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
            "deterministic query-rotated pre-checkpoint live IDs followed by requested duplicate known IDs and requested unknown IDs",
            "knownCount = all: liveVectorCount; broad: ceiling(liveVectorCount * 0.50); selective: ceiling(liveVectorCount * 0.10); very-selective: min(liveVectorCount, topK - 1); empty: 0. For query q, known live IDs start at (seed + q * 2654435761) mod liveVectorCount and advance by one modulo the pre-checkpoint live ID list.",
            "Duplicate input IDs repeat earlier known live IDs when knownCount is greater than zero; empty inputs duplicate unknown IDs so no indexed row becomes visible.",
            "Unknown input IDs are greater than physicalVectorCount and are deliberately ignored by raw allowlist search and candidate-set construction.",
            "Inputs are generated against the pre-checkpoint live view, so tombstoned IDs are excluded and committed delta IDs are eligible before and after checkpoint publication.");

        return new GeneratedCheckpointFilterInputSet(inputs, info);
    }

    private static ExactFlatCandidateSet[] BuildCandidateSets(
        ExactFlatIndex index,
        GeneratedCheckpointFilterInputSet candidateInputs)
    {
        var candidateSets = new ExactFlatCandidateSet[candidateInputs.InputIds.Length];
        for (int queryRow = 0; queryRow < candidateInputs.InputIds.Length; queryRow++)
        {
            candidateSets[queryRow] = index.CreateCandidateSet(candidateInputs.InputIds[queryRow]);
        }

        return candidateSets;
    }

    private static TruthSet GenerateLiveTruth(
        GeneratedDataset dataset,
        GeneratedExactCheckpointOptions options,
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

    private static SearchResult[][] CaptureUnfilteredSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactCheckpointOptions options)
    {
        var results = new SearchResult[options.TopK];
        var allResults = new SearchResult[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), results);
            allResults[queryRow] = results[..written].ToArray();
        }

        return allResults;
    }

    private static SearchResult[][] CaptureRawAllowlistSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactCheckpointOptions options,
        GeneratedCheckpointFilterInputSet rawAllowlists)
    {
        var results = new SearchResult[options.TopK];
        var allResults = new SearchResult[options.QueryCount][];
        var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), rawAllowlists.InputIds[queryRow], results, workspace);
            allResults[queryRow] = results[..written].ToArray();
        }

        return allResults;
    }

    private static SearchResult[][] CaptureCandidateSetSearch(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactCheckpointOptions options,
        ExactFlatCandidateSet[] candidateSets)
    {
        var results = new SearchResult[options.TopK];
        var allResults = new SearchResult[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            int written = index.Search(dataset.GetQuery(queryRow), candidateSets[queryRow], results);
            allResults[queryRow] = results[..written].ToArray();
        }

        return allResults;
    }

    private static bool ValidateStaleCandidateSets(
        ExactFlatIndex index,
        GeneratedDataset dataset,
        GeneratedExactCheckpointOptions options,
        ExactFlatCandidateSet[] staleCandidateSets)
    {
        if (staleCandidateSets.Length == 0)
        {
            return false;
        }

        var sentinel = new SearchResult(ulong.MaxValue, -123f);
        var results = new[] { sentinel };
        try
        {
            _ = index.Search(dataset.GetQuery(0), staleCandidateSets[0], results);
            return false;
        }
        catch (InvalidOperationException)
        {
            return results[0].Equals(sentinel);
        }
    }

    private static bool ValidateDeletedReservedIds(
        GeneratedExactCheckpointOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        VectorMutationResult deletedBase = index.TryAdd(0, dataset.GetVector(0));
        if (deletedBase.Status != VectorMutationStatus.DuplicateId)
        {
            return false;
        }

        ulong deletedDeltaId = (ulong)(options.BaseVectorCount + options.InsertedDeltaCount - 1);
        VectorMutationResult deletedDelta = index.TryDelete(deletedDeltaId);
        if (deletedDelta.Status != VectorMutationStatus.Committed)
        {
            return false;
        }

        VectorMutationResult reservedDelta = index.TryAdd(deletedDeltaId, dataset.GetVector(options.BaseVectorCount + options.InsertedDeltaCount - 1));
        return reservedDelta.Status == VectorMutationStatus.DuplicateId;
    }

    private static GeneratedExactCheckpointOutputInfo InspectCheckpointOutput(string directoryPath, int outputVectorCount)
    {
        long manifestBytes = FileLength(directoryPath, ManifestFileName);
        long idsBytes = FileLength(directoryPath, IdsFileName);
        long vectorsBytes = FileLength(directoryPath, VectorsFileName);
        long totalBytes = manifestBytes + idsBytes + vectorsBytes;
        int fileCount = Directory.Exists(directoryPath) ? Directory.EnumerateFiles(directoryPath).Count() : 0;
        string validationStatus;
        try
        {
            ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(directoryPath);
            validationStatus = opened.VectorCount == outputVectorCount ? "passed" : "failed";
        }
        catch
        {
            validationStatus = "failed";
        }

        return new GeneratedExactCheckpointOutputInfo(
            "written",
            "per-run fresh ignored artifact directory; this path is the final measured run output",
            directoryPath,
            fileCount,
            totalBytes,
            manifestBytes,
            idsBytes,
            vectorsBytes,
            outputVectorCount,
            outputVectorCount == 0 ? 0 : (double)totalBytes / outputVectorCount,
            validationStatus,
            "Directory byte scan and validation open are performed after the timed public Checkpoint call and outside checkpoint duration.");
    }

    private static long FileLength(string directoryPath, string fileName)
    {
        string path = Path.Combine(directoryPath, fileName);
        return File.Exists(path) ? new FileInfo(path).Length : 0;
    }

    private static string CreateCheckpointRoot(string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            directory = "VecNet.BenchmarkRunner.Artifacts";
        }

        string fileName = Path.GetFileNameWithoutExtension(outputPath);
        if (string.IsNullOrWhiteSpace(fileName))
        {
            fileName = "generated-exact-checkpoint";
        }

        return Path.Combine(directory, fileName + "-checkpoint-output-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
    }

    private static GeneratedExactCheckpointCountInfo CreatePreCheckpointCounts(
        GeneratedExactCheckpointOptions options,
        MutationExecution mutationExecution)
    {
        int physical = options.PhysicalVectorCount;
        int live = options.LiveVectorCount;
        return new GeneratedExactCheckpointCountInfo(
            physical,
            live,
            checked(options.BaseVectorCount - options.DeletedBaseCount),
            options.InsertedDeltaCount,
            options.DeletedBaseCount,
            options.DeletedBaseCount,
            physical == 0 ? 0 : (double)options.DeletedBaseCount / physical,
            "physicalVectorCount",
            mutationExecution.GenerationAfterMutations,
            "Deleted/reserved IDs are retained in the writable instance to enforce no ID reuse; reservations are not persisted in checkpoint output.",
            "Pre-checkpoint physicalVectorCount is base vectors plus inserted delta vectors before subtracting deleted/tombstoned base rows.");
    }

    private static GeneratedExactCheckpointCountInfo CreatePostCheckpointCounts(
        GeneratedExactCheckpointOptions options,
        ExactFlatCheckpointResult result) =>
        new(
            result.PhysicalVectorCount,
            result.LiveVectorCount,
            result.BaseVectorCount,
            result.DeltaVectorCount,
            result.TombstoneCount,
            result.DeletedReservedIdCount,
            result.PhysicalVectorCount == 0 ? 0 : (double)result.TombstoneCount / result.PhysicalVectorCount,
            "physicalVectorCount",
            result.Generation,
            "Deleted/reserved IDs remain retained in the original writable instance after checkpoint and are not persisted in reopened read-only checkpoint output.",
            "Post-checkpoint Published physical/base counts equal the pre-checkpoint live visible count; delta and visibility tombstone counts are zero.");

    private static GeneratedExactCheckpointResultInfo CreateCheckpointResultInfo(ExactFlatCheckpointResult result) =>
        new(
            result.Status.ToString(),
            result.Generation,
            result.PhysicalVectorCount,
            result.LiveVectorCount,
            result.BaseVectorCount,
            result.DeltaVectorCount,
            result.TombstoneCount,
            result.DeletedReservedIdCount,
            result.FoldedDeltaVectorCount,
            result.FoldedTombstoneCount);

    private static GeneratedExactCheckpointMutationInfo CreateMutationInfo(MutationExecution mutationExecution)
    {
        int committed = mutationExecution.InsertedCount + mutationExecution.DeletedCount;
        long generationDelta = mutationExecution.GenerationAfterMutations - mutationExecution.GenerationBeforeMutations;
        return new GeneratedExactCheckpointMutationInfo(
            mutationExecution.InsertedCount,
            mutationExecution.DeletedCount,
            mutationExecution.StatusCounts.DuplicateId,
            mutationExecution.StatusCounts.UnknownId,
            mutationExecution.StatusCounts.AlreadyDeleted,
            committed,
            mutationExecution.GenerationBeforeMutations,
            mutationExecution.GenerationAfterMutations,
            generationDelta,
            generationDelta == committed,
            mutationExecution.StatusCounts);
    }

    private static GeneratedExactCheckpointCandidateSetInfo CreateCandidateSetInfo(
        ExactFlatCandidateSet[] candidateSets,
        GeneratedCheckpointFilterInputSet candidateInputs,
        bool staleCandidateSetsRejected)
    {
        int minCount = candidateSets.Length == 0 ? 0 : candidateSets.Min(item => item.Count);
        int maxCount = candidateSets.Length == 0 ? 0 : candidateSets.Max(item => item.Count);
        double meanCount = candidateSets.Length == 0 ? 0 : candidateSets.Average(item => item.Count);
        int totalCount = candidateSets.Sum(item => item.Count);

        return new GeneratedExactCheckpointCandidateSetInfo(
            "preCheckpointSetsStaleAfterPublishedCheckpointAndPostCheckpointSetsConstructedOutsideTiming",
            "public ExactFlatIndex.CreateCandidateSet(allowedIds)",
            "Pre-checkpoint and post-checkpoint candidate-set construction are validation setup and excluded from checkpoint duration.",
            PreCheckpointCandidateSetsConstructed: true,
            staleCandidateSetsRejected,
            PostCheckpointCandidateSetsConstructed: true,
            candidateSets.Length,
            candidateInputs.Info.KnownLiveIdCountPerQuery,
            minCount,
            maxCount,
            meanCount,
            totalCount,
            "Candidate sets are opaque, exact-flat index-bound, generation-bound runtime objects; Published checkpoint advances generation and stales pre-checkpoint sets.",
            "Candidate sets are transient setup artifacts, not persisted filters or public row-ordinal sidecars.");
    }

    private static bool CountsMatch(
        GeneratedExactCheckpointCountInfo preCounts,
        ExactFlatCheckpointResult result) =>
        result.Status == ExactFlatCheckpointStatus.Published &&
        result.PhysicalVectorCount == preCounts.LiveVectorCount &&
        result.LiveVectorCount == preCounts.LiveVectorCount &&
        result.BaseVectorCount == preCounts.LiveVectorCount &&
        result.DeltaVectorCount == 0 &&
        result.TombstoneCount == 0 &&
        result.DeletedReservedIdCount == preCounts.DeletedReservedIdCount &&
        result.FoldedDeltaVectorCount == preCounts.DeltaVectorCount &&
        result.FoldedTombstoneCount == preCounts.VisibilityTombstoneCount;

    private static bool PostCountsMatch(
        GeneratedExactCheckpointCountInfo preCounts,
        GeneratedExactCheckpointCountInfo postCounts,
        ExactFlatCheckpointResult result) =>
        postCounts.PhysicalVectorCount == preCounts.LiveVectorCount &&
        postCounts.LiveVectorCount == preCounts.LiveVectorCount &&
        postCounts.BaseVectorCount == preCounts.LiveVectorCount &&
        postCounts.DeltaVectorCount == 0 &&
        postCounts.VisibilityTombstoneCount == 0 &&
        postCounts.DeletedReservedIdCount == preCounts.DeletedReservedIdCount &&
        postCounts.Generation == result.Generation;

    private static GeneratedExactCheckpointOperationAggregateInfo AggregateCheckpointRuns(
        GeneratedExactCheckpointOperationRunInfo[] runs) =>
        new(
            runs.Length,
            runs.Average(run => run.ElapsedMilliseconds),
            runs.Min(run => run.ElapsedMilliseconds),
            runs.Max(run => run.ElapsedMilliseconds));

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

    private static GeneratedExactCheckpointMemoryEstimateInfo CreateMemoryEstimateInfo(
        GeneratedExactCheckpointOptions options,
        GeneratedExactCheckpointCountInfo preCounts,
        GeneratedExactCheckpointCountInfo postCounts,
        GeneratedExactCheckpointCandidateSetInfo candidateSet) =>
        new(
            "estimatedPayloadLowerBounds",
            "Conservative payload lower-bound estimates only; not managed object overhead, array slack capacity, dictionary/hash-set capacity, GC heap size, working set, private bytes or peak memory.",
            checked((long)preCounts.PhysicalVectorCount * sizeof(ulong)),
            checked((long)preCounts.PhysicalVectorCount * options.Dimension * sizeof(float)),
            checked((long)preCounts.LiveVectorCount * options.Dimension * sizeof(float)),
            checked((long)postCounts.PhysicalVectorCount * sizeof(ulong)),
            checked((long)postCounts.PhysicalVectorCount * options.Dimension * sizeof(float)),
            checked((long)preCounts.LiveVectorCount * sizeof(ulong) + (long)preCounts.LiveVectorCount * options.Dimension * sizeof(float)),
            checked((long)candidateSet.TotalCandidateCount * sizeof(int)),
            NotAvailable("bytes", "Tombstone/deleted-reservation HashSet<ulong> retained capacity is not exposed; no defensible retained-memory byte estimate is reported."),
            NotAvailable("bytes", "Retained HashSet<ulong> capacity is not exposed by ExactFlatIndex and is not estimated."),
            NotMeasured("bytes", "Process resident memory is not measured."),
            NotMeasured("bytes", "GC heap size, GC committed memory and GC fragmented memory are not measured."),
            NotMeasured("bytes", "Working set is OS/cache-sensitive and is not measured."),
            NotMeasured("bytes", "Private bytes are not measured."),
            NotMeasured("bytes", "Peak temporary or process memory is not measured."),
            "These estimates do not establish resident/process memory, managed heap size, object overhead, collection capacity, allocation, peak memory or preview-readiness evidence.");

    private static GeneratedExactCheckpointEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-exact-checkpoint-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            PreviewReadinessEligible: false,
            "Private generated exact-checkpoint smoke evidence is not reviewed public evidence and has no public reporting policy.",
            "No exact-checkpoint baseline-candidate policy is accepted yet.",
            "No exact-checkpoint regression-gate policy, threshold, comparison artifact or hard gate is accepted yet.",
            "One local generated checkpoint smoke report does not establish resource, durability, crash, concurrency, Linux or release-package readiness.",
            [
                "Generated exact-checkpoint smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Checkpoint duration measures only the public ExactFlatIndex.Checkpoint(directoryPath) call.",
                "Output-byte scans, validation searches and report writing are outside checkpoint duration.",
                "Live-view Save, post-checkpoint search timing, checkpoint allocations and resident/process memory are not measured.",
                "Not a public claim, baseline candidate, regression gate, preview-readiness result, Linux x64 validation or BenchmarkDotNet-grade evidence."
            ]);

    private static RunToRunMetricNoiseInfo CreateCheckpointRunToRunNoise(
        GeneratedExactCheckpointOperationRunInfo[] runs)
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
                "Only one measured checkpoint run exists; this field does not establish run-to-run variation.");
        }

        double[] values = runs.Select(run => run.ElapsedMilliseconds).ToArray();
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
            "Computed across measured checkpoint runs using the documented private descriptive-statistics formula.");
    }

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static MeasurementStatusInfo NotAvailable(string unit, string reason) =>
        new("notAvailable", "absent", unit, reason);

    private static MeasurementStatusInfo NotIncluded(string reason) =>
        new("notIncluded", "absent", "notApplicable", reason);

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

    private static int GetKnownCount(string kind, int liveVectorCount, int topK) =>
        kind switch
        {
            "all" => liveVectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.50), 1, liveVectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.10), 1, liveVectorCount),
            "very-selective" => Math.Min(liveVectorCount, topK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact checkpoint selectivity kind.", nameof(kind))
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

    private static void ValidateOptions(GeneratedExactCheckpointOptions options)
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
            throw new ArgumentException("top-k must be less than or equal to the pre-checkpoint live vector count.", nameof(options));
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
            throw new ArgumentException("very-selective checkpoint filters require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactCheckpointOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactCheckpointOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}d-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.Seed:X8}");
    }

    private static double StopwatchTicksToMilliseconds(long ticks) =>
        (double)ticks / Stopwatch.Frequency * 1000;

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private sealed record GeneratedCheckpointFilterInputSet(ulong[][] InputIds, GeneratedExactUpdateFilterInputInfo Info);

    private sealed record CheckpointRunCapture(
        ExactFlatIndex Index,
        MutationExecution MutationExecution,
        ExactFlatCandidateSet[] StaleCandidateSets,
        ExactFlatCheckpointResult CheckpointResult,
        long GenerationBeforeCheckpoint,
        string OutputDirectory);

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
