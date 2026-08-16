using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class HnswAllowlistFilteringScenario
{
    private const string TaskId = "VEC-149";
    private const string SchemaName = "VecNet.HnswAllowlistFilteringBenchmarkReport";
    private const string SchemaVersion = "0.1";

    public static HnswAllowlistFilteringBenchmarkReport Run(
        HnswAllowlistFilteringOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options), options.VectorProfile);
        ValidateFinite(dataset);
        PreparedState state = PrepareState(options, dataset);
        AllowlistSet allowlists = GenerateAllowlists(options, state.LiveIds);
        TruthSet truth = GenerateFilteredTruth(dataset, options, allowlists);

        HnswIndex immutable = BuildLiveHnsw(options, dataset, state.LiveIds);
        Directory.CreateDirectory(options.OpenedIndexDirectory);
        immutable.Save(options.OpenedIndexDirectory);
        HnswIndex opened = HnswIndex.OpenReadOnly(options.OpenedIndexDirectory);

        WarmupHnsw(options, dataset, immutable, allowlists);
        SearchMeasurement immutableSearch = MeasureHnsw(options, dataset, immutable, allowlists, captureResults: true);
        WarmupHnsw(options, dataset, opened, allowlists);
        SearchMeasurement openedSearch = MeasureHnsw(options, dataset, opened, allowlists, captureResults: true);

        WarmupComposite(options, dataset, state.Composite, allowlists);
        SearchMeasurement sourceSearch = MeasureComposite(options, dataset, state.Composite, allowlists, captureResults: true);

        Directory.CreateDirectory(options.CheckpointDirectory);
        HnswBasePlusExactDeltaCheckpointDiagnosticResult checkpoint =
            state.Composite.CheckpointWithDiagnostics(options.CheckpointDirectory);
        HnswIndex checkpointOpened = HnswIndex.OpenReadOnly(options.CheckpointDirectory);

        HnswBasePlusExactDeltaCheckpointCountInfo postCounts = CreateCountInfo(options, state.Composite);
        WarmupComposite(options, dataset, state.Composite, allowlists);
        SearchMeasurement rebuiltSearch = MeasureComposite(options, dataset, state.Composite, allowlists, captureResults: true);
        WarmupHnsw(options, dataset, checkpointOpened, allowlists);
        SearchMeasurement checkpointOpenedSearch = MeasureHnsw(options, dataset, checkpointOpened, allowlists, captureResults: true);

        SectionEvaluation immutableEvaluation = EvaluateSection(options, dataset, truth, allowlists, immutableSearch, isComposite: false, postCheckpointDeltaScan: false);
        SectionEvaluation openedEvaluation = EvaluateSection(options, dataset, truth, allowlists, openedSearch, isComposite: false, postCheckpointDeltaScan: false);
        SectionEvaluation sourceEvaluation = EvaluateSection(options, dataset, truth, allowlists, sourceSearch, isComposite: true, postCheckpointDeltaScan: false);
        SectionEvaluation rebuiltEvaluation = EvaluateSection(options, dataset, truth, allowlists, rebuiltSearch, isComposite: true, postCheckpointDeltaScan: true);
        SectionEvaluation checkpointOpenedEvaluation = EvaluateSection(options, dataset, truth, allowlists, checkpointOpenedSearch, isComposite: false, postCheckpointDeltaScan: true);

        HnswAllowlistParityInfo parity = new(
            CompareSearchParity(immutableSearch.Results, openedSearch.Results, options.Dimension),
            CompareSearchParity(rebuiltSearch.Results, checkpointOpenedSearch.Results, options.Dimension),
            CompareSearchParity(sourceSearch.Results, rebuiltSearch.Results, options.Dimension),
            "Equivalent generated live-view search sections use the same exact filtered truth and caller-owned workspaces. Immutable/opened and rebuilt/checkpoint-opened parity are expected; source/rebuilt parity is recorded as diagnostic because checkpoint rebuild may change the graph.");

        bool fallbackBranch = allowlists.Branches.ExactFallbackQueryCount == options.QueryCount;
        bool exactFallbackPassed = !fallbackBranch ||
            AllExactFallbackPassed(immutableEvaluation, openedEvaluation, sourceEvaluation, rebuiltEvaluation, checkpointOpenedEvaluation);
        bool broadIntegrityPassed = fallbackBranch ||
            AllIntegrityPassed(immutableEvaluation, openedEvaluation, sourceEvaluation, rebuiltEvaluation, checkpointOpenedEvaluation);
        bool allIntegrityPassed = AllIntegrityPassed(immutableEvaluation, openedEvaluation, sourceEvaluation, rebuiltEvaluation, checkpointOpenedEvaluation);
        bool tombstonesSuppressed = AllTombstonesSuppressed(immutableEvaluation, openedEvaluation, sourceEvaluation, rebuiltEvaluation, checkpointOpenedEvaluation);
        bool validationPassed =
            checkpoint.Result.Status == HnswBasePlusExactDeltaCheckpointStatus.Published &&
            allowlists.Branches.BranchConsistencyStatus == "passed" &&
            exactFallbackPassed &&
            broadIntegrityPassed &&
            allIntegrityPassed &&
            tombstonesSuppressed &&
            parity.ImmutableOpenedHnsw.AllResultsMatched &&
            parity.RebuiltCompositeCheckpointOpenedHnsw.AllResultsMatched;

        RepositoryInfo repository = RepositoryInfo.Create();

        return new HnswAllowlistFilteringBenchmarkReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswAllowlistFilteringOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswAllowlistFilteringOptions.ScenarioName, commandArguments.ToArray()),
            new EnvironmentInfo(
                RuntimeInformation.OSDescription,
                RuntimeInformation.ProcessArchitecture.ToString(),
                RuntimeInformation.FrameworkDescription,
                RuntimeInformation.RuntimeIdentifier,
                Environment.ProcessorCount,
                GCSettings.IsServerGC,
                Vector<float>.Count),
            new DatasetInfo(
                dataset.DatasetKind,
                "generated-no-external-source",
                dataset.ProfileDistribution,
                dataset.SeedText,
                options.Metric.ToString(),
                options.Dimension,
                options.PhysicalVectorCount,
                options.QueryCount),
            new TruthInfo(
                "scalar-reference-generated-hnsw-allowlist-live-view",
                options.TopK,
                "allowlist is coalesced to known live generated IDs after base/delta tombstones, then ordered by ascending scalar-reference canonical distance and ascending external ID"),
            new ScenarioInfo(
                HnswAllowlistFilteringOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "generated data setup, HNSW builds, save/open, composite construction, mutation application, checkpoint/rebuild, allowlist generation, exact live-view truth generation, warmup, validation and report writing are excluded from measured search timings"),
            new IndexInfo(
                "InternalHnswAllowlistFiltering",
                "HnswIndex and HnswBasePlusExactDeltaIndex",
                options.Metric.ToString(),
                options.Dimension,
                state.LiveIds.Length,
                "private generated allowlist filtering smoke over immutable/opened HNSW plus source/rebuilt/checkpoint-opened HNSW base-plus-exact-delta; no matrix, external dataset, public docs or public claims"),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated live-view ID order after base tombstones and live delta rows",
                $"{options.Metric} only"),
            new HnswAllowlistFilteringWorkloadInfo(
                options.BaseVectorCount,
                options.InsertedDeltaCount,
                options.DeletedBaseCount,
                options.DeletedDeltaCount,
                options.DuplicateInsertAttempts,
                options.UnknownDeleteAttempts,
                options.RepeatedDeleteAttempts,
                options.QueryCount,
                options.TopK,
                options.Runs,
                options.WarmupQueries,
                dataset.SeedText,
                Path.GetFullPath(options.OpenedIndexDirectory),
                Path.GetFullPath(options.CheckpointDirectory),
                "build base HNSW, insert exact delta rows, delete configured base and delta IDs, apply duplicate/unknown/repeated mutation probes, then checkpoint/rebuild for rebuilt and checkpoint-opened sections",
                "Base IDs are 0..baseVectorCount-1; committed delta IDs continue from baseVectorCount; deleted IDs remain known tombstone probes but are excluded from resolved live allowlists."),
            state.PreCounts,
            CreateMutationInfo(options, state.Mutations),
            CreateCheckpointResultInfo(checkpoint.Result),
            postCounts,
            allowlists.Info,
            allowlists.Branches,
            new HnswAllowlistSearchSectionsInfo(
                CreateSection("immutableHnsw", "HnswIndex.Search(query, allowlist, results, workspace)", options, immutableSearch, immutableEvaluation),
                CreateSection("openedHnsw", "opened read-only HnswIndex.Search(query, allowlist, results, workspace)", options, openedSearch, openedEvaluation),
                CreateSection("sourceComposite", "HnswBasePlusExactDeltaIndex.Search(query, allowlist, results, workspace) before checkpoint", options, sourceSearch, sourceEvaluation),
                CreateSection("rebuiltComposite", "HnswBasePlusExactDeltaIndex.Search(query, allowlist, results, workspace) after checkpoint", options, rebuiltSearch, rebuiltEvaluation),
                CreateSection("checkpointOpenedHnsw", "checkpoint-produced opened read-only HnswIndex.Search(query, allowlist, results, workspace)", options, checkpointOpenedSearch, checkpointOpenedEvaluation)),
            parity,
            NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-149."),
            new HnswAllowlistValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-hnsw-allowlist-filtered-smoke",
                FiniteVectors: true,
                ExactLiveViewTruthGenerated: true,
                ImmutableHnswComparedToTruth: true,
                OpenedHnswComparedToTruth: true,
                SourceCompositeComparedToTruth: true,
                RebuiltCompositeComparedToTruth: true,
                CheckpointOpenedHnswComparedToTruth: true,
                ExactFallbackParityPassedForAllSearches: exactFallbackPassed,
                BroadEmissionIntegrityPassedForAllSearches: broadIntegrityPassed,
                BranchConsistencyPassed: allowlists.Branches.BranchConsistencyStatus == "passed",
                TombstoneSuppressionPassed: tombstonesSuppressed,
                ReturnedResultIntegrityPassedForAllSearches: allIntegrityPassed,
                MemoryNotMeasured: true,
                PublicClaimEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private generated HNSW allowlist filtering smoke evidence only; not a public benchmark claim.",
                "Exact fallback branches must match exact filtered live-view truth; broad emission branches record recall/order and may underfill, but returned-result integrity must pass.",
                "Managed allocation and timing metadata are scoped to individual filtered Search calls with caller-owned result buffers and workspaces.",
                "Memory is explicitly not measured.",
                "Matrix presets, Fashion-MNIST filtering evidence, public docs, package changes, public claims, baseline candidates, comparison artifacts and regression gates are out of scope."
            ]);
    }

    public static void Write(HnswAllowlistFilteringBenchmarkReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static HnswAllowlistSearchSectionInfo CreateSection(
        string name,
        string timedOperation,
        HnswAllowlistFilteringOptions options,
        SearchMeasurement measurement,
        SectionEvaluation evaluation) =>
        new(
            name,
            timedOperation,
            evaluation.Branches,
            new SearchInfo(
                options.QueryCount,
                measurement.Aggregate.MeanElapsedMilliseconds,
                measurement.Aggregate.MeanLatencyP50Milliseconds,
                measurement.Aggregate.MeanLatencyP95Milliseconds,
                measurement.Aggregate.MeanLatencyP99Milliseconds,
                measurement.Aggregate.MeanQps,
                measurement.Runs,
                measurement.Aggregate),
            CreateMeasurement(options, measurement.Runs, timedOperation),
            evaluation.ExactFallback,
            evaluation.BroadEmission,
            evaluation.Integrity,
            evaluation.Underfill,
            evaluation.DeltaScan,
            evaluation.TombstoneSuppression);

    private static MeasurementInfo CreateMeasurement(
        HnswAllowlistFilteringOptions options,
        SearchRunInfo[] runs,
        string timedOperation) =>
        new(
            new LatencyMeasurementInfo(
                "measured",
                "milliseconds",
                "perMeasuredSearchCall",
                timedOperation,
                "generated data setup, HNSW build, save/open, composite construction, mutation application, checkpoint/rebuild, allowlist generation, exact filtered truth generation, warmup, final result validation and report writing",
                "nearest-rank percentile estimator over sorted per-run query latency samples: index = ceil(sampleCount * percentile) - 1, clamped to [0, sampleCount - 1]",
                "Search aggregate percentile fields are arithmetic means across per-run percentile values, not BenchmarkDotNet statistics.",
                "Raw per-query latency samples are not emitted in report JSON."),
            new MeasurementStatusInfo(
                "measured",
                runs.Average(run => run.ManagedAllocatedBytesPerQuery).ToString(CultureInfo.InvariantCulture),
                "bytesPerSearchCall",
                "Measured with GC.GetAllocatedBytesForCurrentThread around each filtered Search(query, allowlist, results, workspace) call using caller-owned SearchResult[] and workspace; setup, allowlist generation, truth, warmup, validation and report writing are excluded."),
            NotMeasured("bytes", "Process resident memory, working set, private bytes, managed heap and peak memory are not measured in VEC-149."),
            new RepeatedRunInfo(
                options.Runs > 1 ? "measured" : "singleRun",
                options.Runs,
                options.Runs > 1,
                options.Runs > 1
                    ? "Multiple measured filtered search runs executed; aggregate mean/min/max timing metadata is recorded without regression thresholds."
                    : "Only one measured search run executed, so cross-run variance/noise is not measured."),
            CreateRunToRunNoise(runs, timedOperation),
            new WarmupInfo(
                options.WarmupQueries > 0 ? "executed" : "absent",
                options.WarmupQueries,
                options.WarmupQueries > 0
                    ? "Warmup filtered searches executed before this measured section and excluded from measured timing and allocation totals."
                    : "No warmup queries were requested."));

    private static PreparedState PrepareState(HnswAllowlistFilteringOptions options, GeneratedDataset dataset)
    {
        HnswIndex baseIndex = BuildBaseHnsw(options, dataset);
        var composite = new HnswBasePlusExactDeltaIndex(baseIndex);
        long generationBeforeMutations = composite.Generation;
        MutationExecution mutations = ExecuteMutations(options, dataset, composite);
        HnswBasePlusExactDeltaCheckpointCountInfo preCounts = CreateCountInfo(options, composite);
        ulong[] liveIds = BuildLiveIds(options);
        return new PreparedState(composite, generationBeforeMutations, mutations, preCounts, liveIds);
    }

    private static HnswIndex BuildBaseHnsw(HnswAllowlistFilteringOptions options, GeneratedDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        var index = new HnswIndex(options.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static HnswIndex BuildLiveHnsw(HnswAllowlistFilteringOptions options, GeneratedDataset dataset, ulong[] liveIds)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        var index = new HnswIndex(options.Dimension, options.Metric, hnswOptions);
        foreach (ulong id in liveIds)
        {
            index.Add(id, dataset.GetVector(checked((int)id)));
        }

        return index;
    }

    private static MutationExecution ExecuteMutations(
        HnswAllowlistFilteringOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite)
    {
        var counts = new MutableMutationStatusCounts();
        int inserted = 0;
        int deletedBase = 0;
        int deletedDelta = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            VectorMutationResult result = composite.TryAdd((ulong)(options.BaseVectorCount + i), dataset.GetVector(options.BaseVectorCount + i));
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                inserted++;
            }
        }

        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            VectorMutationResult result = composite.TryDelete((ulong)i);
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                deletedBase++;
            }
        }

        for (int i = 0; i < options.DeletedDeltaCount; i++)
        {
            VectorMutationResult result = composite.TryDelete((ulong)(options.BaseVectorCount + i));
            counts.Add(result.Status);
            if (result.Status == VectorMutationStatus.Committed)
            {
                deletedDelta++;
            }
        }

        for (int i = 0; i < options.DuplicateInsertAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0 ? (ulong)(i % options.DeletedBaseCount) : (ulong)(i % options.BaseVectorCount);
            counts.Add(composite.TryAdd(id, dataset.GetVector(options.BaseVectorCount + (i % options.InsertedDeltaCount))).Status);
        }

        ulong firstUnknownId = (ulong)options.PhysicalVectorCount + 1UL;
        for (int i = 0; i < options.UnknownDeleteAttempts; i++)
        {
            counts.Add(composite.TryDelete(firstUnknownId + (ulong)i).Status);
        }

        int committedDeleteCount = options.DeletedBaseCount + options.DeletedDeltaCount;
        for (int i = 0; i < options.RepeatedDeleteAttempts; i++)
        {
            ulong id = i % committedDeleteCount < options.DeletedBaseCount
                ? (ulong)(i % options.DeletedBaseCount)
                : (ulong)(options.BaseVectorCount + ((i - options.DeletedBaseCount) % options.DeletedDeltaCount));
            counts.Add(composite.TryDelete(id).Status);
        }

        return new MutationExecution(inserted, deletedBase, deletedDelta, composite.Generation, counts.ToInfo());
    }

    private static AllowlistSet GenerateAllowlists(HnswAllowlistFilteringOptions options, ulong[] liveIds)
    {
        int knownLiveCount = options.FilterProfile switch
        {
            "empty" => 0,
            "very-selective" => Math.Min(liveIds.Length, Math.Max(0, options.TopK - 1)),
            "fallback-boundary" => options.EfSearch,
            "broad" => options.EfSearch + 1,
            "all" => liveIds.Length,
            _ => throw new ArgumentException("Unsupported allowlist profile.", nameof(options))
        };

        var tombstones = BuildTombstoneIds(options);
        bool includeDuplicate = knownLiveCount > 0;
        bool includeUnknown = true;
        bool includeTombstone = tombstones.Length > 0;
        int inputLength = knownLiveCount + (includeDuplicate ? 1 : 0) + (includeUnknown ? 1 : 0) + (includeTombstone ? 1 : 0);
        var allowlists = new ulong[options.QueryCount][];
        int totalDistinct = 0;
        int totalKnown = 0;
        int totalUnknown = 0;
        int totalTombstoned = 0;
        int liveBaseAllowed = 0;
        int liveDeltaAllowed = 0;

        for (int query = 0; query < options.QueryCount; query++)
        {
            var list = new ulong[inputLength];
            int write = 0;
            int start = 0;
            for (int i = 0; i < knownLiveCount; i++)
            {
                list[write++] = liveIds[(start + i) % liveIds.Length];
            }

            if (includeDuplicate)
            {
                list[write++] = list[0];
            }

            if (includeUnknown)
            {
                list[write++] = (ulong)options.PhysicalVectorCount + 1UL + (ulong)query;
            }

            if (includeTombstone)
            {
                list[write++] = tombstones[query % tombstones.Length];
            }

            allowlists[query] = list;
            var distinct = new HashSet<ulong>(list);
            totalDistinct += distinct.Count;
            totalKnown += knownLiveCount + (includeTombstone ? 1 : 0);
            totalUnknown += includeUnknown ? 1 : 0;
            totalTombstoned += includeTombstone ? 1 : 0;
        }

        foreach (ulong id in liveIds.Take(knownLiveCount))
        {
            if (id < (ulong)options.BaseVectorCount)
            {
                liveBaseAllowed++;
            }
            else
            {
                liveDeltaAllowed++;
            }
        }

        int exactFallbackQueries = knownLiveCount <= options.EfSearch ? options.QueryCount : 0;
        int broadQueries = knownLiveCount > options.EfSearch ? options.QueryCount : 0;
        string expectedBranch = exactFallbackQueries == options.QueryCount ? "exactFallback" : "broadEmission";
        HnswAllowlistBranchInfo branches = new(
            exactFallbackQueries,
            broadQueries,
            options.EfSearch,
            expectedBranch,
            "passed",
            0,
            "Known live allowed count is resolved after duplicate coalescing, unknown ID removal and tombstone suppression. Counts <= EfSearch use exact fallback; counts > EfSearch use broad HNSW emission filtering.");

        HnswAllowlistFilteringInfo info = new(
            options.FilterProfile,
            options.QueryCount,
            inputLength,
            options.QueryCount == 0 ? 0 : totalDistinct / options.QueryCount,
            knownLiveCount + (includeTombstone ? 1 : 0),
            includeUnknown ? 1 : 0,
            includeDuplicate ? 1 : 0,
            includeTombstone ? 1 : 0,
            knownLiveCount,
            liveBaseAllowed,
            liveDeltaAllowed,
            knownLiveCount,
            knownLiveCount,
            knownLiveCount,
            checked(inputLength * options.QueryCount),
            totalDistinct,
            totalKnown,
            totalUnknown,
            includeDuplicate ? options.QueryCount : 0,
            totalTombstoned,
            checked(knownLiveCount * options.QueryCount),
            "deterministic live-ID prefix followed by one duplicate live ID when available, one query-specific unknown ID and one query-rotated tombstoned known ID when available",
            "Duplicate input IDs are deliberate caller input and must not change the resolved known live allowed set.",
            "Unknown IDs are greater than the generated physical ID range and must be ignored.",
            "Tombstoned IDs are known generated IDs but must be suppressed before branch selection and result emission.");

        return new AllowlistSet(allowlists, info, branches, knownLiveCount, liveBaseAllowed, liveDeltaAllowed);
    }

    private static TruthSet GenerateFilteredTruth(GeneratedDataset dataset, HnswAllowlistFilteringOptions options, AllowlistSet allowlists)
    {
        HashSet<ulong> live = BuildLiveIds(options).ToHashSet();
        var results = new TruthItem[options.QueryCount][];
        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ulong[] allowed = allowlists.Allowlists[queryRow]
                .Where(live.Contains)
                .Distinct()
                .ToArray();
            if (allowed.Length == 0)
            {
                results[queryRow] = [];
                continue;
            }

            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            var candidates = new TruthItem[allowed.Length];
            for (int i = 0; i < allowed.Length; i++)
            {
                ulong id = allowed[i];
                candidates[i] = new TruthItem(id, ScalarGroundTruth.CalculateDistance(query, dataset.GetVector(checked((int)id)), options.Metric));
            }

            Array.Sort(candidates, CompareTruthItems);
            int count = Math.Min(options.TopK, candidates.Length);
            var top = new TruthItem[count];
            Array.Copy(candidates, top, count);
            results[queryRow] = top;
        }

        return new TruthSet(results, options.TopK);
    }

    private static SectionEvaluation EvaluateSection(
        HnswAllowlistFilteringOptions options,
        GeneratedDataset dataset,
        TruthSet truth,
        AllowlistSet allowlists,
        SearchMeasurement measurement,
        bool isComposite,
        bool postCheckpointDeltaScan)
    {
        HnswAllowlistExactFallbackValidationInfo exactFallback = ValidateExactFallback(options, truth, measurement.Results, allowlists.Branches);
        ResultComparison comparison = ResultComparer.Compare(truth, measurement.Results, options.TopK, options.Dimension, options.Metric);
        HnswAllowlistReturnedResultIntegrityInfo integrity = ValidateReturnedResults(dataset, options, allowlists, measurement.Results);
        int extra = CountExtraResults(truth, measurement.Results, options.TopK);
        HnswAllowlistBroadEmissionValidationInfo broad = new(
            allowlists.Branches.BroadEmissionQueryCount > 0 && integrity.Status == "passed" ? "passed" : allowlists.Branches.BroadEmissionQueryCount == 0 ? "notApplicable" : "failed",
            comparison.RecallAtK,
            comparison.OrderedAgreement,
            comparison.MissingResultCount,
            extra,
            integrity.DistanceMismatchCount,
            "Broad emission filtering records recall/order/underfill against exact filtered truth. Imperfect recall/order is allowed, but returned-result integrity and distance recomputation must pass.");
        HnswAllowlistUnderfillInfo underfill = CreateUnderfill(options, truth, measurement.Results);
        HnswAllowlistDeltaScanInfo deltaScan = CreateDeltaScan(options, measurement.Results, allowlists, isComposite, postCheckpointDeltaScan);
        HnswAllowlistTombstoneSuppressionInfo tombstones = CreateTombstoneSuppression(options, measurement.Results);
        return new SectionEvaluation(allowlists.Branches, exactFallback, broad, integrity, underfill, deltaScan, tombstones);
    }

    private static HnswAllowlistExactFallbackValidationInfo ValidateExactFallback(
        HnswAllowlistFilteringOptions options,
        TruthSet truth,
        SearchResult[][] actual,
        HnswAllowlistBranchInfo branches)
    {
        if (branches.ExactFallbackQueryCount == 0)
        {
            return new HnswAllowlistExactFallbackValidationInfo("notApplicable", 0, 0, 0, 0, "Current profile uses broad emission filtering.");
        }

        int countMismatch = truth.Results.Length == actual.Length ? 0 : 1;
        int idOrOrderMismatch = 0;
        int distanceMismatch = 0;
        int queryCount = Math.Min(truth.Results.Length, actual.Length);
        for (int query = 0; query < queryCount; query++)
        {
            TruthItem[] expected = truth.Results[query];
            SearchResult[] returned = actual[query];
            if (expected.Length != returned.Length)
            {
                countMismatch++;
            }

            int count = Math.Min(expected.Length, returned.Length);
            for (int i = 0; i < count; i++)
            {
                if (expected[i].Id != returned[i].Id)
                {
                    idOrOrderMismatch++;
                }

                if (!ResultComparer.DistanceMatches(expected[i].Distance, returned[i].Distance, options.Dimension, options.Metric))
                {
                    distanceMismatch++;
                }
            }
        }

        bool passed = countMismatch == 0 && idOrOrderMismatch == 0 && distanceMismatch == 0;
        return new HnswAllowlistExactFallbackValidationInfo(
            passed ? "passed" : "failed",
            queryCount,
            countMismatch,
            idOrOrderMismatch,
            distanceMismatch,
            "Exact fallback branches must match exact filtered live-view truth for count, ID/order and distances within D-026 tolerance.");
    }

    private static HnswAllowlistReturnedResultIntegrityInfo ValidateReturnedResults(
        GeneratedDataset dataset,
        HnswAllowlistFilteringOptions options,
        AllowlistSet allowlists,
        SearchResult[][] actual)
    {
        HashSet<ulong> live = BuildLiveIds(options).ToHashSet();
        HashSet<ulong> tombstones = BuildTombstoneIds(options).ToHashSet();
        int queryCountMismatch = actual.Length == options.QueryCount ? 0 : 1;
        int checkedResults = 0;
        int resultCountViolation = 0;
        int nonFinite = 0;
        int duplicate = 0;
        int unknown = 0;
        int tombstoned = 0;
        int notAllowed = 0;
        int distanceMismatch = 0;
        int queryCount = Math.Min(options.QueryCount, actual.Length);
        for (int query = 0; query < queryCount; query++)
        {
            SearchResult[] returned = actual[query];
            if (returned.Length > options.TopK)
            {
                resultCountViolation++;
            }

            HashSet<ulong> allowed = allowlists.Allowlists[query].Where(live.Contains).ToHashSet();
            var seen = new HashSet<ulong>();
            for (int i = 0; i < returned.Length; i++)
            {
                SearchResult result = returned[i];
                checkedResults++;
                if (!float.IsFinite(result.Distance))
                {
                    nonFinite++;
                }

                if (!seen.Add(result.Id))
                {
                    duplicate++;
                }

                if (result.Id >= (ulong)options.PhysicalVectorCount)
                {
                    unknown++;
                    continue;
                }

                if (tombstones.Contains(result.Id) || !live.Contains(result.Id))
                {
                    tombstoned++;
                    continue;
                }

                if (!allowed.Contains(result.Id))
                {
                    notAllowed++;
                    continue;
                }

                float expectedDistance = ScalarGroundTruth.CalculateDistance(dataset.GetQuery(query), dataset.GetVector(checked((int)result.Id)), options.Metric);
                if (!ResultComparer.DistanceMatches(expectedDistance, result.Distance, options.Dimension, options.Metric))
                {
                    distanceMismatch++;
                }
            }
        }

        bool passed = queryCountMismatch == 0 &&
            resultCountViolation == 0 &&
            nonFinite == 0 &&
            duplicate == 0 &&
            unknown == 0 &&
            tombstoned == 0 &&
            notAllowed == 0 &&
            distanceMismatch == 0;

        return new HnswAllowlistReturnedResultIntegrityInfo(
            passed ? "passed" : "failed",
            checkedResults,
            queryCountMismatch,
            resultCountViolation,
            nonFinite,
            duplicate,
            unknown,
            tombstoned,
            notAllowed,
            distanceMismatch,
            "Every returned ID must be known, live, allowed, unique within the query result, finite-distance and distance-recomputable.",
            passed ? "All returned filtered results passed integrity checks." : "One or more returned filtered results failed integrity checks.");
    }

    private static HnswAllowlistUnderfillInfo CreateUnderfill(
        HnswAllowlistFilteringOptions options,
        TruthSet truth,
        SearchResult[][] actual)
    {
        int totalReturned = 0;
        int exactAvailable = 0;
        int underfilledQueries = 0;
        int underfilledSlots = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            int returned = actual[i].Length;
            int available = truth.Results[i].Length;
            totalReturned += returned;
            exactAvailable += available;
            int expected = Math.Min(options.TopK, available);
            if (returned < expected)
            {
                underfilledQueries++;
                underfilledSlots += expected - returned;
            }
        }

        return new HnswAllowlistUnderfillInfo(
            options.QueryCount,
            options.TopK,
            checked(options.QueryCount * options.TopK),
            totalReturned,
            exactAvailable,
            underfilledQueries,
            underfilledSlots,
            "Underfill is counted when returned results are fewer than exact filtered truth can provide up to requested top-k.");
    }

    private static HnswAllowlistDeltaScanInfo CreateDeltaScan(
        HnswAllowlistFilteringOptions options,
        SearchResult[][] actual,
        AllowlistSet allowlists,
        bool isComposite,
        bool postCheckpointDeltaScan)
    {
        if (!isComposite || postCheckpointDeltaScan)
        {
            return new HnswAllowlistDeltaScanInfo(
                isComposite ? "measuredZeroAfterCheckpoint" : "notApplicable",
                0,
                0,
                0,
                0,
                CountDeltaResults(options, actual),
                isComposite
                    ? "After checkpoint publication the composite has no delta rows; exact filtered delta scan count is zero."
                    : "Standalone HNSW sections do not have an exact delta overlay.");
        }

        return new HnswAllowlistDeltaScanInfo(
            "measured",
            options.InsertedDeltaCount - options.DeletedDeltaCount,
            allowlists.LiveDeltaAllowedCountPerQuery,
            checked((options.InsertedDeltaCount - options.DeletedDeltaCount) * options.QueryCount),
            checked(allowlists.LiveDeltaAllowedCountPerQuery * options.QueryCount),
            CountDeltaResults(options, actual),
            "Source composite broad and fallback filtered searches scan allowed live delta rows exactly; this metadata records the generated delta scan boundary outside HNSW base traversal.");
    }

    private static HnswAllowlistTombstoneSuppressionInfo CreateTombstoneSuppression(
        HnswAllowlistFilteringOptions options,
        SearchResult[][] actual)
    {
        HashSet<ulong> baseTombstones = Enumerable.Range(0, options.DeletedBaseCount).Select(i => (ulong)i).ToHashSet();
        HashSet<ulong> deltaTombstones = Enumerable.Range(0, options.DeletedDeltaCount).Select(i => (ulong)(options.BaseVectorCount + i)).ToHashSet();
        int returnedBase = 0;
        int returnedDelta = 0;
        foreach (SearchResult[] query in actual)
        {
            foreach (SearchResult result in query)
            {
                if (baseTombstones.Contains(result.Id))
                {
                    returnedBase++;
                }
                else if (deltaTombstones.Contains(result.Id))
                {
                    returnedDelta++;
                }
            }
        }

        bool passed = returnedBase == 0 && returnedDelta == 0;
        return new HnswAllowlistTombstoneSuppressionInfo(
            passed ? "passed" : "failed",
            options.DeletedBaseCount > 0 ? 1 : 0,
            options.DeletedDeltaCount > 0 ? 1 : 0,
            returnedBase,
            returnedDelta,
            "Known tombstoned allowlist probe IDs must not be returned by any filtered search section.");
    }

    private static void WarmupHnsw(HnswAllowlistFilteringOptions options, GeneratedDataset dataset, HnswIndex index, AllowlistSet allowlists)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(index.Count, options.EfSearch);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int query = i % options.QueryCount;
            index.Search(dataset.GetQuery(query), allowlists.Allowlists[query], results, workspace);
        }
    }

    private static void WarmupComposite(HnswAllowlistFilteringOptions options, GeneratedDataset dataset, HnswBasePlusExactDeltaIndex composite, AllowlistSet allowlists)
    {
        if (options.WarmupQueries == 0)
        {
            return;
        }

        var results = new SearchResult[options.TopK];
        HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(options, composite);
        for (int i = 0; i < options.WarmupQueries; i++)
        {
            int query = i % options.QueryCount;
            composite.Search(dataset.GetQuery(query), allowlists.Allowlists[query], results, workspace);
        }
    }

    private static SearchMeasurement MeasureHnsw(
        HnswAllowlistFilteringOptions options,
        GeneratedDataset dataset,
        HnswIndex index,
        AllowlistSet allowlists,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? captured = null;
        for (int run = 0; run < options.Runs; run++)
        {
            var workspace = new HnswSearchWorkspace(index.Count, options.EfSearch);
            SingleRunMeasurement single = MeasureQueries(
                options,
                dataset,
                allowlists,
                captureResults && run == options.Runs - 1,
                (query, allowlist, destination) => index.Search(query, allowlist, destination, workspace));
            runs[run] = single.Summary with { RunNumber = run + 1 };
            if (single.Results is not null)
            {
                captured = single.Results;
            }
        }

        return new SearchMeasurement(captured ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SearchMeasurement MeasureComposite(
        HnswAllowlistFilteringOptions options,
        GeneratedDataset dataset,
        HnswBasePlusExactDeltaIndex composite,
        AllowlistSet allowlists,
        bool captureResults)
    {
        var runs = new SearchRunInfo[options.Runs];
        SearchResult[][]? captured = null;
        for (int run = 0; run < options.Runs; run++)
        {
            HnswBasePlusExactDeltaSearchWorkspace workspace = CreateCompositeWorkspace(options, composite);
            SingleRunMeasurement single = MeasureQueries(
                options,
                dataset,
                allowlists,
                captureResults && run == options.Runs - 1,
                (query, allowlist, destination) => composite.Search(query, allowlist, destination, workspace));
            runs[run] = single.Summary with { RunNumber = run + 1 };
            if (single.Results is not null)
            {
                captured = single.Results;
            }
        }

        return new SearchMeasurement(captured ?? [], runs, AggregateRuns(runs, options.QueryCount));
    }

    private static SingleRunMeasurement MeasureQueries(
        HnswAllowlistFilteringOptions options,
        GeneratedDataset dataset,
        AllowlistSet allowlists,
        bool captureResults,
        SearchOperation operation)
    {
        var results = new SearchResult[options.TopK];
        SearchResult[][]? allResults = captureResults ? new SearchResult[options.QueryCount][] : null;
        var latencyTicks = new long[options.QueryCount];
        long totalTicks = 0;
        long totalAllocatedBytes = 0;

        for (int queryRow = 0; queryRow < options.QueryCount; queryRow++)
        {
            ReadOnlySpan<float> query = dataset.GetQuery(queryRow);
            ReadOnlySpan<ulong> allowlist = allowlists.Allowlists[queryRow];
            long allocationStart = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            int written = operation(query, allowlist, results);
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

    private static HnswBasePlusExactDeltaSearchWorkspace CreateCompositeWorkspace(
        HnswAllowlistFilteringOptions options,
        HnswBasePlusExactDeltaIndex composite)
    {
        int maxBaseElements = Math.Max(composite.BasePhysicalVectorCount, options.LiveVectorCount);
        return new HnswBasePlusExactDeltaSearchWorkspace(
            maxBaseElements,
            options.EfSearch,
            Math.Min(maxBaseElements, options.EfSearch),
            options.TopK,
            composite.DeltaPhysicalVectorCount);
    }

    private static HnswBasePlusExactDeltaCheckpointCountInfo CreateCountInfo(
        HnswAllowlistFilteringOptions options,
        HnswBasePlusExactDeltaIndex composite)
    {
        int physicalCount = checked(composite.BasePhysicalVectorCount + composite.DeltaPhysicalVectorCount);
        return new HnswBasePlusExactDeltaCheckpointCountInfo(
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
            "Base physical rows remain in the immutable HNSW graph until checkpoint; live count excludes base and delta tombstones; checkpoint folds live rows into a rebuilt immutable base.");
    }

    private static HnswBasePlusExactDeltaCheckpointMutationInfo CreateMutationInfo(
        HnswAllowlistFilteringOptions options,
        MutationExecution mutation) =>
        new(
            mutation.InsertedCount,
            mutation.DeletedBaseCount,
            mutation.DeletedDeltaCount,
            options.DuplicateInsertAttempts,
            options.UnknownDeleteAttempts,
            options.RepeatedDeleteAttempts,
            mutation.InsertedCount + mutation.DeletedBaseCount + mutation.DeletedDeltaCount,
            0,
            mutation.GenerationAfterMutations,
            mutation.GenerationAfterMutations,
            mutation.GenerationAfterMutations == mutation.InsertedCount + mutation.DeletedBaseCount + mutation.DeletedDeltaCount,
            mutation.StatusCounts);

    private static HnswBasePlusExactDeltaCheckpointResultInfo CreateCheckpointResultInfo(HnswBasePlusExactDeltaCheckpointResult result) =>
        new(
            result.Status.ToString(),
            result.Generation,
            result.RebuiltBaseVectorCount,
            result.LiveVectorCount,
            result.BasePhysicalVectorCount,
            result.BaseLiveVectorCount,
            result.DeltaPhysicalVectorCount,
            result.DeltaLiveVectorCount,
            result.BaseTombstoneCount,
            result.DeltaTombstoneCount,
            result.TombstoneCount,
            result.DeletedReservedIdCount,
            result.FoldedDeltaVectorCount,
            result.FoldedBaseTombstoneCount,
            result.FoldedDeltaTombstoneCount);

    private static ulong[] BuildLiveIds(HnswAllowlistFilteringOptions options)
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

    private static ulong[] BuildTombstoneIds(HnswAllowlistFilteringOptions options)
    {
        var ids = new ulong[options.DeletedBaseCount + options.DeletedDeltaCount];
        int write = 0;
        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            ids[write++] = (ulong)i;
        }

        for (int i = 0; i < options.DeletedDeltaCount; i++)
        {
            ids[write++] = (ulong)(options.BaseVectorCount + i);
        }

        return ids;
    }

    private static HnswBasePlusExactDeltaCheckpointParityInfo CompareSearchParity(SearchResult[][] left, SearchResult[][] right, int dimension)
    {
        int countMismatch = left.Length == right.Length ? 0 : 1;
        int idMismatch = 0;
        int orderMismatch = 0;
        int distanceMismatch = 0;
        int queryCount = Math.Min(left.Length, right.Length);
        for (int query = 0; query < queryCount; query++)
        {
            if (left[query].Length != right[query].Length)
            {
                countMismatch++;
            }

            int count = Math.Min(left[query].Length, right[query].Length);
            for (int i = 0; i < count; i++)
            {
                if (left[query][i].Id != right[query][i].Id)
                {
                    idMismatch++;
                    orderMismatch++;
                }

                if (!DistanceMatches(left[query][i].Distance, right[query][i].Distance, dimension))
                {
                    distanceMismatch++;
                }
            }
        }

        return new HnswBasePlusExactDeltaCheckpointParityInfo(
            left.Length,
            countMismatch,
            idMismatch,
            orderMismatch,
            distanceMismatch,
            countMismatch == 0 && idMismatch == 0 && orderMismatch == 0 && distanceMismatch == 0,
            "Compared result count, ID/order and distances within D-026 tolerance for equivalent generated filtered searches.");
    }

    private static int CountDeltaResults(HnswAllowlistFilteringOptions options, SearchResult[][] actual) =>
        actual.Sum(query => query.Count(result => result.Id >= (ulong)options.BaseVectorCount));

    private static int CountExtraResults(TruthSet truth, SearchResult[][] actual, int topK)
    {
        int extra = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            extra += Math.Max(0, actual[i].Length - Math.Min(topK, truth.Results[i].Length));
        }

        return extra;
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

    private static RunToRunNoiseInfo CreateRunToRunNoise(SearchRunInfo[] runs, string operation)
    {
        bool measured = runs.Length > 1;
        string unavailableReason = "Only one measured search run exists; this field does not establish run-to-run variation.";
        return new RunToRunNoiseInfo(
            measured ? "measured" : "notMeasured",
            runs.Length,
            measured,
            $"Across measured generated allowlist-filtered HNSW runs for {operation}; setup, warmup, truth, validation and report writing are excluded.",
            "mean; sample standard deviation when run count is greater than one; coefficient of variation = sampleStandardDeviation / abs(mean) when mean is finite and non-zero; min/max spread = max - min.",
            measured
                ? "Multiple measured runs executed; simple descriptive run-to-run statistics are recorded for private local filtering smoke inspection."
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
            "generated-hnsw-allowlist-filtered-smoke",
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW allowlist filtering smoke output is not reviewed public evidence.",
            "No generated HNSW filtering baseline-candidate policy is accepted.",
            "No generated HNSW filtering regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated squared-L2, inner-product or cosine HNSW allowlist filtering smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Exact fallback parity, broad emission integrity, tombstone suppression and underfill are private methodology evidence.",
                "Managed allocations are measured only for filtered search-call boundaries.",
                "Memory is explicitly not measured.",
                "Not eligible for public filtering, recall, latency, allocation, memory, baseline, comparison, regression-gate, external-dataset or matrix claims."
            ]);

    private static HnswAllowlistEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW allowlist filtering smoke output is not reviewed public evidence.",
            "No generated HNSW filtering baseline-candidate policy is accepted.",
            "No comparison artifact schema or publication policy is accepted for HNSW filtering.",
            "No generated HNSW filtering regression-gate policy, threshold or hard gate is accepted.");

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswAllowlistFilteringOptions options) =>
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

    private static void ValidateOptions(HnswAllowlistFilteringOptions options)
    {
        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct or VectorMetric.Cosine))
        {
            throw new ArgumentException("generated-hnsw-allowlist-filtered supports only SquaredEuclidean, InnerProduct and Cosine.", nameof(options));
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

        if (options.TopK > options.EfSearch)
        {
            throw new ArgumentException("top-k must be less than or equal to ef-search.", nameof(options));
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

        if (options.FilterProfile == "fallback-boundary" && options.LiveVectorCount < options.EfSearch)
        {
            throw new ArgumentException("fallback-boundary profile requires live vector count at least ef-search.", nameof(options));
        }

        if ((options.FilterProfile == "broad" || options.FilterProfile == "all") && options.LiveVectorCount <= options.EfSearch)
        {
            throw new ArgumentException("broad/all profiles require live vector count greater than ef-search.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath) ||
            string.IsNullOrWhiteSpace(options.OpenedIndexDirectory) ||
            string.IsNullOrWhiteSpace(options.CheckpointDirectory))
        {
            throw new ArgumentException("output, opened-index-directory and checkpoint-directory must not be empty.", nameof(options));
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

    private static int CompareTruthItems(TruthItem left, TruthItem right)
    {
        int distanceComparison = left.Distance.CompareTo(right.Distance);
        return distanceComparison != 0 ? distanceComparison : left.Id.CompareTo(right.Id);
    }

    private static bool DistanceMatches(float expected, float actual, int dimension)
    {
        if (!float.IsFinite(actual))
        {
            return false;
        }

        double relative = (8.0 * dimension / 16_777_216.0) * Math.Max(1.0, Math.Abs(expected));
        float tolerance = (float)Math.Max(2e-4, relative);
        return MathF.Abs(expected - actual) <= tolerance;
    }

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static bool AllExactFallbackPassed(params SectionEvaluation[] evaluations) =>
        evaluations.All(evaluation => evaluation.ExactFallback.Status is "passed" or "notApplicable");

    private static bool AllIntegrityPassed(params SectionEvaluation[] evaluations) =>
        evaluations.All(evaluation => evaluation.Integrity.Status == "passed");

    private static bool AllTombstonesSuppressed(params SectionEvaluation[] evaluations) =>
        evaluations.All(evaluation => evaluation.TombstoneSuppression.Status == "passed");

    private static double? FiniteOrNull(double value) => double.IsFinite(value) ? value : null;

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private static string CreateReportId(string? commit, HnswAllowlistFilteringOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswAllowlistFilteringOptions.ScenarioName}-{commitPart}-{options.FilterProfile}-{GeneratedDatasetFactory.GetOptionValue(options.VectorProfile)}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}bd-{options.DeletedDeltaCount}dd-{options.QueryCount}q-{options.TopK}k-{options.Runs}r-{options.WarmupQueries}w-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private delegate int SearchOperation(ReadOnlySpan<float> query, ReadOnlySpan<ulong> allowlist, Span<SearchResult> results);

    private sealed record PreparedState(
        HnswBasePlusExactDeltaIndex Composite,
        long GenerationBeforeMutations,
        MutationExecution Mutations,
        HnswBasePlusExactDeltaCheckpointCountInfo PreCounts,
        ulong[] LiveIds);

    private sealed record MutationExecution(
        int InsertedCount,
        int DeletedBaseCount,
        int DeletedDeltaCount,
        long GenerationAfterMutations,
        GeneratedExactUpdateMutationStatusCountInfo StatusCounts);

    private sealed record AllowlistSet(
        ulong[][] Allowlists,
        HnswAllowlistFilteringInfo Info,
        HnswAllowlistBranchInfo Branches,
        int KnownLiveAllowedCountPerQuery,
        int LiveBaseAllowedCountPerQuery,
        int LiveDeltaAllowedCountPerQuery);

    private sealed record SectionEvaluation(
        HnswAllowlistBranchInfo Branches,
        HnswAllowlistExactFallbackValidationInfo ExactFallback,
        HnswAllowlistBroadEmissionValidationInfo BroadEmission,
        HnswAllowlistReturnedResultIntegrityInfo Integrity,
        HnswAllowlistUnderfillInfo Underfill,
        HnswAllowlistDeltaScanInfo DeltaScan,
        HnswAllowlistTombstoneSuppressionInfo TombstoneSuppression);

    private sealed record SingleRunMeasurement(SearchRunInfo Summary, SearchResult[][]? Results);

    private sealed record SearchMeasurement(SearchResult[][] Results, SearchRunInfo[] Runs, AggregateTimingInfo Aggregate);

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
