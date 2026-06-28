using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class GeneratedExactMemorySmokeScenario
{
    private const string TaskId = "VEC-094";
    private const string SchemaName = "VecNet.ExactMemorySmokeReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "exact-flat.manifest.json";
    private const string IdsFileName = "exact-flat.ids.u64";
    private const string VectorsFileName = "exact-flat.vectors.f32";

    public static GeneratedExactMemorySmokeReport Run(
        GeneratedExactMemorySmokeOptions options,
        IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        var samples = new List<GeneratedExactMemorySampleInfo>();
        MemorySnapshot baseline = CaptureMemorySnapshot();
        samples.Add(CreateSample("baselineProcess", "Runtime after runner startup and before generated scenario allocations where practical.", baseline, baseline));

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        samples.Add(CreateSample("postDatasetGeneration", "Generated vectors and queries are retained by the runner; this is whole-process scenario memory, not VecNet index-only memory.", CaptureMemorySnapshot(), baseline));

        ExactFlatIndex index = BuildBaseIndex(options, dataset);
        samples.Add(CreateSample("postIndexBuildRetained", "Base in-memory ExactFlatIndex is built and retained together with generated inputs.", CaptureMemorySnapshot(), baseline));

        WarmupUnfilteredSearch(options, dataset, index);
        samples.Add(CreateSample("postWarmSearchRetained", "After warm public ExactFlatIndex.Search(query, results) calls with caller-owned result buffer; search is not represented as retained memory.", CaptureMemorySnapshot(), baseline));

        ulong[][] rawAllowlistInputs = GenerateFilterInputs(
            options.BaseVectorCount,
            options.QueryCount,
            options.Seed,
            options.AllowlistKind,
            options.TopK,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            firstUnknownId: (ulong)options.PhysicalVectorCount + 1UL);
        var rawWorkspace = new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount);
        WarmupRawAllowlistSearch(options, dataset, index, rawAllowlistInputs, rawWorkspace);
        samples.Add(CreateSample("rawAllowlistWorkspaceRetained", "After constructing ExactFlatSearchFilterWorkspace and warm raw allowlist searches; caller input ID arrays are runner/application inputs.", CaptureMemorySnapshot(), baseline));

        ulong[][] candidateInputs = GenerateFilterInputs(
            options.BaseVectorCount,
            options.QueryCount,
            options.Seed + 17u,
            options.CandidateSetKind,
            options.TopK,
            options.DuplicateIdsPerQuery,
            options.UnknownIdsPerQuery,
            firstUnknownId: (ulong)options.PhysicalVectorCount + 100UL);
        ExactFlatCandidateSet[] candidateSets = BuildCandidateSets(index, candidateInputs);
        WarmupCandidateSetSearch(options, dataset, index, candidateSets);
        int candidateSetOrdinalCount = candidateSets.Sum(static item => item.Count);
        samples.Add(CreateSample("candidateSetRetained", "After constructing query-scoped ExactFlatCandidateSet instances and warm candidate-set searches; candidate input ID arrays remain runner/application inputs.", CaptureMemorySnapshot(), baseline));

        MutationSummary mutation = ExecuteMutations(options, dataset, index);
        samples.Add(CreateSample("postMutationRetained", "After deterministic TryAdd/TryDelete workload; source index retains base rows, live delta rows, tombstones and deleted/reserved IDs.", CaptureMemorySnapshot(), baseline));

        index.Save(options.SaveDirectory);
        GeneratedExactMemorySmokeOutputInfo saveOutput = InspectOutput(options.SaveDirectory, index.LiveVectorCount, "Save");
        samples.Add(CreateSample("postSaveRetained", "After public ExactFlatIndex.Save(directoryPath); final durable output bytes are reported separately and temporary save peak sampling is not active.", CaptureMemorySnapshot(), baseline));

        ExactFlatIndex opened = ExactFlatIndex.OpenReadOnly(options.SaveDirectory);
        samples.Add(CreateSample("postOpenReadOnlyRetained", "After ExactFlatIndex.OpenReadOnly(directoryPath); source and opened indexes coexist with generated inputs, so this is combined whole-process memory.", CaptureMemorySnapshot(), baseline));

        WarmupUnfilteredSearch(options, dataset, opened);
        samples.Add(CreateSample("openedReadOnlyWarmSearchRetained", "After warm public Search calls on the opened read-only index; working set remains context-only.", CaptureMemorySnapshot(), baseline));

        ExactFlatCheckpointResult checkpointResult = index.Checkpoint(options.CheckpointDirectory);
        GeneratedExactMemorySmokeOutputInfo checkpointOutput = InspectOutput(options.CheckpointDirectory, checkpointResult.LiveVectorCount, "Checkpoint");
        samples.Add(CreateSample("postCheckpointRetained", "After public ExactFlatIndex.Checkpoint(directoryPath); final checkpoint output bytes are reported separately and active peak sampling is not implemented.", CaptureMemorySnapshot(), baseline));

        RepositoryInfo repository = RepositoryInfo.Create();
        int rawKnownCount = GetKnownCount(options.AllowlistKind, options.BaseVectorCount, options.TopK);
        int candidateKnownCount = GetKnownCount(options.CandidateSetKind, options.BaseVectorCount, options.TopK);
        GeneratedExactMemorySmokeActualMemoryInfo actualMemory = CreateActualMemory(samples);
        GeneratedExactMemorySmokeLayoutLowerBoundsInfo lowerBounds = CreateLowerBounds(
            options,
            mutation,
            rawWorkspace,
            candidateSetOrdinalCount,
            saveOutput);
        GeneratedExactMemorySmokeOutputsInfo outputs = CreateOutputs(saveOutput, checkpointOutput);

        bool validationPassed =
            mutation.InsertedCount == options.InsertedDeltaCount &&
            mutation.DeletedCount == options.DeletedBaseCount &&
            saveOutput.FinalOutputBytes > 0 &&
            checkpointResult.Status == ExactFlatCheckpointStatus.Published &&
            checkpointOutput.FinalOutputBytes > 0;

        return new GeneratedExactMemorySmokeReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            GeneratedExactMemorySmokeOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(GeneratedExactMemorySmokeOptions.ScenarioName, commandArguments.ToArray()),
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
            new ScenarioInfo(
                GeneratedExactMemorySmokeOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "This mode samples actual process/GC memory at retained-state boundaries; it does not time search, save, open or checkpoint operations."),
            new IndexInfo(
                "ExactMemorySmoke",
                nameof(ExactFlatIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.PhysicalVectorCount,
                "Existing public ExactFlatIndex APIs only: constructor/Add/Search/raw allowlist workspace/CreateCandidateSet/TryAdd/TryDelete/Save/OpenReadOnly/Checkpoint. No src/VecNet changes or public memory claim."),
            new GeneratedExactMemorySmokeWorkloadInfo(
                options.BaseVectorCount,
                mutation.PhysicalVectorCount,
                mutation.LiveVectorCount,
                mutation.LiveBaseVectorCount,
                mutation.DeltaVectorCount,
                mutation.TombstoneCount,
                mutation.DeletedReservedIdCount,
                mutation.PhysicalVectorCount == 0 ? 0 : (double)mutation.TombstoneCount / mutation.PhysicalVectorCount,
                options.BaseVectorCount == 0 ? 0 : (double)mutation.DeltaVectorCount / options.BaseVectorCount,
                options.QueryCount,
                options.TopK,
                options.WarmupQueries,
                options.AllowlistKind,
                options.CandidateSetKind,
                rawKnownCount,
                candidateKnownCount,
                candidateSets.Length,
                candidateSetOrdinalCount,
                "base Add calls, warm unfiltered search, raw allowlist workspace/search, candidate-set construction/search, committed TryAdd delta inserts, committed TryDelete base tombstones, duplicate/reserved inserts, unknown deletes, repeated deletes, Save, OpenReadOnly, opened warm search, Checkpoint",
                "Actual samples are whole-process boundary samples. Lower-bound layout estimates are separate payload floors and are not actual retained-memory claims."),
            actualMemory,
            lowerBounds,
            outputs,
            new GeneratedExactMemorySmokeValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-exact-memory-smoke",
                FiniteVectors: true,
                BaseIndexBuilt: true,
                WarmSearchExecuted: options.WarmupQueries > 0,
                RawAllowlistWorkspaceConstructed: true,
                CandidateSetsConstructed: true,
                MutationCountsMatched: mutation.InsertedCount == options.InsertedDeltaCount && mutation.DeletedCount == options.DeletedBaseCount,
                SaveOutputWritten: saveOutput.FinalOutputBytes > 0,
                OpenReadOnlyCompleted: opened.LiveVectorCount == mutation.LiveVectorCount,
                CheckpointPublished: checkpointResult.Status == ExactFlatCheckpointStatus.Published,
                ActualAndEstimateSectionsSeparated: true,
                UnsupportedFieldsExplicitlyMarked: true,
                WorkingSetContextOnly: true,
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private generated exact memory smoke evidence only; not a public memory, capacity, latency, QPS, allocation, package, platform or preview-readiness claim.",
                "Actual memory samples are whole-process local samples and include retained generated inputs, runner objects and any coexisting source/opened indexes.",
                "Working set and process peak working set are OS/cache-sensitive context only and are not VecNet retained-memory claims.",
                "Lower-bound layout estimates are payload floors in layoutLowerBounds and are separate from actualMemory samples.",
                "Object-accurate Dictionary/HashSet retained memory, object headers, array headers, slack capacity, index-only private bytes and opened-only retained memory are not available in this mode.",
                "Peak temporary disk and peak temporary process memory during save/checkpoint are not measured because active sampling is not implemented.",
                "Public claim, preview-readiness, baseline-candidate, comparison-artifact and regression-gate eligibility are false."
            ]);
    }

    public static void Write(GeneratedExactMemorySmokeReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(GeneratedExactMemorySmokeOptions options) =>
        new(
            options.Metric,
            options.Dimension,
            options.PhysicalVectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            Runs: 1,
            options.WarmupQueries);

    private static ExactFlatIndex BuildBaseIndex(GeneratedExactMemorySmokeOptions options, GeneratedDataset dataset)
    {
        var index = new ExactFlatIndex(options.Dimension, options.Metric);
        for (int row = 0; row < options.BaseVectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static void WarmupUnfilteredSearch(GeneratedExactMemorySmokeOptions options, GeneratedDataset dataset, ExactFlatIndex index)
    {
        var results = new SearchResult[options.TopK];
        int count = Math.Max(1, options.WarmupQueries);
        for (int i = 0; i < count; i++)
        {
            index.Search(dataset.GetQuery(i % dataset.QueryCount), results);
        }
    }

    private static void WarmupRawAllowlistSearch(
        GeneratedExactMemorySmokeOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ulong[][] inputs,
        ExactFlatSearchFilterWorkspace workspace)
    {
        var results = new SearchResult[options.TopK];
        int count = Math.Max(1, options.WarmupQueries);
        for (int i = 0; i < count; i++)
        {
            int queryRow = i % dataset.QueryCount;
            index.Search(dataset.GetQuery(queryRow), inputs[queryRow], results, workspace);
        }
    }

    private static void WarmupCandidateSetSearch(
        GeneratedExactMemorySmokeOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index,
        ExactFlatCandidateSet[] candidateSets)
    {
        var results = new SearchResult[options.TopK];
        int count = Math.Max(1, options.WarmupQueries);
        for (int i = 0; i < count; i++)
        {
            int queryRow = i % dataset.QueryCount;
            index.Search(dataset.GetQuery(queryRow), candidateSets[queryRow], results);
        }
    }

    private static ExactFlatCandidateSet[] BuildCandidateSets(ExactFlatIndex index, ulong[][] inputs)
    {
        var candidateSets = new ExactFlatCandidateSet[inputs.Length];
        for (int queryRow = 0; queryRow < inputs.Length; queryRow++)
        {
            candidateSets[queryRow] = index.CreateCandidateSet(inputs[queryRow]);
        }

        return candidateSets;
    }

    private static MutationSummary ExecuteMutations(
        GeneratedExactMemorySmokeOptions options,
        GeneratedDataset dataset,
        ExactFlatIndex index)
    {
        int inserted = 0;
        int deleted = 0;
        int duplicateFailures = 0;
        int unknownFailures = 0;
        int repeatedFailures = 0;

        for (int i = 0; i < options.InsertedDeltaCount; i++)
        {
            VectorMutationResult result = index.TryAdd((ulong)(options.BaseVectorCount + i), dataset.GetVector(options.BaseVectorCount + i));
            if (result.Status == VectorMutationStatus.Committed)
            {
                inserted++;
            }
        }

        for (int i = 0; i < options.DeletedBaseCount; i++)
        {
            VectorMutationResult result = index.TryDelete((ulong)i);
            if (result.Status == VectorMutationStatus.Committed)
            {
                deleted++;
            }
        }

        for (int i = 0; i < options.DuplicateInsertAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0 ? (ulong)(i % options.DeletedBaseCount) : 0UL;
            VectorMutationResult result = index.TryAdd(id, dataset.GetVector((int)id));
            if (result.Status == VectorMutationStatus.DuplicateId)
            {
                duplicateFailures++;
            }
        }

        for (int i = 0; i < options.UnknownDeleteAttempts; i++)
        {
            VectorMutationResult result = index.TryDelete((ulong)options.PhysicalVectorCount + 10_000UL + (ulong)i);
            if (result.Status == VectorMutationStatus.UnknownId)
            {
                unknownFailures++;
            }
        }

        for (int i = 0; i < options.RepeatedDeleteAttempts; i++)
        {
            ulong id = options.DeletedBaseCount > 0 ? (ulong)(i % options.DeletedBaseCount) : 0UL;
            VectorMutationResult result = index.TryDelete(id);
            if (result.Status == VectorMutationStatus.AlreadyDeleted)
            {
                repeatedFailures++;
            }
        }

        return new MutationSummary(
            inserted,
            deleted,
            duplicateFailures,
            unknownFailures,
            repeatedFailures,
            index.PhysicalVectorCount,
            index.LiveVectorCount,
            index.BaseVectorCount,
            index.DeltaVectorCount,
            index.TombstoneCount,
            index.DeletedReservedIdCount);
    }

    private static ulong[][] GenerateFilterInputs(
        int liveVectorCount,
        int queryCount,
        uint seed,
        string kind,
        int topK,
        int duplicateIdsPerQuery,
        int unknownIdsPerQuery,
        ulong firstUnknownId)
    {
        int knownPerQuery = GetKnownCount(kind, liveVectorCount, topK);
        int inputLength = checked(knownPerQuery + duplicateIdsPerQuery + unknownIdsPerQuery);
        var inputs = new ulong[queryCount][];
        for (int queryRow = 0; queryRow < queryCount; queryRow++)
        {
            var input = new ulong[inputLength];
            int write = 0;
            ulong start = liveVectorCount == 0
                ? 0
                : (seed + ((ulong)queryRow * 2_654_435_761UL)) % (ulong)liveVectorCount;
            for (int i = 0; i < knownPerQuery; i++)
            {
                input[write++] = (start + (ulong)i) % (ulong)liveVectorCount;
            }

            for (int i = 0; i < duplicateIdsPerQuery; i++)
            {
                input[write++] = knownPerQuery == 0 ? firstUnknownId : input[i % knownPerQuery];
            }

            for (int i = 0; i < unknownIdsPerQuery; i++)
            {
                input[write++] = firstUnknownId + ((ulong)queryRow * (ulong)Math.Max(1, unknownIdsPerQuery)) + (ulong)i;
            }

            inputs[queryRow] = input;
        }

        return inputs;
    }

    private static int GetKnownCount(string kind, int liveVectorCount, int topK) =>
        kind switch
        {
            "all" => liveVectorCount,
            "broad" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.50), 1, liveVectorCount),
            "selective" => Math.Clamp((int)Math.Ceiling(liveVectorCount * 0.10), 1, liveVectorCount),
            "very-selective" => Math.Min(liveVectorCount, topK - 1),
            "empty" => 0,
            _ => throw new ArgumentException("Unsupported generated exact memory smoke selectivity kind.", nameof(kind))
        };

    private static GeneratedExactMemorySmokeActualMemoryInfo CreateActualMemory(IReadOnlyList<GeneratedExactMemorySampleInfo> samples) =>
        new(
            "measured",
            "wholeProcessBoundarySamples",
            "Samples use GC.GetGCMemoryInfo and Process.GetCurrentProcess after each retained-state boundary. No active high-frequency sampling and no forced full-GC stabilization are applied.",
            "Actual memory samples are separate from layout lower-bound estimates and are private local smoke evidence only.",
            samples[0],
            samples[1],
            samples[2],
            samples[3],
            samples[4],
            samples[5],
            samples[6],
            samples[7],
            samples[8],
            samples[9],
            samples[10],
            new GeneratedExactMemoryUnsupportedInfo(
                NotAvailable("bytes", "Object-accurate Dictionary<ulong,int> retained bytes, buckets, entries, object headers and slack capacity are not exposed by ExactFlatIndex."),
                NotAvailable("bytes", "Object-accurate HashSet<ulong> tombstone retained bytes, buckets, entries, object headers and slack capacity are not exposed by ExactFlatIndex."),
                NotAvailable("bytes", "Object-accurate HashSet<ulong> deleted/reserved-ID retained bytes, buckets, entries, object headers and slack capacity are not exposed by ExactFlatIndex."),
                NotMeasured("bytes", "Index-only private bytes are not measured because generated inputs, runner objects and other managed/runtime state coexist in the same process."),
                NotMeasured("bytes", "Opened-only retained memory is not measured because the source index, opened index and generated inputs coexist in this runner process."),
                NotMeasured("bytes", "Peak temporary process memory is not measured because active sampling around build/open/save/checkpoint is not implemented."),
                NotMeasured("bytes", "Peak temporary disk is not measured because active directory sampling during save/checkpoint is not implemented.")),
            [
                "Whole-process samples cannot isolate VecNet index-only retained private bytes from generated input arrays, result buffers, runner objects or runtime state.",
                "Working set and process peak working set are OS/cache-sensitive context only.",
                "Peak observed private bytes and peak observed working set require active sampling and are not measured by this foundation mode.",
                "GC committed and fragmented values are runtime counters, not object-accurate VecNet retained memory attribution."
            ]);

    private static GeneratedExactMemorySmokeLayoutLowerBoundsInfo CreateLowerBounds(
        GeneratedExactMemorySmokeOptions options,
        MutationSummary mutation,
        ExactFlatSearchFilterWorkspace rawWorkspace,
        int candidateSetOrdinalCount,
        GeneratedExactMemorySmokeOutputInfo saveOutput) =>
        new(
            "estimatedLowerBound",
            "payload-only; not actual retained memory",
            checked((long)mutation.PhysicalVectorCount * sizeof(ulong)),
            checked((long)mutation.PhysicalVectorCount * options.Dimension * sizeof(float)),
            checked((long)mutation.LiveVectorCount * options.Dimension * sizeof(float)),
            checked((long)mutation.PhysicalVectorCount * (sizeof(ulong) + sizeof(int))),
            checked((long)rawWorkspace.MaxVectorCount * sizeof(int)),
            checked((long)candidateSetOrdinalCount * sizeof(int)),
            checked((long)mutation.LiveVectorCount * sizeof(ulong) + (long)mutation.LiveVectorCount * options.Dimension * sizeof(float)),
            saveOutput.DurableIdPayloadBytes,
            saveOutput.DurableVectorPayloadBytes,
            checked(saveOutput.DurableIdPayloadBytes + saveOutput.DurableVectorPayloadBytes),
            NotAvailable("bytes", "Tombstone HashSet<ulong> retained capacity is not exposed; no object-accurate retained-memory byte estimate is reported."),
            NotAvailable("bytes", "Deleted/reserved-ID HashSet<ulong> retained capacity is not exposed; no object-accurate retained-memory byte estimate is reported."),
            "Excludes managed object headers, array headers, Dictionary/HashSet buckets and entries, free-list/slack capacity, alignment, JIT/runtime overhead, generated runner input arrays, result buffers, temporary allocations and process/GC fragmentation.");

    private static GeneratedExactMemorySmokeOutputsInfo CreateOutputs(
        GeneratedExactMemorySmokeOutputInfo saveOutput,
        GeneratedExactMemorySmokeOutputInfo checkpointOutput) =>
        new(
            saveOutput,
            checkpointOutput,
            NotMeasured("bytes", "Save final output bytes are measured after the public Save call, but peak output-directory bytes during Save are not actively sampled."),
            NotMeasured("bytes", "Checkpoint final output bytes are measured after the public Checkpoint call, but peak output-directory bytes during Checkpoint are not actively sampled."),
            NotMeasured("bytes", "Peak temporary disk bytes are not measured because active directory sampling is not implemented."),
            NotMeasured("bytes", "Peak observed private bytes during Save are not measured because active process sampling is not implemented."),
            NotMeasured("bytes", "Peak observed private bytes during Checkpoint are not measured because active process sampling is not implemented."),
            "Final output bytes are directory scans after successful operations. Peak temporary disk/process fields remain notMeasured and are not inferred from final output bytes.");

    private static GeneratedExactMemorySampleInfo CreateSample(string name, string boundary, MemorySnapshot current, MemorySnapshot baseline) =>
        new(
            name,
            boundary,
            Measured(current.ManagedHeapSizeBytes, baseline.ManagedHeapSizeBytes, contextOnly: false, "GC.GetGCMemoryInfo().HeapSizeBytes at sample boundary."),
            Measured(current.GcCommittedBytes, baseline.GcCommittedBytes, contextOnly: false, "GC.GetGCMemoryInfo().TotalCommittedBytes at sample boundary where exposed by the runtime."),
            Measured(current.GcFragmentedBytes, baseline.GcFragmentedBytes, contextOnly: false, "GC.GetGCMemoryInfo().FragmentedBytes at sample boundary where exposed by the runtime."),
            Measured(current.ProcessPrivateBytes, baseline.ProcessPrivateBytes, contextOnly: false, "Process.PrivateMemorySize64 at sample boundary; whole-process local value, not index-only attribution."),
            Measured(current.ProcessWorkingSetBytes, baseline.ProcessWorkingSetBytes, contextOnly: true, "Process.WorkingSet64 at sample boundary; OS/cache-sensitive context only, not a retained-memory claim."),
            Measured(current.ProcessPeakWorkingSetBytes, baseline.ProcessPeakWorkingSetBytes, contextOnly: true, "Process.PeakWorkingSet64 at sample boundary; process-lifetime OS/cache-sensitive context only."),
            NotMeasuredMetric("Active process sampling is not implemented; no peak observed private bytes are reported for this boundary."),
            NotMeasuredMetric("Active process sampling is not implemented; no peak observed working set is reported for this boundary."));

    private static MemorySnapshot CaptureMemorySnapshot()
    {
        GCMemoryInfo gc = GC.GetGCMemoryInfo();
        using Process process = Process.GetCurrentProcess();
        process.Refresh();
        return new MemorySnapshot(
            gc.HeapSizeBytes,
            gc.TotalCommittedBytes,
            gc.FragmentedBytes,
            process.PrivateMemorySize64,
            process.WorkingSet64,
            process.PeakWorkingSet64);
    }

    private static GeneratedExactMemoryMetricInfo Measured(
        long value,
        long baseline,
        bool contextOnly,
        string reason) =>
        new("measured", value, value - baseline, "bytes", contextOnly, reason);

    private static GeneratedExactMemoryMetricInfo NotMeasuredMetric(string reason) =>
        new("notMeasured", null, null, "bytes", ContextOnly: false, reason);

    private static GeneratedExactMemorySmokeOutputInfo InspectOutput(string directoryPath, int outputVectorCount, string operation)
    {
        long manifestBytes = FileLength(directoryPath, ManifestFileName);
        long idsBytes = FileLength(directoryPath, IdsFileName);
        long vectorsBytes = FileLength(directoryPath, VectorsFileName);
        long idPayloadBytes = checked((long)outputVectorCount * sizeof(ulong));
        long vectorPayloadBytes = outputVectorCount == 0
            ? 0
            : Math.Max(0, vectorsBytes - 48);

        return new GeneratedExactMemorySmokeOutputInfo(
            "written",
            directoryPath,
            Directory.Exists(directoryPath) ? Directory.EnumerateFiles(directoryPath).Count() : 0,
            checked(manifestBytes + idsBytes + vectorsBytes),
            manifestBytes,
            idsBytes,
            vectorsBytes,
            outputVectorCount,
            idPayloadBytes,
            vectorPayloadBytes,
            string.Create(CultureInfo.InvariantCulture, $"File byte lengths are scanned after the public {operation} call; output scans are not part of active peak temporary disk sampling."));
    }

    private static long FileLength(string directoryPath, string fileName)
    {
        string path = Path.Combine(directoryPath, fileName);
        return File.Exists(path) ? new FileInfo(path).Length : 0;
    }

    private static GeneratedExactMemorySmokeEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-exact-memory-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated exact memory smoke output is not reviewed public evidence and has no public reporting policy.",
            "One local whole-process memory smoke report does not establish Phase 6D preview readiness.",
            "No exact memory smoke baseline-candidate policy is accepted.",
            "Exact memory smoke reports are not accepted comparison artifacts.",
            "No exact memory smoke regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated exact memory smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Actual samples are local whole-process samples and are separated from lower-bound layout estimates.",
                "Working set is OS/cache-sensitive context only.",
                "Peak temporary memory/disk are not measured without active sampling.",
                "Not a public claim, preview-readiness result, baseline candidate, comparison artifact, regression gate, Linux validation or BenchmarkDotNet-grade evidence."
            ]);

    private static GeneratedExactMemorySmokeEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated exact memory smoke output is not reviewed public evidence.",
            "This private local smoke report does not establish preview API/package readiness.",
            "No exact memory baseline-candidate policy is accepted.",
            "No exact memory comparison-artifact policy is accepted.",
            "No exact memory regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static MeasurementStatusInfo NotAvailable(string unit, string reason) =>
        new("notAvailable", "absent", unit, reason);

    private static void ValidateOptions(GeneratedExactMemorySmokeOptions options)
    {
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
            throw new ArgumentException("very-selective memory-smoke filters require top-k greater than 1.", nameof(options));
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

    private static string CreateReportId(string? commit, GeneratedExactMemorySmokeOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{GeneratedExactMemorySmokeOptions.ScenarioName}-{commitPart}-{options.Metric}-{options.Dimension}d-{options.BaseVectorCount}b-{options.InsertedDeltaCount}i-{options.DeletedBaseCount}d-{options.QueryCount}q-{options.TopK}k-{options.Seed:X8}");
    }

    private sealed record MemorySnapshot(
        long ManagedHeapSizeBytes,
        long GcCommittedBytes,
        long GcFragmentedBytes,
        long ProcessPrivateBytes,
        long ProcessWorkingSetBytes,
        long ProcessPeakWorkingSetBytes);

    private sealed record MutationSummary(
        int InsertedCount,
        int DeletedCount,
        int DuplicateInsertFailures,
        int UnknownDeleteFailures,
        int RepeatedDeleteFailures,
        int PhysicalVectorCount,
        int LiveVectorCount,
        int LiveBaseVectorCount,
        int DeltaVectorCount,
        int TombstoneCount,
        int DeletedReservedIdCount);
}
