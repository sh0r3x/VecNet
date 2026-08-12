using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;

namespace VecNet.BenchmarkRunner;

public static class HnswMemorySmokeScenario
{
    private const string TaskId = "VEC-113";
    private const string SchemaName = "VecNet.HnswMemorySmokeReport";
    private const string SchemaVersion = "0.1";
    private const string ManifestFileName = "hnsw.manifest.json";
    private const string IdsFileName = "hnsw.ids.u64";
    private const string VectorsFileName = "hnsw.vectors.f32";
    private const string LevelsFileName = "hnsw.levels.i32";
    private const string GraphFileName = "hnsw.graph.bin";

    public static HnswMemorySmokeReport Run(HnswMemorySmokeOptions options, IReadOnlyList<string> commandArguments)
    {
        ValidateOptions(options);

        MemorySnapshot baseline = CaptureMemorySnapshot();
        HnswMemorySampleInfo baselineSample = CreateSample(
            "baselineProcess",
            "Runtime after runner startup and before generated scenario allocations where practical.",
            baseline,
            baseline);

        GeneratedDataset dataset = GeneratedDatasetFactory.Create(ToGeneratedOptions(options));
        ValidateFinite(dataset);
        HnswMemorySampleInfo postDataset = CreateSample(
            "postDatasetGeneration",
            "Generated base vectors and query vectors are retained by the runner; this is whole-process input memory, not HNSW index-only memory.",
            CaptureMemorySnapshot(),
            baseline);

        HnswIndex sourceIndex = null!;
        HnswMemoryPeakOperationInfo buildPeak = SampleOperation(
            "build",
            options.SampleIntervalMilliseconds,
            baseline,
            "new HnswIndex(...) plus generated base-vector Add calls",
            "generated data creation, exact truth construction, save, open, warm search, validation and report writing",
            () =>
            {
                sourceIndex = BuildIndex(options, dataset);
            });
        HnswMemorySampleInfo postBuild = CreateSample(
            "postSourceBuildRetained",
            "Source internal HNSW is built and retained together with generated inputs; build peak is reported separately under peakMemory.build.",
            CaptureMemorySnapshot(),
            baseline);

        SearchResult[][] sourceResults = WarmSearch(options, dataset, sourceIndex);
        HnswMemorySampleInfo postSourceSearch = CreateSample(
            "postSourceWarmSearchRetained",
            "After warm source HNSW Search calls with caller-owned SearchResult[] and HnswSearchWorkspace; captured parity result arrays are runner validation state.",
            CaptureMemorySnapshot(),
            baseline);

        HnswMemoryPeakOperationInfo savePeak = SampleOperation(
            "save",
            options.SampleIntervalMilliseconds,
            baseline,
            "source HnswIndex.Save(snapshotDirectory)",
            "generated data creation, source build, target path selection, open, warm search, validation, file-byte scans and report writing",
            () => sourceIndex.Save(options.SnapshotDirectory));
        HnswMemorySampleInfo postSave = CreateSample(
            "postSaveRetained",
            "After HnswIndex.Save(directoryPath); final durable file-size facts are reported separately under storageSize.",
            CaptureMemorySnapshot(),
            baseline);

        HnswIndex openedIndex = null!;
        HnswMemoryPeakOperationInfo openPeak = SampleOperation(
            "open",
            options.SampleIntervalMilliseconds,
            baseline,
            "HnswIndex.OpenReadOnly(snapshotDirectory)",
            "source build, save, generated inputs, validation searches, output-byte scans and report writing; source index coexists with opened index",
            () =>
            {
                openedIndex = HnswIndex.OpenReadOnly(options.SnapshotDirectory);
            });
        HnswMemorySampleInfo postOpen = CreateSample(
            "postOpenReadOnlyRetained",
            "After HnswIndex.OpenReadOnly(directoryPath); source and opened indexes coexist with generated inputs in this process.",
            CaptureMemorySnapshot(),
            baseline);

        SearchResult[][] openedResults = WarmSearch(options, dataset, openedIndex);
        HnswMemorySampleInfo postOpenedSearch = CreateSample(
            "postOpenedWarmSearchRetained",
            "After warm opened read-only HNSW Search calls with caller-owned SearchResult[] and HnswSearchWorkspace.",
            CaptureMemorySnapshot(),
            baseline);

        DurableHnswReadOnlyMutationInfo readOnlyMutation = ValidateOpenedReadOnlyMutation(openedIndex, options.Dimension);
        DurableHnswParityInfo parity = CompareSourceOpenedParity(sourceResults, openedResults);
        HnswReturnedResultIntegrityInfo sourceIntegrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, options.Metric, sourceResults, options.TopK);
        HnswReturnedResultIntegrityInfo openedIntegrity = HnswGeneratedScenario.ValidateReturnedResults(dataset, options.Metric, openedResults, options.TopK);
        HnswMemorySmokeStorageSizeInfo storageSize = InspectStorageSize(options.SnapshotDirectory, options.VectorCount);
        HnswMemorySampleInfo postValidation = CreateSample(
            "postValidationRetained",
            "After source/opened parity validation, returned-result integrity checks and opened read-only mutation rejection.",
            CaptureMemorySnapshot(),
            baseline);

        RepositoryInfo repository = RepositoryInfo.Create();
        HnswMemorySmokeActualMemoryInfo actualMemory = CreateActualMemory(
            baselineSample,
            postDataset,
            postBuild,
            postSourceSearch,
            postSave,
            postOpen,
            postOpenedSearch,
            postValidation);
        HnswMemorySmokePeakMemoryInfo peakMemory = CreatePeakMemory(options, buildPeak, savePeak, openPeak);
        HnswMemorySmokeLayoutLowerBoundsInfo lowerBounds = CreateLowerBounds(options, sourceIndex);
        bool validationPassed =
            readOnlyMutation.Status == "passed" &&
            parity.AllResultsMatched &&
            sourceIntegrity.Status == "passed" &&
            openedIntegrity.Status == "passed" &&
            storageSize.TotalBytes > 0 &&
            buildPeak.Status == "sampled" &&
            savePeak.Status == "sampled" &&
            openPeak.Status == "sampled";

        return new HnswMemorySmokeReport(
            SchemaName,
            SchemaVersion,
            CreateReportId(repository.Commit, options),
            DateTimeOffset.UtcNow,
            TaskId,
            HnswMemorySmokeOptions.ScenarioName,
            "local-evidence",
            "private-raw",
            CreateEvidence(),
            repository,
            new RunnerInfo("VecNet.BenchmarkRunner", "0.1", commandArguments.ToArray()),
            new CommandInfo(HnswMemorySmokeOptions.ScenarioName, commandArguments.ToArray()),
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
                options.VectorCount,
                options.QueryCount),
            new ScenarioInfo(
                HnswMemorySmokeOptions.ScenarioName,
                options.TopK,
                options.QueryCount,
                1,
                "This mode samples actual whole-process memory at retained-state boundaries and actively samples observed process peaks for build, save and open; it does not produce latency, recall, baseline or regression evidence."),
            new IndexInfo(
                "InternalHnswMemorySmoke",
                nameof(HnswIndex),
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                $"internal/evaluation-only HnswIndex; generated {options.Metric} build/search, Save, OpenReadOnly and opened search are exercised for private memory methodology only"),
            new HnswMemorySmokeWorkloadInfo(
                options.Metric.ToString(),
                options.Dimension,
                options.VectorCount,
                options.QueryCount,
                options.TopK,
                options.WarmupQueries,
                dataset.SeedText,
                FormatHex(options.HnswSeed),
                options.M,
                options.EfConstruction,
                options.EfSearch,
                options.SampleIntervalMilliseconds,
                "generated vector row order, external ids 0..vectorCount-1",
                "source HNSW saved to the requested private snapshot directory, opened read-only, and searched with caller-owned buffers/workspace",
                "Actual samples are whole-process boundary samples. Peak values are observed sampled whole-process peaks. Layout lower bounds and durable file bytes are separate sections."),
            new HnswConfigurationInfo(
                options.M,
                MMax: options.M,
                MMax0: checked(options.M * 2),
                options.EfConstruction,
                options.EfSearch,
                FormatHex(options.HnswSeed),
                "generated vector row order, external ids 0..vectorCount-1",
                $"{options.Metric} memory-smoke metric"),
            actualMemory,
            peakMemory,
            lowerBounds,
            storageSize,
            new HnswMemorySmokeValidationInfo(
                validationPassed ? "passed" : "failed",
                "generated-hnsw-memory-smoke",
                FiniteVectors: true,
                SourceHnswBuilt: true,
                SourceWarmSearchExecuted: sourceResults.Length == options.QueryCount,
                SourceHnswSaved: storageSize.TotalBytes > 0,
                OpenedHnswOpened: openedIndex.Count == options.VectorCount,
                OpenedIndexReadOnly: readOnlyMutation.Status == "passed",
                OpenedWarmSearchExecuted: openedResults.Length == options.QueryCount,
                SourceOpenedParityChecked: true,
                parity,
                sourceIntegrity,
                openedIntegrity,
                ActualPeakLowerBoundAndStorageSectionsSeparated: true,
                UnsupportedFieldsExplicitlyMarked: true,
                WorkingSetContextOnly: true,
                SampledPeakLabelsPresent: true,
                PublicClaimEligible: false,
                PreviewReadinessEligible: false,
                BaselineCandidateEligible: false,
                ComparisonArtifactEligible: false,
                RegressionGateEligible: false,
                ReportIsPrivateRaw: true),
            CreateEligibility(),
            [
                "Private generated HNSW memory smoke evidence only; not a public memory, capacity, storage-size, latency, recall, allocation, package, platform or preview-readiness claim.",
                "Actual memory samples and sampled peaks are whole-process local samples and include generated inputs, runner objects, runtime state and coexisting source/opened indexes.",
                "Build, save and open peak fields are observed sampled peaks; short-lived peaks between samples can be missed and values are not true maxima.",
                "Working set and process peak working set are OS/cache-sensitive context only and are not VecNet retained-memory claims.",
                "layoutLowerBounds contains payload floors only and is separate from actualMemory and peakMemory.",
                "storageSize contains final durable file facts only and is separate from memory measurements and lower-bound estimates.",
                "Object-accurate graph/dictionary memory, object headers, array slack, index-only private bytes, opened-only retained memory, save/open allocation and peak temporary disk remain notMeasured or notAvailable.",
                "Public claim, preview-readiness, baseline-candidate, comparison-artifact and regression-gate eligibility are false."
            ]);
    }

    public static void Write(HnswMemorySmokeReport report, string outputPath) =>
        ReportWriter.WriteJson(report, outputPath);

    private static GeneratedExactSearchOptions ToGeneratedOptions(HnswMemorySmokeOptions options) =>
        new(
            options.Metric,
            options.Dimension,
            options.VectorCount,
            options.QueryCount,
            options.TopK,
            options.Seed,
            options.OutputPath,
            BaselineReportId: null,
            Runs: 1,
            options.WarmupQueries);

    private static HnswIndex BuildIndex(HnswMemorySmokeOptions options, GeneratedDataset dataset)
    {
        var hnswOptions = new HnswIndexOptions(options.M, options.EfConstruction, options.EfSearch, options.HnswSeed);
        var index = new HnswIndex(options.Dimension, options.Metric, hnswOptions);
        for (int row = 0; row < dataset.VectorCount; row++)
        {
            index.Add((ulong)row, dataset.GetVector(row));
        }

        return index;
    }

    private static SearchResult[][] WarmSearch(HnswMemorySmokeOptions options, GeneratedDataset dataset, HnswIndex index)
    {
        int searchCount = Math.Max(options.QueryCount, options.WarmupQueries);
        var results = new SearchResult[options.TopK];
        var workspace = new HnswSearchWorkspace(options.VectorCount, options.EfSearch);
        var captured = new SearchResult[options.QueryCount][];

        for (int i = 0; i < searchCount; i++)
        {
            int queryRow = i % options.QueryCount;
            int written = index.Search(dataset.GetQuery(queryRow), results, workspace);
            if (i < options.QueryCount)
            {
                var queryResults = new SearchResult[written];
                results.AsSpan(0, written).CopyTo(queryResults);
                captured[queryRow] = queryResults;
            }
        }

        return captured;
    }

    private static HnswMemoryPeakOperationInfo SampleOperation(
        string name,
        int sampleIntervalMilliseconds,
        MemorySnapshot baseline,
        string timedScope,
        string excludedOperations,
        Action operation)
    {
        using var sampler = new ProcessMemorySampler(name, sampleIntervalMilliseconds);
        sampler.Start();
        operation();
        ProcessMemorySamplerResult result = sampler.Stop();

        return new HnswMemoryPeakOperationInfo(
            name,
            "sampled",
            sampleIntervalMilliseconds,
            result.SampleCount,
            CreateSample(name + "Start", "Whole-process sample immediately before " + timedScope + ".", result.Start, baseline),
            CreateSample(name + "End", "Whole-process sample immediately after " + timedScope + ".", result.End, baseline),
            SampledPeak(result.Peak.ManagedHeapSizeBytes, baseline.ManagedHeapSizeBytes, contextOnly: false, "Highest sampled GC heap size during " + name + "; observed sampled peak, not a true maximum."),
            SampledPeak(result.Peak.GcCommittedBytes, baseline.GcCommittedBytes, contextOnly: false, "Highest sampled GC committed bytes during " + name + "; observed sampled peak, not a true maximum."),
            SampledPeak(result.Peak.ProcessPrivateBytes, baseline.ProcessPrivateBytes, contextOnly: false, "Highest sampled Process.PrivateMemorySize64 during " + name + "; whole-process observed sampled peak, not index-only attribution."),
            SampledPeak(result.Peak.ProcessWorkingSetBytes, baseline.ProcessWorkingSetBytes, contextOnly: true, "Highest sampled Process.WorkingSet64 during " + name + "; OS/cache-sensitive context-only observed sampled peak."),
            "Sampling can miss short-lived peaks between samples; this field is an observed sampled peak, not a mathematical maximum.",
            "Samples are whole-process values and cannot attribute bytes only to VecNet HNSW structures.",
            timedScope,
            excludedOperations);
    }

    private static HnswMemorySmokeActualMemoryInfo CreateActualMemory(
        HnswMemorySampleInfo baseline,
        HnswMemorySampleInfo postDataset,
        HnswMemorySampleInfo postBuild,
        HnswMemorySampleInfo postSourceSearch,
        HnswMemorySampleInfo postSave,
        HnswMemorySampleInfo postOpen,
        HnswMemorySampleInfo postOpenedSearch,
        HnswMemorySampleInfo postValidation) =>
        new(
            "measured",
            "wholeProcessBoundarySamples",
            "Samples use GC.GetGCMemoryInfo and Process.GetCurrentProcess after retained-state boundaries. No forced full-GC stabilization is applied.",
            "Actual samples are separate from observed sampled peaks, payload lower-bound estimates and durable file-size facts.",
            baseline,
            postDataset,
            postBuild,
            postSourceSearch,
            postSave,
            postOpen,
            postOpenedSearch,
            postValidation,
            new HnswMemoryUnsupportedInfo(
                NotAvailable("bytes", "Object-accurate Dictionary<ulong,int> retained bytes, buckets, entries, object headers and slack capacity are not exposed by HnswIndex."),
                NotAvailable("bytes", "Object-accurate HnswGraphLayer object/header/alignment retained bytes are not exposed by HnswIndex."),
                NotAvailable("bytes", "Managed object headers, array headers, alignment and backing-array slack cannot be attributed by VecNet structure in this runner."),
                NotAvailable("bytes", "NeighborCandidate array element/object layout is not reported as an object-accurate retained-memory value."),
                NotMeasured("bytes", "Index-only private bytes are not measured because generated inputs, result arrays, runner objects and runtime state coexist in the same process."),
                NotMeasured("bytes", "Opened-only retained memory is not measured because the source index, opened index and generated inputs coexist in this runner process."),
                NotMeasured("bytesPerSaveCall", "Managed allocation for HnswIndex.Save(directoryPath) is not measured by this memory smoke mode."),
                NotMeasured("bytesPerOpenCall", "Managed allocation for HnswIndex.OpenReadOnly(directoryPath) is not measured by this memory smoke mode."),
                NotMeasured("bytes", "True process peak memory is not measured; sampled peak fields can miss short-lived peaks between samples."),
                NotMeasured("bytes", "Peak temporary disk usage is not measured because active directory/temp-file sampling is not implemented.")),
            [
                "Whole-process samples cannot isolate VecNet index-only retained private bytes from generated input arrays, result buffers, validation arrays, runner objects or runtime state.",
                "Working set and process peak working set are OS/cache-sensitive context only.",
                "GC committed and fragmented values are runtime counters, not object-accurate VecNet retained memory attribution.",
                "Source and opened indexes intentionally coexist after OpenReadOnly in this single-process smoke."
            ]);

    private static HnswMemorySmokePeakMemoryInfo CreatePeakMemory(
        HnswMemorySmokeOptions options,
        HnswMemoryPeakOperationInfo build,
        HnswMemoryPeakOperationInfo save,
        HnswMemoryPeakOperationInfo open) =>
        new(
            "sampled",
            "observedSampledWholeProcessPeaks",
            "Peak memory is actively sampled whole-process process/GC memory for build, save and open only. It is not index-only attribution and not a true maximum.",
            build,
            save,
            open,
            NotMeasured("bytes", "Search warmup peak memory is not actively sampled in VEC-113; retained boundary samples are reported under actualMemory."),
            NotMeasured("bytes", "Opened search warmup peak memory is not actively sampled in VEC-113; retained boundary samples are reported under actualMemory."),
            NotMeasured("bytes", "Peak temporary disk is not measured; final durable snapshot bytes are reported under storageSize only."),
            [
                string.Create(CultureInfo.InvariantCulture, $"Build/save/open are sampled every {options.SampleIntervalMilliseconds} ms plus explicit start/end samples."),
                "Observed sampled peaks can miss short-lived allocations between samples.",
                "Working-set peaks are context-only and OS/cache-sensitive.",
                "Peak values are whole-process values and include runner/runtime state."
            ]);

    private static HnswMemorySmokeLayoutLowerBoundsInfo CreateLowerBounds(HnswMemorySmokeOptions options, HnswIndex index)
    {
        int layerCount = Math.Max(0, index.MaxLayer + 1);
        var layers = new HnswMemoryLayerLowerBoundInfo[layerCount];
        long graphCountBytes = 0;
        long graphNeighborBytes = 0;
        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = layer == 0 ? checked(options.M * 2) : options.M;
            long layerCountBytes = checked((long)options.VectorCount * sizeof(int));
            long layerNeighborBytes = checked((long)options.VectorCount * stride * sizeof(int));
            layers[layer] = new HnswMemoryLayerLowerBoundInfo(layer, stride, layerCountBytes, layerNeighborBytes);
            graphCountBytes = checked(graphCountBytes + layerCountBytes);
            graphNeighborBytes = checked(graphNeighborBytes + layerNeighborBytes);
        }

        long vectorBytes = checked((long)options.VectorCount * options.Dimension * sizeof(float));
        long idBytes = checked((long)options.VectorCount * sizeof(ulong));
        long levelBytes = checked((long)options.VectorCount * sizeof(int));
        long graphBytes = checked(graphCountBytes + graphNeighborBytes);
        long idMapEntryBytes = checked((long)options.VectorCount * (sizeof(ulong) + sizeof(int)));
        long searchWorkspaceBytes = EstimateWorkspaceBytes(options.VectorCount, options.EfSearch);
        long buildScratchBytes = checked(
            EstimateWorkspaceBytes(options.VectorCount, options.EfConstruction) +
            ((long)Math.Max(options.EfConstruction, (options.M * 2) + 1) * sizeof(int)) +
            ((long)options.M * sizeof(int)));
        long openedRetained = checked(vectorBytes + idBytes + levelBytes + graphBytes + idMapEntryBytes);
        long sourceRetained = checked(openedRetained + buildScratchBytes);

        return new HnswMemorySmokeLayoutLowerBoundsInfo(
            "estimatedLowerBound",
            "payload-only; not actual retained memory",
            vectorBytes,
            idBytes,
            levelBytes,
            graphCountBytes,
            graphNeighborBytes,
            graphBytes,
            idMapEntryBytes,
            new MeasurementStatusInfo(
                "estimatedLowerBound",
                searchWorkspaceBytes.ToString(CultureInfo.InvariantCulture),
                "bytes",
                "Caller-owned HnswSearchWorkspace array payload floor for vectorCount and efSearch; excludes object/array headers, alignment and slack."),
            new MeasurementStatusInfo(
                "estimatedLowerBound",
                buildScratchBytes.ToString(CultureInfo.InvariantCulture),
                "bytes",
                "Retained build scratch payload floor for workspace plus candidate/selected ordinal int arrays; excludes NeighborCandidate layout, object/array headers, alignment and slack."),
            sourceRetained,
            openedRetained,
            layers,
            "Excludes managed object headers, array headers, Dictionary buckets/entries/capacity overhead, graph layer object overhead, backing-array slack from growth, NeighborCandidate layout, generated input arrays, captured validation results, JSON serialization objects, runtime/JIT state, process fragmentation and temporary save/open copies except where observed by sampled peak fields.");
    }

    private static HnswMemorySmokeStorageSizeInfo InspectStorageSize(string snapshotDirectory, int vectorCount)
    {
        long manifestBytes = FileLength(snapshotDirectory, ManifestFileName);
        long idsBytes = FileLength(snapshotDirectory, IdsFileName);
        long vectorsBytes = FileLength(snapshotDirectory, VectorsFileName);
        long levelsBytes = FileLength(snapshotDirectory, LevelsFileName);
        long graphBytes = FileLength(snapshotDirectory, GraphFileName);
        long total = checked(manifestBytes + idsBytes + vectorsBytes + levelsBytes + graphBytes);

        return new HnswMemorySmokeStorageSizeInfo(
            "fileFacts",
            "Final durable HNSW snapshot file lengths scanned after successful Save and outside save/open peak sampling.",
            "private ignored benchmark-runner artifact path",
            snapshotDirectory,
            Directory.Exists(snapshotDirectory) ? Directory.EnumerateFiles(snapshotDirectory).Count() : 0,
            total,
            manifestBytes,
            idsBytes,
            vectorsBytes,
            levelsBytes,
            graphBytes,
            vectorCount == 0 ? 0 : (double)total / vectorCount,
            "Directory scan occurs after Save and OpenReadOnly; it is outside save/open timing and outside peak memory sampling.",
            NotMeasured("bytes", "Output directory bytes are scanned after Save only; active directory-size sampling during Save is not implemented."),
            NotMeasured("bytes", "Peak temporary disk bytes are not measured and are not inferred from final snapshot bytes."));
    }

    private static DurableHnswReadOnlyMutationInfo ValidateOpenedReadOnlyMutation(HnswIndex openedIndex, int dimension)
    {
        try
        {
            openedIndex.Add(ulong.MaxValue, new float[dimension]);
        }
        catch (InvalidOperationException ex)
        {
            return new DurableHnswReadOnlyMutationInfo(
                "passed",
                ex.GetType().Name,
                RejectedBeforeVectorValidation: true,
                "Add(ulong.MaxValue, zeroVector)",
                "Opened HNSW index rejected mutation before accepting new vector data.");
        }

        return new DurableHnswReadOnlyMutationInfo(
            "failed",
            "none",
            RejectedBeforeVectorValidation: false,
            "Add(ulong.MaxValue, zeroVector)",
            "Opened HNSW index accepted a mutation unexpectedly.");
    }

    private static DurableHnswParityInfo CompareSourceOpenedParity(SearchResult[][] source, SearchResult[][] opened)
    {
        int queryCount = Math.Min(source.Length, opened.Length);
        int writtenMismatch = source.Length == opened.Length ? 0 : 1;
        int idMismatch = 0;
        int orderMismatch = 0;
        int distanceMismatch = 0;

        for (int query = 0; query < queryCount; query++)
        {
            SearchResult[] left = source[query];
            SearchResult[] right = opened[query];
            if (left.Length != right.Length)
            {
                writtenMismatch++;
            }

            int resultCount = Math.Min(left.Length, right.Length);
            for (int i = 0; i < resultCount; i++)
            {
                if (left[i].Id != right[i].Id)
                {
                    idMismatch++;
                }

                if (left[i].Id != right[i].Id || left[i].Distance != right[i].Distance)
                {
                    orderMismatch++;
                }

                if (left[i].Distance != right[i].Distance)
                {
                    distanceMismatch++;
                }
            }
        }

        bool matched = writtenMismatch == 0 && idMismatch == 0 && orderMismatch == 0 && distanceMismatch == 0;
        return new DurableHnswParityInfo(
            queryCount,
            writtenMismatch,
            idMismatch,
            orderMismatch,
            distanceMismatch,
            matched,
            "Source and opened HNSW searches run over the same deterministic generated queries with independent caller-owned result buffers and HnswSearchWorkspace instances; exact ID order and distance equality are required for durable parity.");
    }

    private static HnswMemorySampleInfo CreateSample(string name, string boundary, MemorySnapshot current, MemorySnapshot baseline) =>
        new(
            name,
            boundary,
            Measured(current.ManagedHeapSizeBytes, baseline.ManagedHeapSizeBytes, contextOnly: false, "GC.GetGCMemoryInfo().HeapSizeBytes at sample boundary."),
            Measured(current.GcCommittedBytes, baseline.GcCommittedBytes, contextOnly: false, "GC.GetGCMemoryInfo().TotalCommittedBytes at sample boundary where exposed by the runtime."),
            Measured(current.GcFragmentedBytes, baseline.GcFragmentedBytes, contextOnly: false, "GC.GetGCMemoryInfo().FragmentedBytes at sample boundary where exposed by the runtime."),
            Measured(current.ProcessPrivateBytes, baseline.ProcessPrivateBytes, contextOnly: false, "Process.PrivateMemorySize64 at sample boundary; whole-process local value, not index-only attribution."),
            Measured(current.ProcessWorkingSetBytes, baseline.ProcessWorkingSetBytes, contextOnly: true, "Process.WorkingSet64 at sample boundary; OS/cache-sensitive context only, not a retained-memory claim."),
            Measured(current.ProcessPeakWorkingSetBytes, baseline.ProcessPeakWorkingSetBytes, contextOnly: true, "Process.PeakWorkingSet64 at sample boundary; process-lifetime OS/cache-sensitive context only."));

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

    private static HnswMemoryMetricInfo Measured(long value, long baseline, bool contextOnly, string reason) =>
        new("measured", value, value - baseline, "bytes", contextOnly, reason);

    private static HnswMemoryMetricInfo SampledPeak(long value, long baseline, bool contextOnly, string reason) =>
        new("sampled", value, value - baseline, "bytes", contextOnly, reason);

    private static MeasurementStatusInfo NotMeasured(string unit, string reason) =>
        new("notMeasured", "absent", unit, reason);

    private static MeasurementStatusInfo NotAvailable(string unit, string reason) =>
        new("notAvailable", "absent", unit, reason);

    private static long EstimateWorkspaceBytes(int maxElements, int maxEf) =>
        checked(
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(int)) +
            ((long)maxElements * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)) +
            ((long)maxEf * sizeof(int)) +
            ((long)maxEf * sizeof(float)));

    private static long FileLength(string directoryPath, string fileName)
    {
        string path = Path.Combine(directoryPath, fileName);
        return File.Exists(path) ? new FileInfo(path).Length : 0;
    }

    private static HnswMemorySmokeEvidenceInfo CreateEvidence() =>
        new(
            "smoke",
            "generated-hnsw-memory-smoke",
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW memory smoke output is not reviewed public evidence and has no public reporting policy.",
            "One private local generated memory smoke does not establish HNSW preview readiness.",
            "No generated HNSW memory baseline-candidate policy is accepted.",
            "Generated HNSW memory smoke reports are not accepted comparison artifacts.",
            "No generated HNSW memory regression-gate policy, threshold, comparison artifact or hard gate is accepted.",
            [
                "Generated squared-L2 or inner-product HNSW memory smoke evidence only; no external dataset source, license, version or checksum applies.",
                "Actual samples are local whole-process boundary samples and are separated from sampled peaks, lower-bound layout estimates and durable file facts.",
                "Build/save/open peaks are observed sampled peaks, not true maxima.",
                "Working set is OS/cache-sensitive context only.",
                "Not a public claim, preview-readiness result, baseline candidate, comparison artifact, regression gate, Linux validation or BenchmarkDotNet-grade evidence."
            ]);

    private static HnswMemorySmokeEligibilityInfo CreateEligibility() =>
        new(
            PublicClaimEligible: false,
            PreviewReadinessEligible: false,
            BaselineCandidateEligible: false,
            ComparisonArtifactEligible: false,
            RegressionGateEligible: false,
            "Private generated HNSW memory smoke output is not reviewed public evidence.",
            "This private local smoke report does not establish public HNSW API/package preview readiness.",
            "No HNSW memory baseline-candidate policy is accepted.",
            "No HNSW memory comparison-artifact policy is accepted.",
            "No HNSW memory regression-gate policy, threshold, comparison artifact or hard gate is accepted.");

    private static void ValidateOptions(HnswMemorySmokeOptions options)
    {
        if (options.Metric is not (VectorMetric.SquaredEuclidean or VectorMetric.InnerProduct))
        {
            throw new ArgumentException("generated-hnsw-memory-smoke supports only SquaredEuclidean and InnerProduct.", nameof(options));
        }

        if (options.TopK > options.VectorCount)
        {
            throw new ArgumentException("top-k must be less than or equal to the vector count.", nameof(options));
        }

        if (options.WarmupQueries < 0)
        {
            throw new ArgumentException("warmup queries must be non-negative.", nameof(options));
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

        if (options.SampleIntervalMilliseconds is < 1 or > 1000)
        {
            throw new ArgumentException("sample interval must be in the range 1..1000 milliseconds.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.OutputPath))
        {
            throw new ArgumentException("output path must not be empty.", nameof(options));
        }

        if (string.IsNullOrWhiteSpace(options.SnapshotDirectory))
        {
            throw new ArgumentException("snapshot directory must not be empty.", nameof(options));
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

    private static string CreateReportId(string? commit, HnswMemorySmokeOptions options)
    {
        string commitPart = string.IsNullOrWhiteSpace(commit) ? "unknown" : commit[..Math.Min(12, commit.Length)];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{HnswMemorySmokeOptions.ScenarioName}-{commitPart}-{options.Dimension}d-{options.VectorCount}v-{options.QueryCount}q-{options.TopK}k-m{options.M}-efc{options.EfConstruction}-efs{options.EfSearch}-{options.Seed:X8}-{options.HnswSeed:X16}");
    }

    private static string FormatHex(ulong value) => string.Create(CultureInfo.InvariantCulture, $"0x{value:X16}");

    private sealed record MemorySnapshot(
        long ManagedHeapSizeBytes,
        long GcCommittedBytes,
        long GcFragmentedBytes,
        long ProcessPrivateBytes,
        long ProcessWorkingSetBytes,
        long ProcessPeakWorkingSetBytes);

    private sealed record ProcessMemorySamplerResult(
        MemorySnapshot Start,
        MemorySnapshot End,
        MemorySnapshot Peak,
        int SampleCount);

    private sealed class ProcessMemorySampler : IDisposable
    {
        private readonly string _name;
        private readonly int _intervalMilliseconds;
        private readonly List<MemorySnapshot> _samples = [];
        private readonly object _gate = new();
        private volatile bool _stopRequested;
        private Thread? _thread;

        internal ProcessMemorySampler(string name, int intervalMilliseconds)
        {
            _name = name;
            _intervalMilliseconds = intervalMilliseconds;
        }

        internal void Start()
        {
            lock (_gate)
            {
                _samples.Add(CaptureMemorySnapshot());
            }

            _thread = new Thread(Run)
            {
                IsBackground = true,
                Name = "VecNet.HnswMemorySmoke." + _name
            };
            _thread.Start();
        }

        internal ProcessMemorySamplerResult Stop()
        {
            _stopRequested = true;
            _thread?.Join();
            lock (_gate)
            {
                _samples.Add(CaptureMemorySnapshot());
                MemorySnapshot start = _samples[0];
                MemorySnapshot end = _samples[^1];
                MemorySnapshot peak = new(
                    _samples.Max(sample => sample.ManagedHeapSizeBytes),
                    _samples.Max(sample => sample.GcCommittedBytes),
                    _samples.Max(sample => sample.GcFragmentedBytes),
                    _samples.Max(sample => sample.ProcessPrivateBytes),
                    _samples.Max(sample => sample.ProcessWorkingSetBytes),
                    _samples.Max(sample => sample.ProcessPeakWorkingSetBytes));
                return new ProcessMemorySamplerResult(start, end, peak, _samples.Count);
            }
        }

        public void Dispose()
        {
            if (_thread is not null && _thread.IsAlive)
            {
                _stopRequested = true;
                _thread.Join();
            }
        }

        private void Run()
        {
            while (!_stopRequested)
            {
                Thread.Sleep(_intervalMilliseconds);
                MemorySnapshot sample = CaptureMemorySnapshot();
                lock (_gate)
                {
                    _samples.Add(sample);
                }
            }
        }
    }
}
