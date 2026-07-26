using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;

namespace VecNet;

internal static class HnswIndexStorage
{
    internal const string ManifestFileName = "hnsw.manifest.json";
    internal const string IdsFileName = "hnsw.ids.u64";
    internal const string VectorsFileName = "hnsw.vectors.f32";
    internal const string LevelsFileName = "hnsw.levels.i32";
    internal const string GraphFileName = "hnsw.graph.bin";

    internal const string ManifestSchemaName = "VecNet.HnswIndexManifest";
    internal const string ManifestSchemaVersion = "1.0";
    internal const string FormatFamily = "hnsw";

    internal const string IdsMagicText = "VNETHI01";
    internal const string VectorsMagicText = "VNETHV01";
    internal const string LevelsMagicText = "VNETHL01";
    internal const string GraphMagicText = "VNETHG01";
    internal const ushort BinaryMajorVersion = 1;
    internal const ushort BinaryMinorVersion = 0;
    internal const int IdsHeaderLength = 32;
    internal const int VectorsHeaderLength = 48;
    internal const int LevelsHeaderLength = 32;
    internal const int GraphHeaderLength = 64;
    internal const int GraphLayerDirectoryEntryLength = 48;
    internal const uint SquaredEuclideanMetricCode = 1;
    internal const uint InnerProductMetricCode = 2;
    internal const uint CosineMetricCode = 3;
    internal const uint NoNormalizationCode = 0;
    internal const uint CosineUnitNormalizedCode = 1;
    internal const double CosineStoredRowSquaredLengthTolerance = 1e-4;

    private const int MinM = 2;
    private const int MaxM = 64;
    private const int MaxEf = 4096;
    private const int MaxManifestBytes = 1024 * 1024;
    private const string CreatedByTask = "VEC-176";
    private const string BinaryVersionText = "1.0";
    private const string GraphAdjacencyLayout = "dense-layer0-sparse-upper-v1";
    private const string NormalizationNone = "none";
    private const string NormalizationCosineUnit = "cosine-unit-normalized";
    private const uint GraphLayerLayoutDense = 0;
    private const uint GraphLayerLayoutSparse = 1;

    private static readonly byte[] IdsMagic = "VNETHI01"u8.ToArray();
    private static readonly byte[] VectorsMagic = "VNETHV01"u8.ToArray();
    private static readonly byte[] LevelsMagic = "VNETHL01"u8.ToArray();
    private static readonly byte[] GraphMagic = "VNETHG01"u8.ToArray();

    internal static void Save(string directoryPath, HnswIndex index)
    {
        ArgumentNullException.ThrowIfNull(index);

        string directory = PrepareSaveDirectory(directoryPath, out bool createdDirectory);
        string tempSuffix = ".tmp-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);
        string idsTempPath = Path.Combine(directory, IdsFileName + tempSuffix);
        string vectorsTempPath = Path.Combine(directory, VectorsFileName + tempSuffix);
        string levelsTempPath = Path.Combine(directory, LevelsFileName + tempSuffix);
        string graphTempPath = Path.Combine(directory, GraphFileName + tempSuffix);
        string manifestTempPath = Path.Combine(directory, ManifestFileName + tempSuffix);

        try
        {
            WriteIdsFile(idsTempPath, index.InternalIds);
            WriteVectorsFile(vectorsTempPath, index.Dimension, index.Metric, index.Count, index.InternalVectors);
            WriteLevelsFile(levelsTempPath, index.InternalLevels);
            WriteGraphFile(graphTempPath, index);

            var idsMetadata = CreateBinaryFileMetadata(idsTempPath, IdsFileName, IdsMagicText);
            var vectorsMetadata = CreateBinaryFileMetadata(vectorsTempPath, VectorsFileName, VectorsMagicText);
            var levelsMetadata = CreateBinaryFileMetadata(levelsTempPath, LevelsFileName, LevelsMagicText);
            var graphMetadata = CreateBinaryFileMetadata(graphTempPath, GraphFileName, GraphMagicText);
            WriteManifest(manifestTempPath, index, idsMetadata, vectorsMetadata, levelsMetadata, graphMetadata);

            File.Move(idsTempPath, Path.Combine(directory, IdsFileName));
            File.Move(vectorsTempPath, Path.Combine(directory, VectorsFileName));
            File.Move(levelsTempPath, Path.Combine(directory, LevelsFileName));
            File.Move(graphTempPath, Path.Combine(directory, GraphFileName));
            File.Move(manifestTempPath, Path.Combine(directory, ManifestFileName));
        }
        catch
        {
            TryDelete(idsTempPath);
            TryDelete(vectorsTempPath);
            TryDelete(levelsTempPath);
            TryDelete(graphTempPath);
            TryDelete(manifestTempPath);
            if (createdDirectory)
            {
                TryDeleteDirectoryIfEmpty(directory);
            }

            throw;
        }
    }

    internal static HnswIndex OpenReadOnly(string directoryPath)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        string directory = Path.GetFullPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("HNSW index path is an existing file, not a directory.");
        }

        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"HNSW index directory was not found: {directoryPath}");
        }

        string manifestPath = Path.Combine(directory, ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            throw new FileNotFoundException("HNSW index manifest was not found.", manifestPath);
        }

        Manifest manifest = ReadManifest(manifestPath);
        string idsPath = ResolveManifestFilePath(directory, manifest.IdsFile.RelativePath, IdsFileName);
        string vectorsPath = ResolveManifestFilePath(directory, manifest.VectorsFile.RelativePath, VectorsFileName);
        string levelsPath = ResolveManifestFilePath(directory, manifest.LevelsFile.RelativePath, LevelsFileName);
        string graphPath = ResolveManifestFilePath(directory, manifest.GraphFile.RelativePath, GraphFileName);

        ValidateFileExistsLengthAndHash(idsPath, manifest.IdsFile, "ID");
        ValidateFileExistsLengthAndHash(vectorsPath, manifest.VectorsFile, "vector");
        ValidateFileExistsLengthAndHash(levelsPath, manifest.LevelsFile, "level");
        ValidateFileExistsLengthAndHash(graphPath, manifest.GraphFile, "graph");

        ulong[] ids = ReadIdsFile(idsPath, manifest.VectorCount);
        float[] vectors = ReadVectorsFile(vectorsPath, manifest.Dimension, manifest.Metric, manifest.VectorCount);
        int[] levels = ReadLevelsFile(levelsPath, manifest.VectorCount, manifest.MaxLayer);
        HnswIndex.HnswLayerSnapshot[] layers = ReadGraphFile(graphPath, manifest);

        ValidateRowsAndGraph(ids, vectors, levels, layers, manifest);

        var snapshot = new HnswIndex.HnswStorageSnapshot(
            manifest.Dimension,
            manifest.Metric,
            manifest.Options,
            manifest.MMax,
            manifest.MMax0,
            manifest.LevelMultiplier,
            manifest.EntryPoint,
            manifest.MaxLayer,
            ids,
            vectors,
            levels,
            layers);
        return HnswIndex.HydrateReadOnly(snapshot);
    }

    internal static void ValidateSavedIndex(string directoryPath, HnswIndex expected)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        ArgumentNullException.ThrowIfNull(expected);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        string directory = Path.GetFullPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("HNSW index path is an existing file, not a directory.");
        }

        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"HNSW index directory was not found: {directoryPath}");
        }

        string manifestPath = Path.Combine(directory, ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            throw new FileNotFoundException("HNSW index manifest was not found.", manifestPath);
        }

        Manifest manifest = ReadManifest(manifestPath);
        ValidateManifestMatchesExpected(manifest, expected);

        string idsPath = ResolveManifestFilePath(directory, manifest.IdsFile.RelativePath, IdsFileName);
        string vectorsPath = ResolveManifestFilePath(directory, manifest.VectorsFile.RelativePath, VectorsFileName);
        string levelsPath = ResolveManifestFilePath(directory, manifest.LevelsFile.RelativePath, LevelsFileName);
        string graphPath = ResolveManifestFilePath(directory, manifest.GraphFile.RelativePath, GraphFileName);

        ValidateFileExistsLengthAndHash(idsPath, manifest.IdsFile, "ID");
        ValidateFileExistsLengthAndHash(vectorsPath, manifest.VectorsFile, "vector");
        ValidateFileExistsLengthAndHash(levelsPath, manifest.LevelsFile, "level");
        ValidateFileExistsLengthAndHash(graphPath, manifest.GraphFile, "graph");

        ValidateIdsFileMatches(idsPath, expected.InternalIds);
        ValidateVectorsFileMatches(vectorsPath, expected.Dimension, expected.Metric, expected.Count, expected.InternalVectors);
        ValidateLevelsFileMatches(levelsPath, expected.InternalLevels, expected.MaxLayer);
        ValidateGraphFileMatches(graphPath, manifest, expected);
    }

    private static void ValidateManifestMatchesExpected(Manifest manifest, HnswIndex expected)
    {
        int expectedLayerCount = expected.Count == 0 ? 0 : expected.MaxLayer + 1;
        if (manifest.Dimension != expected.Dimension ||
            manifest.Metric != expected.Metric ||
            manifest.VectorCount != expected.Count ||
            manifest.Options != expected.Options ||
            manifest.MMax != expected.InternalMMax ||
            manifest.MMax0 != expected.InternalMMax0 ||
            Math.Abs(manifest.LevelMultiplier - expected.InternalLevelMultiplier) > 1e-12 ||
            manifest.EntryPoint != expected.EntryPoint ||
            manifest.MaxLayer != expected.MaxLayer ||
            manifest.LayerCount != expectedLayerCount ||
            manifest.Layer0Stride != expected.InternalMMax0 ||
            manifest.UpperLayerStride != expected.InternalMMax)
        {
            throw new InvalidDataException("HNSW checkpoint manifest does not match the rebuilt base.");
        }
    }

    private static void ValidateIdsFileMatches(string path, ReadOnlySpan<ulong> expectedIds)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        ValidateIdsFileHeader(stream, expectedIds.Length);

        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        for (int i = 0; i < expectedIds.Length; i++)
        {
            stream.ReadExactly(buffer);
            if (BinaryPrimitives.ReadUInt64LittleEndian(buffer) != expectedIds[i])
            {
                throw new InvalidDataException("HNSW checkpoint ID payload does not match the rebuilt base.");
            }
        }
    }

    private static void ValidateVectorsFileMatches(
        string path,
        int expectedDimension,
        VectorMetric expectedMetric,
        int expectedRowCount,
        ReadOnlySpan<float> expectedVectors)
    {
        if (expectedVectors.Length != checked(expectedRowCount * expectedDimension))
        {
            throw new InvalidOperationException("HNSW expected vector payload length is invalid.");
        }

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        ValidateVectorsFileHeader(stream, expectedDimension, expectedMetric, expectedRowCount);

        Span<byte> buffer = stackalloc byte[sizeof(float)];
        for (int i = 0; i < expectedVectors.Length; i++)
        {
            stream.ReadExactly(buffer);
            float actual = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(buffer));
            if (BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expectedVectors[i]))
            {
                throw new InvalidDataException("HNSW checkpoint vector payload does not match the rebuilt base.");
            }
        }
    }

    private static void ValidateLevelsFileMatches(
        string path,
        ReadOnlySpan<int> expectedLevels,
        int expectedMaxLayer)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        ValidateLevelsFileHeader(stream, expectedLevels.Length);

        Span<byte> buffer = stackalloc byte[sizeof(int)];
        bool hasMaxLayer = expectedLevels.Length == 0;
        for (int i = 0; i < expectedLevels.Length; i++)
        {
            stream.ReadExactly(buffer);
            int actual = BinaryPrimitives.ReadInt32LittleEndian(buffer);
            if (actual != expectedLevels[i] || actual < 0 || actual > expectedMaxLayer)
            {
                throw new InvalidDataException("HNSW checkpoint level payload does not match the rebuilt base.");
            }

            hasMaxLayer |= actual == expectedMaxLayer;
        }

        if (!hasMaxLayer)
        {
            throw new InvalidDataException("HNSW checkpoint levels do not contain the max layer.");
        }
    }

    private static void ValidateGraphFileMatches(string path, Manifest manifest, HnswIndex expected)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < GraphHeaderLength)
        {
            throw new InvalidDataException("HNSW index graph file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[GraphHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(GraphMagic))
        {
            throw new InvalidDataException("HNSW index graph file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "graph");
        RequireHeaderLength(header[12..], GraphHeaderLength, "graph");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        uint layerCount = BinaryPrimitives.ReadUInt32LittleEndian(header[24..]);
        int entryPoint = BinaryPrimitives.ReadInt32LittleEndian(header[28..]);
        int maxLayer = BinaryPrimitives.ReadInt32LittleEndian(header[32..]);
        uint m = BinaryPrimitives.ReadUInt32LittleEndian(header[36..]);
        uint mMax0 = BinaryPrimitives.ReadUInt32LittleEndian(header[40..]);
        uint mMax = BinaryPrimitives.ReadUInt32LittleEndian(header[44..]);
        ulong directoryOffset = BinaryPrimitives.ReadUInt64LittleEndian(header[48..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[56..]);
        if (rowCount != checked((ulong)manifest.VectorCount) ||
            layerCount != checked((uint)manifest.LayerCount) ||
            entryPoint != manifest.EntryPoint ||
            maxLayer != manifest.MaxLayer ||
            m != checked((uint)manifest.Options.M) ||
            mMax0 != checked((uint)manifest.MMax0) ||
            mMax != checked((uint)manifest.MMax) ||
            directoryOffset != GraphHeaderLength ||
            reserved != 0)
        {
            throw new InvalidDataException("HNSW index graph file header is inconsistent.");
        }

        long directoryLength = checked((long)manifest.LayerCount * GraphLayerDirectoryEntryLength);
        if (stream.Length < GraphHeaderLength + directoryLength)
        {
            throw new InvalidDataException("HNSW index graph layer directory is truncated.");
        }

        var entries = new GraphLayerEntry[manifest.LayerCount];
        Span<byte> entryBuffer = stackalloc byte[GraphLayerDirectoryEntryLength];
        for (int layer = 0; layer < manifest.LayerCount; layer++)
        {
            stream.ReadExactly(entryBuffer);
            uint layerNumber = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer);
            uint stride = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[4..]);
            uint storedRows = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[8..]);
            uint layout = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[12..]);
            long ordinalsOffset = ReadGraphOffset(entryBuffer[16..]);
            long countsOffset = ReadGraphOffset(entryBuffer[24..]);
            long neighborsOffset = ReadGraphOffset(entryBuffer[32..]);
            ulong entryReserved = BinaryPrimitives.ReadUInt64LittleEndian(entryBuffer[40..]);
            int expectedStride = layer == 0 ? manifest.MMax0 : manifest.MMax;
            int expectedStoredRows = layer == 0
                ? manifest.VectorCount
                : CountParticipatingRows(expected.InternalLevels, layer);
            uint expectedLayout = layer == 0 ? GraphLayerLayoutDense : GraphLayerLayoutSparse;
            if (layerNumber != checked((uint)layer) ||
                stride != checked((uint)expectedStride) ||
                storedRows != checked((uint)expectedStoredRows) ||
                layout != expectedLayout ||
                (layer == 0 && ordinalsOffset != 0) ||
                entryReserved != 0)
            {
                throw new InvalidDataException("HNSW checkpoint graph layer directory entry is invalid.");
            }

            entries[layer] = new GraphLayerEntry((int)stride, (int)storedRows, ordinalsOffset, countsOffset, neighborsOffset);
        }

        ValidateGraphRanges(entries, stream.Length);

        Span<byte> intBuffer = stackalloc byte[sizeof(int)];
        for (int layer = 0; layer < manifest.LayerCount; layer++)
        {
            GraphLayerEntry entry = entries[layer];
            if (layer > 0)
            {
                stream.Position = entry.OrdinalsOffset;
                for (int ordinal = 0; ordinal < expected.Count; ordinal++)
                {
                    if (expected.InternalLevels[ordinal] < layer)
                    {
                        continue;
                    }

                    stream.ReadExactly(intBuffer);
                    if (BinaryPrimitives.ReadInt32LittleEndian(intBuffer) != ordinal)
                    {
                        throw new InvalidDataException("HNSW checkpoint compact graph row ordinal is invalid.");
                    }
                }
            }

            stream.Position = entry.CountsOffset;
            for (int ordinal = 0; ordinal < expected.Count; ordinal++)
            {
                if (layer > 0 && expected.InternalLevels[ordinal] < layer)
                {
                    continue;
                }

                stream.ReadExactly(intBuffer);
                int actualCount = BinaryPrimitives.ReadInt32LittleEndian(intBuffer);
                if (actualCount != expected.InternalGetGraphNeighbors(layer, ordinal).Length)
                {
                    throw new InvalidDataException("HNSW checkpoint graph neighbor count does not match the rebuilt base.");
                }
            }

            stream.Position = entry.NeighborsOffset;
            for (int ordinal = 0; ordinal < expected.Count; ordinal++)
            {
                if (layer > 0 && expected.InternalLevels[ordinal] < layer)
                {
                    continue;
                }

                ReadOnlySpan<int> expectedNeighbors = expected.InternalGetGraphNeighbors(layer, ordinal);
                for (int i = 0; i < entry.Stride; i++)
                {
                    stream.ReadExactly(intBuffer);
                    int actual = BinaryPrimitives.ReadInt32LittleEndian(intBuffer);
                    int expectedNeighbor = i < expectedNeighbors.Length ? expectedNeighbors[i] : 0;
                    if (actual != expectedNeighbor)
                    {
                        throw new InvalidDataException("HNSW checkpoint graph neighbor payload does not match the rebuilt base.");
                    }
                }
            }
        }
    }

    private static void ValidateIdsFileHeader(FileStream stream, int expectedRowCount)
    {
        if (stream.Length < IdsHeaderLength)
        {
            throw new InvalidDataException("HNSW index ID file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[IdsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(IdsMagic))
        {
            throw new InvalidDataException("HNSW index ID file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "ID");
        RequireHeaderLength(header[12..], IdsHeaderLength, "ID");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[24..]);
        if (reserved != 0 || rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("HNSW index ID file header is inconsistent.");
        }

        if (stream.Length != checked(IdsHeaderLength + (long)expectedRowCount * sizeof(ulong)))
        {
            throw new InvalidDataException("HNSW index ID file payload length is invalid.");
        }
    }

    private static void ValidateVectorsFileHeader(
        FileStream stream,
        int expectedDimension,
        VectorMetric expectedMetric,
        int expectedRowCount)
    {
        if (stream.Length < VectorsHeaderLength)
        {
            throw new InvalidDataException("HNSW index vector file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[VectorsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(VectorsMagic))
        {
            throw new InvalidDataException("HNSW index vector file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "vector");
        RequireHeaderLength(header[12..], VectorsHeaderLength, "vector");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        uint dimension = BinaryPrimitives.ReadUInt32LittleEndian(header[24..]);
        uint representation = BinaryPrimitives.ReadUInt32LittleEndian(header[28..]);
        uint metric = BinaryPrimitives.ReadUInt32LittleEndian(header[32..]);
        uint normalization = BinaryPrimitives.ReadUInt32LittleEndian(header[36..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[40..]);
        if (rowCount != checked((ulong)expectedRowCount) ||
            dimension != checked((uint)expectedDimension) ||
            representation != 1 ||
            metric != GetMetricCode(expectedMetric) ||
            normalization != GetNormalizationCode(expectedMetric) ||
            reserved != 0)
        {
            throw new InvalidDataException("HNSW index vector file header is inconsistent.");
        }

        int valueCount = checked(expectedRowCount * expectedDimension);
        if (stream.Length != checked(VectorsHeaderLength + (long)valueCount * sizeof(float)))
        {
            throw new InvalidDataException("HNSW index vector file payload length is invalid.");
        }
    }

    private static void ValidateLevelsFileHeader(FileStream stream, int expectedRowCount)
    {
        if (stream.Length < LevelsHeaderLength)
        {
            throw new InvalidDataException("HNSW index level file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[LevelsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(LevelsMagic))
        {
            throw new InvalidDataException("HNSW index level file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "level");
        RequireHeaderLength(header[12..], LevelsHeaderLength, "level");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[24..]);
        if (reserved != 0 || rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("HNSW index level file header is inconsistent.");
        }

        if (stream.Length != checked(LevelsHeaderLength + (long)expectedRowCount * sizeof(int)))
        {
            throw new InvalidDataException("HNSW index level file payload length is invalid.");
        }
    }

    private static string PrepareSaveDirectory(string directoryPath, out bool createdDirectory)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        string directory = Path.GetFullPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("HNSW index save path is an existing file, not a directory.");
        }

        if (Directory.Exists(directory))
        {
            if (Directory.EnumerateFileSystemEntries(directory).Any())
            {
                throw new IOException("HNSW index save directory must be empty.");
            }

            createdDirectory = false;
        }
        else
        {
            Directory.CreateDirectory(directory);
            createdDirectory = true;
        }

        return directory;
    }

    private static void WriteIdsFile(string path, ReadOnlySpan<ulong> ids)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[IdsHeaderLength];
        IdsMagic.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], BinaryMajorVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], BinaryMinorVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], IdsHeaderLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], checked((ulong)ids.Length));
        BinaryPrimitives.WriteUInt64LittleEndian(header[24..], 0);
        stream.Write(header);

        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        foreach (ulong id in ids)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(buffer, id);
            stream.Write(buffer);
        }
    }

    private static void WriteVectorsFile(
        string path,
        int dimension,
        VectorMetric metric,
        int rowCount,
        ReadOnlySpan<float> vectors)
    {
        if (vectors.Length != checked(rowCount * dimension))
        {
            throw new InvalidOperationException("HNSW vector payload length does not match index metadata.");
        }

        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[VectorsHeaderLength];
        VectorsMagic.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], BinaryMajorVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], BinaryMinorVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], VectorsHeaderLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], checked((ulong)rowCount));
        BinaryPrimitives.WriteUInt32LittleEndian(header[24..], checked((uint)dimension));
        BinaryPrimitives.WriteUInt32LittleEndian(header[28..], 1);
        BinaryPrimitives.WriteUInt32LittleEndian(header[32..], GetMetricCode(metric));
        BinaryPrimitives.WriteUInt32LittleEndian(header[36..], GetNormalizationCode(metric));
        BinaryPrimitives.WriteUInt64LittleEndian(header[40..], 0);
        stream.Write(header);

        Span<byte> buffer = stackalloc byte[sizeof(float)];
        foreach (float value in vectors)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buffer, BitConverter.SingleToInt32Bits(value));
            stream.Write(buffer);
        }
    }

    private static void WriteLevelsFile(string path, ReadOnlySpan<int> levels)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[LevelsHeaderLength];
        LevelsMagic.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], BinaryMajorVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], BinaryMinorVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], LevelsHeaderLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], checked((ulong)levels.Length));
        BinaryPrimitives.WriteUInt64LittleEndian(header[24..], 0);
        stream.Write(header);

        Span<byte> buffer = stackalloc byte[sizeof(int)];
        foreach (int level in levels)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buffer, level);
            stream.Write(buffer);
        }
    }

    private static void WriteGraphFile(string path, HnswIndex index)
    {
        int rowCount = index.Count;
        int layerCount = rowCount == 0 ? 0 : index.MaxLayer + 1;
        ReadOnlySpan<int> levels = index.InternalLevels;
        long directoryOffset = GraphHeaderLength;
        long payloadOffset = checked(directoryOffset + (long)layerCount * GraphLayerDirectoryEntryLength);
        var ordinalOffsets = new long[layerCount];
        var countOffsets = new long[layerCount];
        var neighborOffsets = new long[layerCount];
        var storedRowsByLayer = new int[layerCount];

        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = index.InternalGetGraphStride(layer);
            int storedRows = layer == 0 ? rowCount : CountParticipatingRows(levels, layer);
            storedRowsByLayer[layer] = storedRows;
            if (layer > 0)
            {
                ordinalOffsets[layer] = payloadOffset;
                payloadOffset = checked(payloadOffset + (long)storedRows * sizeof(int));
            }

            countOffsets[layer] = payloadOffset;
            payloadOffset = checked(payloadOffset + (long)storedRows * sizeof(int));
            neighborOffsets[layer] = payloadOffset;
            payloadOffset = checked(payloadOffset + (long)storedRows * stride * sizeof(int));
        }

        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[GraphHeaderLength];
        GraphMagic.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], BinaryMajorVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], BinaryMinorVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], GraphHeaderLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], checked((ulong)rowCount));
        BinaryPrimitives.WriteUInt32LittleEndian(header[24..], checked((uint)layerCount));
        BinaryPrimitives.WriteInt32LittleEndian(header[28..], index.EntryPoint);
        BinaryPrimitives.WriteInt32LittleEndian(header[32..], index.MaxLayer);
        BinaryPrimitives.WriteUInt32LittleEndian(header[36..], checked((uint)index.Options.M));
        BinaryPrimitives.WriteUInt32LittleEndian(header[40..], checked((uint)index.InternalMMax0));
        BinaryPrimitives.WriteUInt32LittleEndian(header[44..], checked((uint)index.InternalMMax));
        BinaryPrimitives.WriteUInt64LittleEndian(header[48..], checked((ulong)directoryOffset));
        BinaryPrimitives.WriteUInt64LittleEndian(header[56..], 0);
        stream.Write(header);

        Span<byte> entry = stackalloc byte[GraphLayerDirectoryEntryLength];
        for (int layer = 0; layer < layerCount; layer++)
        {
            entry.Clear();
            int stride = index.InternalGetGraphStride(layer);
            BinaryPrimitives.WriteUInt32LittleEndian(entry, checked((uint)layer));
            BinaryPrimitives.WriteUInt32LittleEndian(entry[4..], checked((uint)stride));
            BinaryPrimitives.WriteUInt32LittleEndian(entry[8..], checked((uint)storedRowsByLayer[layer]));
            BinaryPrimitives.WriteUInt32LittleEndian(entry[12..], layer == 0 ? GraphLayerLayoutDense : GraphLayerLayoutSparse);
            BinaryPrimitives.WriteUInt64LittleEndian(entry[16..], checked((ulong)ordinalOffsets[layer]));
            BinaryPrimitives.WriteUInt64LittleEndian(entry[24..], checked((ulong)countOffsets[layer]));
            BinaryPrimitives.WriteUInt64LittleEndian(entry[32..], checked((ulong)neighborOffsets[layer]));
            BinaryPrimitives.WriteUInt64LittleEndian(entry[40..], 0);
            stream.Write(entry);
        }

        Span<byte> buffer = stackalloc byte[sizeof(int)];
        for (int layer = 0; layer < layerCount; layer++)
        {
            int stride = index.InternalGetGraphStride(layer);
            if (layer > 0)
            {
                for (int ordinal = 0; ordinal < rowCount; ordinal++)
                {
                    if (levels[ordinal] < layer)
                    {
                        continue;
                    }

                    BinaryPrimitives.WriteInt32LittleEndian(buffer, ordinal);
                    stream.Write(buffer);
                }
            }

            for (int ordinal = 0; ordinal < rowCount; ordinal++)
            {
                if (layer > 0 && levels[ordinal] < layer)
                {
                    continue;
                }

                ReadOnlySpan<int> neighbors = index.InternalGetGraphNeighbors(layer, ordinal);
                BinaryPrimitives.WriteInt32LittleEndian(buffer, neighbors.Length);
                stream.Write(buffer);
            }

            for (int ordinal = 0; ordinal < rowCount; ordinal++)
            {
                if (layer > 0 && levels[ordinal] < layer)
                {
                    continue;
                }

                ReadOnlySpan<int> neighbors = index.InternalGetGraphNeighbors(layer, ordinal);
                for (int i = 0; i < neighbors.Length; i++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(buffer, neighbors[i]);
                    stream.Write(buffer);
                }

                BinaryPrimitives.WriteInt32LittleEndian(buffer, 0);
                for (int i = neighbors.Length; i < stride; i++)
                {
                    stream.Write(buffer);
                }
            }
        }
    }

    private static BinaryFileMetadata CreateBinaryFileMetadata(string path, string relativePath, string magic)
    {
        var file = new FileInfo(path);
        return new BinaryFileMetadata(relativePath, file.Length, ComputeSha256Hex(path), magic, BinaryVersionText);
    }

    private static void WriteManifest(
        string path,
        HnswIndex index,
        BinaryFileMetadata idsFile,
        BinaryFileMetadata vectorsFile,
        BinaryFileMetadata levelsFile,
        BinaryFileMetadata graphFile)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });
        int layerCount = index.Count == 0 ? 0 : index.MaxLayer + 1;

        writer.WriteStartObject();
        writer.WriteString("schemaName", ManifestSchemaName);
        writer.WriteString("schemaVersion", ManifestSchemaVersion);
        writer.WriteString("formatFamily", FormatFamily);
        writer.WriteString("snapshotId", Guid.NewGuid().ToString("D", CultureInfo.InvariantCulture));
        writer.WriteString("createdUtc", DateTimeOffset.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'", CultureInfo.InvariantCulture));
        writer.WriteString("createdByTask", CreatedByTask);

        writer.WriteStartObject("writer");
        writer.WriteString("product", "VecNet");
        writer.WriteString("formatWriter", "HnswIndex.Save");
        writer.WriteString("assemblyVersion", typeof(HnswIndex).Assembly.GetName().Version?.ToString() ?? "unknown");
        writer.WriteEndObject();

        writer.WriteStartObject("index");
        writer.WriteNumber("dimension", index.Dimension);
        writer.WriteString("metric", ToMetricText(index.Metric));
        writer.WriteNumber("vectorCount", index.Count);
        writer.WriteString("idType", "uint64");
        writer.WriteString("ordinalType", "int32");
        writer.WriteString("vectorElementType", "float32");
        writer.WriteString("vectorLayout", "row-major-dense");
        writer.WriteString("normalizationState", GetNormalizationState(index.Metric));
        writer.WriteEndObject();

        writer.WriteStartObject("hnsw");
        writer.WriteStartObject("options");
        writer.WriteNumber("m", index.Options.M);
        writer.WriteNumber("efConstruction", index.Options.EfConstruction);
        writer.WriteNumber("efSearch", index.Options.EfSearch);
        writer.WriteNumber("randomSeed", index.Options.RandomSeed);
        writer.WriteEndObject();
        writer.WriteStartObject("derivedParameters");
        writer.WriteNumber("mMax", index.InternalMMax);
        writer.WriteNumber("mMax0", index.InternalMMax0);
        writer.WriteNumber("levelMultiplier", index.InternalLevelMultiplier);
        writer.WriteBoolean("extendCandidates", false);
        writer.WriteBoolean("keepPrunedConnections", false);
        writer.WriteEndObject();
        writer.WriteStartObject("graph");
        writer.WriteNumber("entryPoint", index.EntryPoint);
        writer.WriteNumber("maxLayer", index.MaxLayer);
        writer.WriteNumber("layerCount", layerCount);
        writer.WriteNumber("layer0Stride", index.InternalMMax0);
        writer.WriteNumber("upperLayerStride", index.InternalMMax);
        writer.WriteString("adjacencyLayout", GraphAdjacencyLayout);
        writer.WriteString("levelGenerator", "SplitMix64.VEC-034");
        writer.WriteString("insertionOrder", "ordinal-row-order");
        writer.WriteEndObject();
        writer.WriteEndObject();

        writer.WriteStartObject("semantics");
        writer.WriteString("distanceContract", "VecNet.CanonicalDistance.1");
        writer.WriteString("tiePolicy", "DistanceThenExternalId");
        writer.WriteString("squaredEuclideanExecutionPolicy", "CurrentPublicDefault");
        writer.WriteString("resultKind", "approximate");
        writer.WriteString("openedLifecycle", "read-only");
        writer.WriteString("mutationPolicy", "reject");
        writer.WriteString("workspacePolicy", "caller-owned-independent-workspace");
        writer.WriteEndObject();

        writer.WriteStartObject("files");
        WriteFileMetadata(writer, "ids", idsFile);
        WriteFileMetadata(writer, "vectors", vectorsFile);
        WriteFileMetadata(writer, "levels", levelsFile);
        WriteFileMetadata(writer, "graph", graphFile);
        writer.WriteEndObject();

        writer.WriteStartObject("compatibility");
        writer.WriteStartArray("requiredFeatures");
        writer.WriteEndArray();
        writer.WriteStartArray("optionalFeatures");
        writer.WriteEndArray();
        writer.WriteNumber("minimumReaderMajorVersion", 1);
        writer.WriteString("unsupportedFeaturePolicy", "reject-unknown-required-features");
        writer.WriteEndObject();

        writer.WriteStartObject("evidence");
        writer.WriteString("privacyClass", "private-raw");
        writer.WriteString("claimClass", "local-evidence");
        writer.WriteBoolean("publicClaimEligible", false);
        writer.WriteBoolean("baselineCandidateEligible", false);
        writer.WriteBoolean("regressionGateEligible", false);
        writer.WriteBoolean("previewReadinessEligible", false);
        writer.WriteEndObject();

        writer.WriteEndObject();
    }

    private static void WriteFileMetadata(Utf8JsonWriter writer, string propertyName, BinaryFileMetadata file)
    {
        writer.WriteStartObject(propertyName);
        writer.WriteString("path", file.RelativePath);
        writer.WriteNumber("byteLength", file.ByteLength);
        writer.WriteString("sha256", file.Sha256);
        writer.WriteString("binaryMagic", file.BinaryMagic);
        writer.WriteString("binaryVersion", file.BinaryVersion);
        writer.WriteEndObject();
    }

    private static Manifest ReadManifest(string manifestPath)
    {
        var file = new FileInfo(manifestPath);
        if (file.Length > MaxManifestBytes)
        {
            throw new InvalidDataException("HNSW index manifest is too large.");
        }

        try
        {
            using FileStream stream = File.OpenRead(manifestPath);
            using JsonDocument document = JsonDocument.Parse(stream);
            JsonElement root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("HNSW index manifest root must be a JSON object.");
            }

            RequireString(root, "schemaName", ManifestSchemaName);
            RequireString(root, "schemaVersion", ManifestSchemaVersion);
            RequireString(root, "formatFamily", FormatFamily);
            ValidateGuidText(GetRequiredString(root, "snapshotId"));
            ValidateCreatedUtc(GetRequiredString(root, "createdUtc"));
            RequireString(root, "createdByTask", CreatedByTask);
            ValidateWriter(GetRequiredObject(root, "writer"));

            JsonElement index = GetRequiredObject(root, "index");
            int dimension = GetRequiredInt32(index, "dimension", minimumValue: 1);
            VectorMetric metric = ParseMetric(GetRequiredString(index, "metric"));
            int vectorCount = GetRequiredInt32(index, "vectorCount", minimumValue: 0);
            RequireString(index, "idType", "uint64");
            RequireString(index, "ordinalType", "int32");
            RequireString(index, "vectorElementType", "float32");
            RequireString(index, "vectorLayout", "row-major-dense");
            string normalizationState = GetRequiredString(index, "normalizationState");
            if (!string.Equals(normalizationState, GetNormalizationState(metric), StringComparison.Ordinal))
            {
                throw new InvalidDataException("HNSW index manifest metric and normalization state are inconsistent.");
            }

            JsonElement hnsw = GetRequiredObject(root, "hnsw");
            HnswIndexOptions options = ReadOptions(GetRequiredObject(hnsw, "options"));
            JsonElement derived = GetRequiredObject(hnsw, "derivedParameters");
            int mMax = GetRequiredInt32(derived, "mMax", minimumValue: 1);
            int mMax0 = GetRequiredInt32(derived, "mMax0", minimumValue: 1);
            double levelMultiplier = GetRequiredDouble(derived, "levelMultiplier");
            RequireBoolean(derived, "extendCandidates", expected: false);
            RequireBoolean(derived, "keepPrunedConnections", expected: false);
            if (mMax != options.M || mMax0 != checked(options.M * 2))
            {
                throw new InvalidDataException("HNSW index manifest derived graph parameters are inconsistent.");
            }

            double expectedLevelMultiplier = 1.0 / Math.Log(options.M);
            if (Math.Abs(levelMultiplier - expectedLevelMultiplier) > 1e-12)
            {
                throw new InvalidDataException("HNSW index manifest level multiplier is inconsistent.");
            }

            JsonElement graph = GetRequiredObject(hnsw, "graph");
            int entryPoint = GetRequiredInt32(graph, "entryPoint", minimumValue: -1);
            int maxLayer = GetRequiredInt32(graph, "maxLayer", minimumValue: -1);
            int layerCount = GetRequiredInt32(graph, "layerCount", minimumValue: 0);
            int layer0Stride = GetRequiredInt32(graph, "layer0Stride", minimumValue: 1);
            int upperLayerStride = GetRequiredInt32(graph, "upperLayerStride", minimumValue: 1);
            RequireString(graph, "adjacencyLayout", GraphAdjacencyLayout);
            RequireString(graph, "levelGenerator", "SplitMix64.VEC-034");
            RequireString(graph, "insertionOrder", "ordinal-row-order");
            ValidateGraphHeaderValues(vectorCount, entryPoint, maxLayer, layerCount, layer0Stride, upperLayerStride, mMax0, mMax);

            ValidateSemantics(GetRequiredObject(root, "semantics"));

            JsonElement files = GetRequiredObject(root, "files");
            BinaryFileMetadata ids = ReadFileMetadata(GetRequiredObject(files, "ids"), IdsFileName, IdsMagicText);
            BinaryFileMetadata vectors = ReadFileMetadata(GetRequiredObject(files, "vectors"), VectorsFileName, VectorsMagicText);
            BinaryFileMetadata levels = ReadFileMetadata(GetRequiredObject(files, "levels"), LevelsFileName, LevelsMagicText);
            BinaryFileMetadata graphFile = ReadFileMetadata(GetRequiredObject(files, "graph"), GraphFileName, GraphMagicText);

            ValidateCompatibility(GetRequiredObject(root, "compatibility"));
            ValidateEvidence(GetRequiredObject(root, "evidence"));

            return new Manifest(
                dimension,
                metric,
                vectorCount,
                options,
                mMax,
                mMax0,
                levelMultiplier,
                entryPoint,
                maxLayer,
                layerCount,
                layer0Stride,
                upperLayerStride,
                ids,
                vectors,
                levels,
                graphFile);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("HNSW index manifest JSON is malformed.", exception);
        }
    }

    private static HnswIndexOptions ReadOptions(JsonElement options)
    {
        int m = GetRequiredInt32(options, "m", minimumValue: MinM);
        int efConstruction = GetRequiredInt32(options, "efConstruction", minimumValue: 1);
        int efSearch = GetRequiredInt32(options, "efSearch", minimumValue: 1);
        ulong randomSeed = GetRequiredUInt64(options, "randomSeed");
        if (m > MaxM || efConstruction < m || efConstruction > MaxEf || efSearch > MaxEf)
        {
            throw new InvalidDataException("HNSW index manifest options are unsupported.");
        }

        return new HnswIndexOptions(m, efConstruction, efSearch, randomSeed);
    }

    private static void ValidateWriter(JsonElement writer)
    {
        RequireString(writer, "product", "VecNet");
        RequireString(writer, "formatWriter", "HnswIndex.Save");
        _ = GetRequiredString(writer, "assemblyVersion");
    }

    private static void ValidateSemantics(JsonElement semantics)
    {
        RequireString(semantics, "distanceContract", "VecNet.CanonicalDistance.1");
        RequireString(semantics, "tiePolicy", "DistanceThenExternalId");
        RequireString(semantics, "squaredEuclideanExecutionPolicy", "CurrentPublicDefault");
        RequireString(semantics, "resultKind", "approximate");
        RequireString(semantics, "openedLifecycle", "read-only");
        RequireString(semantics, "mutationPolicy", "reject");
        RequireString(semantics, "workspacePolicy", "caller-owned-independent-workspace");
    }

    private static BinaryFileMetadata ReadFileMetadata(JsonElement file, string expectedPath, string expectedMagic)
    {
        string relativePath = GetRequiredString(file, "path");
        long byteLength = GetRequiredInt64(file, "byteLength", minimumValue: 0);
        string sha256 = GetRequiredString(file, "sha256");
        ValidateSha256Text(sha256);
        RequireString(file, "binaryMagic", expectedMagic);
        RequireString(file, "binaryVersion", BinaryVersionText);
        if (!string.Equals(relativePath, expectedPath, StringComparison.Ordinal))
        {
            throw new InvalidDataException("HNSW index manifest file path is not the pinned format path.");
        }

        return new BinaryFileMetadata(relativePath, byteLength, sha256, expectedMagic, BinaryVersionText);
    }

    private static void ValidateCompatibility(JsonElement compatibility)
    {
        JsonElement requiredFeatures = GetRequiredArray(compatibility, "requiredFeatures");
        foreach (JsonElement feature in requiredFeatures.EnumerateArray())
        {
            if (feature.ValueKind != JsonValueKind.String)
            {
                throw new InvalidDataException("HNSW index required features must be strings.");
            }

            throw new InvalidDataException("HNSW index manifest contains an unknown required feature.");
        }

        JsonElement optionalFeatures = GetRequiredArray(compatibility, "optionalFeatures");
        if (optionalFeatures.GetArrayLength() != 0)
        {
            throw new InvalidDataException("HNSW index manifest contains an unsupported optional feature.");
        }

        int minimumReaderMajorVersion = GetRequiredInt32(compatibility, "minimumReaderMajorVersion", minimumValue: 1);
        if (minimumReaderMajorVersion > BinaryMajorVersion)
        {
            throw new InvalidDataException("HNSW index requires a newer reader major version.");
        }

        RequireString(compatibility, "unsupportedFeaturePolicy", "reject-unknown-required-features");
    }

    private static void ValidateEvidence(JsonElement evidence)
    {
        RequireString(evidence, "privacyClass", "private-raw");
        RequireString(evidence, "claimClass", "local-evidence");
        RequireBoolean(evidence, "publicClaimEligible", expected: false);
        RequireBoolean(evidence, "baselineCandidateEligible", expected: false);
        RequireBoolean(evidence, "regressionGateEligible", expected: false);
        RequireBoolean(evidence, "previewReadinessEligible", expected: false);
    }

    private static void ValidateGraphHeaderValues(
        int vectorCount,
        int entryPoint,
        int maxLayer,
        int layerCount,
        int layer0Stride,
        int upperLayerStride,
        int mMax0,
        int mMax)
    {
        if (layer0Stride != mMax0 || upperLayerStride != mMax)
        {
            throw new InvalidDataException("HNSW index graph stride metadata is inconsistent.");
        }

        if (vectorCount == 0)
        {
            if (entryPoint != -1 || maxLayer != -1 || layerCount != 0)
            {
                throw new InvalidDataException("HNSW empty graph metadata is inconsistent.");
            }

            return;
        }

        if ((uint)entryPoint >= (uint)vectorCount || maxLayer < 0 || layerCount != maxLayer + 1)
        {
            throw new InvalidDataException("HNSW non-empty graph metadata is inconsistent.");
        }
    }

    private static string ResolveManifestFilePath(string directory, string relativePath, string expectedFileName)
    {
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath))
        {
            throw new InvalidDataException("HNSW index manifest file paths must be relative file names.");
        }

        if (!string.Equals(relativePath, expectedFileName, StringComparison.Ordinal))
        {
            throw new InvalidDataException("HNSW index manifest file path is not the pinned format path.");
        }

        string resolved = Path.GetFullPath(Path.Combine(directory, relativePath));
        string root = Path.GetFullPath(directory);
        string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
            ? root
            : root + Path.DirectorySeparatorChar;
        if (!resolved.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("HNSW index manifest file path escapes the index directory.");
        }

        return resolved;
    }

    private static void ValidateFileExistsLengthAndHash(string path, BinaryFileMetadata metadata, string artifactName)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"HNSW index {artifactName} file was not found.", path);
        }

        long actualLength = new FileInfo(path).Length;
        if (actualLength != metadata.ByteLength)
        {
            throw new InvalidDataException($"HNSW index {artifactName} file byte length does not match the manifest.");
        }

        string actualSha256 = ComputeSha256Hex(path);
        if (!string.Equals(actualSha256, metadata.Sha256, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"HNSW index {artifactName} file checksum does not match the manifest.");
        }
    }

    private static ulong[] ReadIdsFile(string path, int expectedRowCount)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < IdsHeaderLength)
        {
            throw new InvalidDataException("HNSW index ID file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[IdsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(IdsMagic))
        {
            throw new InvalidDataException("HNSW index ID file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "ID");
        RequireHeaderLength(header[12..], IdsHeaderLength, "ID");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[24..]);
        if (reserved != 0 || rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("HNSW index ID file header is inconsistent.");
        }

        if (stream.Length != checked(IdsHeaderLength + (long)expectedRowCount * sizeof(ulong)))
        {
            throw new InvalidDataException("HNSW index ID file payload length is invalid.");
        }

        var ids = new ulong[expectedRowCount];
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        for (int i = 0; i < ids.Length; i++)
        {
            stream.ReadExactly(buffer);
            ids[i] = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
        }

        return ids;
    }

    private static float[] ReadVectorsFile(
        string path,
        int expectedDimension,
        VectorMetric expectedMetric,
        int expectedRowCount)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < VectorsHeaderLength)
        {
            throw new InvalidDataException("HNSW index vector file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[VectorsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(VectorsMagic))
        {
            throw new InvalidDataException("HNSW index vector file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "vector");
        RequireHeaderLength(header[12..], VectorsHeaderLength, "vector");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        uint dimension = BinaryPrimitives.ReadUInt32LittleEndian(header[24..]);
        uint representation = BinaryPrimitives.ReadUInt32LittleEndian(header[28..]);
        uint metric = BinaryPrimitives.ReadUInt32LittleEndian(header[32..]);
        uint normalization = BinaryPrimitives.ReadUInt32LittleEndian(header[36..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[40..]);
        if (rowCount != checked((ulong)expectedRowCount) ||
            dimension != checked((uint)expectedDimension) ||
            representation != 1 ||
            metric != GetMetricCode(expectedMetric) ||
            normalization != GetNormalizationCode(expectedMetric) ||
            reserved != 0)
        {
            throw new InvalidDataException("HNSW index vector file header is inconsistent.");
        }

        int valueCount = checked(expectedRowCount * expectedDimension);
        if (stream.Length != checked(VectorsHeaderLength + (long)valueCount * sizeof(float)))
        {
            throw new InvalidDataException("HNSW index vector file payload length is invalid.");
        }

        var vectors = new float[valueCount];
        Span<byte> buffer = stackalloc byte[sizeof(float)];
        for (int i = 0; i < vectors.Length; i++)
        {
            stream.ReadExactly(buffer);
            vectors[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(buffer));
        }

        return vectors;
    }

    private static int[] ReadLevelsFile(string path, int expectedRowCount, int maxLayer)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < LevelsHeaderLength)
        {
            throw new InvalidDataException("HNSW index level file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[LevelsHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(LevelsMagic))
        {
            throw new InvalidDataException("HNSW index level file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "level");
        RequireHeaderLength(header[12..], LevelsHeaderLength, "level");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[24..]);
        if (reserved != 0 || rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("HNSW index level file header is inconsistent.");
        }

        if (stream.Length != checked(LevelsHeaderLength + (long)expectedRowCount * sizeof(int)))
        {
            throw new InvalidDataException("HNSW index level file payload length is invalid.");
        }

        var levels = new int[expectedRowCount];
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        bool hasMaxLayer = expectedRowCount == 0;
        for (int i = 0; i < levels.Length; i++)
        {
            stream.ReadExactly(buffer);
            int level = BinaryPrimitives.ReadInt32LittleEndian(buffer);
            if (level < 0 || level > maxLayer)
            {
                throw new InvalidDataException("HNSW index level payload is invalid.");
            }

            hasMaxLayer |= level == maxLayer;
            levels[i] = level;
        }

        if (!hasMaxLayer)
        {
            throw new InvalidDataException("HNSW index levels do not contain the max layer.");
        }

        return levels;
    }

    private static HnswIndex.HnswLayerSnapshot[] ReadGraphFile(string path, Manifest manifest)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < GraphHeaderLength)
        {
            throw new InvalidDataException("HNSW index graph file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[GraphHeaderLength];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(GraphMagic))
        {
            throw new InvalidDataException("HNSW index graph file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "graph");
        RequireHeaderLength(header[12..], GraphHeaderLength, "graph");
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        uint layerCount = BinaryPrimitives.ReadUInt32LittleEndian(header[24..]);
        int entryPoint = BinaryPrimitives.ReadInt32LittleEndian(header[28..]);
        int maxLayer = BinaryPrimitives.ReadInt32LittleEndian(header[32..]);
        uint m = BinaryPrimitives.ReadUInt32LittleEndian(header[36..]);
        uint mMax0 = BinaryPrimitives.ReadUInt32LittleEndian(header[40..]);
        uint mMax = BinaryPrimitives.ReadUInt32LittleEndian(header[44..]);
        ulong directoryOffset = BinaryPrimitives.ReadUInt64LittleEndian(header[48..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[56..]);
        if (rowCount != checked((ulong)manifest.VectorCount) ||
            layerCount != checked((uint)manifest.LayerCount) ||
            entryPoint != manifest.EntryPoint ||
            maxLayer != manifest.MaxLayer ||
            m != checked((uint)manifest.Options.M) ||
            mMax0 != checked((uint)manifest.MMax0) ||
            mMax != checked((uint)manifest.MMax) ||
            directoryOffset != GraphHeaderLength ||
            reserved != 0)
        {
            throw new InvalidDataException("HNSW index graph file header is inconsistent.");
        }

        long directoryLength = checked((long)manifest.LayerCount * GraphLayerDirectoryEntryLength);
        if (stream.Length < GraphHeaderLength + directoryLength)
        {
            throw new InvalidDataException("HNSW index graph layer directory is truncated.");
        }

        var entries = new GraphLayerEntry[manifest.LayerCount];
        Span<byte> entryBuffer = stackalloc byte[GraphLayerDirectoryEntryLength];
        for (int layer = 0; layer < manifest.LayerCount; layer++)
        {
            stream.ReadExactly(entryBuffer);
            uint layerNumber = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer);
            uint stride = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[4..]);
            uint storedRows = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[8..]);
            uint layout = BinaryPrimitives.ReadUInt32LittleEndian(entryBuffer[12..]);
            long ordinalsOffset = ReadGraphOffset(entryBuffer[16..]);
            long countsOffset = ReadGraphOffset(entryBuffer[24..]);
            long neighborsOffset = ReadGraphOffset(entryBuffer[32..]);
            ulong entryReserved = BinaryPrimitives.ReadUInt64LittleEndian(entryBuffer[40..]);
            int expectedStride = layer == 0 ? manifest.MMax0 : manifest.MMax;
            uint expectedLayout = layer == 0 ? GraphLayerLayoutDense : GraphLayerLayoutSparse;
            uint expectedRows = layer == 0 ? checked((uint)manifest.VectorCount) : checked((uint)manifest.VectorCount);
            if (layerNumber != checked((uint)layer) ||
                stride != checked((uint)expectedStride) ||
                layout != expectedLayout ||
                storedRows > expectedRows ||
                (layer == 0 && storedRows != expectedRows) ||
                (layer == 0 && ordinalsOffset != 0) ||
                entryReserved != 0)
            {
                throw new InvalidDataException("HNSW index graph layer directory entry is invalid.");
            }

            entries[layer] = new GraphLayerEntry((int)stride, (int)storedRows, ordinalsOffset, countsOffset, neighborsOffset);
        }

        ValidateGraphRanges(entries, stream.Length);

        var layers = new HnswIndex.HnswLayerSnapshot[manifest.LayerCount];
        Span<byte> intBuffer = stackalloc byte[sizeof(int)];
        for (int layer = 0; layer < manifest.LayerCount; layer++)
        {
            GraphLayerEntry entry = entries[layer];
            int[] rowOrdinals = layer == 0 ? [] : new int[entry.StoredRows];
            var counts = new int[entry.StoredRows];
            var neighbors = new int[checked(entry.StoredRows * entry.Stride)];

            if (layer > 0)
            {
                stream.Position = entry.OrdinalsOffset;
                for (int i = 0; i < rowOrdinals.Length; i++)
                {
                    stream.ReadExactly(intBuffer);
                    rowOrdinals[i] = BinaryPrimitives.ReadInt32LittleEndian(intBuffer);
                }
            }

            stream.Position = entry.CountsOffset;
            for (int i = 0; i < counts.Length; i++)
            {
                stream.ReadExactly(intBuffer);
                counts[i] = BinaryPrimitives.ReadInt32LittleEndian(intBuffer);
            }

            stream.Position = entry.NeighborsOffset;
            for (int i = 0; i < neighbors.Length; i++)
            {
                stream.ReadExactly(intBuffer);
                neighbors[i] = BinaryPrimitives.ReadInt32LittleEndian(intBuffer);
            }

            layers[layer] = new HnswIndex.HnswLayerSnapshot(entry.Stride, rowOrdinals, counts, neighbors);
        }

        return layers;
    }

    private static void ValidateGraphRanges(GraphLayerEntry[] entries, long fileLength)
    {
        var ranges = new List<(long Start, long End)>(1 + entries.Length * 2)
        {
            (0, checked(GraphHeaderLength + (long)entries.Length * GraphLayerDirectoryEntryLength))
        };

        foreach (GraphLayerEntry entry in entries)
        {
            if (entry.OrdinalsOffset != 0)
            {
                long ordinalsLength = checked((long)entry.StoredRows * sizeof(int));
                ranges.Add((entry.OrdinalsOffset, checked(entry.OrdinalsOffset + ordinalsLength)));
            }

            long countsLength = checked((long)entry.StoredRows * sizeof(int));
            long neighborsLength = checked((long)entry.StoredRows * entry.Stride * sizeof(int));
            ranges.Add((entry.CountsOffset, checked(entry.CountsOffset + countsLength)));
            ranges.Add((entry.NeighborsOffset, checked(entry.NeighborsOffset + neighborsLength)));
        }

        ranges.Sort(static (left, right) => left.Start.CompareTo(right.Start));
        long expectedStart = 0;
        foreach ((long start, long end) in ranges)
        {
            if (start != expectedStart || start < 0 || end < start || end > fileLength || (start % sizeof(int)) != 0)
            {
                throw new InvalidDataException("HNSW index graph payload ranges are invalid.");
            }

            expectedStart = end;
        }

        if (expectedStart != fileLength)
        {
            throw new InvalidDataException("HNSW index graph file contains trailing or missing bytes.");
        }
    }

    private static long ReadGraphOffset(ReadOnlySpan<byte> bytes)
    {
        ulong offset = BinaryPrimitives.ReadUInt64LittleEndian(bytes);
        if (offset > long.MaxValue)
        {
            throw new InvalidDataException("HNSW index graph payload offset is invalid.");
        }

        return (long)offset;
    }

    private static void ValidateRowsAndGraph(
        ulong[] ids,
        float[] vectors,
        int[] levels,
        HnswIndex.HnswLayerSnapshot[] layers,
        Manifest manifest)
    {
        var seen = new HashSet<ulong>();
        foreach (ulong id in ids)
        {
            if (!seen.Add(id))
            {
                throw new InvalidDataException("HNSW index contains duplicate external IDs.");
            }
        }

        for (int row = 0; row < ids.Length; row++)
        {
            double squaredMagnitude = 0;
            int offset = row * manifest.Dimension;
            for (int i = 0; i < manifest.Dimension; i++)
            {
                float value = vectors[offset + i];
                if (!float.IsFinite(value))
                {
                    throw new InvalidDataException("HNSW index vector payload contains a non-finite component.");
                }

                if (manifest.Metric == VectorMetric.Cosine)
                {
                    squaredMagnitude += (double)value * value;
                }
            }

            if (manifest.Metric == VectorMetric.Cosine &&
                (squaredMagnitude == 0 ||
                 Math.Abs(squaredMagnitude - 1) > CosineStoredRowSquaredLengthTolerance))
            {
                throw new InvalidDataException("HNSW index cosine stored row is not within the unit-length tolerance.");
            }
        }

        if (manifest.VectorCount == 0)
        {
            if (layers.Length != 0)
            {
                throw new InvalidDataException("HNSW empty graph contains layers.");
            }

            return;
        }

        if (levels[manifest.EntryPoint] != manifest.MaxLayer)
        {
            throw new InvalidDataException("HNSW index entry point is not on the max layer.");
        }

        for (int layer = 0; layer < layers.Length; layer++)
        {
            HnswIndex.HnswLayerSnapshot graphLayer = layers[layer];
            int expectedStride = layer == 0 ? manifest.MMax0 : manifest.MMax;
            int expectedRows = layer == 0 ? manifest.VectorCount : CountParticipatingRows(levels, layer);
            if (graphLayer.Stride != expectedStride ||
                graphLayer.Counts.Length != expectedRows ||
                graphLayer.Neighbors.Length != checked(expectedRows * expectedStride) ||
                (layer == 0 && graphLayer.RowOrdinals.Length != 0) ||
                (layer > 0 && graphLayer.RowOrdinals.Length != expectedRows))
            {
                throw new InvalidDataException("HNSW index graph layer shape is invalid.");
            }

            int previousOrdinal = -1;
            for (int row = 0; row < expectedRows; row++)
            {
                int ordinal = layer == 0 ? row : graphLayer.RowOrdinals[row];
                if ((uint)ordinal >= (uint)manifest.VectorCount ||
                    ordinal <= previousOrdinal ||
                    levels[ordinal] < layer)
                {
                    throw new InvalidDataException("HNSW index compact graph row ordinal is invalid.");
                }

                previousOrdinal = ordinal;
                int count = graphLayer.Counts[row];
                if (count < 0 || count > graphLayer.Stride)
                {
                    throw new InvalidDataException("HNSW index graph neighbor count is invalid.");
                }

                var seenNeighbors = new HashSet<int>();
                int offset = row * graphLayer.Stride;
                for (int i = 0; i < count; i++)
                {
                    int neighbor = graphLayer.Neighbors[offset + i];
                    if ((uint)neighbor >= (uint)manifest.VectorCount ||
                        neighbor == ordinal ||
                        levels[neighbor] < layer ||
                        !seenNeighbors.Add(neighbor))
                    {
                        throw new InvalidDataException("HNSW index graph neighbor reference is invalid.");
                    }
                }
            }
        }
    }

    private static int CountParticipatingRows(ReadOnlySpan<int> levels, int layer)
    {
        int count = 0;
        foreach (int level in levels)
        {
            if (level >= layer)
            {
                count++;
            }
        }

        return count;
    }

    private static void ValidateBinaryVersion(ReadOnlySpan<byte> versionBytes, string artifactName)
    {
        ushort major = BinaryPrimitives.ReadUInt16LittleEndian(versionBytes);
        ushort minor = BinaryPrimitives.ReadUInt16LittleEndian(versionBytes[2..]);
        if (major != BinaryMajorVersion || minor != BinaryMinorVersion)
        {
            throw new InvalidDataException($"HNSW index {artifactName} file binary version is unsupported.");
        }
    }

    private static void RequireHeaderLength(ReadOnlySpan<byte> bytes, uint expected, string artifactName)
    {
        if (BinaryPrimitives.ReadUInt32LittleEndian(bytes) != expected)
        {
            throw new InvalidDataException($"HNSW index {artifactName} file header length is invalid.");
        }
    }

    private static string ComputeSha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static void TryDeleteDirectoryIfEmpty(string path)
    {
        try
        {
            if (Directory.Exists(path) && !Directory.EnumerateFileSystemEntries(path).Any())
            {
                Directory.Delete(path);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static JsonElement GetRequiredObject(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' must be an object.");
        }

        return value;
    }

    private static JsonElement GetRequiredArray(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' must be an array.");
        }

        return value;
    }

    private static JsonElement GetRequiredProperty(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement value))
        {
            throw new InvalidDataException($"HNSW index manifest is missing required property '{propertyName}'.");
        }

        return value;
    }

    private static string GetRequiredString(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' must be a string.");
        }

        return value.GetString()!;
    }

    private static int GetRequiredInt32(JsonElement element, string propertyName, int minimumValue)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int result) || result < minimumValue)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' has an invalid integer value.");
        }

        return result;
    }

    private static long GetRequiredInt64(JsonElement element, string propertyName, long minimumValue)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt64(out long result) || result < minimumValue)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' has an invalid integer value.");
        }

        return result;
    }

    private static ulong GetRequiredUInt64(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetUInt64(out ulong result))
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' has an invalid unsigned integer value.");
        }

        return result;
    }

    private static double GetRequiredDouble(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out double result) || !double.IsFinite(result))
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' has an invalid number value.");
        }

        return result;
    }

    private static void RequireString(JsonElement element, string propertyName, string expected)
    {
        string actual = GetRequiredString(element, propertyName);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' is unsupported.");
        }
    }

    private static void RequireBoolean(JsonElement element, string propertyName, bool expected)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.True && value.ValueKind != JsonValueKind.False)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' must be a boolean.");
        }

        if (value.GetBoolean() != expected)
        {
            throw new InvalidDataException($"HNSW index manifest property '{propertyName}' is unsupported.");
        }
    }

    private static void ValidateCreatedUtc(string value)
    {
        if (!DateTimeOffset.TryParseExact(
                value,
                "yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'",
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                out DateTimeOffset parsed) ||
            parsed.Offset != TimeSpan.Zero)
        {
            throw new InvalidDataException("HNSW index manifest createdUtc is not in the pinned UTC format.");
        }
    }

    private static void ValidateGuidText(string value)
    {
        if (!Guid.TryParseExact(value, "D", out _))
        {
            throw new InvalidDataException("HNSW index manifest snapshotId is invalid.");
        }
    }

    private static void ValidateSha256Text(string value)
    {
        if (value.Length != 64 || value.Any(static c => !IsLowerHex(c)))
        {
            throw new InvalidDataException("HNSW index manifest SHA-256 must be lowercase hexadecimal text.");
        }
    }

    private static bool IsLowerHex(char value) =>
        value is >= '0' and <= '9' or >= 'a' and <= 'f';

    private static string ToMetricText(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => "squared-euclidean",
            VectorMetric.Cosine => "cosine",
            VectorMetric.InnerProduct => "inner-product",
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static VectorMetric ParseMetric(string metric) =>
        metric switch
        {
            "squared-euclidean" => VectorMetric.SquaredEuclidean,
            "cosine" => VectorMetric.Cosine,
            "inner-product" => throw new InvalidDataException("HNSW index manifest metric is unsupported."),
            _ => throw new InvalidDataException("HNSW index manifest metric is unsupported.")
        };

    private static string GetNormalizationState(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? NormalizationCosineUnit : NormalizationNone;

    private static uint GetMetricCode(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => SquaredEuclideanMetricCode,
            VectorMetric.Cosine => CosineMetricCode,
            VectorMetric.InnerProduct => InnerProductMetricCode,
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static uint GetNormalizationCode(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? CosineUnitNormalizedCode : NoNormalizationCode;

    private sealed record Manifest(
        int Dimension,
        VectorMetric Metric,
        int VectorCount,
        HnswIndexOptions Options,
        int MMax,
        int MMax0,
        double LevelMultiplier,
        int EntryPoint,
        int MaxLayer,
        int LayerCount,
        int Layer0Stride,
        int UpperLayerStride,
        BinaryFileMetadata IdsFile,
        BinaryFileMetadata VectorsFile,
        BinaryFileMetadata LevelsFile,
        BinaryFileMetadata GraphFile);

    private sealed record BinaryFileMetadata(
        string RelativePath,
        long ByteLength,
        string Sha256,
        string BinaryMagic,
        string BinaryVersion);

    private readonly record struct GraphLayerEntry(
        int Stride,
        int StoredRows,
        long OrdinalsOffset,
        long CountsOffset,
        long NeighborsOffset);
}
