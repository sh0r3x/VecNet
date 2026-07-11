using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;

namespace VecNet;

public sealed partial class ExactFlatIndex
{
    /// <summary>
    /// Writes a compact live exact-flat checkpoint to a new or empty directory and publishes it in memory.
    /// </summary>
    /// <remarks>
    /// Checkpoint writes the current live view only, validates the newly written output, and then
    /// publishes the compact view in this index instance. It does not replace an active directory,
    /// coordinate with other processes, provide crash recovery for caller-managed replacement, or
    /// make the opened directory writable.
    /// </remarks>
    /// <param name="directoryPath">
    /// The target directory path. It must not be null or whitespace, must not name an existing file,
    /// and must either not exist or name an empty directory. Existing index directories are not overwritten.
    /// </param>
    /// <returns>A checkpoint result describing whether a compact generation was published.</returns>
    public ExactFlatCheckpointResult Checkpoint(string directoryPath)
    {
        if (_isReadOnly)
        {
            throw new InvalidOperationException("This exact flat index was opened read-only and cannot be checkpointed.");
        }

        ExactFlatIndexStorage.ValidateNewOrEmptyDirectoryPath(directoryPath);

        int foldedDeltaCount = DeltaVectorCount;
        int foldedTombstoneCount = _tombstoneCount;
        if (foldedDeltaCount == 0 && foldedTombstoneCount == 0)
        {
            return CreateCheckpointResult(
                ExactFlatCheckpointStatus.NoChanges,
                foldedDeltaCount,
                foldedTombstoneCount);
        }

        (ulong[] compactIds, float[] compactVectors) = CreateLiveSnapshot();
        ExactFlatIndexStorage.Save(directoryPath, Dimension, Metric, compactIds, compactVectors);

        ExactFlatIndex validated = ExactFlatIndexStorage.OpenReadOnly(directoryPath);
        ValidateCheckpointOutput(validated, compactIds, compactVectors);

        Dictionary<ulong, int> compactMap = BuildIdToOrdinalMap(compactIds);
        _ids = compactIds;
        _vectors = compactVectors;
        _rowDeleted = new byte[compactIds.Length];
        _idToOrdinal = compactMap;
        _count = compactIds.Length;
        _baseRowCount = compactIds.Length;
        _tombstoneCount = 0;
        _generation++;

        return CreateCheckpointResult(
            ExactFlatCheckpointStatus.Published,
            foldedDeltaCount,
            foldedTombstoneCount);
    }

    /// <summary>
    /// Saves this exact flat index to a new or empty durable exact-flat directory.
    /// </summary>
    /// <remarks>
    /// Save writes the current live view only. It requires a new or empty target location and does
    /// not replace an active index directory, coordinate with other processes, or provide
    /// caller-level crash recovery for directory swaps.
    /// </remarks>
    /// <param name="directoryPath">
    /// The target directory path. It must not be null or whitespace, must not name an existing file,
    /// and must either not exist or name an empty directory. Existing index directories are not overwritten.
    /// </param>
    public void Save(string directoryPath)
    {
        ExactFlatIndexStorage.ValidateNewOrEmptyDirectoryPath(directoryPath);
        (ulong[] ids, float[] vectors) = CreateLiveSnapshot();
        ExactFlatIndexStorage.Save(directoryPath, Dimension, Metric, ids, vectors);
    }

    /// <summary>
    /// Opens a durable exact-flat index directory as an immutable read-only index.
    /// </summary>
    /// <remarks>
    /// Open validates the manifest and binary files using broad preview failure categories such as
    /// invalid data, missing files, unsupported format, or I/O errors. It does not establish a
    /// stable complete exception taxonomy and does not open the index for mutation.
    /// </remarks>
    /// <param name="directoryPath">
    /// The exact-flat index directory path. It must not be null or whitespace and must name an
    /// existing exact-flat directory containing a valid manifest and binary files.
    /// </param>
    /// <returns>A searchable read-only exact flat index.</returns>
    public static ExactFlatIndex OpenReadOnly(string directoryPath) =>
        ExactFlatIndexStorage.OpenReadOnly(directoryPath);

    private (ulong[] Ids, float[] Vectors) CreateLiveSnapshot()
    {
        int count = LiveVectorCount;
        var ids = new ulong[count];
        var vectors = new float[checked(count * Dimension)];
        int destinationRow = 0;
        for (int sourceRow = 0; sourceRow < _count; sourceRow++)
        {
            if (IsDeleted(sourceRow))
            {
                continue;
            }

            ids[destinationRow] = _ids[sourceRow];
            _vectors
                .AsSpan(sourceRow * Dimension, Dimension)
                .CopyTo(vectors.AsSpan(destinationRow * Dimension, Dimension));
            destinationRow++;
        }

        return (ids, vectors);
    }

    private ExactFlatCheckpointResult CreateCheckpointResult(
        ExactFlatCheckpointStatus status,
        int foldedDeltaCount,
        int foldedTombstoneCount) =>
        new(
            status,
            _generation,
            _count,
            LiveVectorCount,
            BaseVectorCount,
            DeltaVectorCount,
            _tombstoneCount,
            _deletedReservedIds.Count,
            foldedDeltaCount,
            foldedTombstoneCount);

    private void ValidateCheckpointOutput(
        ExactFlatIndex validated,
        ReadOnlySpan<ulong> expectedIds,
        ReadOnlySpan<float> expectedVectors)
    {
        if (validated.Dimension != Dimension ||
            validated.Metric != Metric ||
            validated.VectorCount != expectedIds.Length ||
            !validated._ids.AsSpan(0, validated._count).SequenceEqual(expectedIds) ||
            !validated._vectors.AsSpan(0, checked(expectedIds.Length * Dimension)).SequenceEqual(expectedVectors))
        {
            throw new InvalidDataException("Exact flat checkpoint output failed read-only validation.");
        }
    }
}

internal static class ExactFlatIndexStorage
{
    internal const string ManifestFileName = "exact-flat.manifest.json";
    internal const string IdsFileName = "exact-flat.ids.u64";
    internal const string VectorsFileName = "exact-flat.vectors.f32";

    internal const string ManifestSchemaName = "VecNet.ExactFlatIndexManifest";
    internal const string ManifestSchemaVersion = "1.0";
    internal const string FormatFamily = "exact-flat";

    internal const string IdsMagicText = "VNETID01";
    internal const string VectorsMagicText = "VNETVF01";
    internal const ushort BinaryMajorVersion = 1;
    internal const ushort BinaryMinorVersion = 0;
    internal const int IdsHeaderLength = 32;
    internal const int VectorsHeaderLength = 48;
    internal const uint Float32RowMajorRepresentationCode = 1;
    internal const uint NoNormalizationCode = 0;
    internal const uint CosineUnitNormalizedCode = 1;
    internal const double CosineStoredRowSquaredLengthTolerance = 1e-4;

    private const int MaxManifestBytes = 1024 * 1024;
    private const string CreatedByTask = "VEC-031";
    private const string IdType = "uint64";
    private const string VectorElementType = "float32";
    private const string VectorLayout = "row-major-dense";
    private const string NormalizationNone = "none";
    private const string NormalizationCosineUnit = "cosine-unit-normalized";
    private const string DistanceContract = "VecNet.CanonicalDistance.1";
    private const string TiePolicy = "DistanceThenExternalId";
    private const string SquaredEuclideanExecutionPolicy = "CurrentPublicDefault";
    private const string BinaryVersionText = "1.0";

    private static readonly byte[] IdsMagic = "VNETID01"u8.ToArray();
    private static readonly byte[] VectorsMagic = "VNETVF01"u8.ToArray();

    internal static void Save(
        string directoryPath,
        int dimension,
        VectorMetric metric,
        ReadOnlySpan<ulong> ids,
        ReadOnlySpan<float> vectors)
    {
        string directory = PrepareSaveDirectory(directoryPath, out bool createdDirectory);
        string tempSuffix = ".tmp-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);
        string idsTempPath = Path.Combine(directory, IdsFileName + tempSuffix);
        string vectorsTempPath = Path.Combine(directory, VectorsFileName + tempSuffix);
        string manifestTempPath = Path.Combine(directory, ManifestFileName + tempSuffix);

        try
        {
            WriteIdsFile(idsTempPath, ids);
            WriteVectorsFile(vectorsTempPath, dimension, metric, ids.Length, vectors);

            var idsMetadata = CreateBinaryFileMetadata(idsTempPath, IdsFileName, IdsMagicText);
            var vectorsMetadata = CreateBinaryFileMetadata(vectorsTempPath, VectorsFileName, VectorsMagicText);
            WriteManifest(
                manifestTempPath,
                dimension,
                metric,
                ids.Length,
                idsMetadata,
                vectorsMetadata);

            File.Move(idsTempPath, Path.Combine(directory, IdsFileName));
            File.Move(vectorsTempPath, Path.Combine(directory, VectorsFileName));
            File.Move(manifestTempPath, Path.Combine(directory, ManifestFileName));
        }
        catch
        {
            TryDelete(idsTempPath);
            TryDelete(vectorsTempPath);
            TryDelete(manifestTempPath);
            if (createdDirectory)
            {
                TryDeleteDirectoryIfEmpty(directory);
            }

            throw;
        }
    }

    internal static ExactFlatIndex OpenReadOnly(string directoryPath)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        string directory = GetFullDirectoryPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("Exact flat index path is an existing file, not a directory.");
        }

        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Exact flat index directory was not found: {directoryPath}");
        }

        string manifestPath = Path.Combine(directory, ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            throw new FileNotFoundException("Exact flat index manifest was not found.", manifestPath);
        }

        Manifest manifest = ReadManifest(manifestPath);
        string idsPath = ResolveManifestFilePath(directory, manifest.IdsFile.RelativePath, IdsFileName);
        string vectorsPath = ResolveManifestFilePath(directory, manifest.VectorsFile.RelativePath, VectorsFileName);

        ValidateFileExistsLengthAndHash(idsPath, manifest.IdsFile, "ID");
        ValidateFileExistsLengthAndHash(vectorsPath, manifest.VectorsFile, "vector");

        ulong[] ids = ReadIdsFile(idsPath, manifest.VectorCount);
        float[] vectors = ReadVectorsFile(vectorsPath, manifest.Dimension, manifest.Metric, manifest.VectorCount);

        ValidateHydratedRows(ids, vectors, manifest.Dimension, manifest.Metric);

        return ExactFlatIndex.HydrateReadOnly(manifest.Dimension, manifest.Metric, ids, vectors);
    }

    internal static void ValidateNewOrEmptyDirectoryPath(string directoryPath)
    {
        string directory = GetFullDirectoryPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("Exact flat index save path is an existing file, not a directory.");
        }

        if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
        {
            throw new IOException("Exact flat index save directory must be empty.");
        }
    }

    private static string PrepareSaveDirectory(string directoryPath, out bool createdDirectory)
    {
        string directory = GetFullDirectoryPath(directoryPath);
        if (File.Exists(directory))
        {
            throw new IOException("Exact flat index save path is an existing file, not a directory.");
        }

        if (Directory.Exists(directory))
        {
            if (Directory.EnumerateFileSystemEntries(directory).Any())
            {
                throw new IOException("Exact flat index save directory must be empty.");
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

    private static string GetFullDirectoryPath(string directoryPath)
    {
        ArgumentNullException.ThrowIfNull(directoryPath);
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Directory path must not be empty.", nameof(directoryPath));
        }

        return Path.GetFullPath(directoryPath);
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
            throw new InvalidOperationException("Vector payload length does not match index metadata.");
        }

        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[VectorsHeaderLength];
        VectorsMagic.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], BinaryMajorVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], BinaryMinorVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], VectorsHeaderLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], checked((ulong)rowCount));
        BinaryPrimitives.WriteUInt32LittleEndian(header[24..], checked((uint)dimension));
        BinaryPrimitives.WriteUInt32LittleEndian(header[28..], Float32RowMajorRepresentationCode);
        BinaryPrimitives.WriteUInt32LittleEndian(
            header[32..],
            metric == VectorMetric.Cosine ? CosineUnitNormalizedCode : NoNormalizationCode);
        BinaryPrimitives.WriteUInt32LittleEndian(header[36..], 0);
        BinaryPrimitives.WriteUInt64LittleEndian(header[40..], 0);
        stream.Write(header);

        Span<byte> buffer = stackalloc byte[sizeof(float)];
        foreach (float value in vectors)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buffer, BitConverter.SingleToInt32Bits(value));
            stream.Write(buffer);
        }
    }

    private static BinaryFileMetadata CreateBinaryFileMetadata(string path, string relativePath, string magic)
    {
        var file = new FileInfo(path);
        return new BinaryFileMetadata(
            relativePath,
            file.Length,
            ComputeSha256Hex(path),
            magic,
            BinaryVersionText);
    }

    private static void WriteManifest(
        string path,
        int dimension,
        VectorMetric metric,
        int vectorCount,
        BinaryFileMetadata idsFile,
        BinaryFileMetadata vectorsFile)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });

        writer.WriteStartObject();
        writer.WriteString("schemaName", ManifestSchemaName);
        writer.WriteString("schemaVersion", ManifestSchemaVersion);
        writer.WriteString("formatFamily", FormatFamily);
        writer.WriteString("createdUtc", DateTimeOffset.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'", CultureInfo.InvariantCulture));
        writer.WriteString("createdByTask", CreatedByTask);

        writer.WriteStartObject("writer");
        writer.WriteString("product", "VecNet");
        writer.WriteString("formatWriter", "ExactFlatIndex.Save");
        writer.WriteString("assemblyVersion", typeof(ExactFlatIndex).Assembly.GetName().Version?.ToString() ?? "unknown");
        writer.WriteEndObject();

        writer.WriteStartObject("index");
        writer.WriteNumber("dimension", dimension);
        writer.WriteString("metric", ToMetricText(metric));
        writer.WriteNumber("vectorCount", vectorCount);
        writer.WriteString("idType", IdType);
        writer.WriteString("vectorElementType", VectorElementType);
        writer.WriteString("vectorLayout", VectorLayout);
        writer.WriteString("normalizationState", GetNormalizationState(metric));
        writer.WriteEndObject();

        writer.WriteStartObject("semantics");
        writer.WriteString("distanceContract", DistanceContract);
        writer.WriteString("tiePolicy", TiePolicy);
        writer.WriteString("squaredEuclideanExecutionPolicy", SquaredEuclideanExecutionPolicy);
        writer.WriteBoolean("cosineQueryNormalization", true);
        writer.WriteEndObject();

        writer.WriteStartObject("files");
        WriteFileMetadata(writer, "ids", idsFile);
        WriteFileMetadata(writer, "vectors", vectorsFile);
        writer.WriteEndObject();

        writer.WriteStartObject("compatibility");
        writer.WriteStartArray("requiredFeatures");
        writer.WriteEndArray();
        writer.WriteStartArray("optionalFeatures");
        writer.WriteEndArray();
        writer.WriteNumber("minimumReaderMajorVersion", 1);
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
            throw new InvalidDataException("Exact flat index manifest is too large.");
        }

        try
        {
            using FileStream stream = File.OpenRead(manifestPath);
            using JsonDocument document = JsonDocument.Parse(stream);
            JsonElement root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("Exact flat index manifest root must be a JSON object.");
            }

            RequireString(root, "schemaName", ManifestSchemaName);
            RequireString(root, "schemaVersion", ManifestSchemaVersion);
            RequireString(root, "formatFamily", FormatFamily);
            ValidateCreatedUtc(GetRequiredString(root, "createdUtc"));
            _ = GetRequiredString(root, "createdByTask");
            ValidateWriter(GetRequiredObject(root, "writer"));

            JsonElement index = GetRequiredObject(root, "index");
            int dimension = GetRequiredInt32(index, "dimension", minimumValue: 1);
            VectorMetric metric = ParseMetric(GetRequiredString(index, "metric"));
            int vectorCount = GetRequiredInt32(index, "vectorCount", minimumValue: 0);
            RequireString(index, "idType", IdType);
            RequireString(index, "vectorElementType", VectorElementType);
            RequireString(index, "vectorLayout", VectorLayout);
            RequireString(index, "normalizationState", GetNormalizationState(metric));

            JsonElement semantics = GetRequiredObject(root, "semantics");
            RequireString(semantics, "distanceContract", DistanceContract);
            RequireString(semantics, "tiePolicy", TiePolicy);
            RequireString(semantics, "squaredEuclideanExecutionPolicy", SquaredEuclideanExecutionPolicy);
            RequireBoolean(semantics, "cosineQueryNormalization", expected: true);

            JsonElement files = GetRequiredObject(root, "files");
            BinaryFileMetadata ids = ReadFileMetadata(GetRequiredObject(files, "ids"), IdsFileName, IdsMagicText);
            BinaryFileMetadata vectors = ReadFileMetadata(GetRequiredObject(files, "vectors"), VectorsFileName, VectorsMagicText);

            ValidateCompatibility(GetRequiredObject(root, "compatibility"));

            return new Manifest(dimension, metric, vectorCount, ids, vectors);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("Exact flat index manifest JSON is malformed.", exception);
        }
    }

    private static void ValidateWriter(JsonElement writer)
    {
        _ = GetRequiredString(writer, "product");
        _ = GetRequiredString(writer, "formatWriter");
        _ = GetRequiredString(writer, "assemblyVersion");
    }

    private static BinaryFileMetadata ReadFileMetadata(
        JsonElement file,
        string expectedPath,
        string expectedMagic)
    {
        string relativePath = GetRequiredString(file, "path");
        long byteLength = GetRequiredInt64(file, "byteLength", minimumValue: 0);
        string sha256 = GetRequiredString(file, "sha256");
        ValidateSha256Text(sha256);
        RequireString(file, "binaryMagic", expectedMagic);
        RequireString(file, "binaryVersion", BinaryVersionText);

        if (!string.Equals(relativePath, expectedPath, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Exact flat index manifest file path is not the pinned format path.");
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
                throw new InvalidDataException("Exact flat index required features must be strings.");
            }

            throw new InvalidDataException("Exact flat index manifest contains an unknown required feature.");
        }

        JsonElement optionalFeatures = GetRequiredArray(compatibility, "optionalFeatures");
        foreach (JsonElement feature in optionalFeatures.EnumerateArray())
        {
            if (feature.ValueKind != JsonValueKind.String)
            {
                throw new InvalidDataException("Exact flat index optional features must be strings.");
            }
        }

        int minimumReaderMajorVersion = GetRequiredInt32(compatibility, "minimumReaderMajorVersion", minimumValue: 1);
        if (minimumReaderMajorVersion > BinaryMajorVersion)
        {
            throw new InvalidDataException("Exact flat index requires a newer reader major version.");
        }
    }

    private static string ResolveManifestFilePath(string directory, string relativePath, string expectedFileName)
    {
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath))
        {
            throw new InvalidDataException("Exact flat index manifest file paths must be relative file names.");
        }

        if (!string.Equals(relativePath, expectedFileName, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Exact flat index manifest file path is not the pinned format path.");
        }

        string resolved = Path.GetFullPath(Path.Combine(directory, relativePath));
        string root = Path.GetFullPath(directory);
        string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
            ? root
            : root + Path.DirectorySeparatorChar;
        if (!resolved.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Exact flat index manifest file path escapes the index directory.");
        }

        return resolved;
    }

    private static void ValidateFileExistsLengthAndHash(
        string path,
        BinaryFileMetadata metadata,
        string artifactName)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Exact flat index {artifactName} file was not found.", path);
        }

        long actualLength = new FileInfo(path).Length;
        if (actualLength != metadata.ByteLength)
        {
            throw new InvalidDataException($"Exact flat index {artifactName} file byte length does not match the manifest.");
        }

        string actualSha256 = ComputeSha256Hex(path);
        if (!string.Equals(actualSha256, metadata.Sha256, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Exact flat index {artifactName} file checksum does not match the manifest.");
        }
    }

    private static ulong[] ReadIdsFile(string path, int expectedRowCount)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < IdsHeaderLength)
        {
            throw new InvalidDataException("Exact flat index ID file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[IdsHeaderLength];
        stream.ReadExactly(header);

        if (!header[..8].SequenceEqual(IdsMagic))
        {
            throw new InvalidDataException("Exact flat index ID file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "ID");
        uint headerLength = BinaryPrimitives.ReadUInt32LittleEndian(header[12..]);
        if (headerLength != IdsHeaderLength)
        {
            throw new InvalidDataException("Exact flat index ID file header length is invalid.");
        }

        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(header[24..]);
        if (reserved != 0)
        {
            throw new InvalidDataException("Exact flat index ID file reserved header field must be zero.");
        }

        if (rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("Exact flat index ID file row count does not match the manifest.");
        }

        long expectedLength = checked(IdsHeaderLength + (long)expectedRowCount * sizeof(ulong));
        if (stream.Length != expectedLength)
        {
            throw new InvalidDataException("Exact flat index ID file payload length is invalid.");
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
        VectorMetric metric,
        int expectedRowCount)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < VectorsHeaderLength)
        {
            throw new InvalidDataException("Exact flat index vector file is shorter than the pinned header length.");
        }

        Span<byte> header = stackalloc byte[VectorsHeaderLength];
        stream.ReadExactly(header);

        if (!header[..8].SequenceEqual(VectorsMagic))
        {
            throw new InvalidDataException("Exact flat index vector file magic is invalid.");
        }

        ValidateBinaryVersion(header[8..], "vector");
        uint headerLength = BinaryPrimitives.ReadUInt32LittleEndian(header[12..]);
        if (headerLength != VectorsHeaderLength)
        {
            throw new InvalidDataException("Exact flat index vector file header length is invalid.");
        }

        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
        uint dimension = BinaryPrimitives.ReadUInt32LittleEndian(header[24..]);
        uint representationCode = BinaryPrimitives.ReadUInt32LittleEndian(header[28..]);
        uint normalizationCode = BinaryPrimitives.ReadUInt32LittleEndian(header[32..]);
        uint reserved0 = BinaryPrimitives.ReadUInt32LittleEndian(header[36..]);
        ulong reserved1 = BinaryPrimitives.ReadUInt64LittleEndian(header[40..]);

        if (rowCount != checked((ulong)expectedRowCount))
        {
            throw new InvalidDataException("Exact flat index vector file row count does not match the manifest.");
        }

        if (dimension != checked((uint)expectedDimension))
        {
            throw new InvalidDataException("Exact flat index vector file dimension does not match the manifest.");
        }

        if (representationCode != Float32RowMajorRepresentationCode)
        {
            throw new InvalidDataException("Exact flat index vector file representation code is unsupported.");
        }

        uint expectedNormalizationCode = metric == VectorMetric.Cosine ? CosineUnitNormalizedCode : NoNormalizationCode;
        if (normalizationCode != expectedNormalizationCode)
        {
            throw new InvalidDataException("Exact flat index vector file normalization code does not match the manifest.");
        }

        if (reserved0 != 0 || reserved1 != 0)
        {
            throw new InvalidDataException("Exact flat index vector file reserved header fields must be zero.");
        }

        int valueCount = checked(expectedRowCount * expectedDimension);
        long expectedLength = checked(VectorsHeaderLength + (long)valueCount * sizeof(float));
        if (stream.Length != expectedLength)
        {
            throw new InvalidDataException("Exact flat index vector file payload length is invalid.");
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

    private static void ValidateBinaryVersion(ReadOnlySpan<byte> versionBytes, string artifactName)
    {
        ushort major = BinaryPrimitives.ReadUInt16LittleEndian(versionBytes);
        ushort minor = BinaryPrimitives.ReadUInt16LittleEndian(versionBytes[2..]);
        if (major != BinaryMajorVersion || minor != BinaryMinorVersion)
        {
            throw new InvalidDataException($"Exact flat index {artifactName} file binary version is unsupported.");
        }
    }

    private static void ValidateHydratedRows(
        ulong[] ids,
        float[] vectors,
        int dimension,
        VectorMetric metric)
    {
        var seen = new HashSet<ulong>();
        foreach (ulong id in ids)
        {
            if (!seen.Add(id))
            {
                throw new InvalidDataException("Exact flat index contains duplicate external IDs.");
            }
        }

        for (int row = 0; row < ids.Length; row++)
        {
            double squaredMagnitude = 0;
            int offset = row * dimension;
            for (int i = 0; i < dimension; i++)
            {
                float value = vectors[offset + i];
                if (!float.IsFinite(value))
                {
                    throw new InvalidDataException("Exact flat index vector payload contains a non-finite component.");
                }

                if (metric == VectorMetric.Cosine)
                {
                    squaredMagnitude += (double)value * value;
                }
            }

            if (metric == VectorMetric.Cosine &&
                (squaredMagnitude == 0 ||
                 Math.Abs(squaredMagnitude - 1) > CosineStoredRowSquaredLengthTolerance))
            {
                throw new InvalidDataException("Exact flat index cosine stored row is not within the unit-length tolerance.");
            }
        }
    }

    private static string ComputeSha256Hex(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
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

    private static string ToMetricText(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => "squared-euclidean",
            VectorMetric.InnerProduct => "inner-product",
            VectorMetric.Cosine => "cosine",
            _ => throw new ArgumentOutOfRangeException(nameof(metric))
        };

    private static VectorMetric ParseMetric(string metric) =>
        metric switch
        {
            "squared-euclidean" => VectorMetric.SquaredEuclidean,
            "inner-product" => VectorMetric.InnerProduct,
            "cosine" => VectorMetric.Cosine,
            _ => throw new InvalidDataException("Exact flat index manifest metric is unsupported.")
        };

    private static string GetNormalizationState(VectorMetric metric) =>
        metric == VectorMetric.Cosine ? NormalizationCosineUnit : NormalizationNone;

    private static JsonElement GetRequiredObject(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' must be an object.");
        }

        return value;
    }

    private static JsonElement GetRequiredArray(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' must be an array.");
        }

        return value;
    }

    private static JsonElement GetRequiredProperty(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement value))
        {
            throw new InvalidDataException($"Exact flat index manifest is missing required property '{propertyName}'.");
        }

        return value;
    }

    private static string GetRequiredString(JsonElement element, string propertyName)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' must be a string.");
        }

        return value.GetString()!;
    }

    private static int GetRequiredInt32(JsonElement element, string propertyName, int minimumValue)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int result) || result < minimumValue)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' has an invalid integer value.");
        }

        return result;
    }

    private static long GetRequiredInt64(JsonElement element, string propertyName, long minimumValue)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt64(out long result) || result < minimumValue)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' has an invalid integer value.");
        }

        return result;
    }

    private static void RequireString(JsonElement element, string propertyName, string expected)
    {
        string actual = GetRequiredString(element, propertyName);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' is unsupported.");
        }
    }

    private static void RequireBoolean(JsonElement element, string propertyName, bool expected)
    {
        JsonElement value = GetRequiredProperty(element, propertyName);
        if (value.ValueKind != JsonValueKind.True && value.ValueKind != JsonValueKind.False)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' must be a boolean.");
        }

        if (value.GetBoolean() != expected)
        {
            throw new InvalidDataException($"Exact flat index manifest property '{propertyName}' is unsupported.");
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
            throw new InvalidDataException("Exact flat index manifest createdUtc is not in the pinned UTC format.");
        }
    }

    private static void ValidateSha256Text(string value)
    {
        if (value.Length != 64 || value.Any(static c => !IsLowerHex(c)))
        {
            throw new InvalidDataException("Exact flat index manifest SHA-256 must be lowercase hexadecimal text.");
        }
    }

    private static bool IsLowerHex(char value) =>
        value is >= '0' and <= '9' or >= 'a' and <= 'f';

    private sealed record Manifest(
        int Dimension,
        VectorMetric Metric,
        int VectorCount,
        BinaryFileMetadata IdsFile,
        BinaryFileMetadata VectorsFile);

    private sealed record BinaryFileMetadata(
        string RelativePath,
        long ByteLength,
        string Sha256,
        string BinaryMagic,
        string BinaryVersion);
}
