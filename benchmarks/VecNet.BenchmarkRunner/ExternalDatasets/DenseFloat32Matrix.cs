using System.Buffers.Binary;
using System.Text;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record DenseFloat32MatrixHeader(string SchemaName, string SchemaVersion, ulong RowCount, uint Dimension);

public static class DenseFloat32Matrix
{
    public const string SchemaName = "VecNetDenseFloat32Matrix";
    public const string SchemaVersion = "0.1";
    public const string MagicText = "VNDM001\0";

    private static readonly byte[] Magic = Encoding.ASCII.GetBytes(MagicText);

    public static float[] ConvertImages(IdxImageSet images)
    {
        float[] values = new float[checked(images.Count * images.Dimension)];
        for (int i = 0; i < images.Pixels.Length; i++)
        {
            values[i] = images.Pixels[i];
        }

        return values;
    }

    public static void Write(string path, int rowCount, int dimension, ReadOnlySpan<float> values)
    {
        if (rowCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rowCount), "Row count must be non-negative.");
        }

        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive.");
        }

        if (values.Length != checked(rowCount * dimension))
        {
            throw new ArgumentException("Matrix payload length does not match row count and dimension.", nameof(values));
        }

        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using FileStream stream = File.Create(path);
        Write(stream, rowCount, dimension, values);
    }

    public static void Write(Stream stream, int rowCount, int dimension, ReadOnlySpan<float> values)
    {
        Span<byte> buffer = stackalloc byte[8];
        stream.Write(Magic);
        BinaryPrimitives.WriteUInt64LittleEndian(buffer, (ulong)rowCount);
        stream.Write(buffer);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer[..4], (uint)dimension);
        stream.Write(buffer[..4]);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer[..4], 0);
        stream.Write(buffer[..4]);

        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer[..4], BitConverter.SingleToUInt32Bits(values[i]));
            stream.Write(buffer[..4]);
        }
    }

    public static DenseFloat32MatrixHeader ReadHeader(Stream stream)
    {
        Span<byte> magic = stackalloc byte[8];
        ReadExactly(stream, magic);
        if (!magic.SequenceEqual(Magic))
        {
            throw new InvalidDataException("Dense matrix magic is invalid.");
        }

        Span<byte> buffer = stackalloc byte[8];
        ReadExactly(stream, buffer);
        ulong rowCount = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
        ReadExactly(stream, buffer[..4]);
        uint dimension = BinaryPrimitives.ReadUInt32LittleEndian(buffer[..4]);
        ReadExactly(stream, buffer[..4]);
        uint reserved = BinaryPrimitives.ReadUInt32LittleEndian(buffer[..4]);
        if (reserved != 0)
        {
            throw new InvalidDataException("Dense matrix reserved header field must be zero.");
        }

        return new DenseFloat32MatrixHeader(SchemaName, SchemaVersion, rowCount, dimension);
    }

    public static float[] Read(string path, ulong expectedRowCount, uint expectedDimension)
    {
        using FileStream stream = File.OpenRead(path);
        DenseFloat32MatrixHeader header = ReadHeader(stream);
        if (header.RowCount != expectedRowCount)
        {
            throw new InvalidDataException($"Dense matrix row count mismatch for '{path}'. Expected {expectedRowCount}, got {header.RowCount}.");
        }

        if (header.Dimension != expectedDimension)
        {
            throw new InvalidDataException($"Dense matrix dimension mismatch for '{path}'. Expected {expectedDimension}, got {header.Dimension}.");
        }

        ulong elementCount = checked(header.RowCount * header.Dimension);
        if (elementCount > int.MaxValue)
        {
            throw new InvalidDataException("Dense matrix is too large to load into the benchmark runner process.");
        }

        ulong payloadBytes = checked(elementCount * sizeof(float));
        if (payloadBytes > int.MaxValue)
        {
            throw new InvalidDataException("Dense matrix payload is too large to load into the benchmark runner process.");
        }

        if (stream.CanSeek && stream.Length - stream.Position != (long)payloadBytes)
        {
            throw new InvalidDataException("Dense matrix payload byte length does not match header row count and dimension.");
        }

        byte[] payload = new byte[(int)payloadBytes];
        ReadExactly(stream, payload);
        if (stream.ReadByte() != -1)
        {
            throw new InvalidDataException("Dense matrix contains extra bytes after the expected payload.");
        }

        var values = new float[(int)elementCount];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(i * sizeof(float), sizeof(float))));
        }

        return values;
    }

    private static void ReadExactly(Stream stream, Span<byte> destination)
    {
        int totalRead = 0;
        while (totalRead < destination.Length)
        {
            int read = stream.Read(destination[totalRead..]);
            if (read == 0)
            {
                break;
            }

            totalRead += read;
        }

        if (totalRead != destination.Length)
        {
            throw new EndOfStreamException("Dense matrix content is truncated.");
        }
    }
}
