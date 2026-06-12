using System.Buffers.Binary;
using System.IO.Compression;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record IdxImageSet(int Count, int Rows, int Columns, byte[] Pixels)
{
    public int Dimension => Rows * Columns;

    public ReadOnlySpan<byte> GetImage(int row) => Pixels.AsSpan(row * Dimension, Dimension);
}

public sealed record IdxLabelSet(int Count, byte MinValue, byte MaxValue, int[] Histogram);

public static class IdxFileReader
{
    private const int ImageMagic = 2051;
    private const int LabelMagic = 2049;

    public static IdxImageSet ReadImages(string path, int expectedCount, int expectedRows, int expectedColumns)
    {
        using FileStream file = File.OpenRead(path);
        return ReadImages(file, expectedCount, expectedRows, expectedColumns);
    }

    public static IdxImageSet ReadImages(Stream gzipStream, int expectedCount, int expectedRows, int expectedColumns)
    {
        byte[] decoded = Decompress(gzipStream);
        if (decoded.Length < 16)
        {
            throw new InvalidDataException("IDX image file is truncated before the header.");
        }

        int magic = ReadInt32(decoded, 0);
        int count = ReadInt32(decoded, 4);
        int rows = ReadInt32(decoded, 8);
        int columns = ReadInt32(decoded, 12);
        if (magic != ImageMagic)
        {
            throw new InvalidDataException($"IDX image magic must be {ImageMagic}.");
        }

        if (count != expectedCount)
        {
            throw new InvalidDataException($"IDX image count must be {expectedCount}.");
        }

        if (rows != expectedRows || columns != expectedColumns)
        {
            throw new InvalidDataException($"IDX image dimensions must be {expectedRows} x {expectedColumns}.");
        }

        int payloadLength = checked(count * rows * columns);
        if (decoded.Length != 16 + payloadLength)
        {
            throw new InvalidDataException("IDX image payload length does not match the header.");
        }

        byte[] pixels = new byte[payloadLength];
        Buffer.BlockCopy(decoded, 16, pixels, 0, payloadLength);
        return new IdxImageSet(count, rows, columns, pixels);
    }

    public static IdxLabelSet ReadLabels(string path, int expectedCount)
    {
        using FileStream file = File.OpenRead(path);
        return ReadLabels(file, expectedCount);
    }

    public static IdxLabelSet ReadLabels(Stream gzipStream, int expectedCount)
    {
        byte[] decoded = Decompress(gzipStream);
        if (decoded.Length < 8)
        {
            throw new InvalidDataException("IDX label file is truncated before the header.");
        }

        int magic = ReadInt32(decoded, 0);
        int count = ReadInt32(decoded, 4);
        if (magic != LabelMagic)
        {
            throw new InvalidDataException($"IDX label magic must be {LabelMagic}.");
        }

        if (count != expectedCount)
        {
            throw new InvalidDataException($"IDX label count must be {expectedCount}.");
        }

        if (decoded.Length != 8 + count)
        {
            throw new InvalidDataException("IDX label payload length does not match the header.");
        }

        var histogram = new int[10];
        byte min = byte.MaxValue;
        byte max = byte.MinValue;
        for (int i = 0; i < count; i++)
        {
            byte value = decoded[8 + i];
            if (value > 9)
            {
                throw new InvalidDataException("IDX label values must be in the range 0..9.");
            }

            min = Math.Min(min, value);
            max = Math.Max(max, value);
            histogram[value]++;
        }

        return new IdxLabelSet(count, count == 0 ? (byte)0 : min, count == 0 ? (byte)0 : max, histogram);
    }

    private static byte[] Decompress(Stream gzipStream)
    {
        using var gzip = new GZipStream(gzipStream, CompressionMode.Decompress, leaveOpen: true);
        using var decoded = new MemoryStream();
        gzip.CopyTo(decoded);
        return decoded.ToArray();
    }

    private static int ReadInt32(byte[] bytes, int offset) =>
        BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(offset, 4));
}
