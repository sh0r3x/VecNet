using System.Security.Cryptography;

namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record RawFileVerification(
    string FileName,
    string Role,
    string SourceUrl,
    int ExpectedCount,
    string OfficialMd5,
    string ComputedMd5,
    string ComputedSha256,
    long ByteSize,
    string VerificationStatus);

public static class FileChecksum
{
    public static RawFileVerification VerifyRawFile(string path, FashionMnistRawFileSpec spec)
    {
        string computedMd5 = ComputeMd5(path);
        if (!string.Equals(computedMd5, spec.OfficialMd5, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Raw file '{spec.FileName}' MD5 mismatch. Expected {spec.OfficialMd5}, got {computedMd5}.");
        }

        return new RawFileVerification(
            spec.FileName,
            spec.Role,
            spec.SourceUrl,
            spec.ExpectedCount,
            spec.OfficialMd5,
            computedMd5,
            ComputeSha256(path),
            new FileInfo(path).Length,
            "passed");
    }

    public static string ComputeMd5(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] hash = MD5.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    public static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
