namespace VecNet.BenchmarkRunner.ExternalDatasets;

public sealed record FashionMnistRawFileSpec(
    string FileName,
    string Role,
    int ExpectedCount,
    string OfficialMd5,
    string SourceUrl);

public sealed record FashionMnistDatasetSpecification(
    string DatasetId,
    string MaintainerUrl,
    string DownloadRoot,
    string OfficialReadmeUrl,
    string LicenseUrl,
    string LicenseName,
    string Copyright,
    string AccessDate,
    string CitationDate,
    int BaseCount,
    int QueryCount,
    int ImageRows,
    int ImageColumns,
    int Dimension,
    FashionMnistRawFileSpec TrainImages,
    FashionMnistRawFileSpec TrainLabels,
    FashionMnistRawFileSpec QueryImages,
    FashionMnistRawFileSpec QueryLabels)
{
    public const string EuclideanDatasetId = "fashion-mnist-784-euclidean";
    public const string CosineDatasetId = "fashion-mnist-784-cosine";
    public const string InnerProductDatasetId = "fashion-mnist-784-inner-product";
    public const string RawDatasetId = EuclideanDatasetId;

    public static FashionMnistDatasetSpecification Official { get; } = CreateOfficial();

    public FashionMnistRawFileSpec[] RawFiles => [TrainImages, TrainLabels, QueryImages, QueryLabels];

    public static string GetDatasetId(VectorMetric metric) =>
        metric switch
        {
            VectorMetric.SquaredEuclidean => EuclideanDatasetId,
            VectorMetric.InnerProduct => InnerProductDatasetId,
            VectorMetric.Cosine => CosineDatasetId,
            _ => throw new ArgumentException("Fashion-MNIST external runners support only SquaredEuclidean, InnerProduct and Cosine.", nameof(metric))
        };

    public FashionMnistDatasetSpecification WithMetricIdentity(VectorMetric metric) =>
        this with { DatasetId = GetDatasetId(metric) };

    private static FashionMnistDatasetSpecification CreateOfficial()
    {
        const string downloadRoot = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";

        static FashionMnistRawFileSpec RawFile(string fileName, string role, int expectedCount, string officialMd5) =>
            new(fileName, role, expectedCount, officialMd5, downloadRoot + fileName);

        return new FashionMnistDatasetSpecification(
            EuclideanDatasetId,
            MaintainerUrl: "https://github.com/zalandoresearch/fashion-mnist",
            DownloadRoot: downloadRoot,
            OfficialReadmeUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
            LicenseUrl: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
            LicenseName: "MIT",
            Copyright: "Copyright 2017 Zalando SE",
            AccessDate: "2026-06-12",
            CitationDate: "2017-08-28",
            BaseCount: 60_000,
            QueryCount: 10_000,
            ImageRows: 28,
            ImageColumns: 28,
            Dimension: 784,
            TrainImages: RawFile("train-images-idx3-ubyte.gz", "base-images", 60_000, "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
            TrainLabels: RawFile("train-labels-idx1-ubyte.gz", "base-labels", 60_000, "25c81989df183df01b3e8a0aad5dffbe"),
            QueryImages: RawFile("t10k-images-idx3-ubyte.gz", "query-images", 10_000, "bef4ecab320f06d8554ea6380940ec79"),
            QueryLabels: RawFile("t10k-labels-idx1-ubyte.gz", "query-labels", 10_000, "bb300cfdad3c16e7a12a480ee83cd310"));
    }
}
