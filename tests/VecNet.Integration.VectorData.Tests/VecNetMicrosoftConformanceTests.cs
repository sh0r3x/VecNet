using Microsoft.Extensions.VectorData;
using VecNet.Integration.VectorData;
using VectorData.ConformanceTests;
using VectorData.ConformanceTests.ModelTests;
using VectorData.ConformanceTests.Support;
using VectorData.ConformanceTests.TypeTests;

namespace VecNet.Integration.VectorData.Tests;

public sealed class VecNetBasicModelConformanceTests(VecNetBasicModelConformanceTests.Fixture fixture) :
    BasicModelTests<string>(fixture),
    IClassFixture<VecNetBasicModelConformanceTests.Fixture>
{
    public new sealed class Fixture : BasicModelTests<string>.Fixture
    {
        public override TestStore TestStore => VecNetConformanceTestStore.Instance;
    }
}

public sealed class VecNetCollectionManagementConformanceTests(
    VecNetCollectionManagementConformanceTests.Fixture fixture) :
    CollectionManagementTests<string>(fixture),
    IClassFixture<VecNetCollectionManagementConformanceTests.Fixture>
{
    public sealed class Fixture : VectorStoreFixture
    {
        public override TestStore TestStore => VecNetConformanceTestStore.Instance;
    }
}

public sealed class VecNetConformanceSuiteImplementationTests : TestSuiteImplementationTests
{
    protected override ICollection<Type> IgnoredTestBases =>
    [
        typeof(DependencyInjectionTests<,,,>),
        typeof(DependencyInjectionTests<>),
        typeof(DistanceFunctionTests<>),
        typeof(EmbeddingGenerationTests<>),
        typeof(FilterTests<>),
        typeof(HybridSearchTests<>),
        typeof(IndexKindTests<>),
        typeof(MultiVectorModelTests<>),
        typeof(DynamicModelTests<>),
        typeof(NoDataModelTests<>),
        typeof(NoVectorModelTests<>),
        typeof(DataTypeTests<,>),
        typeof(DataTypeTests<>),
        typeof(EmbeddingTypeTests<>),
        typeof(KeyTypeTests)
    ];
}

internal sealed class VecNetConformanceTestStore : TestStore
{
    public static readonly VecNetConformanceTestStore Instance = new();

    public override string DefaultDistanceFunction => DistanceFunction.EuclideanSquaredDistance;

    protected override Task StartAsync()
    {
        DefaultVectorStore = new VecNetVectorStore();
        return Task.CompletedTask;
    }
}
