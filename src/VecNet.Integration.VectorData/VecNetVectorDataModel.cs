using System.Linq.Expressions;
using System.Reflection;
using Microsoft.Extensions.VectorData;

namespace VecNet.Integration.VectorData;

internal sealed class VecNetVectorDataModel<TRecord>
    where TRecord : class
{
    private VecNetVectorDataModel(
        PropertyInfo keyProperty,
        PropertyInfo vectorProperty,
        int dimensions,
        string distanceFunction,
        VectorMetric metric)
    {
        KeyProperty = keyProperty;
        VectorProperty = vectorProperty;
        Dimensions = dimensions;
        DistanceFunction = distanceFunction;
        Metric = metric;
        IsSimilarityScore =
            distanceFunction == Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity ||
            distanceFunction == Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity;
        ProjectionConstructor = typeof(TRecord).GetConstructor(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            Type.EmptyTypes,
            modifiers: null);
        ProjectionProperties = typeof(TRecord)
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.GetIndexParameters().Length == 0 && property.GetMethod is not null)
            .ToArray();
    }

    public PropertyInfo KeyProperty { get; }

    public PropertyInfo VectorProperty { get; }

    public int Dimensions { get; }

    public string DistanceFunction { get; }

    public VectorMetric Metric { get; }

    public bool IsSimilarityScore { get; }

    private ConstructorInfo? ProjectionConstructor { get; }

    private PropertyInfo[] ProjectionProperties { get; }

    public static VecNetVectorDataModel<TRecord> Create(VectorStoreCollectionDefinition? definition)
    {
        if (definition?.EmbeddingGenerator is not null)
        {
            throw new NotSupportedException("VecNet VectorData does not support collection-level embedding generation.");
        }

        if (definition is not null)
        {
            return CreateFromDefinition(definition);
        }

        return CreateFromAttributes();
    }

    public object? GetKey(TRecord record) => KeyProperty.GetValue(record);

    public ReadOnlyMemory<float> GetVector(TRecord record)
    {
        object? value = VectorProperty.GetValue(record);
        return value switch
        {
            ReadOnlyMemory<float> memory => memory,
            float[] array => array,
            null => throw new VectorStoreException("The VectorData vector property value must not be null.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "VectorMapping"
            },
            _ => throw new NotSupportedException(
                $"VecNet VectorData supports only float[] and ReadOnlyMemory<float> vector properties, not '{value.GetType()}'.")
        };
    }

    public static ReadOnlyMemory<float> GetSearchVector<TInput>(TInput searchValue)
    {
        return searchValue switch
        {
            ReadOnlyMemory<float> memory => memory,
            float[] array => array,
            null => throw new ArgumentNullException(nameof(searchValue), "Search vector must not be null."),
            _ => throw new NotSupportedException(
                $"VecNet VectorData supports only float[] and ReadOnlyMemory<float> search vectors, not '{typeof(TInput)}'.")
        };
    }

    public void ValidateVectorPropertySelector(Expression<Func<TRecord, object?>>? vectorProperty)
    {
        if (vectorProperty is null)
        {
            return;
        }

        string selectedName = GetSelectedPropertyName(vectorProperty);
        if (!StringComparer.Ordinal.Equals(selectedName, VectorProperty.Name))
        {
            throw new NotSupportedException(
                $"VecNet VectorData collection '{typeof(TRecord).Name}' has one vector property, '{VectorProperty.Name}', and cannot search '{selectedName}'.");
        }
    }

    public double ProjectScore(float vecNetDistance)
    {
        return DistanceFunction switch
        {
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance => vecNetDistance,
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanDistance => Math.Sqrt(vecNetDistance),
            Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance => vecNetDistance,
            Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity => 1 - vecNetDistance,
            Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity => -vecNetDistance,
            _ => throw new InvalidOperationException("Unsupported VectorData distance function.")
        };
    }

    public bool PassesThreshold(double score, double? scoreThreshold)
    {
        if (!scoreThreshold.HasValue)
        {
            return true;
        }

        return IsSimilarityScore
            ? score >= scoreThreshold.Value
            : score <= scoreThreshold.Value;
    }

    public TRecord ProjectRecord(TRecord record, bool includeVectors)
    {
        if (includeVectors)
        {
            return record;
        }

        if (ProjectionConstructor is null)
        {
            throw new NotSupportedException(
                $"VecNet VectorData cannot omit vectors for record type '{typeof(TRecord)}' because it does not have a parameterless constructor.");
        }

        var projected = (TRecord)ProjectionConstructor.Invoke(null);
        foreach (PropertyInfo property in ProjectionProperties)
        {
            if (property.SetMethod is null || !property.SetMethod.IsPublic)
            {
                throw new NotSupportedException(
                    $"VecNet VectorData cannot omit vectors for record type '{typeof(TRecord)}' because public property '{property.Name}' is not publicly settable.");
            }

            object? value = property == VectorProperty ? GetDefaultValue(property.PropertyType) : property.GetValue(record);
            property.SetValue(projected, value);
        }

        return projected;
    }

    private static VecNetVectorDataModel<TRecord> CreateFromDefinition(VectorStoreCollectionDefinition definition)
    {
        VectorStoreKeyProperty keyPropertyDefinition = GetSingleDefinitionProperty<VectorStoreKeyProperty>(
            definition,
            "key");
        VectorStoreVectorProperty vectorPropertyDefinition = GetSingleDefinitionProperty<VectorStoreVectorProperty>(
            definition,
            "vector");

        if (keyPropertyDefinition.IsAutoGenerated == true)
        {
            throw new NotSupportedException("VecNet VectorData does not support store-generated keys.");
        }

        if (vectorPropertyDefinition.EmbeddingGenerator is not null)
        {
            throw new NotSupportedException("VecNet VectorData does not support vector-property embedding generation.");
        }

        PropertyInfo keyProperty = GetRecordProperty(keyPropertyDefinition.Name);
        PropertyInfo vectorProperty = GetRecordProperty(vectorPropertyDefinition.Name);
        ValidateVectorType(vectorProperty);
        ValidateConfiguredVectorType(vectorPropertyDefinition.Type, vectorProperty);
        string indexKind = vectorPropertyDefinition.IndexKind ?? Microsoft.Extensions.VectorData.IndexKind.Flat;
        string distanceFunction =
            vectorPropertyDefinition.DistanceFunction ??
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance;

        return CreateValidated(
            keyProperty,
            vectorProperty,
            vectorPropertyDefinition.Dimensions,
            indexKind,
            distanceFunction);
    }

    private static VecNetVectorDataModel<TRecord> CreateFromAttributes()
    {
        PropertyInfo keyProperty = GetSingleAttributedProperty<VectorStoreKeyAttribute>("key");
        PropertyInfo vectorProperty = GetSingleAttributedProperty<VectorStoreVectorAttribute>("vector");
        ValidateVectorType(vectorProperty);

        var keyAttribute = keyProperty.GetCustomAttribute<VectorStoreKeyAttribute>(inherit: true)!;
        if (keyAttribute.IsAutoGenerated)
        {
            throw new NotSupportedException("VecNet VectorData does not support store-generated keys.");
        }

        var vectorAttribute = vectorProperty.GetCustomAttribute<VectorStoreVectorAttribute>(inherit: true)!;
        string indexKind = vectorAttribute.IndexKind ?? Microsoft.Extensions.VectorData.IndexKind.Flat;
        string distanceFunction =
            vectorAttribute.DistanceFunction ??
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance;

        return CreateValidated(
            keyProperty,
            vectorProperty,
            vectorAttribute.Dimensions,
            indexKind,
            distanceFunction);
    }

    private static VecNetVectorDataModel<TRecord> CreateValidated(
        PropertyInfo keyProperty,
        PropertyInfo vectorProperty,
        int dimensions,
        string indexKind,
        string distanceFunction)
    {
        if (dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Vector dimensions must be positive.");
        }

        if (indexKind != Microsoft.Extensions.VectorData.IndexKind.Flat)
        {
            throw new NotSupportedException(
                $"VecNet VectorData supports only IndexKind.Flat, not '{indexKind}'.");
        }

        VectorMetric metric = distanceFunction switch
        {
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanSquaredDistance => VectorMetric.SquaredEuclidean,
            Microsoft.Extensions.VectorData.DistanceFunction.EuclideanDistance => VectorMetric.SquaredEuclidean,
            Microsoft.Extensions.VectorData.DistanceFunction.CosineDistance => VectorMetric.Cosine,
            Microsoft.Extensions.VectorData.DistanceFunction.CosineSimilarity => VectorMetric.Cosine,
            Microsoft.Extensions.VectorData.DistanceFunction.DotProductSimilarity => VectorMetric.InnerProduct,
            Microsoft.Extensions.VectorData.DistanceFunction.NegativeDotProductSimilarity =>
                throw new NotSupportedException(
                    "VecNet VectorData does not support NegativeDotProductSimilarity in the first exact-flat prototype."),
            Microsoft.Extensions.VectorData.DistanceFunction.HammingDistance =>
                throw new NotSupportedException("VecNet VectorData does not support HammingDistance."),
            Microsoft.Extensions.VectorData.DistanceFunction.ManhattanDistance =>
                throw new NotSupportedException("VecNet VectorData does not support ManhattanDistance."),
            _ => throw new NotSupportedException(
                $"VecNet VectorData does not support distance function '{distanceFunction}'.")
        };

        return new VecNetVectorDataModel<TRecord>(
            keyProperty,
            vectorProperty,
            dimensions,
            distanceFunction,
            metric);
    }

    private static TProperty GetSingleDefinitionProperty<TProperty>(
        VectorStoreCollectionDefinition definition,
        string logicalName)
        where TProperty : VectorStoreProperty
    {
        List<TProperty> properties = definition.Properties.OfType<TProperty>().ToList();
        return properties.Count switch
        {
            1 => properties[0],
            0 => throw new VectorStoreException(
                $"VecNet VectorData requires exactly one {logicalName} property in the collection definition.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "Schema"
            },
            _ => throw new NotSupportedException(
                $"VecNet VectorData supports exactly one {logicalName} property, not {properties.Count}.")
        };
    }

    private static PropertyInfo GetSingleAttributedProperty<TAttribute>(string logicalName)
        where TAttribute : Attribute
    {
        List<PropertyInfo> properties = typeof(TRecord)
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.GetCustomAttribute<TAttribute>(inherit: true) is not null)
            .ToList();

        return properties.Count switch
        {
            1 => properties[0],
            0 => throw new VectorStoreException(
                $"VecNet VectorData requires exactly one [{typeof(TAttribute).Name}] {logicalName} property on '{typeof(TRecord)}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "Schema"
            },
            _ => throw new NotSupportedException(
                $"VecNet VectorData supports exactly one {logicalName} property, not {properties.Count}.")
        };
    }

    private static PropertyInfo GetRecordProperty(string propertyName)
    {
        PropertyInfo? property = typeof(TRecord).GetProperty(
            propertyName,
            BindingFlags.Instance | BindingFlags.Public);
        if (property is null)
        {
            throw new VectorStoreException(
                $"The VectorData collection definition refers to property '{propertyName}', which was not found on '{typeof(TRecord)}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "Schema"
            };
        }

        return property;
    }

    private static void ValidateVectorType(PropertyInfo vectorProperty)
    {
        if (vectorProperty.PropertyType != typeof(ReadOnlyMemory<float>) &&
            vectorProperty.PropertyType != typeof(float[]))
        {
            throw new NotSupportedException(
                $"VecNet VectorData supports only float[] and ReadOnlyMemory<float> vector properties, not '{vectorProperty.PropertyType}'.");
        }
    }

    private static void ValidateConfiguredVectorType(Type? configuredType, PropertyInfo vectorProperty)
    {
        if (configuredType is null)
        {
            return;
        }

        if (configuredType != typeof(ReadOnlyMemory<float>) && configuredType != typeof(float[]))
        {
            throw new NotSupportedException(
                $"VecNet VectorData supports only float[] and ReadOnlyMemory<float> configured vector property types, not '{configuredType}'.");
        }

        if (configuredType != vectorProperty.PropertyType)
        {
            throw new VectorStoreException(
                $"The configured vector type '{configuredType}' does not match record property '{vectorProperty.Name}' type '{vectorProperty.PropertyType}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "Schema"
            };
        }
    }

    private static string GetSelectedPropertyName(Expression<Func<TRecord, object?>> selector)
    {
        Expression expression = selector.Body;
        if (expression is UnaryExpression { NodeType: ExpressionType.Convert } unary)
        {
            expression = unary.Operand;
        }

        if (expression is MemberExpression { Member: PropertyInfo property })
        {
            return property.Name;
        }

        throw new NotSupportedException(
            "VecNet VectorData supports VectorSearchOptions.VectorProperty only when it selects the configured vector property.");
    }

    private static object? GetDefaultValue(Type type) => type.IsValueType ? Activator.CreateInstance(type) : null;
}
