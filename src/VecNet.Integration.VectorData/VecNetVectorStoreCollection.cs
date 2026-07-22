using System.Collections.Generic;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.VectorData;

namespace VecNet.Integration.VectorData;

/// <summary>
/// Represents an in-memory exact-flat VecNet collection exposed through Microsoft.Extensions.VectorData.
/// </summary>
/// <remarks>
/// Retrieval honors VectorData <c>IncludeVectors</c> options. Null/default options and explicit
/// <c>IncludeVectors = false</c> omit vector values from returned records; explicit
/// <c>IncludeVectors = true</c> includes vector values. When vector omission requires projecting a
/// record shape the adapter does not support, the operation throws <see cref="NotSupportedException"/>.
/// </remarks>
/// <typeparam name="TKey">The VectorData record key type.</typeparam>
/// <typeparam name="TRecord">The VectorData record type.</typeparam>
public sealed class VecNetVectorStoreCollection<TKey, TRecord> : VectorStoreCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly VecNetVectorStoreCollectionState<TKey, TRecord> _state;

    internal VecNetVectorStoreCollection(VecNetVectorStoreCollectionState<TKey, TRecord> state)
    {
        _state = state;
    }

    /// <inheritdoc />
    public override string Name => _state.Name;

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(_state.Exists);
    }

    /// <inheritdoc />
    public override Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _state.EnsureCollectionExists();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _state.DeleteCollection();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task<TRecord?> GetAsync(
        TKey key,
        RecordRetrievalOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(_state.Get(key, IncludeVectors(options)));
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<TRecord> GetAsync(
        IEnumerable<TKey> keys,
        RecordRetrievalOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(keys);
        IReadOnlyList<TRecord> records = _state.Get(keys, IncludeVectors(options), cancellationToken);
        foreach (TRecord record in records)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return record;
        }
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<TRecord> GetAsync(
        Expression<Func<TRecord, bool>> filter,
        int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        IReadOnlyList<TRecord> records = _state.Get(filter, top, IncludeVectors(options), options, cancellationToken);
        foreach (TRecord record in records)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return record;
        }
    }

    /// <inheritdoc />
    public override Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _state.Delete(key);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task DeleteAsync(IEnumerable<TKey> keys, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(keys);
        _state.Delete(keys, cancellationToken);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _state.Upsert(record);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        _state.Upsert(records, cancellationToken);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        IReadOnlyList<VectorSearchResult<TRecord>> results = _state.Search(searchValue, top, options, cancellationToken);
        foreach (VectorSearchResult<TRecord> result in results)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return result;
        }
    }

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        if (serviceType == typeof(VectorStoreCollectionMetadata))
        {
            return new VectorStoreCollectionMetadata
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                CollectionName = Name
            };
        }

        return null;
    }

    private static bool IncludeVectors(RecordRetrievalOptions? options) => options?.IncludeVectors == true;

    private static bool IncludeVectors(FilteredRecordRetrievalOptions<TRecord>? options) =>
        options?.IncludeVectors == true;
}
