using System.Collections.Generic;
using Microsoft.Extensions.VectorData;

namespace VecNet.Integration.VectorData;

/// <summary>
/// Provides in-memory exact-flat VecNet collections through the Microsoft.Extensions.VectorData abstraction.
/// </summary>
/// <remarks>
/// Collections created by this store are in-memory exact-flat VecNet collections. The adapter is
/// not HNSW storage, durable VectorData storage, embedding generation, hybrid search, or an
/// application record store.
/// </remarks>
public sealed class VecNetVectorStore : VectorStore
{
    private readonly object _gate = new();
    private readonly Dictionary<string, IVecNetCollectionState> _collections = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(
        string name,
        VectorStoreCollectionDefinition? definition = null)
        where TRecord : class
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        lock (_gate)
        {
            if (_collections.TryGetValue(name, out IVecNetCollectionState? existing))
            {
                if (existing is VecNetVectorStoreCollectionState<TKey, TRecord> typedState)
                {
                    return new VecNetVectorStoreCollection<TKey, TRecord>(typedState);
                }

                throw new VectorStoreException(
                    $"Collection '{name}' was already requested with a different key or record type.")
                {
                    VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                    CollectionName = name,
                    OperationName = nameof(GetCollection)
                };
            }

            var state = VecNetVectorStoreCollectionState<TKey, TRecord>.Create(name, definition);
            _collections.Add(name, state);
            return new VecNetVectorStoreCollection<TKey, TRecord>(state);
        }
    }

    /// <inheritdoc />
    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(
        string name,
        VectorStoreCollectionDefinition definition) =>
        throw new NotSupportedException("VecNet VectorData dynamic dictionary collections are not supported.");

    /// <inheritdoc />
    public override async IAsyncEnumerable<string> ListCollectionNamesAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string[] names;
        lock (_gate)
        {
            names = _collections
                .Where(pair => pair.Value.Exists)
                .Select(pair => pair.Key)
                .Order(StringComparer.Ordinal)
                .ToArray();
        }

        foreach (string name in names)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return name;
        }
    }

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate)
        {
            return Task.FromResult(_collections.TryGetValue(name, out IVecNetCollectionState? state) && state.Exists);
        }
    }

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate)
        {
            if (_collections.TryGetValue(name, out IVecNetCollectionState? state))
            {
                state.DeleteCollection();
            }
        }

        return Task.CompletedTask;
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

        if (serviceType == typeof(VectorStoreMetadata))
        {
            return new VectorStoreMetadata
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName
            };
        }

        return null;
    }
}
