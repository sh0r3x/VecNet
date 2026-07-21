using System.Collections.Generic;
using System.Linq.Expressions;
using Microsoft.Extensions.VectorData;

namespace VecNet.Integration.VectorData;

internal interface IVecNetCollectionState
{
    bool Exists { get; }

    void DeleteCollection();
}

internal sealed class VecNetVectorStoreCollectionState<TKey, TRecord> : IVecNetCollectionState
    where TKey : notnull
    where TRecord : class
{
    private readonly object _gate = new();
    private readonly VecNetVectorDataModel<TRecord> _model;
    private ExactFlatIndex _index;
    private Dictionary<object, VecNetVectorDataEntry<TRecord>> _entriesByKey = [];
    private Dictionary<ulong, object> _keysByVectorId = [];
    private ulong _nextVectorId = 1;
    private bool _exists;

    private VecNetVectorStoreCollectionState(string name, VecNetVectorDataModel<TRecord> model)
    {
        Name = name;
        _model = model;
        _index = new ExactFlatIndex(model.Dimensions, model.Metric);
    }

    public string Name { get; }

    public bool Exists
    {
        get
        {
            lock (_gate)
            {
                return _exists;
            }
        }
    }

    public static VecNetVectorStoreCollectionState<TKey, TRecord> Create(
        string name,
        VectorStoreCollectionDefinition? definition)
    {
        var model = VecNetVectorDataModel<TRecord>.Create(definition);
        return new VecNetVectorStoreCollectionState<TKey, TRecord>(name, model);
    }

    public void EnsureCollectionExists()
    {
        lock (_gate)
        {
            _exists = true;
        }
    }

    public void DeleteCollection()
    {
        lock (_gate)
        {
            _entriesByKey = [];
            _keysByVectorId = [];
            _index = new ExactFlatIndex(_model.Dimensions, _model.Metric);
            _nextVectorId = 1;
            _exists = false;
        }
    }

    public TRecord? Get(TKey key, bool includeVectors)
    {
        object normalizedKey = NormalizeKey(key);
        lock (_gate)
        {
            EnsureExists();
            return _entriesByKey.TryGetValue(normalizedKey, out VecNetVectorDataEntry<TRecord>? entry)
                ? _model.ProjectRecord(entry.Record, includeVectors)
                : null;
        }
    }

    public IReadOnlyList<TRecord> Get(
        IEnumerable<TKey> keys,
        bool includeVectors,
        CancellationToken cancellationToken)
    {
        var records = new List<TRecord>();
        lock (_gate)
        {
            EnsureExists();
            foreach (TKey key in keys)
            {
                cancellationToken.ThrowIfCancellationRequested();
                object normalizedKey = NormalizeKey(key);
                if (_entriesByKey.TryGetValue(normalizedKey, out VecNetVectorDataEntry<TRecord>? entry))
                {
                    records.Add(_model.ProjectRecord(entry.Record, includeVectors));
                }
            }
        }

        return records;
    }

    public IReadOnlyList<TRecord> Get(
        Expression<Func<TRecord, bool>> filter,
        int top,
        bool includeVectors,
        FilteredRecordRetrievalOptions<TRecord>? options,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(filter);
        if (top < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(top), "Result count must not be negative.");
        }

        int skip = options?.Skip ?? 0;
        if (skip < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Skip must not be negative.");
        }

        var predicate = CompileFilter(filter);
        var records = new List<TRecord>(top);
        lock (_gate)
        {
            EnsureExists();
            foreach (VecNetVectorDataEntry<TRecord> entry in _entriesByKey.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!predicate(entry.Record))
                {
                    continue;
                }

                if (skip > 0)
                {
                    skip--;
                    continue;
                }

                if (records.Count == top)
                {
                    break;
                }

                records.Add(_model.ProjectRecord(entry.Record, includeVectors));
            }
        }

        return records;
    }

    public void Delete(TKey key)
    {
        object normalizedKey = NormalizeKey(key);
        lock (_gate)
        {
            EnsureExists();
            DeleteCore(normalizedKey);
        }
    }

    public void Delete(IEnumerable<TKey> keys, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            EnsureExists();
            foreach (TKey key in keys)
            {
                cancellationToken.ThrowIfCancellationRequested();
                DeleteCore(NormalizeKey(key));
            }
        }
    }

    public void Upsert(TRecord record)
    {
        ArgumentNullException.ThrowIfNull(record);
        object key = NormalizeKey(_model.GetKey(record));
        ReadOnlyMemory<float> vector = _model.GetVector(record);

        lock (_gate)
        {
            EnsureExists();
            UpsertCore(key, record, vector);
        }
    }

    public void Upsert(IEnumerable<TRecord> records, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            EnsureExists();
            foreach (TRecord record in records)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ArgumentNullException.ThrowIfNull(record);
                object key = NormalizeKey(_model.GetKey(record));
                ReadOnlyMemory<float> vector = _model.GetVector(record);
                UpsertCore(key, record, vector);
            }
        }
    }

    public IReadOnlyList<VectorSearchResult<TRecord>> Search<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (top < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(top), "Result count must not be negative.");
        }

        int skip = options?.Skip ?? 0;
        if (skip < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Skip must not be negative.");
        }

        _model.ValidateVectorPropertySelector(options?.VectorProperty);
        ReadOnlyMemory<float> query = VecNetVectorDataModel<TRecord>.GetSearchVector(searchValue);

        if (top == 0)
        {
            return [];
        }

        Expression<Func<TRecord, bool>>? filter = options?.Filter;
        Func<TRecord, bool>? predicate = filter is null ? null : CompileFilter(filter);
        double? scoreThreshold = options?.ScoreThreshold;

        lock (_gate)
        {
            EnsureExists();
            int requested = CalculateRequestedSearchResultCount(top, skip, scoreThreshold);
            if (requested == 0 || _entriesByKey.Count == 0)
            {
                return [];
            }

            SearchResult[] coreResults = new SearchResult[requested];
            int written;
            if (predicate is null)
            {
                written = _index.Search(query.Span, coreResults);
            }
            else
            {
                ulong[] allowedIds = BuildAllowlist(predicate, cancellationToken);
                if (allowedIds.Length == 0)
                {
                    return [];
                }

                var workspace = new ExactFlatSearchFilterWorkspace(_index.PhysicalVectorCount);
                written = _index.Search(query.Span, allowedIds, coreResults, workspace);
            }

            return ProjectResults(
                coreResults.AsSpan(0, written),
                skip,
                top,
                options?.IncludeVectors == true,
                scoreThreshold,
                cancellationToken);
        }
    }

    private static Func<TRecord, bool> CompileFilter(Expression<Func<TRecord, bool>> filter)
    {
        try
        {
            return filter.Compile();
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new VectorStoreException("The VectorData filter expression could not be compiled.", ex)
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "Filter"
            };
        }
    }

    private void UpsertCore(object key, TRecord record, ReadOnlyMemory<float> vector)
    {
        ulong vectorId = _nextVectorId++;
        VectorMutationResult addResult = _index.TryAdd(vectorId, vector.Span);
        if (addResult.Status != VectorMutationStatus.Committed)
        {
            throw new VectorStoreException(
                $"VecNet rejected the vector for key '{key}' with status '{addResult.Status}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                CollectionName = Name,
                OperationName = nameof(Upsert)
            };
        }

        if (_entriesByKey.TryGetValue(key, out VecNetVectorDataEntry<TRecord>? existing))
        {
            VectorMutationResult deleteResult = _index.TryDelete(existing.VectorId);
            if (deleteResult.Status is not VectorMutationStatus.Committed and not VectorMutationStatus.AlreadyDeleted)
            {
                _index.TryDelete(vectorId);
                throw new VectorStoreException(
                    $"VecNet failed to tombstone the previous vector for key '{key}'.")
                {
                    VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                    CollectionName = Name,
                    OperationName = nameof(Upsert)
                };
            }

            _keysByVectorId.Remove(existing.VectorId);
        }

        _entriesByKey[key] = new VecNetVectorDataEntry<TRecord>(record, vectorId);
        _keysByVectorId.Add(vectorId, key);
    }

    private void DeleteCore(object key)
    {
        if (!_entriesByKey.Remove(key, out VecNetVectorDataEntry<TRecord>? entry))
        {
            return;
        }

        _keysByVectorId.Remove(entry.VectorId);
        VectorMutationResult result = _index.TryDelete(entry.VectorId);
        if (result.Status is not VectorMutationStatus.Committed and not VectorMutationStatus.AlreadyDeleted)
        {
            throw new VectorStoreException($"VecNet failed to delete the vector for key '{key}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                CollectionName = Name,
                OperationName = nameof(Delete)
            };
        }
    }

    private ulong[] BuildAllowlist(Func<TRecord, bool> predicate, CancellationToken cancellationToken)
    {
        var allowedIds = new ulong[_entriesByKey.Count];
        int count = 0;
        foreach (VecNetVectorDataEntry<TRecord> entry in _entriesByKey.Values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (predicate(entry.Record))
            {
                allowedIds[count++] = entry.VectorId;
            }
        }

        if (count != allowedIds.Length)
        {
            Array.Resize(ref allowedIds, count);
        }

        return allowedIds;
    }

    private IReadOnlyList<VectorSearchResult<TRecord>> ProjectResults(
        ReadOnlySpan<SearchResult> coreResults,
        int skip,
        int top,
        bool includeVectors,
        double? scoreThreshold,
        CancellationToken cancellationToken)
    {
        var results = new List<VectorSearchResult<TRecord>>(Math.Min(top, coreResults.Length));
        for (int i = 0; i < coreResults.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            SearchResult coreResult = coreResults[i];
            double score = _model.ProjectScore(coreResult.Distance);
            if (!_model.PassesThreshold(score, scoreThreshold))
            {
                continue;
            }

            if (skip > 0)
            {
                skip--;
                continue;
            }

            if (results.Count == top)
            {
                break;
            }

            if (!_keysByVectorId.TryGetValue(coreResult.Id, out object? key) ||
                !_entriesByKey.TryGetValue(key, out VecNetVectorDataEntry<TRecord>? entry))
            {
                throw new VectorStoreException("VecNet returned a vector ID that is not in the adapter key map.")
                {
                    VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                    CollectionName = Name,
                    OperationName = nameof(Search)
                };
            }

            results.Add(new VectorSearchResult<TRecord>(_model.ProjectRecord(entry.Record, includeVectors), score));
        }

        return results;
    }

    private int CalculateRequestedSearchResultCount(int top, int skip, double? scoreThreshold)
    {
        if (top == 0)
        {
            return 0;
        }

        if (scoreThreshold.HasValue)
        {
            return _entriesByKey.Count;
        }

        try
        {
            return Math.Min(checked(top + skip), _entriesByKey.Count);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(nameof(top), "The requested top plus skip count is too large.");
        }
    }

    private void EnsureExists()
    {
        if (!_exists)
        {
            throw new VectorStoreException($"Collection '{Name}' does not exist.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                CollectionName = Name
            };
        }
    }

    private static object NormalizeKey(TKey key)
    {
        if (key is null)
        {
            throw new ArgumentNullException(nameof(key), "VectorData record keys must not be null.");
        }

        return key;
    }

    private static object NormalizeKey(object? key)
    {
        if (key is null)
        {
            throw new VectorStoreException("VectorData record keys must not be null.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "KeyMapping"
            };
        }

        if (key is not TKey)
        {
            throw new VectorStoreException(
                $"The record key type '{key.GetType()}' does not match the collection key type '{typeof(TKey)}'.")
            {
                VectorStoreSystemName = VecNetVectorDataConstants.SystemName,
                OperationName = "KeyMapping"
            };
        }

        return key;
    }
}

internal sealed record VecNetVectorDataEntry<TRecord>(TRecord Record, ulong VectorId)
    where TRecord : class;
