namespace VecNet;

/// <summary>
/// Status returned by a status-reporting vector mutation operation.
/// </summary>
public enum VectorMutationStatus
{
    /// <summary>
    /// The mutation was committed and published a new visible generation.
    /// </summary>
    Committed,

    /// <summary>
    /// The inserted external ID is already visible or reserved by a prior delete.
    /// </summary>
    DuplicateId,

    /// <summary>
    /// The deleted external ID is not known to the current index instance.
    /// </summary>
    UnknownId,

    /// <summary>
    /// The deleted external ID is already tombstoned.
    /// </summary>
    AlreadyDeleted,

    /// <summary>
    /// The mutation was rejected because the index instance is read-only.
    /// </summary>
    ReadOnly,

    /// <summary>
    /// The mutation is unsupported by the selected index mode.
    /// </summary>
    Unsupported
}
