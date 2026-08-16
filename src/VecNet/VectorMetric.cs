namespace VecNet;

/// <summary>
/// Defines the canonical distance computation used by an index.
/// </summary>
public enum VectorMetric
{
    /// <summary>
    /// Squared Euclidean distance: lower squared L2 values are nearer.
    /// </summary>
    SquaredEuclidean,

    /// <summary>
    /// Negative inner product distance: larger dot products become lower distances.
    /// </summary>
    /// <remarks>
    /// Vectors are used as supplied; VecNet does not normalize or clamp inner-product distances.
    /// Finite inputs can therefore produce positive or negative infinity when the negative dot
    /// product exceeds the <see cref="float"/> range.
    /// </remarks>
    InnerProduct,

    /// <summary>
    /// Cosine distance after index-normalized insertion and query inputs: lower values are nearer.
    /// </summary>
    Cosine
}
