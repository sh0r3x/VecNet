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
    /// Vectors are used as supplied; VecNet does not normalize or clamp inner-product vectors.
    /// Magnitude affects ranking, zero vectors are valid, and larger dot products become lower
    /// negative distances. An indexed vector is not necessarily its own nearest result under
    /// maximum inner product; use <see cref="Cosine"/> when direction-only similarity is desired.
    /// Finite inputs can produce positive or negative infinity when the negative dot product
    /// exceeds the <see cref="float"/> range.
    /// </remarks>
    InnerProduct,

    /// <summary>
    /// Cosine distance after index-normalized insertion and query inputs: lower values are nearer.
    /// </summary>
    /// <remarks>
    /// The canonical cosine distance range is [0, 2]. Tiny floating-point excursions up to 1e-6
    /// below zero or above two are tolerated.
    /// </remarks>
    Cosine
}
