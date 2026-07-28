# Changelog

## 1.2.1 - 2026-07-28

### Changed

- `VecNet` and `VecNet.Integration.VectorData` are aligned at package version
  `1.2.1`.
- Packages are validated against the previous stable package line by default
  and against the first stable `1.x` package line for broader package/API
  compatibility coverage.

### Compatibility

- The stable `1.x` package line follows semantic versioning. Patch and minor
  releases preserve existing public API and package compatibility while
  allowing additive APIs, diagnostics, documentation, validation hardening and
  bug fixes.
- Current `1.x` packages should open and search durable exact-flat and
  immutable HNSW snapshots written by earlier stable `1.x` packages for the
  same supported surface and metric.
- Future major package lines are not guaranteed to read every `1.x` durable
  directory indefinitely. Applications should keep source vectors and
  application records so they can rebuild or export indexes when adopting a
  future major line.

### Fixed

- Cosine query distance calculation in exact-flat and HNSW search avoids
  repeated per-component query division while preserving the public distance
  contract.

### Known limitations

- The optional VectorData adapter remains exact-flat and in-memory. It does
  not add HNSW VectorData collections, durable VectorData storage, embedding
  generation, hybrid search or multiple vector properties.
- Mutable HNSW durable compatibility is represented through checkpoint output.
  The mutable overlay state itself is not reopened as a durable mutable index.

## 1.2.0

### Added

- Added HNSW cosine support for immutable build/search, durable save/open and
  opened read-only search.
- Added update-oriented mutable HNSW cosine support through an immutable HNSW
  base plus exact delta rows, tombstones, allowlist search and checkpoint
  rebuild into a new immutable HNSW snapshot.

### Compatibility

- The adapter package was version-aligned with the core package. The adapter
  remained exact-flat-only and did not add HNSW VectorData support.

### Known limitations

- HNSW inner product remained unsupported. Use exact-flat indexes for
  inner-product retrieval.

## 1.1.0

### Added

- Added public benchmark documentation for bounded exact-flat and HNSW
  squared-L2 search measurements.
- Strengthened VectorData adapter behavior for exact-flat use, including
  vector inclusion behavior and in-memory expression filtering within the
  supported adapter surface.

### Compatibility

- The package line preserved the stable `1.0` public API while adding
  documentation, validation and adapter hardening.

## 1.0.1

### Added

- Added workspace helper APIs that size caller-owned search workspaces from
  the current index shape.
- Added clearer count terminology for physical, live, delta, tombstone and
  reserved-ID states while preserving compatibility names.

### Fixed

- Corrected VectorData `IncludeVectors` behavior so default and explicit
  vector-omission options omit vectors, while explicit inclusion keeps them
  in returned records.

### Changed

- Expanded README and XML documentation for workspace sizing, count names,
  persistence, checkpoints, deleted-ID reservation and VectorData boundaries.

## 1.0.0

### Added

- First stable package line for the dependency-free core `VecNet` package and
  the optional `VecNet.Integration.VectorData` adapter package.
- Exact-flat indexing for squared L2, inner product and cosine, including
  durable save/open, allowlist filtering, reusable candidate sets, mutation
  with tombstones and checkpoint compaction.
- HNSW approximate indexing for squared L2, including build/search, durable
  save/open, caller-owned workspaces and caller-owned external-ID allowlist
  filtering.
- Update-oriented HNSW workflow for squared L2 using an immutable HNSW base,
  exact delta rows, tombstones and checkpoint rebuild into a new immutable
  HNSW snapshot.
- Exact-flat VectorData adapter for in-memory collections over pregenerated
  vectors.

### Known limitations

- VecNet is an embedded vector indexing library, not a vector database,
  metadata store, authorization system, embedding host, full-text engine,
  distributed service or GPU library.
- The optional VectorData adapter is exact-flat and in-memory only.
