# VecNet

VecNet is an embedded vector indexing library for .NET applications. It is
designed for applications that keep their records, metadata, authorization,
and application storage outside the vector engine while using VecNet for dense
vector retrieval.

This package is a `0.1` preview. The currently documented surface focuses on
exact flat indexing, canonical distance semantics, durable exact-flat save and
open, exact allowlist filtering, reusable exact candidate sets, exact count
inspection, exact mutation/checkpoint workflows, and preview squared-L2 HNSW
surfaces.

## Current Support

- Target framework: `net10.0`.
- Dense `float` vectors with fixed per-index dimension.
- External vector IDs as caller-owned `ulong` values.
- Exact exhaustive search with these canonical distances:
  - `VectorMetric.SquaredEuclidean`: squared L2 distance.
  - `VectorMetric.InnerProduct`: negative dot product, so lower distance is
    still better.
  - `VectorMetric.Cosine`: `1 - dot` after VecNet normalizes inserted and
    query vectors.
- Results ordered by ascending distance, then ascending external ID when the
  computed distance is equal.
- Durable exact-flat `Save` and `OpenReadOnly` over a directory containing a
  manifest and binary vector/ID files.
- Exact raw allowlist filtering and reusable exact candidate sets.
- Exact-flat `TryAdd`, `TryDelete`, and `Checkpoint` for visible generation
  updates in the current process.
- Preview HNSW approximate indexing for `VectorMetric.SquaredEuclidean` with
  build ingestion, caller-owned workspace search, caller-owned external-ID
  allowlist filtering, preview durable `Save`/`OpenReadOnly`, opened read-only
  search, and read-only concurrent search when each caller uses independent
  result buffers and workspaces.
- Preview update-oriented HNSW mode for squared L2 using an immutable HNSW
  base plus exact in-memory delta rows, tombstones, search merge/rerank, and
  caller-initiated checkpoint/rebuild into a new immutable HNSW snapshot.

## Preview Limitations

- APIs and file formats are still preview and may change before a stable
  release.
- VecNet stores vector IDs and vectors, not application records or payloads.
- Application metadata filtering, authorization, transactions, backups, and
  record hydration remain the responsibility of the host application.
- HNSW support is squared-L2-only and approximate. Cosine HNSW and
  inner-product HNSW are not supported for 1.0; use exact-flat indexes for
  those metrics.
- HNSW `Add` is build ingestion for an immutable graph, not upsert,
  replacement, delete, repair, direct graph mutation, or live graph update.
  HNSW indexes opened with `OpenReadOnly` are searchable but reject mutation.
- HNSW allowlist filtering uses caller-owned external `ulong` IDs only.
  VecNet does not store labels, metadata, authorization rules, records,
  payloads, durable graph-aware filter metadata, persisted candidate sets,
  public graph ordinals, or reusable HNSW candidate sets.
- For selective HNSW allowlists where the known live allowed count is within
  `EfSearch`, VecNet uses exact filtered fallback. For broader allowlists,
  HNSW traversal remains approximate and unfiltered; non-allowed candidates are
  suppressed at emission and fewer than the requested number of results may be
  returned.
- Read-only HNSW searches may overlap only over a logically frozen index or
  generation with independent caller-owned result buffers and independent
  workspaces. Concurrent mutation/search, concurrent checkpoint/search, and
  shared scratch are not supported.
- The update-oriented HNSW mode does not mutate the graph in place. Delta rows
  are exact in-memory rows, deletes are tombstones, checkpoint/rebuild writes a
  new immutable HNSW snapshot after validation, and mutable overlay state is
  not durably reopened.
- HNSW durable files are a preview round-trip format and have no stable
  compatibility promise.
- Compressed indexes, SSD-scale indexes, richer key mapping, optional
  integration adapters, and release-grade operational tooling are planned
  work, not supported public package capabilities in this preview.
- This README does not make public HNSW recall, latency, throughput,
  allocation, memory, capacity, storage-size, comparison, stable file-format,
  stable API, production-readiness, or platform support claims.

## Installation

Add the published preview package to a .NET 10 project:

```bash
dotnet add package VecNet --version 0.1.0-preview.4
```

## Basic Usage

```csharp
using VecNet;

var index = new ExactFlatIndex(dimension: 3, VectorMetric.SquaredEuclidean);

index.Add(1001, [1.0f, 0.0f, 0.0f]);
index.Add(1002, [0.0f, 1.0f, 0.0f]);
index.Add(1003, [0.0f, 0.0f, 1.0f]);

Span<SearchResult> results = stackalloc SearchResult[2];
int written = index.Search([0.9f, 0.1f, 0.0f], results);

for (int i = 0; i < written; i++)
{
    Console.WriteLine($"{results[i].Id}: {results[i].Distance}");
}
```

## Persistence

Use `Save` as the initial persistence operation for an exact-flat index. It
writes the current live view to a new or empty directory. Use `OpenReadOnly` to
open that directory as an immutable searchable index.

```csharp
using VecNet;

var index = new ExactFlatIndex(3, VectorMetric.Cosine);
index.Add(1, [1.0f, 0.0f, 0.0f]);
index.Add(2, [0.0f, 1.0f, 0.0f]);

string path = Path.Combine(Environment.CurrentDirectory, "vecnet-index");
index.Save(path);

ExactFlatIndex reopened = ExactFlatIndex.OpenReadOnly(path);

Span<SearchResult> results = stackalloc SearchResult[1];
int written = reopened.Search([1.0f, 0.0f, 0.0f], results);
```

`Save` does not overwrite an existing non-empty directory. It writes only the
current live view, so deleted rows are not searchable in the saved output.

## HNSW Preview

`HnswIndex` is a developer-preview approximate index for squared L2 only. Use
`HnswIndexOptions` to choose preview build/search parameters, and pass a
caller-owned `HnswSearchWorkspace` to every search.

```csharp
using VecNet;

var options = new HnswIndexOptions(
    M: 16,
    EfConstruction: 200,
    EfSearch: 50,
    RandomSeed: 0x564543_034UL);

var index = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);

index.Add(1001, [1.0f, 0.0f, 0.0f]);
index.Add(1002, [0.0f, 1.0f, 0.0f]);
index.Add(1003, [0.0f, 0.0f, 1.0f]);

var workspace = new HnswSearchWorkspace(index.Count, index.Options.EfSearch);
Span<SearchResult> results = stackalloc SearchResult[2];

int written = index.Search([0.9f, 0.1f, 0.0f], results, workspace);
```

For caller-owned external-ID allowlist filtering, pass the allowlist to
`Search` with the same caller-owned result buffer and workspace pattern.

```csharp
ulong[] allowedIds = [1001, 1003];
int filteredWritten = index.Search(
    [0.9f, 0.1f, 0.0f],
    allowedIds,
    results,
    workspace);
```

The allowlist contains application-owned external IDs. Unknown IDs are ignored
and duplicates are coalesced. For selective allowlists within the configured
`EfSearch` budget, VecNet uses exact filtered fallback. For broader allowlists,
HNSW traversal remains approximate and may return fewer than the requested
number of results even when exact filtered truth has enough live matches.

For HNSW preview persistence, save to a new or empty directory and open it as
read-only.

```csharp
string path = Path.Combine(Environment.CurrentDirectory, "vecnet-hnsw");
index.Save(path);

HnswIndex opened = HnswIndex.OpenReadOnly(path);

var openedWorkspace = new HnswSearchWorkspace(opened.Count, opened.Options.EfSearch);
int openedWritten = opened.Search([0.9f, 0.1f, 0.0f], results, openedWorkspace);
```

Opened HNSW indexes reject `Add`. HNSW callers own synchronization, result
buffers, and workspaces; do not share a result buffer or workspace between
overlapping searches.

## HNSW Update-Oriented Preview

The HNSW update-oriented preview searches an immutable HNSW base plus exact
in-memory delta rows. Deletes are represented as tombstones over base or
delta IDs. Checkpoint rebuilds the current live view into a new immutable HNSW
snapshot and publishes that rebuilt base in the current instance after
validation.

```csharp
using VecNet;

var baseIndex = new HnswIndex(3, VectorMetric.SquaredEuclidean);
baseIndex.Add(1001, [1.0f, 0.0f, 0.0f]);
baseIndex.Add(1002, [0.0f, 1.0f, 0.0f]);

var mutable = new HnswMutableIndex(baseIndex);

VectorMutationResult add = mutable.TryAdd(1003, [0.0f, 0.0f, 1.0f]);
VectorMutationResult delete = mutable.TryDelete(1002);

var mutableWorkspace = new HnswMutableSearchWorkspace(mutable, maxResults: 2);
Span<SearchResult> mutableResults = stackalloc SearchResult[2];
int mutableWritten = mutable.Search(
    [0.9f, 0.1f, 0.0f],
    mutableResults,
    mutableWorkspace);

if (add.Status == VectorMutationStatus.Committed ||
    delete.Status == VectorMutationStatus.Committed)
{
    HnswMutableCheckpointResult checkpoint =
        mutable.Checkpoint("vecnet-hnsw-checkpoint");
    Console.WriteLine(checkpoint.Status);
}
```

Create mutable HNSW workspaces from the current mutable index shape. Recreate
them after a committed `TryAdd`, committed `TryDelete`, or published
`Checkpoint`. The mutable wrapper does not expose direct graph mutation,
upsert, replacement, graph repair, checkpoint diagnostics, or durable mutable
overlay reopen.

## Filtering

For one-off filtering, pass an allowlist of external vector IDs plus a
caller-owned workspace sized for the current physical index rows.

```csharp
using VecNet;

var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
index.Add(10, [1.0f, 0.0f]);
index.Add(20, [0.0f, 1.0f]);
index.Add(30, [1.0f, 1.0f]);

ulong[] allowedIds = [10, 30];
var workspace = new ExactFlatSearchFilterWorkspace(index.PhysicalVectorCount);
Span<SearchResult> results = stackalloc SearchResult[2];

int written = index.Search(
    [1.0f, 0.5f],
    allowedIds,
    results,
    workspace);
```

`VectorCount` is a compatibility name for the physical stored-row count and has
the same meaning as `PhysicalVectorCount`. Size raw allowlist workspaces from
`PhysicalVectorCount`/`VectorCount`, not from `LiveVectorCount`.

For a filter reused across searches on the same visible generation, create an
exact candidate set from external IDs.

```csharp
ExactFlatCandidateSet candidates = index.CreateCandidateSet([10, 30]);
int written = index.Search([1.0f, 0.5f], candidates, results);
```

Candidate sets are bound to the creating index instance and its current
`Generation`. Rebuild them after a committed `TryAdd` or `TryDelete`, and
after a `Checkpoint` that publishes a compact generation. Failed or no-op
mutation results and `Checkpoint` results with `NoChanges` do not advance
`Generation` by themselves.

## Counts And Deleted IDs

`ExactFlatIndex` exposes physical, live, tombstone, and reserved-ID counts:

- `VectorCount` and `PhysicalVectorCount` report physical stored rows used for
  workspace sizing. This can be larger than the searchable live count.
- `LiveVectorCount` reports vectors currently visible to search.
- `TombstoneCount` reports deleted physical rows that are hidden from search
  until compaction.
- `DeletedReservedIdCount` reports deleted IDs that remain unavailable for
  reuse.

Deleting an ID reserves that ID permanently for the lifetime of the index
state. A later `TryAdd` with the same ID reports a reuse conflict even after a
checkpoint compacts away tombstoned rows.

## Updates And Checkpoints

`TryAdd` and `TryDelete` report status instead of throwing for expected
mutation conflicts such as duplicate add, unknown delete, repeated delete,
deleted-ID reuse, and read-only mutation cases. They still throw for invalid
arguments, such as vectors with the wrong dimension or invalid numeric values.

```csharp
VectorMutationResult add = index.TryAdd(40, [0.25f, 0.75f]);
VectorMutationResult delete = index.TryDelete(20);

if (add.Status == VectorMutationStatus.Committed ||
    delete.Status == VectorMutationStatus.Committed)
{
    ExactFlatCheckpointResult checkpoint = index.Checkpoint("vecnet-checkpoint");
    Console.WriteLine(checkpoint.Status);
}
```

`Checkpoint` is mutation compaction and publication. It writes a compact live
exact-flat generation to a new or empty directory and publishes that compact
generation in the current index instance. When there are no delta rows or
tombstones to fold, it returns `NoChanges`.

Use `Save` for the initial persisted live view. Use `Checkpoint` after
committed mutations when the application wants to publish a compacted live
view and continue using the current index instance.

## Thread Safety And Workspaces

Treat `ExactFlatIndex` and `HnswMutableIndex` as externally synchronized. Do
not run mutation, checkpoint, save, candidate-set creation, or search
concurrently against the same mutable instance unless your application provides
its own coordination. Caller-owned result buffers and workspaces must not be
shared by overlapping calls. Candidate sets and mutable HNSW workspaces are
transient handles for one owner index and generation; rebuild rather than
sharing stale handles across mutation boundaries.

## Floating-Point Comparisons

VecNet ranks by the distances computed by the executing index. If an
application test compares VecNet distances with an independent implementation,
use a tolerance appropriate for the metric and data scale. Optimized
floating-point accumulation can differ slightly from another implementation,
and near-tie ordering is only guaranteed when distances compare equal in the
executing path, where external ID breaks the tie.

## Planned Work

VecNet is being built toward a broader embedded indexing engine with additional
index strategies, richer filtering and update workflows, package polish,
consumer documentation, and integration tooling. Those capabilities will be
documented when they are admitted to the public preview surface.

## Repository

- Source: https://github.com/sh0r3x/VecNet
- Issues: https://github.com/sh0r3x/VecNet/issues

## License

VecNet is licensed under the MIT License. See [LICENSE](LICENSE).
