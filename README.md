# VecNet

VecNet is an embedded vector indexing library for .NET applications that need
dense-vector retrieval without adopting a vector database service. It stores
vectors under caller-owned `ulong` IDs and returns IDs with distances, while
your application keeps records, metadata, authorization, embedding generation,
and durable application storage.

## Install

Use VecNet from a `.NET 10` project. Add the core package:

```bash
dotnet add package VecNet --version 1.4.2
```

For `Microsoft.Extensions.VectorData` applications, add the separate optional
exact-flat adapter package in addition to the core package:

```bash
dotnet add package VecNet.Integration.VectorData --version 1.4.2
```

The core `VecNet` package intentionally has no runtime package dependencies
and no dependency on `Microsoft.Extensions.VectorData`.

## Which Surface Should I Start With?

| Need | Start with | Boundary |
| --- | --- | --- |
| Exhaustive search for squared L2, inner product, or cosine | `ExactFlatIndex` | Scans live rows and ranks by canonical distance. |
| Save/open an exact snapshot | `ExactFlatIndex.Save` and `ExactFlatIndex.OpenReadOnly` | Opens as immutable read-only exact search. |
| Exact search over changing data in one process | `ExactFlatIndex.TryAdd`, `TryDelete`, and `Checkpoint` | Deletes are tombstones until checkpoint compaction; deleted IDs remain reserved. |
| Approximate in-memory graph search | `HnswIndex` | Squared-L2, inner product, and cosine immutable build/search plus durable save/open. |
| Update-oriented HNSW workflow | `HnswMutableIndex` | Supported-metric immutable HNSW base plus exact delta/tombstones; checkpoint rebuilds a new immutable HNSW base with the same metric. |
| VectorData integration | `VecNet.Integration.VectorData` | Separate optional exact-flat in-memory adapter, not HNSW or durable VectorData storage. |

## Small Exact-Flat Example

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

## HNSW First-Use Examples

For approximate graph search, create `HnswIndex` with the metric you want, add
vectors, and search with your own result buffer and workspace.

Squared L2 keeps ordinary distance semantics: lower squared distance is nearer.

```csharp
using VecNet;

var options = new HnswIndexOptions(
    M: 16,
    EfConstruction: 200,
    EfSearch: 50,
    RandomSeed: 12345UL);

var index = new HnswIndex(dimension: 3, VectorMetric.SquaredEuclidean, options);

index.Add(1001, [1.0f, 0.0f, 0.0f]);
index.Add(1002, [0.0f, 1.0f, 0.0f]);
index.Add(1003, [0.0f, 0.0f, 1.0f]);

Span<SearchResult> results = stackalloc SearchResult[2];
HnswSearchWorkspace workspace = index.CreateSearchWorkspace();

int written = index.Search([0.9f, 0.1f, 0.0f], results, workspace);
```

Cosine is for direction-only similarity. VecNet normalizes inserted and query
vectors for cosine indexes, and cosine rejects zero vectors.

```csharp
using VecNet;

var options = new HnswIndexOptions(M: 16, EfConstruction: 200, EfSearch: 50);
var index = new HnswIndex(dimension: 3, VectorMetric.Cosine, options);

index.Add(2001, [1.0f, 0.0f, 0.0f]);
index.Add(2002, [0.0f, 1.0f, 0.0f]);
index.Add(2003, [1.0f, 1.0f, 0.0f]);

Span<SearchResult> results = stackalloc SearchResult[2];
HnswSearchWorkspace workspace = index.CreateSearchWorkspace();

int written = index.Search([0.8f, 0.2f, 0.0f], results, workspace);
```

Inner product uses raw vectors as supplied. VecNet does not normalize them,
zero vectors are valid, and magnitude affects ranking: a larger dot product
becomes a lower negative distance.

```csharp
using VecNet;

var options = new HnswIndexOptions(M: 16, EfConstruction: 200, EfSearch: 50);
var index = new HnswIndex(dimension: 3, VectorMetric.InnerProduct, options);

index.Add(3001, [1.0f, 0.0f, 0.0f]);
index.Add(3002, [2.0f, 0.0f, 0.0f]);
index.Add(3003, [0.0f, 1.0f, 0.0f]);
index.Add(3004, [0.0f, 0.0f, 0.0f]);

Span<SearchResult> results = stackalloc SearchResult[2];
HnswSearchWorkspace workspace = index.CreateSearchWorkspace();

int written = index.Search([1.0f, 0.0f, 0.0f], results, workspace);
```

## Where Next?

- Durable exact-flat save/open: [Persistence](#persistence).
- HNSW squared-L2, inner-product, and cosine search, including
  update-oriented mutable workflows where supported: [HNSW](#hnsw).
- Filtering with caller-owned IDs: [Filtering](#filtering).
- Mutation and checkpoint workflows: [Updates And Checkpoints](#updates-and-checkpoints).
- Benchmark summaries: [Benchmarks](#benchmarks).
- VectorData applications: [Optional VectorData Adapter](#optional-vectordata-adapter).
- Support and no-claim boundaries: [Limitations And Unsupported Claims](#limitations-and-unsupported-claims).

## Benchmarks

VecNet has bounded benchmark summaries:

- `ExactFlatIndex` squared-L2 generated-data search:
  [docs/benchmarks/exact-flat.md](docs/benchmarks/exact-flat.md).
- `HnswIndex` squared-L2 recall versus latency over generated data and
  Fashion-MNIST:
  [docs/benchmarks/hnsw-squared-l2.md](docs/benchmarks/hnsw-squared-l2.md).
- `HnswIndex` cosine recall versus latency:
  [docs/benchmarks/hnsw-cosine.md](docs/benchmarks/hnsw-cosine.md).
- `HnswIndex` inner-product recall versus latency:
  [docs/benchmarks/hnsw-inner-product.md](docs/benchmarks/hnsw-inner-product.md).

Read each methodology and limit section before using the numbers. The
summaries are narrow search measurements, not capacity, package-wide,
platform-wide, adapter, competitor, NativeAOT/trimming, regression-threshold,
or semantic-relevance claims.

## Things To Know

- Prefer the self-sizing workspace helpers when a workspace belongs to one
  index: `ExactFlatIndex.CreateSearchFilterWorkspace()` for exact allowlists
  and `HnswIndex.CreateSearchWorkspace()` for immutable HNSW search. If you
  construct workspaces manually, exact allowlist workspaces are sized from
  `PhysicalVectorCount`/`VectorCount`, not `LiveVectorCount`; immutable HNSW
  workspaces are sized from `Count` and the intended `maxEfSearch`.
- Count names are intentionally explicit where possible. On `ExactFlatIndex`,
  `VectorCount` is a compatibility name for physical stored rows and matches
  `PhysicalVectorCount`. On mutation result records, `VectorCount` is a
  compatibility alias for live/searchable count. Prefer
  `PhysicalVectorCount`, `LiveVectorCount`, `TombstoneCount`, and
  `DeletedReservedIdCount` when the distinction matters.
- Use `Save` for the first persisted live view in a new or empty directory.
  Use `Checkpoint` after committed mutations when you want compaction and
  publication into a new or empty directory. Neither operation edits an
  existing durable index directory in place.
- Deleting an ID reserves that external ID for the lifetime of the current
  index state, including after checkpoint compaction. Deleted IDs cannot be
  reused later as replacement IDs.
- VecNet returns vector IDs and distances. It does not expose vector
  read-back or vector enumeration APIs. Keep your source vectors and
  application records if you need rebuild, export, display, reranking, or
  non-index storage.
- To grow a saved HNSW index with a supported metric, use the
  [HNSW Saved-Index Growth Recipe](#hnsw-saved-index-growth-recipe): open the
  saved index read-only, wrap it in `HnswMutableIndex`, apply `TryAdd` and
  `TryDelete`, `Checkpoint` to a new or empty directory, then reopen the new
  durable HNSW output.
- Immutable HNSW durable files are the current `1.x` round-trip format for
  `Save` and `OpenReadOnly`. Mutable checkpoint output for supported HNSW
  metrics uses the same immutable snapshot format. Current stable `1.x`
  packages should open supported earlier stable `1.x` snapshots for the same
  supported surface and metric; unsupported, future, corrupt, or incompatible
  formats fail closed. There is no cross-major durable-format promise.
- The optional VectorData adapter follows `IncludeVectors`: null/default
  options and `IncludeVectors = false` omit vectors, while
  `IncludeVectors = true` includes vectors. Omitting vectors may require a
  projectable class record shape; unsupported projection shapes throw a clear
  `NotSupportedException`.

## Supported 1.4.2 Feature List

- Target framework: `net10.0`.
- The core `VecNet` package is dependency-free and ships managed `lib/net10.0`
  assets without native, RID-specific, build, analyzer, or content assets.
- Dense `float` vectors with fixed per-index dimension.
- External vector IDs as caller-owned `ulong` values.
- Exact exhaustive search with these canonical distances:
  - `VectorMetric.SquaredEuclidean`: squared L2 distance.
  - `VectorMetric.InnerProduct`: raw-vector negative dot product, so larger
    dot product produces a lower distance and magnitude affects ranking.
  - `VectorMetric.Cosine`: `1 - dot` after VecNet normalizes inserted and
    query vectors. The canonical range is `[0, 2]`; tiny floating-point
    excursions up to `1e-6` below zero or above two are tolerated.
- Results ordered by ascending distance, then ascending external ID when the
  computed distance is equal.
- Durable exact-flat `Save` and `OpenReadOnly` over a directory containing a
  manifest and binary vector/ID files.
- Exact raw allowlist filtering and reusable exact candidate sets.
- Exact-flat `TryAdd`, `TryDelete`, and `Checkpoint` for visible generation
  updates in the current process.
- HNSW approximate indexing for `VectorMetric.SquaredEuclidean`,
  `VectorMetric.InnerProduct`, and `VectorMetric.Cosine` with immutable
  `HnswIndex` build/search, durable `Save`/`OpenReadOnly`, opened read-only
  search, opened read-only concurrent unfiltered and caller-owned allowlist
  search when each overlapping call uses its own result buffer and
  `HnswSearchWorkspace`, and update-oriented mutable functional support.
- Update-oriented HNSW mode for squared L2, inner product, and cosine using an
  immutable HNSW base plus exact in-memory delta rows, tombstones, search merge/rerank,
  allowlist search, and caller-initiated checkpoint/rebuild into a new
  immutable HNSW snapshot with the same metric.
- Optional `Microsoft.Extensions.VectorData` support through the separate
  `VecNet.Integration.VectorData` adapter package. The adapter is exact-flat
  and in-memory, supports pregenerated `float[]` or `ReadOnlyMemory<float>`
  vectors, adapter-owned `TKey` mapping to VecNet `ulong` IDs, adapter-owned
  records, VectorData CRUD/search, and expression filters evaluated in memory
  and converted to VecNet allowlists.

## Limitations And Unsupported Claims

- VecNet is not a vector database, distributed service, metadata query engine,
  authorization system, embedding model host, full-text index, GPU library, or
  application record store.
- VecNet stores vector IDs and vectors, not application records or payloads.
  The calling application owns record hydration, final permission checks,
  grouping, deduplication, freshness checks, reranking, and presentation.
- Application metadata filtering, authorization, transactions, backups, and
  record hydration remain the responsibility of the host application.
- Immutable HNSW support is approximate. Exact-flat remains the exhaustive
  retrieval option when approximate search is not appropriate.
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
- Read-only HNSW searches for supported metrics may overlap when each
  overlapping unfiltered or caller-owned allowlist search uses its own result
  buffer and its own `HnswSearchWorkspace`. Shared result buffers, shared
  search workspaces, shared scratch, concurrent mutation/search, concurrent
  checkpoint/search, and mutable HNSW concurrency are not supported.
- The update-oriented HNSW mode does not mutate the graph in place. Delta rows
  are exact in-memory rows, deletes are tombstones, checkpoint/rebuild writes a
  new immutable HNSW snapshot after validation, and mutable overlay state is
  not durably reopened.
- Stable `1.x` releases follow semantic versioning. Patch and minor releases
  preserve public API/package compatibility while allowing additive APIs,
  diagnostics, documentation, validation hardening and bug fixes.
- Current stable `1.x` packages should open and search durable exact-flat and
  immutable HNSW snapshots written by earlier stable `1.x` packages for the
  same supported surface and metric. Unsupported, future, corrupt, or
  incompatible formats fail closed. There is no cross-major durable-format
  promise unless a later release states one. Keep source vectors and
  application records so you can rebuild or export indexes when adopting a
  future major line.
- HNSW durable files are a current `1.x` round-trip format. Mutable HNSW
  durable compatibility is represented by checkpoint output in the immutable
  HNSW snapshot format; mutable overlay state itself is not reopened as a
  durable mutable index.
- The optional VectorData adapter does not support HNSW VectorData indexes,
  durable VectorData collection open/reopen, durable record or key-map
  storage, embedding generation, hybrid search, multiple vector properties,
  store-generated keys, sparse vectors, binary vectors, quantized vectors, or
  provider-specific vector types.
- Compressed indexes, SSD-scale indexes, richer core key mapping, broader
  integration adapters, and release-grade operational tooling are planned
  work, not supported public package capabilities in the current package line.
- Package installation and basic consumer checks do not create a public
  performance, platform support, NativeAOT, trimming, or universal deployment
  claim.
- `1.4.2` is the stable package version for the supported public API surfaces
  described here. Except for the dedicated benchmark documents linked above,
  this README does not make public HNSW recall, latency, throughput,
  allocation, memory, capacity, storage-size, update-profile, concurrency,
  comparison, stable file-format, production-readiness, platform support,
  NativeAOT, or trimming claims.

## Metric Selection

Use `VectorMetric.SquaredEuclidean` when lower squared L2 distance is the
desired ranking. VecNet keeps L2 squared; it does not take a square root merely
for display.

Use `VectorMetric.InnerProduct` when larger raw dot product should rank better.
VecNet uses vectors as supplied, does not normalize inner-product vectors, and
reports the canonical distance as negative dot product so all result ordering
remains "lower distance is better." Magnitude affects ranking, zero vectors are
valid, and an indexed vector is not necessarily its own nearest result under
maximum inner product.

Use `VectorMetric.Cosine` when direction-only similarity should rank better.
VecNet normalizes inserted and query vectors for cosine indexes, rejects zero
vectors, and reports `1 - dot(normalizedQuery, normalizedStored)`. The
canonical cosine distance range is `[0, 2]`; tiny floating-point excursions up
to `1e-6` below zero or above two are tolerated.

HNSW supports squared L2, inner product, and cosine across immutable search,
durable save/open, opened read-only search, caller-owned allowlist search, and
the update-oriented mutable wrapper.

## Optional VectorData Adapter

`VecNet.Integration.VectorData` wraps VecNet exact-flat collections behind
`Microsoft.Extensions.VectorData` abstractions. It is intended for applications
that already own records, metadata, authorization, embedding generation, and
durable application storage.

The adapter creates in-memory exact-flat collections from `IndexKind.Flat`
VectorData definitions or attributes. It accepts pregenerated `float[]` or
`ReadOnlyMemory<float>` vectors, maps VectorData keys to internal VecNet
`ulong` IDs, keeps records in adapter-owned memory, supports upsert, delete,
get, search, `Skip`, `ScoreThreshold`, and in-memory expression filters, and
projects scores for Euclidean squared distance, Euclidean distance, cosine
distance, cosine similarity, and dot-product similarity.

Retrieval options follow VectorData `IncludeVectors` behavior. Null/default
options and explicit `IncludeVectors = false` return records without vectors.
Explicit `IncludeVectors = true` includes vectors. In the no-vector return
mode, VecNet projects shallow record copies for supported class record shapes;
if a record shape cannot be projected and that mode is required, the adapter
throws `NotSupportedException`.

The adapter is not a durable record store and does not add dependencies to the
core `VecNet` package. It does not provide HNSW VectorData collections,
durable VectorData collection open/reopen, durable records or key maps,
embedding generation, hybrid search, multiple vector properties, or public
performance, platform, NativeAOT, or trimming claims.

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
Current stable `1.x` packages should open supported earlier stable `1.x`
exact-flat snapshots for the same surface and metric. Unsupported, future,
corrupt, or incompatible formats fail closed. There is no cross-major
durable-format promise.

## HNSW

`HnswIndex` is an approximate index for squared L2, inner product, and cosine.
The example below uses squared L2; the first-use examples above show cosine
and inner product as well. Inner product uses vectors as supplied, so VecNet
does not normalize away magnitude and a larger dot product becomes a lower
negative distance. Cosine is the direction-only metric. All three metrics use
the same immutable `HnswIndex` build/search and durable save/open surface, and
each can also be wrapped by `HnswMutableIndex` for the update-oriented
workflow described below.

Use `HnswIndexOptions` to choose build/search parameters, and pass a
caller-owned `HnswSearchWorkspace` to every search.

```csharp
using VecNet;

var options = new HnswIndexOptions(
    M: 16,
    EfConstruction: 200,
    EfSearch: 50,
    RandomSeed: 12345UL);

var index = new HnswIndex(3, VectorMetric.SquaredEuclidean, options);

index.Add(1001, [1.0f, 0.0f, 0.0f]);
index.Add(1002, [0.0f, 1.0f, 0.0f]);
index.Add(1003, [0.0f, 0.0f, 1.0f]);

var workspace = index.CreateSearchWorkspace();
Span<SearchResult> results = stackalloc SearchResult[2];

int written = index.Search([0.9f, 0.1f, 0.0f], results, workspace);
```

For supported HNSW caller-owned external-ID allowlist filtering, pass the
allowlist to `Search` with the same caller-owned result buffer and workspace
pattern.

```csharp
ulong[] allowedIds = [1001, 1003];
int filteredWritten = index.Search(
    [0.9f, 0.1f, 0.0f],
    allowedIds,
    results,
    workspace);
```

The allowlist contains application-owned external IDs. Unknown IDs are ignored
and duplicates are coalesced. For selective supported-metric HNSW allowlists
within the configured `EfSearch` budget, VecNet uses exact filtered fallback.
For broader allowlists, HNSW traversal remains approximate and may return
fewer than the requested number of results even when exact filtered truth has
enough live matches.

For HNSW persistence, save to a new or empty directory and open it as read-only.
Current stable `1.x` packages should open supported earlier stable `1.x`
immutable HNSW snapshots for the same surface and metric. Unsupported, future,
corrupt, or incompatible formats fail closed. Mutable HNSW checkpoint output is
an immutable HNSW snapshot in the current supported format. There is no
cross-major durable-format promise.

```csharp
string path = Path.Combine(Environment.CurrentDirectory, "vecnet-hnsw");
index.Save(path);

HnswIndex opened = HnswIndex.OpenReadOnly(path);

var openedWorkspace = opened.CreateSearchWorkspace();
int openedWritten = opened.Search([0.9f, 0.1f, 0.0f], results, openedWorkspace);
```

Opened HNSW indexes reject `Add`. HNSW callers own result buffers and
workspaces. Overlapping read-only unfiltered or allowlist search uses one
result buffer and one `HnswSearchWorkspace` per overlapping call.

### HNSW Capacity, Workspace, And Scratch Guidance

Plan HNSW construction around the number of rows the application expects to
ingest. The `initialCapacity` constructors and `EnsureCapacity` reserve vector
row, graph, ID-map, and build-scratch storage for mutable/buildable HNSW
instances. `Capacity` is storage reservation, not vector cardinality or a
published supported scale limit.

Capacity reservation preserves logical durable index contents, counts, search
behavior, and durable compatibility. Saved physical files may still contain
normal manifest metadata differences.

Buildable HNSW instances retain build scratch so additional `Add`
operations can continue without a separate seal step. Applications that want
to serve a logically frozen HNSW generation without build scratch
should save the generation and open it with `OpenReadOnly`.

Create one `HnswSearchWorkspace` per overlapping search.
Size immutable HNSW workspaces from the current `Count` and the intended
`maxEfSearch`; recreate a workspace when either value can exceed the
workspace's recorded capacity. High row counts, high `maxEfSearch` values, and
high concurrent HNSW search counts increase caller-owned workspace memory that
the application must budget.

HNSW `Save`, `OpenReadOnly`, and mutable checkpoint/rebuild operate
over the index state and may need temporary memory and disk space while
publishing or validating output. This README gives qualitative planning
guidance only; it does not publish numeric memory, capacity, storage-size,
latency, throughput, or recall claims.

## HNSW Update-Oriented Mode

The HNSW update-oriented mode is documented for squared L2, inner product, and
cosine. It searches an immutable HNSW base plus exact in-memory delta rows with
the same metric as the base. Deletes are represented as tombstones over base or
delta IDs. Checkpoint rebuilds the current live view into a new immutable HNSW
snapshot with the same metric and publishes that rebuilt base in the current
instance after validation.

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
`Checkpoint`. For mutable HNSW, `maxEfSearch` is both the workspace capacity
and the adaptive retry ceiling when base tombstones would otherwise underfill
results. Latency-sensitive callers should use bounded workspace ceilings for
their recall/latency tiers instead of reusing one oversized workspace for all
queries. The mutable wrapper does not expose direct graph mutation, upsert,
replacement, graph repair, checkpoint diagnostics, or durable mutable overlay
reopen.

For planned mutable HNSW checkpoint/rebuild workflows, capacity-plan the HNSW
base around the expected live row count after folding delta rows and
tombstones. Mutable HNSW workspaces are tied to the wrapper generation and
must be recreated after any generation-changing mutation or checkpoint. This
README makes no public mutable-HNSW benchmark, memory, capacity,
update-profile, concurrency, NativeAOT, trimming, or platform claim.

### HNSW Saved-Index Growth Recipe

Saved HNSW directories with supported metrics open as immutable read-only graph
generations. To add or delete IDs after saving, create a new durable
generation with the same supported metric instead of editing the saved
directory in place:

1. Open the saved HNSW directory with `HnswIndex.OpenReadOnly`.
2. Create `new HnswMutableIndex(opened)`.
3. Apply changes with `TryAdd` and `TryDelete`.
4. Call `Checkpoint` with a new or empty output directory.
5. Reopen the checkpoint output with `HnswIndex.OpenReadOnly`.

```csharp
HnswIndex openedBase = HnswIndex.OpenReadOnly("vecnet-hnsw");
var mutable = new HnswMutableIndex(openedBase);

mutable.TryAdd(1004, [0.25f, 0.25f, 0.5f]);
mutable.TryDelete(1001);

HnswMutableCheckpointResult checkpoint =
    mutable.Checkpoint("vecnet-hnsw-next");

HnswIndex nextBase = HnswIndex.OpenReadOnly("vecnet-hnsw-next");
```

## Filtering

Evaluate tenant, permission, category, availability, and other business
predicates in your own record or metadata store first. Pass the matching
VecNet vector IDs to VecNet as an allowlist or reusable exact candidate set,
then hydrate returned IDs through your own store and run final authorization
and freshness checks before showing results.

For one-off filtering, pass an allowlist of external vector IDs plus a
caller-owned workspace sized for the current physical index rows.

```csharp
using VecNet;

var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
index.Add(10, [1.0f, 0.0f]);
index.Add(20, [0.0f, 1.0f]);
index.Add(30, [1.0f, 1.0f]);

ulong[] allowedIds = [10, 30];
var workspace = index.CreateSearchFilterWorkspace();
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

## Application Keys And VecNet IDs

Core VecNet APIs use opaque caller-assigned `ulong` vector IDs. They are not
database primary keys, document IDs, tenant IDs, graph ordinals, or row
positions. If the application uses `string`, `Guid`, database primary keys,
compound tenant/document keys, chunk IDs, or embedding-slot IDs, keep durable
maps in application storage, for example:

- `RecordKey -> VecNetId(s)` for vectors that represent a record or chunks of
  a record.
- `VecNetId -> (RecordKey, ChunkKey, EmbeddingSlot)` for hydrating search
  results.

VecNet durable directories persist vector retrieval state only: vector IDs,
vectors, and graph assets where applicable. They do not persist application
records, payloads, authorization policy, rich metadata, or caller-owned key
maps.

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

Read-only `HnswIndex` searches for supported metrics may overlap for
unfiltered and caller-owned allowlist search when each overlapping call uses
its own result buffer and its own `HnswSearchWorkspace`.

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
documented when they become supported public features.

## Repository

- Source: https://github.com/sh0r3x/VecNet
- Issues: https://github.com/sh0r3x/VecNet/issues

## License

VecNet is licensed under the MIT License. See [LICENSE](LICENSE).
