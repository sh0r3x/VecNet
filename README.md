# VecNet

VecNet is an embedded vector indexing library for .NET applications. It is
designed for applications that keep their records, metadata, authorization,
and application storage outside the vector engine while using VecNet for dense
vector retrieval.

This package is a `0.1` preview. The currently documented surface focuses on
exact flat indexing, canonical distance semantics, durable exact-flat save and
open, exact allowlist filtering, reusable exact candidate sets, and exact
mutation/checkpoint workflows.

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

## Preview Limitations

- APIs and file formats are still preview and may change before a stable
  release.
- VecNet stores vector IDs and vectors, not application records or payloads.
- Application metadata filtering, authorization, transactions, backups, and
  record hydration remain the responsibility of the host application.
- Approximate indexing, compressed indexes, SSD-scale indexes, richer key
  mapping, optional integration adapters, and release-grade operational
  tooling are planned work, not supported public package capabilities in this
  preview.
- This README does not make public performance, memory, allocation, recall, or
  platform support claims.

## Installation

When a preview package is available from your configured NuGet source, add it
to a .NET 10 project:

```bash
dotnet add package VecNet --version 0.1.0-preview.1
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

Use `Save` to write an exact-flat index to a new or empty directory. Use
`OpenReadOnly` to open that directory as an immutable searchable index.

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

`Save` does not overwrite an existing non-empty directory.

## Filtering

For one-off filtering, pass an allowlist of external vector IDs plus a
caller-owned workspace sized for the current index.

```csharp
using VecNet;

var index = new ExactFlatIndex(2, VectorMetric.InnerProduct);
index.Add(10, [1.0f, 0.0f]);
index.Add(20, [0.0f, 1.0f]);
index.Add(30, [1.0f, 1.0f]);

ulong[] allowedIds = [10, 30];
var workspace = new ExactFlatSearchFilterWorkspace(index.VectorCount);
Span<SearchResult> results = stackalloc SearchResult[2];

int written = index.Search(
    [1.0f, 0.5f],
    allowedIds,
    results,
    workspace);
```

For a filter reused across searches on the same visible generation, create an
exact candidate set from external IDs.

```csharp
ExactFlatCandidateSet candidates = index.CreateCandidateSet([10, 30]);
int written = index.Search([1.0f, 0.5f], candidates, results);
```

Candidate sets are bound to the creating index instance and generation. Rebuild
them after successful mutation or checkpoint operations.

## Updates And Checkpoints

`TryAdd` and `TryDelete` report status instead of throwing for duplicate,
unknown, repeated-delete, and read-only mutation cases.

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

`Checkpoint` writes a compact live exact-flat generation to a new or empty
directory and publishes that compact generation in the current index instance.

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
