# HNSW Cosine Benchmark

This page publishes a bounded benchmark summary for VecNet `HnswIndex`
approximate cosine search. It centers recall@k versus latency across
`efSearch`.

## Methodology

This is a VecNet-only benchmark. It measures immutable and durable
`HnswIndex` for `VectorMetric.Cosine`; it does not measure `ExactFlatIndex`,
`HnswMutableIndex`, the optional VectorData adapter, HNSW inner product, or
checkpoint/update workflows. It does not compare VecNet with any other
package, database, service, or native library.

Measured at commit
`2d3ae68db7197f04ce00a174c56a00b82fb24a77`.

Generated-data cases used uniform generated `float32` vectors and queries
from the VecNet benchmark runner, scalar-reference canonical cosine truth,
and external IDs `0..n-1`.

Fashion-MNIST cases used Zalando Research Fashion-MNIST with MIT source
metadata. The dataset shape was `60000` base vectors at `784` dimensions,
with `1000` measured queries and `topK=10`. Raw `uint8` pixels were converted
to unnormalized `float32` caller input; VecNet normalized vectors for cosine
indexing and search.

Cosine distance is canonical `1 - dot(normalizedQuery, normalizedStored)`, with
an expected range of `[0, 2]`. Tiny floating-point excursions up to `1e-6`
below zero or above two are tolerated. Results are ordered by ascending
distance, then ascending external ID when distances compare equal.

The selected search matrix used these cases:

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch values | Runs | Warmup queries | Exact truth source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 32, 64, 128, 256 | 5 | 100 | scalar-reference canonical cosine truth |
| generated | 25000 | 384 | 1000 | 10 | 16 | 200 | 64, 128, 256, 512 | 5 | 100 | scalar-reference canonical cosine truth |
| generated | 10000 | 386 | 1000 | 10 | 16 | 200 | 64, 128, 256 | 5 | 100 | scalar-reference canonical cosine truth |
| generated | 10000 | 768 | 1000 | 100 | 16 | 200 | 128, 256, 512, 1024 | 5 | 100 | scalar-reference canonical cosine truth |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 32, 64, 128, 256 | 5 | 100 | Fashion-MNIST canonical cosine truth |

The `10000 x 386/topK=10` generated rows are a tail-dimension generated case.

The durable context rows show save/open/persisted-byte context for the cases
where those measurements were collected:

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch values | Runs | Warmup queries | Exact truth source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 32, 64, 128, 256 | 5 | 100 | Fashion-MNIST canonical cosine truth |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 256 | 5 | 100 | scalar-reference canonical cosine truth |

Environment: Windows `10.0.26200`, `.NET 10.0.9`, X64, workstation GC
(`Server GC=False`), `Vector<float>.Count=8`, `6` physical cores, `12`
logical processors, and `13.86 GiB` RAM.

Search latency and QPS measure only
`HnswIndex.Search(query, results, workspace)` with caller-owned result buffers
and caller-owned search workspaces. Build, exact-truth generation, dataset or
cache loading, warmup, result capture/comparison, save/open, and report
writing are excluded from search timing. Durable build, save, open, and
opened-search timings are reported separately where the durable runners
measured them.

Latency percentiles are nearest-rank per-run query latencies; the summary
values below are arithmetic means across measured runs. QPS is mean measured
search-call throughput and is supporting context, not the headline. Recall@k
is measured against exact truth for the same query set and `topK`.

Managed allocation is measured around the `HnswIndex.Search` call boundary
with caller-owned buffers and workspaces. Durable persisted bytes, save time,
and open time are present only for durable context rows. Process resident
bytes, process private bytes, GC heap size, sampled peak memory, and broad
capacity limits were not measured.

## Recall Versus Latency

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch | Recall@k | p50 ms | p95 ms | p99 ms | QPS | Build ms | Managed B/query | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 32 | 0.9821 | 0.43638 | 0.70014 | 1.04044 | 2158.25 | 104058.29 | 0 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 64 | 0.9908 | 0.67256 | 1.04042 | 1.545 | 1409.35 | 107774.86 | 0 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 128 | 0.9944 | 1.11574 | 1.79878 | 2.69362 | 850.15 | 106358.48 | 0 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 16 | 200 | 256 | 0.9972 | 1.68078 | 2.3927 | 3.09362 | 577.52 | 104795.95 | 0 | passed |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 32 | 0.2565 | 0.53778 | 0.92516 | 1.61666 | 1698.21 | 24221.48 | 1.55 | passed |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 64 | 0.416 | 0.93152 | 1.62106 | 2.16354 | 973.23 | 25346.51 | 1.45 | passed |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 128 | 0.616 | 1.56916 | 2.76092 | 4.35486 | 574.49 | 25231.47 | 1.25 | passed |
| generated | 10000 | 384 | 1000 | 10 | 16 | 200 | 256 | 0.8235 | 2.64244 | 3.82642 | 4.54712 | 355.08 | 25627.11 | 0.84 | passed |
| generated | 25000 | 384 | 1000 | 10 | 16 | 200 | 64 | 0.2408 | 1.1353 | 2.14264 | 3.32914 | 778.01 | 88613.71 | 0 | passed |
| generated | 25000 | 384 | 1000 | 10 | 16 | 200 | 128 | 0.3931 | 1.89396 | 2.6464 | 3.6838 | 501.9 | 87306.87 | 0 | passed |
| generated | 25000 | 384 | 1000 | 10 | 16 | 200 | 256 | 0.5933 | 3.47638 | 5.43948 | 8.22478 | 266.34 | 85771.74 | 0 | passed |
| generated | 25000 | 384 | 1000 | 10 | 16 | 200 | 512 | 0.8017 | 6.09862 | 8.03826 | 9.40332 | 156.43 | 84473.85 | 0 | passed |
| generated | 10000 | 386 | 1000 | 10 | 16 | 200 | 64 | 0.4205 | 0.85678 | 1.02766 | 1.18074 | 1147.17 | 23338.71 | 1.45 | passed |
| generated | 10000 | 386 | 1000 | 10 | 16 | 200 | 128 | 0.6186 | 1.44468 | 1.72666 | 2.22356 | 672.43 | 23215 | 1.25 | passed |
| generated | 10000 | 386 | 1000 | 10 | 16 | 200 | 256 | 0.8272 | 2.446 | 2.67748 | 2.91188 | 404.7 | 23308.12 | 0.84 | passed |
| generated | 10000 | 768 | 1000 | 100 | 16 | 200 | 128 | 0.4659 | 3.03824 | 3.92636 | 4.67004 | 318.09 | 52195.29 | 0 | passed |
| generated | 10000 | 768 | 1000 | 100 | 16 | 200 | 256 | 0.68416 | 4.85868 | 5.30364 | 5.82878 | 203.46 | 48343.29 | 0 | passed |
| generated | 10000 | 768 | 1000 | 100 | 16 | 200 | 512 | 0.88005 | 7.41664 | 8.07512 | 8.75364 | 133.2 | 46813.5 | 0 | passed |
| generated | 10000 | 768 | 1000 | 100 | 16 | 200 | 1024 | 0.97898 | 10.79562 | 15.17928 | 17.99306 | 87.58 | 47581.81 | 0 | passed |

The generated and Fashion-MNIST rows show the expected approximate-search
tradeoff: higher `efSearch` improved recall@k and usually increased latency.
QPS, build time, allocation, and durable storage values are supporting
context for interpreting those recall/latency curves.

## Durable Context

These rows are not a separate durability benchmark suite. They report the
durable context cases where persisted bytes, save time, open time, build
time, opened-index search latency, source/opened parity, and opened read-only
behavior were measured.

| Dataset | Vectors | Dim | Queries | topK | efSearch | Recall@k | Opened p50 ms | Build ms | Save ms | Open ms | Persisted bytes | Bytes/vector | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 32 | 0.9803 | 0.42232 | 104031.64 | 1364.51 | 1341.71 | 197094300 | 3284.91 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 64 | 0.9905 | 0.68 | 105610.45 | 1465.43 | 1319.41 | 197094300 | 3284.91 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 128 | 0.9945 | 1.13176 | 111893.04 | 1306.83 | 1370.31 | 197094301 | 3284.91 | passed |
| Fashion-MNIST | 60000 | 784 | 1000 | 10 | 256 | 0.9972 | 1.70176 | 108742.52 | 1368.77 | 1400.43 | 197094301 | 3284.91 | passed |
| generated | 10000 | 384 | 1000 | 10 | 256 | 0.8156 | 2.51674 | 22965.9 | 113.23 | 130.05 | 16848202 | 1684.82 | passed |

## Current 1.3.x Update

A current `1.3.x` refresh reran the same comparable matrices from source
commit `74896b9a6d9192d1a33a66e947f481a6eff8accf`, with the same datasets,
query counts, `M`, `efConstruction`, `efSearch` values, run counts, warmup
shape, caller-owned result buffers, and caller-owned search workspaces. The
tables above remain the original public summary; this section describes the
comparison against the current source line.

Recall@k was unchanged for every comparable generated, Fashion-MNIST, and
durable row. The recall/latency shape also stayed the same: higher
`efSearch` improved recall and cost more search time.

| Area | Recall result | Current latency result | Allocation note |
| --- | --- | --- | --- |
| generated immutable rows | unchanged for every comparable row | lower across the matrix; p50 was `0.67x` to `0.76x` of the original summary, and p99 was `0.34x` to `0.85x` | generated rows that reported small B/query values stayed at the same small values; the other generated rows stayed at `0 B/query` |
| Fashion-MNIST immutable rows | unchanged for every comparable row | p50 was lower for all `efSearch` values; p95 and p99 were lower through `efSearch=128` | `0 B/query` |
| durable opened-search rows | unchanged for every comparable row | generated opened p50/p95/p99 were `0.75x`/`0.71x`/`0.73x`; Fashion-MNIST opened p50 was `0.72x` to `0.76x`, p95 was `0.62x` to `0.79x`, and p99 was `0.68x` to `0.88x` | opened-search rows stayed at `0 B/query` |

The main caveat is Fashion-MNIST immutable search at `efSearch=256`: recall
was unchanged and p50 improved, but p95 was `1.04x` and p99 was `1.18x` of
the original summary. Build times were lower for every comparable current row,
and durable persisted bytes were nearly unchanged.

## Measurement Availability

Measured for the selected search cases: recall@k, p50/p95/p99 search
latency, QPS, HNSW build time, and search-call managed allocation.

Durable persisted bytes, save time, open time, source/opened search parity,
and opened read-only mutation rejection were measured only for the durable
context rows.

Process resident bytes, process private bytes, GC heap size, sampled peak
memory, checkpoint time, mutable update profile, HNSW inner product,
NativeAOT/trimming behavior, concurrency behavior, and competitor results
were not measured.

Layout estimates are runner estimates only. They are not observed process
usage and are not a memory-capacity claim.

## Do Not Generalize These Numbers

These numbers are not a package-wide claim. They cover only `HnswIndex`
cosine for the listed immutable and durable cases, not exact-flat, filters,
mutable HNSW, the optional VectorData adapter, application workflows, or the
package as a whole.

These numbers are not a platform-wide claim. They do not claim equivalent
results on other operating systems, runtimes, architectures, processors, GC
modes, RAM sizes, or deployment environments.

These numbers are not a NativeAOT or trimming claim, not a competitor
comparison, not a regression threshold, not a broad memory or capacity
support envelope, and not a public-dataset-general semantic relevance claim.

These numbers do not describe HNSW inner-product results, mutable HNSW cosine,
checkpoint behavior, update-profile behavior, advanced filtering, or
concurrency.
