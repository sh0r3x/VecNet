# HNSW Squared-L2 Benchmark

This page publishes a bounded benchmark summary for VecNet `HnswIndex`
approximate squared-L2 search. It centers recall@k versus latency across
`efSearch`.

## Methodology

This is a VecNet-only benchmark. It measures only `HnswIndex` for
`VectorMetric.SquaredEuclidean`; it does not measure `ExactFlatIndex`,
`HnswMutableIndex`, the optional VectorData adapter, cosine HNSW, or
inner-product HNSW. It does not compare VecNet with any other package,
database, service, or native library.

Measured against VecNet `1.0.1` at commit
`4c84a7c9a029e00af8249e5f65bb9332b9b2f35b` on branch
`public-benchmarking`.

The selected generated-data cases used uniform generated `float32` vectors
and queries from the VecNet benchmark runner, with scalar-reference exact
truth and external IDs `0..n-1`.

The Fashion-MNIST cases used Zalando Research Fashion-MNIST with MIT license
metadata. The dataset shape was `60000` base vectors at `784` dimensions.
The Fashion-MNIST exact-truth subset used for this run covered `50` queries
at truth depth `100`, so the Fashion-MNIST search and durable context cases
use `50` measured queries.

The generated search matrix used these cases:

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch values | Runs | Warmup queries | Exact truth source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| generated | 10000 | 384 | 300 | 10 | 16 | 128 | 32, 64, 128 | 3 | 30 | scalar-reference exact truth |
| generated | 25000 | 384 | 300 | 10 | 16 | 128 | 32, 64, 128 | 3 | 30 | scalar-reference exact truth |
| generated | 10000 | 768 | 200 | 100 | 16 | 128 | 128, 256, 512 | 3 | 20 | scalar-reference exact truth |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 16 | 128 | 32, 64, 128 | 3 | 30 | Fashion-MNIST exact-truth subset |

The durable context cases show save/open/persisted-byte context for the
cases where those measurements were collected:

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch | Runs | Warmup queries | Exact truth source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| generated | 10000 | 384 | 200 | 10 | 16 | 128 | 128 | 2 | 20 | scalar-reference exact truth |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 16 | 128 | 128 | 1 | 30 | Fashion-MNIST exact-truth subset |

Environment: Windows `10.0.26200`, `.NET 10.0.9`, X64, workstation GC
(`Server GC=False`), `Vector<float>.Count=8`, `6` physical cores, `12`
logical processors, and `13.86 GiB` RAM.

Search latency and QPS measure only
`HnswIndex.Search(query, results, workspace)` with caller-owned result buffers
and caller-owned search workspaces. Build, exact-truth generation, dataset or
cache loading, warmup, result capture/comparison, save/open, and report
writing are excluded from search timing. Durable build, save, open, and
opened-search timings are reported separately where the durable runners
already measured them.

Latency percentiles are nearest-rank per-run query latencies; the summary
values below are arithmetic means across measured runs. QPS is mean measured
search-call throughput and is supporting context, not the headline.
Recall@k is measured against exact truth for the same query set and `topK`.

Managed allocation is measured around the `HnswIndex.Search` call boundary
with caller-owned buffers and workspaces. Layout/payload memory values are
runner estimates or payload lower bounds, not observed process memory.
Durable persisted bytes, save time, and open time are present only for the
durable context cases.

## Recall Versus Latency

| Dataset | Vectors | Dim | Queries | topK | M | efConstruction | efSearch | Recall@k | p50 ms | p95 ms | p99 ms | QPS | Build ms | Managed B/query | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 16 | 128 | 32 | 0.99 | 0.2025 | 0.45583 | 0.65653 | 4197.25 | 37570 | 0 | passed |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 16 | 128 | 64 | 1 | 0.3259 | 0.6637 | 1.02387 | 2739.75 | 35563.16 | 0 | passed |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 16 | 128 | 128 | 1 | 0.52277 | 0.8148 | 1.24887 | 1831.52 | 31953.3 | 0 | passed |
| generated | 10000 | 384 | 300 | 10 | 16 | 128 | 32 | 0.399 | 0.21163 | 0.4527 | 0.71697 | 4122.47 | 7841.25 | 0 | passed |
| generated | 10000 | 384 | 300 | 10 | 16 | 128 | 64 | 0.554 | 0.4317 | 0.9291 | 1.35783 | 1963.1 | 8172.89 | 0 | passed |
| generated | 10000 | 384 | 300 | 10 | 16 | 128 | 128 | 0.73567 | 0.5981 | 0.8786 | 0.99517 | 1658.73 | 7297.77 | 0 | passed |
| generated | 25000 | 384 | 300 | 10 | 16 | 128 | 32 | 0.24167 | 0.27247 | 0.52493 | 0.77117 | 3240.12 | 26781.1 | 0 | passed |
| generated | 25000 | 384 | 300 | 10 | 16 | 128 | 64 | 0.37633 | 0.40773 | 0.6707 | 0.91553 | 2239.57 | 23139.93 | 0 | passed |
| generated | 25000 | 384 | 300 | 10 | 16 | 128 | 128 | 0.533 | 1.1191 | 1.9815 | 2.783 | 832.47 | 28737.61 | 0 | passed |
| generated | 10000 | 768 | 200 | 100 | 16 | 128 | 128 | 0.5992 | 1.42157 | 2.9025 | 3.54007 | 615.35 | 17177.97 | 0 | passed |
| generated | 10000 | 768 | 200 | 100 | 16 | 128 | 256 | 0.7729 | 2.726 | 5.2115 | 8.06233 | 329.39 | 16509.44 | 0 | passed |
| generated | 10000 | 768 | 200 | 100 | 16 | 128 | 512 | 0.90515 | 3.88047 | 6.1316 | 7.42217 | 244 | 17037.75 | 0 | passed |

The generated rows show the expected approximate-search tradeoff: higher
`efSearch` improved recall@k and usually increased latency. QPS, build time,
allocation, memory estimates, and durable storage values are supporting
context for interpreting those recall/latency curves.

## Durable Context

These rows are not a separate durability benchmark suite. They report the two
durable context cases where persisted bytes, save time, open time, build
time, and opened-index search latency were measured.

| Dataset | Vectors | Dim | Queries | topK | efSearch | Recall@k | Opened p50 ms | Build ms | Save ms | Open ms | Persisted bytes | Bytes/vector | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST | 60000 | 784 | 50 | 10 | 128 | 1 | 0.5396 | 33574.87 | 1910.39 | 2030.95 | 197085222 | 3284.75 | passed |
| generated | 10000 | 384 | 200 | 10 | 128 | 0.7485 | 0.8324 | 9487.22 | 199.49 | 188.86 | 16850211 | 1685.02 | passed |

## Measurement Availability

Measured for the selected search cases: recall@k, p50/p95/p99 search
latency, QPS, HNSW build time, and search-call managed allocation.
Search-call managed allocation is `0` managed bytes per query in the rows
above.

Layout/payload memory values are runner estimates or lower bounds. They are
not observed resident memory, private bytes, GC heap values, sampled peaks,
or total application memory.

Durable persisted bytes, save time, and open time are available only for the
two durable context cases. Process resident bytes, process private bytes, GC
heap size, GC committed/fragmented values, sampled peak memory, and
checkpoint time were not measured.

## Excluded Cases

The first three generated `10000 x 384/topK=10` runs were accidentally
launched concurrently. Their outputs were overwritten by sequential reruns,
and the concurrent runs are not used here because process contention can
distort latency.

Fashion-MNIST uses `50` measured queries because the Fashion-MNIST exact-truth
subset used for this run covered `50` queries at truth depth `100`.
`query-count=100` and `query-count=300` Fashion-MNIST variants are not
included.

## Do Not Generalize These Numbers

These numbers are not a package-wide claim. They cover only `HnswIndex`
squared-L2 for the cases listed above, not exact-flat, filters, mutable HNSW,
the optional VectorData adapter, application workflows, or the package as a
whole.

These numbers are not a platform-wide claim. They do not claim equivalent
results on other operating systems, runtimes, architectures, processors, GC
modes, RAM sizes, or deployment environments.

These numbers are not a NativeAOT or trimming claim, not a competitor
comparison, not a regression threshold, and not a public-dataset-general
semantic relevance claim.

These numbers do not describe advanced HNSW filtering, storage, concurrency,
mutation/checkpoint behavior, cosine HNSW, or inner-product HNSW.
