# Exact-Flat Generated-Data Benchmark

This page publishes one reviewed benchmark summary for VecNet
`ExactFlatIndex` squared Euclidean search. It is intentionally narrow.

## Methodology

This is a generated-data benchmark only. It used generated uniform `float32`
vectors and queries from the VecNet benchmark runner, with scalar-reference
generated exact top-k truth and VecNet's ascending distance, then ascending
external ID tie policy. It is not a public dataset result and does not imply
semantic-search relevance.

The measured surface is only `ExactFlatIndex` exact-flat search for
`VectorMetric.SquaredEuclidean`. It does not measure cosine, inner product,
filters, persistence, updates, HNSW, or the optional VectorData adapter. It
does not compare VecNet with any other package, database, service, or native
library.

Package/source identity: VecNet `1.0.1` source behavior at commit
`ea3094a624cc6054b0507805acd438f09e53694d`. Product source status was clean
for `src/VecNet` in the reviewed evidence.

The run was single-process and local, with query concurrency `1`. Each case
used `1000` measured queries, `5` measured runs, and `100` warmup queries.
The selected generated matrix was declared before the numbers: vector-count
tiers `1000`, `10000`, and `50000`; dimensions `128`, `384`, and `768`;
topK values `1`, `10`, and attempted `100` in the listed cases. Follow-up
probes were limited to `topK=100` at `128d` and `384d`, plus `768d/topK=10`,
after earlier strict-validation failures.

Environment context: Microsoft Windows `10.0.26200`, `.NET 10.0.9`, X64,
RID `win-x64`, workstation GC (`Server GC=False`), `Vector<float>.Count=8`,
`6` physical cores, `12` logical processors, and `13.86 GiB` installed RAM
context. CPU model, owner/account names, host name, and local paths are not
published.

Latency and throughput measure the public
`ExactFlatIndex.Search(query, results)` call only. Setup, index build,
scalar-reference truth generation, warmup queries, result comparison, and
report writing are excluded. Latency percentiles are nearest-rank per-run
query latencies; the summary values below are arithmetic means across the
five per-run percentile values. QPS is mean per-run measured query
throughput for the public search call only.

Managed allocation is measured with `GC.GetAllocatedBytesForCurrentThread`
around each public search call boundary. Process/private bytes, GC heap
values, sampled peaks, layout estimates, persisted bytes, open time, and
checkpoint time were not measured.

## Results

Strict exact validation passed for `7` cases and failed for `5` cases.
After the VEC-215 public evidence policy, all `12` cases were accepted for
this bounded summary. The strict-failed cases remain visible in the table
instead of being omitted.

| Tier | Dimension | Vectors | Queries | topK | Strict validation | Public evidence | Recall@k | Ordered agreement | Distance tolerance | Missing | Duplicate | Wrong ID away | p50 ms | p95 ms | p99 ms | QPS | Managed B/query |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 128 | 1000 | 1000 | 1 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 0.0231 | 0.22308 | 0.26622 | 33139.73 | 0 |
| small | 384 | 1000 | 1000 | 10 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 0.0636 | 0.09632 | 0.1226 | 15470.43 | 0 |
| small | 768 | 1000 | 1000 | 100 | failed | accepted-near-tie-order-only | 0.99999 | 0.99989 | passed | 0 | 0 | 0 | 0.14254 | 0.20938 | 0.25242 | 6594.4 | 0 |
| medium | 128 | 10000 | 1000 | 1 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 0.22826 | 0.32966 | 0.43426 | 4159.16 | 0 |
| medium | 384 | 10000 | 1000 | 10 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 0.74512 | 1.10366 | 1.26298 | 1252.76 | 0 |
| medium | 768 | 10000 | 1000 | 100 | failed | accepted-near-tie-order-only | 0.99999 | 0.99969 | passed | 0 | 0 | 0 | 1.76028 | 2.30332 | 2.75476 | 541.83 | 0 |
| larger-local-feasible | 128 | 50000 | 1000 | 1 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 1.69512 | 2.3804 | 3.22916 | 552.24 | 0 |
| larger-local-feasible | 384 | 50000 | 1000 | 10 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 4.95256 | 5.84686 | 6.91674 | 198.91 | 0 |
| larger-local-feasible | 768 | 50000 | 1000 | 100 | failed | accepted-near-tie-order-only | 1 | 0.99976 | passed | 0 | 0 | 0 | 9.84714 | 11.41124 | 13.97954 | 99.71 | 0 |
| small | 128 | 1000 | 1000 | 100 | failed | accepted-near-tie-order-only | 1 | 0.99994 | passed | 0 | 0 | 0 | 0.04166 | 0.26638 | 0.34424 | 17856.49 | 0 |
| small | 384 | 1000 | 1000 | 100 | failed | accepted-near-tie-order-only | 1 | 0.99972 | passed | 0 | 0 | 0 | 0.07706 | 0.11976 | 0.14932 | 11904.55 | 0 |
| small | 768 | 1000 | 1000 | 10 | passed | passed-strict | 1 | 1 | passed | 0 | 0 | 0 | 0.10284 | 0.15254 | 0.19 | 9357.14 | 0 |

## Validation Notes

Strict validation passed for `7` cases. The `5` `topK=100` cases were
accepted under VEC-215 as deterministic near-tie/order-only cases. In those
cases, returned distances passed tolerance, with no missing results, no
duplicate results, no distance mismatches, and no wrong IDs away from
near-tie boundaries.

Ordered agreement remains visible because these cases are not presented as
perfectly ordered scalar-reference matches. The near-tie explanation is
limited to generated exact-flat evidence where scalar-reference truth and
the measured production search path can order near-boundary or near-adjacent
results differently while still returning tolerated distances for the
returned IDs.

## Do Not Generalize These Numbers

These numbers are not a capacity claim. They cover only the listed local
generated cases, vector counts, dimensions, `topK` values, query count, run
count, warmup count, runtime, and hardware context.

These numbers are not a platform-wide claim. They do not claim equivalent
results on other operating systems, runtimes, architectures, CPUs, GC modes,
RAM sizes, or deployment environments.

These numbers are not a package-wide claim. They do not describe HNSW,
filters, persistence, updates, checkpointing, the optional VectorData adapter,
or any application-level semantic-search workflow.

These numbers are not a NativeAOT or trimming claim, not a regression
threshold, and not a public-dataset result.

They are also not a competitor comparison. No other library, database,
service, or native vector-search package was measured here.
