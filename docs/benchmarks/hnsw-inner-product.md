# HNSW Inner-Product Benchmark

This page publishes a bounded benchmark summary for VecNet `HnswIndex`
approximate inner-product search. It centers recall@k versus latency across
`efSearch`.

## Methodology

This is a VecNet-only benchmark. It measures immutable and durable/opened
read-only `HnswIndex` for `VectorMetric.InnerProduct`; it does not measure
`ExactFlatIndex`, `HnswMutableIndex`, update profiles, checkpoint cost,
advanced filtering, advanced concurrency, the optional VectorData adapter, or
the package as a whole. It does not compare VecNet with any other package,
database, service, or native library.

Inner-product distance is canonical negative dot product: lower distance is
better, vectors are used exactly as supplied, and VecNet does not normalize or
clamp inner-product inputs. Zero vectors are valid. A zero query or zero
indexed vector produces a dot product of `0` with the corresponding zero row
and therefore canonical distance `0`; ranking remains magnitude-sensitive, so
nonzero vectors with larger positive dot products can rank ahead of
zero-vector matches.

Measured at commit `081fb61ce1b96ca8f699ffa9b9d2a0c764c4f643` from a clean
working tree. This is a source-state benchmark summary, not a package-wide
claim.

Generated profiles used deterministic benchmark-runner data:

| Profile | Distribution | Seed | Notes |
| --- | --- | --- | --- |
| generated uniform | `uniform[-1,1)` | `0x5EED3671` | Mixed-sign generated inner-product workload. |
| generated norm-skewed | `uniform[-1,1)` scaled by deterministic row norm factors `[0.125,8]` | `0x5EED3672` | Magnitude-sensitive workload for raw inner product. |
| generated zero-vector | `uniform[-1,1)` with deterministic all-zero vector rows every fifth indexed row and all-zero query rows every third query row | `0x5EED3673` | Explicit zero-vector-validity context for inner product. |

Fashion-MNIST rows used the admitted `fashion-mnist-784-inner-product` dataset
identity with `60000` base vectors, `10000` available query vectors, `784`
dimensions, raw `uint8` pixels converted to unnormalized `float32`, `50`
measured queries, `topK=10`, and exact inner-product truth at depth `100`.
The existing cache, manifest, converted matrices, and truth/provenance checks
were validated by the runner without network acquisition before the rows were
admitted.

Fashion-MNIST public provenance:

| Field | Value |
| --- | --- |
| Dataset source | Zalando Research Fashion-MNIST |
| Maintainer URL | `https://github.com/zalandoresearch/fashion-mnist` |
| Download root | `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/` |
| Official README | `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md` |
| License | MIT, Copyright 2017 Zalando SE |
| Access date | 2026-06-12 |
| Version/status | No release tag available; official raw file URLs plus source MD5 checksums are pinned. |
| Stable dataset identity | `fashion-mnist-784-inner-product` |
| Conversion rules | Source IDX `uint8` image pixels are converted deterministically to row-major little-endian `float32` matrix values `0..255` without normalization. Labels are validated but are not stored in converted vectors or truth artifacts. |
| Exact-truth method | VecNet scalar-reference inner-product truth: canonical distance is `-dot(rawQuery, rawBase)`, query subset is the first `50` admitted query vectors, truth depth is `100`, and ties are ordered by ascending scalar-reference canonical distance then ascending base ordinal. |

Fashion-MNIST checksums:

| Artifact | Role | Official MD5 | SHA-256 |
| --- | --- | --- | --- |
| `train-images-idx3-ubyte.gz` | base images | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | base labels | `25c81989df183df01b3e8a0aad5dffbe` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | query images | `bef4ecab320f06d8554ea6380940ec79` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | query labels | `bb300cfdad3c16e7a12a480ee83cd310` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |
| converted base matrix | converted vectors | n/a | `e5cc5fd9a5b6acaa953ec6ef7340f927d343ae9d643b964b4f7c04625622c9c4` |
| converted query matrix | converted queries | n/a | `8170f945abd71b67cb22da95206e6e110afb57bb8e7939af1e424b4cf925659c` |
| conversion manifest | conversion record | n/a | `ddaac377d99c7de4611d52be144ed5b622d2fdf51be099c5f672b80dff9a8d81` |
| dataset admission manifest | dataset record | n/a | `1288a59e02dc91f52f442bdbbfa0f0c8a98a0928a94dd4e60aa5f204e6bfe77e` |
| exact truth | scalar-reference truth | n/a | `51abdca2a872328199b159bf62a5efe68819a9adb42f503c4013227f54e8f99a` |

All rows used single-query search calls with caller-owned `SearchResult[]`
buffers and caller-owned `HnswSearchWorkspace` instances. Search latency and
QPS time only `HnswIndex.Search(query, results, workspace)`. Build,
exact-truth generation or truth loading, dataset generation or matrix loading,
warmup queries, result capture/comparison, save/open, output-byte scans, and
report writing are excluded from search timing.

Latency percentiles are nearest-rank per-run query latencies; summary values
are arithmetic means across measured runs. QPS is supporting context.
Recall@k is set recall against exact inner-product truth for the same query
set and `topK`. Managed allocation is measured around the search-call
boundary.

Environment: Windows `10.0.26200`, `.NET 10.0.9`, X64, workstation GC
(`Server GC=False`), `Vector<float>.Count=8`, and `12` logical processors.
Available RAM observed after benchmark collection was `3458` MiB.

The immutable search matrix used `M=16`, `efConstruction=200`, and `efSearch`
values `32,64,128,256`. Generated rows used `10000` vectors, dimension `384`,
`300` queries, `topK=10`, `3` measured runs, and `30` warmup queries.
Fashion-MNIST rows used `60000` vectors, dimension `784`, `50` queries,
`topK=10`, `3` measured runs, and `30` warmup queries.

## Reproducibility Command Templates

These templates list the public scenario options used for the measured rows.
Choose report and durable snapshot directories that fit your environment. For
Fashion-MNIST, use a prepared and checksum-validated dataset cache.

Generated immutable recall curve:

```text
dotnet run --project <benchmark-runner-project> --configuration Release --no-build -- hnsw-generated --metric InnerProduct --vector-profile <uniform|norm-skewed|zero-vector> --dimension 384 --vectors 10000 --queries 300 --top-k 10 --runs 3 --warmup-queries 30 --seed <profile-seed> --m 16 --ef-construction 200 --ef-search <32|64|128|256> --hnsw-seed <profile-hnsw-seed>
```

Generated durable/opened context:

```text
dotnet run --project <benchmark-runner-project> --configuration Release --no-build -- hnsw-generated-durable --metric InnerProduct --vector-profile <uniform|norm-skewed|zero-vector> --dimension 384 --vectors 10000 --queries 200 --top-k 10 --runs 2 --warmup-queries 20 --seed <profile-durable-seed> --m 16 --ef-construction 200 --ef-search 128 --hnsw-seed <profile-durable-hnsw-seed>
```

Fashion-MNIST immutable recall curve:

```text
dotnet run --project <benchmark-runner-project> --configuration Release --no-build -- external-fashion-mnist-hnsw --cache-root <prepared-fashion-mnist-cache> --metric inner-product --query-count 50 --top-k 10 --runs 3 --warmup-queries 30 --m 16 --ef-construction 200 --ef-search <32|64|128|256> --hnsw-seed 0x484E535700036780
```

Fashion-MNIST durable/opened context:

```text
dotnet run --project <benchmark-runner-project> --configuration Release --no-build -- external-fashion-mnist-hnsw-durable --cache-root <prepared-fashion-mnist-cache> --metric inner-product --query-count 50 --top-k 10 --runs 1 --warmup-queries 30 --m 16 --ef-construction 200 --ef-search 128 --hnsw-seed 0x484E535700036790
```

Generated profile seeds:

| Profile | Immutable data seed | Immutable HNSW seed | Durable data seed | Durable HNSW seed |
| --- | --- | --- | --- | --- |
| generated uniform | `0x5EED3671` | `0x484E535700036710` | `0x5EED3674` | `0x484E535700036740` |
| generated norm-skewed | `0x5EED3672` | `0x484E535700036720` | `0x5EED3675` | `0x484E535700036750` |
| generated zero-vector | `0x5EED3673` | `0x484E535700036730` | `0x5EED3676` | `0x484E535700036760` |

## Recall Versus Latency

| Dataset/profile | Vectors | Dim | Queries | topK | M | efConstruction | efSearch | Recall@k | p50 ms | p95 ms | p99 ms | QPS | Build ms | Managed B/query | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST raw inner product | 60000 | 784 | 50 | 10 | 16 | 200 | 32 | 0.522 | 0.16473 | 0.2753 | 0.29103 | 5665.74 | 55284.81 | 0 | passed |
| Fashion-MNIST raw inner product | 60000 | 784 | 50 | 10 | 16 | 200 | 64 | 0.544 | 0.2286 | 0.41357 | 0.44957 | 4068.35 | 54456.7 | 0 | passed |
| Fashion-MNIST raw inner product | 60000 | 784 | 50 | 10 | 16 | 200 | 128 | 0.58 | 0.3412 | 0.66627 | 1.2654 | 2484.07 | 52268.3 | 0 | passed |
| Fashion-MNIST raw inner product | 60000 | 784 | 50 | 10 | 16 | 200 | 256 | 0.602 | 0.51783 | 1.01383 | 1.94533 | 1658.19 | 55175.26 | 0 | passed |
| generated norm-skewed | 10000 | 384 | 300 | 10 | 16 | 200 | 32 | 0.67067 | 0.24063 | 0.51813 | 0.6465 | 3759.71 | 11947.91 | 0 | passed |
| generated norm-skewed | 10000 | 384 | 300 | 10 | 16 | 200 | 64 | 0.86233 | 0.32903 | 0.68477 | 0.90183 | 2676.33 | 11323.61 | 0 | passed |
| generated norm-skewed | 10000 | 384 | 300 | 10 | 16 | 200 | 128 | 0.97167 | 0.58717 | 1.29107 | 1.64337 | 1477.12 | 11700.39 | 0 | passed |
| generated norm-skewed | 10000 | 384 | 300 | 10 | 16 | 200 | 256 | 0.99667 | 0.86857 | 1.8075 | 2.35147 | 1004.97 | 11453.98 | 0 | passed |
| generated uniform | 10000 | 384 | 300 | 10 | 16 | 200 | 32 | 0.273 | 0.30443 | 0.64967 | 0.87777 | 2909.36 | 15481.72 | 0 | passed |
| generated uniform | 10000 | 384 | 300 | 10 | 16 | 200 | 64 | 0.41867 | 0.529 | 1.1644 | 1.58897 | 1641.58 | 15176.18 | 0 | passed |
| generated uniform | 10000 | 384 | 300 | 10 | 16 | 200 | 128 | 0.63433 | 0.85423 | 1.76703 | 2.0797 | 1036.96 | 14185.06 | 0 | passed |
| generated uniform | 10000 | 384 | 300 | 10 | 16 | 200 | 256 | 0.83067 | 1.55483 | 3.19993 | 3.92347 | 551.07 | 13896.87 | 0 | passed |
| generated zero-vector | 10000 | 384 | 300 | 10 | 16 | 200 | 32 | 0.339 | 0.35357 | 1.73763 | 2.1227 | 1555.91 | 13915.32 | 0 | passed |
| generated zero-vector | 10000 | 384 | 300 | 10 | 16 | 200 | 64 | 0.48433 | 0.5861 | 2.50843 | 2.96593 | 1007.73 | 14043.39 | 0 | passed |
| generated zero-vector | 10000 | 384 | 300 | 10 | 16 | 200 | 128 | 0.689 | 1.32367 | 4.1048 | 5.0451 | 567.81 | 14995.29 | 0 | passed |
| generated zero-vector | 10000 | 384 | 300 | 10 | 16 | 200 | 256 | 0.84267 | 1.9063 | 3.8064 | 4.23597 | 467.67 | 14323.15 | 0 | passed |

Across these rows, increasing `efSearch` generally improved recall and
increased latency. The generated zero-vector rows validate that explicit
all-zero query and indexed rows are accepted for inner product and that
returned distances remain finite and recomputed-distance validation passes.

## Durable Context

These rows are durable context, not a broad durability benchmark suite. They
report save/open/opened-search context for immutable `HnswIndex.Save` followed
by `HnswIndex.OpenReadOnly`, source/opened result parity, opened read-only
mutation rejection, persisted bytes, and opened-index search validation.

| Dataset/profile | Vectors | Dim | Queries | topK | efSearch | Recall@k | Opened p50 ms | Opened p95 ms | Build ms | Save ms | Open ms | Persisted bytes | Bytes/vector | Managed B/query | Source/opened parity | Opened read-only | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Fashion-MNIST raw inner product | 60000 | 784 | 50 | 10 | 128 | 0.566 | 0.344 | 0.7282 | 51823.02 | 1694.45 | 1351.16 | 197086570 | 3284.78 | 0 | matched | passed | passed |
| generated norm-skewed | 10000 | 384 | 200 | 10 | 128 | 0.974 | 0.5888 | 1.4672 | 11309.05 | 138 | 155.11 | 16852999 | 1685.3 | 0 | matched | passed | passed |
| generated uniform | 10000 | 384 | 200 | 10 | 128 | 0.6155 | 0.92695 | 1.82415 | 14550.53 | 164.26 | 154.54 | 16848317 | 1684.83 | 0 | matched | passed | passed |
| generated zero-vector | 10000 | 384 | 200 | 10 | 128 | 0.7345 | 1.072 | 3.23165 | 14869.08 | 161.71 | 151.99 | 16848583 | 1684.86 | 0 | matched | passed | passed |

## Measurement Availability

Measured: recall@k, p50/p95/p99 search latency, QPS, HNSW build time,
search-call managed allocation, generated durable save/open time,
Fashion-MNIST durable save/open time, opened-search latency, persisted bytes,
source/opened parity, and opened read-only mutation rejection.

Reported only as layout or payload lower-bound context: vector payload bytes,
ID payload bytes, level payload bytes, graph payload bytes, and
search-workspace bytes from the runner's durable reports.

Not measured: process resident memory, process private bytes, GC heap size,
sampled peak memory, temporary disk peak, checkpoint time, mutable-update
behavior, advanced filtering, advanced concurrency, package-consumer behavior,
NativeAOT/trimming behavior, and external implementation comparisons.

## Do Not Generalize These Numbers

These numbers measure immutable and durable/opened read-only `HnswIndex`
inner-product search only. They do not measure `ExactFlatIndex`,
`HnswMutableIndex`, updates, checkpoint cost, adaptive retry, tombstone
profiles, allowlist filtering, advanced graph-aware filtering, concurrent
search, the optional VectorData adapter, package installation, or the package
as a whole.

These numbers are not a platform-wide claim. They do not claim equivalent
results on other operating systems, architectures, processors, GC modes,
memory sizes, storage devices, runtimes, or deployment environments.

These numbers are not a semantic relevance claim for Fashion-MNIST. Raw
Fashion-MNIST inner product is useful here as a reproducible public-data
vector workload with non-negative, norm-dominated pixel vectors; generated
mixed-sign and norm-skewed rows are included because raw Fashion-MNIST does
not exercise every inner-product shape.
