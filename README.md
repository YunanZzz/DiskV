# DiskV

Vector databases have recently drawn significant attention due to the rapid rise of large language models (LLMs). To improve the cost-efficiency of vector search, several disk-based vector indexes—such as SPANN and DiskANN—have been proposed. However, these indexes are mainly optimized for fast SSDs and often perform poorly on low-cost storage devices such as HDDs or cloud-based storage, which are widely adopted in practice to reduce infrastructure costs.

In this work, we introduce **DiskV**, the first disk-based vector index designed to deliver high search performance across heterogeneous storage media, including both **low-cost devices** (e.g., HDDs and cloud storage) and **high-speed devices** (e.g., local SSDs). The core design of DiskV builds upon a **quantization-based indexing structure** (similar to SPANN), which partitions the dataset into large buckets and leverages **sequential access patterns** to better utilize low-cost storage.

To further achieve high performance on fast devices, DiskV introduces several novel optimizations, including **query-aware adaptive search**, **segment-based pruning**, and an **optimized asynchronous I/O engine**. Extensive experiments on three billion-scale datasets demonstrate that DiskV significantly outperforms existing disk-based indexes on low-cost storage (by up to **17×**) while maintaining comparable or superior performance on fast storage (by up to **2.5×**).

We believe DiskV provides a **practical and cost-efficient foundation** for large-scale AI applications that require scalable and high-performance vector search.

---



## Installation

### Dependencies

```
Basic requirements:

- A C++17 compiler (with OpenMP version 2.0 or higher)
- A BLAS implementation (Intel MKL is strongly recommended for best performance on Intel machines)
```

### Compilation

You can quickly build DiskV with the following commands:

```
cd DiskV

cmake -DFAISS_OPT_LEVEL=generic -DBUILD_SHARED_LIBS=ON \
      -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=ON \
      -DCMAKE_BUILD_TYPE=Release -B build

make -C build install
```

### Basic Testing

```
make -C build demo_ivfpq_indexing
./build/demos/demo_ivfpq_indexing
```

------

## Experiments

### Usage

DiskV provides two demo scripts: `demo_script.cpp` and `demo_script_hybrid.cpp`.
 The hybrid version supports **searching data stored on two different disks**.

#### Case 1: Search on a single storage device

First, compile the demo script:

```
make -C build demo_script
```

Then run the build and search phases with a parameter script:

```
# Build index
./build/demos/demo_script 0 ./demos/dataset_sift100m.sh

# Search
./build/demos/demo_script 1 ./demos/dataset_sift100m.sh
```

------

#### Case 2: Search across two storage devices

First, compile the hybrid version:

```
make -C build demo_script_hybrid
```

Then run the build and search phases:

```
# Build index
./build/demos/demo_script_hybrid 0 ./demos/dataset_sift1b_hybrid.sh

# Search
./build/demos/demo_script_hybrid 1 ./demos/dataset_sift1b_hybrid.sh
```

------

### Parameters

Recommended parameters for the **SIFT1B**, **DEEP1B**, and **Text2Image1B** datasets are listed below:

| Parameter   | Meaning and Default Value                                    |
| ----------- | ------------------------------------------------------------ |
| *partition* | Number of segments **Default:** 10                           |
| *nlist*     | Number of buckets in each segment **Default:** 200,000 (for all datasets) |
| *ratio*     | Sampling ratio used during IVF_PQ clustering **Default:** 4 (i.e., sample 25%) |
| $m$         | Number of sub-vectors in IVF_PQ **Default:** 16 (SIFT1B); 24 (DEEP1B); 100 (Text2Image1B) |
| $c_{pq}$    | Number of clusters per sub-vector space **Default:** 256     |
| $M$         | Maximum neighbor count for the centroid graph **Default:** 16 |
| *efb*       | Priority queue length during centroid-graph construction **Default:** 75 |
| *efs*       | Priority queue length during centroid-graph search **Default:** 150 |
| $N$         | Number of lists to probe during search **Default:** 200      |
| $f$         | Number of lists for full posting-list search **Default:** Automatically adjusted based on disk performance (typically 3 for SSD/GP3, 30 for HDD) |
| $\epsilon$  | Estimated filtering factor during full posting-list search **Default:** 0.03 |

------

### Experimental Results

(Refer to the paper or `docs/` directory for detailed experimental results.)