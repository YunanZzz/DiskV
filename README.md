# DiskV
Vector databases have recently attracted significant attention largely due to the rise of large language models (LLMs). To improve the cost-efficiency of vector search, several disk-based vector indexes, such as SPANN and DiskANN, have been developed. However, these indexes are primarily optimized for fast SSDs and perform poorly on low-cost storage devices such as HDDs and cloud storage, which are widely used in practice to reduce costs.

In this work, we present DiskV, the first disk-based vector index designed to achieve high search performance across different storage devices, including both low-cost storage (e.g., HDDs and cloud storage) and high-speed storage (e.g., local SSDs). The core idea of DiskV builds upon a quantization-based indexing structure (similar to SPANN) that partitions the dataset into large buckets, leveraging sequential access patterns to favor low-cost storage. To achieve high performance on fast storage, DiskV introduces a suite of novel optimizations, including query-aware adaptive search, segment-based pruning, and optimized asynchronous I/O. Extensive experiments on three billion-scale datasets show that DiskV substantially outperforms existing disk-based indexes on low-cost storage (by up to **17X**), while achieving comparable or superior performance on high-speed storage devices (by up to **2.5X**). We believe DiskV provides a more practical and cost-effective foundation for supporting large-scale AI applications.


## Installing:

### Dependency
```
xxxxx
```
### Compilation
```
xxxxx
```

## Experiment:

### Usage

### Parameter


### Experiment Results



## Acknowledgments
DiskV is implemented on Faiss, we really appreciate your work.

## Acknowledgments
DiskV is implemented on [Faiss](https://github.com/facebookresearch/faiss), and we sincerely appreciate their excellent work.