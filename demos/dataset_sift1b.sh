# Data files
base_filepath=/scratch_ssd/dataset/sift1b/bigann_100m_base.ubin
query_filepath=/scratch_ssd/dataset/sift1b/bigann_query.ubin
ground_truth_filepath=/scratch_ssd/dataset/sift1b/gnd/idx_1000M.ibin

# Data formats / types
dataset_fmt=u8bin     # bvecs/ivecs/fbin/ibin
vector_type=uint8     # uint8/int/float
queryset_fmt=u8bin     # bvecs/ivecs/fbin/ibin
truthset_fmt=ibin     # ivecs/ibin

# Basic dataset properties
d=128
nb=100000000
nq=1000
ratio=4

# Partitions and paths
partitions=10

build_index_store_path=
build_disk_store_path=
build_centroid_index_path=


# search_disk_store_path=/hdd_root/he923/indices/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
# search_centroid_index_path=/hdd_root/he923/indices/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

search_disk_store_path=/ssd_root/he923/indices/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
search_centroid_index_path=/ssd_root/he923/indices/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

# search_disk_store_path=/home/he923/indices/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
# search_centroid_index_path=/home/he923/indices/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

# search_index_store_path=/hdd_root/he923/diskv/sift100M/sift100M_pq248_list200k_c115
# search_disk_store_path=/hdd_root/he923/diskv/sift100M/sift100M_pq248_ivfpqdisk_list200k_c115.index
# search_centroid_index_path=/hdd_root/he923/diskv/sift100M/sift100M_pq248_centroid_hnsw_list200k_c115


test_speed_file=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115.index_0.clustered
#test_speed_file=/scratch_ssd/dataset/sift1b/bigann_10m_base.bvecs

# Build phase parameters
nlist=200000         # IVF cluster count
m=16                 # PQ sub-quantizers
nbits=8               # PQ bit width
replicas=2
shrink_replicas=1.15
build_estimate_factor=1.2
build_prune_factor=1.9
build_metric_type=L2   # IP/L2

build_threads=128
#ratio=10              #sample ratio = 1/ratio


# Search phase parameters

k=100                 # candidates to retrieve
#k_per_partition=40      # retrieve k per partition candidates from each partition
k_per_partition=40

search_estimate_factor=1.05
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
search_top=2

search_nprobes=200
search_estimate_factors_partial=1.04

max_continous_pages=5
full_decode_volume=5
partial_one_decode_volume=5
partial_two_decode_volume=5
submit_per_round=5
search_threads=16
cache_vectors=0
query_for_warm_up=0


use_recommend=true
