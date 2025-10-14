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

search_index_store_path=/home/he923/indices/sift1B/sift1B_r2_ivfpqdisk_list200k_c115
search_disk_store_path=/home/he923/indices/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
search_centroid_index_path=/home/he923/indices/sift1B/sift1B_r2_centroid_hnsw_list20k_c115


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

search_estimate_factor=1.1
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
search_top=2

search_nprobes=250,300,350
search_estimate_factors_partial=1.01,1.03,1.05

max_continous_pages=5
full_decode_volume=3
partial_one_decode_volume=10
partial_two_decode_volume=8
submit_per_round=10
search_threads=16
cache_vectors=0
query_for_warm_up=0

############################
# --- Key addition: split partitions across disks ---
# First 3 partitions use disk1, remaining partitions use disk2
############################
disk1_count=5

# Disk 1 base paths
search_disk1_disk_store_path=/ssd_root/he923/indices/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
search_disk1_centroid_index_path=/ssd_root/he923/indices/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

# Disk 2 base paths
search_disk2_disk_store_path=/hdd_root/he923/indices/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
search_disk2_centroid_index_path=/hdd_root/he923/indices/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

############################
# --- Key addition: per-disk load parameters ---
# (Used by load_indices; falls back to global search_* if unset)
############################
# Disk 1 specific
search1_estimate_factor=1.1
search1_estimate_factor_high_dim=1.1
search1_prune_factor=5
search1_top=2
search1_replicas=2
search1_max_continous_pages=10
search1_full_decode_volume=5
search1_partial_one_decode_volume=10
search1_partial_two_decode_volume=8
search1_submit_per_round=10

# Disk 2 specific
search2_estimate_factor=1.1
search2_estimate_factor_high_dim=1.1
search2_prune_factor=5
search2_top=25
search2_replicas=2
search2_max_continous_pages=6
search2_full_decode_volume=8
search2_partial_one_decode_volume=8
search2_partial_two_decode_volume=8
search2_submit_per_round=10

############################
# --- Optional: override nprobe / partial per disk at runtime ---
# If the four keys below are set, the loop keeps nprobe/partial fixed to each disk value;
# To keep sweeping, leave them commented and rely on search_nprobes / search_estimate_factors_partial.
############################
# search1_nprobe=80
# search1_estimate_factor_partial=0.99

# search2_nprobe=60
# search2_estimate_factor_partial=1.01


use_recommend=true
