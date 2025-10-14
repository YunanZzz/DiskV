# Data files
base_filepath=/scratch_ssd/dataset/sift1b/bigann_base.bvecs
query_filepath=/scratch_ssd/dataset/sift1b/bigann_query.bvecs
ground_truth_filepath=/scratch_ssd/dataset/sift1b/gnd/idx_1000M.ivecs

# Data formats / types
data_format=bvecs        # fvecs, ivecs, bvecs, fbin, ibin
base_data_type=uint8     # float, int, uint8, int8
query_data_type=uint8

# Basic dataset properties
d=128
nb=1000000000
nq=10000
ratio=4

# Partitions and paths
partitions=1

# build_index_store_path=/scratch1/zhan4404/dataset/index_folder/DiskV/sift1B/sift1B_r2_list60k_c115
# build_disk_store_path=/scratch1/zhan4404/dataset/index_folder/DiskV/sift1B/sift1B_r2_ivfpqdisk_list60k_c115.index
# build_centroid_index_path=/scratch1/zhan4404/dataset/index_folder/DiskV/sift1B/sift1B_r2_centroid_hnsw_list60k_c115

build_index_store_path=
build_disk_store_path=
build_centroid_index_path=

search_index_store_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_list200k_c115
search_disk_store_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_ivfpqdisk_list200k_c115.index
search_centroid_index_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_centroid_hnsw_list20k_c115

# search_index_store_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115
# search_disk_store_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115.index
# search_centroid_index_path=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_c1_centroid_hnsw_list2M_c115

# search_index_store_path=/hdd_root/he923/diskv/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115
# search_disk_store_path=/hdd_root/he923/diskv/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115.index
# search_centroid_index_path=/hdd_root/he923/diskv/sift1B/sift1B_r2_c1_centroid_hnsw_list2M_c115

test_speed_file=/ssd_root/dataset/index_folder/DiskV/sift1B/sift1B_r2_c1_ivfpqdisk_list2M_c115.index_0.clustered
#test_speed_file=/scratch_ssd/dataset/sift1b/bigann_10m_base.bvecs

# Build phase parameters
nlist=200000          # IVF cluster count
m=16                  # PQ sub-quantizers
nbits=8               # PQ bit width
replicas=2
shrink_replicas=1.15
build_estimate_factor=1.2
build_prune_factor=1.9
build_metric_type=L2   # IP/L2
dataset_fmt=bvecs     # bvecs/ivecs/fbin/ibin
vector_type=uint8     # uint8/int/float
build_threads=128
ratio=10              #sample ratio = 1/ratio


# Search phase parameters
queryset_fmt=bvecs     # bvecs/ivecs/fbin/ibin
truthset_fmt=ivecs     # ivecs/ibin
k=100                  # candidates to retrieve
#k_per_partition=40      # retrieve k per partition candidates from each partition 
k_per_partition=20

search_estimate_factor=1.1
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
search_top=3

search_nprobes=200
search_estimate_factors_partial=0.98,0.99,1.00,1.01,1.02,1.04,1.06,1.08,1.1
max_continous_pages=3
full_decode_volume=6
partial_one_decode_volume=15
partial_two_decode_volume=10
submit_per_round=5
search_threads=16
cache_vectors=0
query_for_warm_up=0



use_recommend=true
