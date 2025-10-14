# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
find_package(OpenMP REQUIRED)

# add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
# target_link_libraries(demo_imi_flat PRIVATE faiss)

# add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
# target_link_libraries(demo_imi_pq PRIVATE faiss)

# add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
# target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

# add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
# target_link_libraries(demo_nndescent PRIVATE faiss)

# add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
# target_link_libraries(demo_sift1M PRIVATE faiss)

# add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
# target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

# add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
# target_link_libraries(demo_residual_quantizer PRIVATE faiss)

# add_executable(demo_hnswflat EXCLUDE_FROM_ALL demo_hnswflat.cpp)
# target_link_libraries(demo_hnswflat PRIVATE faiss)


# add_executable(demo_ivf_pq EXCLUDE_FROM_ALL demo_ivf_pq.cpp)
# target_link_libraries(demo_ivf_pq PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivfflat_replicas EXCLUDE_FROM_ALL demo_ivfflat_replicas.cpp)
# target_link_libraries(demo_ivfflat_replicas PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivfpq_refine EXCLUDE_FROM_ALL demo_ivfpq_refine.cpp)
# target_link_libraries(demo_ivfpq_refine PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivfpq_common EXCLUDE_FROM_ALL demo_ivfpq_common.cpp)
# target_link_libraries(demo_ivfpq_common PRIVATE faiss_avx2)

# add_executable(demo_fastscan_refine EXCLUDE_FROM_ALL demo_fastscan_refine.cpp)
# target_link_libraries(demo_fastscan_refine PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivf_pq_disk EXCLUDE_FROM_ALL demo_ivf_pq_disk.cpp)
# target_link_libraries(demo_ivf_pq_disk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivfpqfs EXCLUDE_FROM_ALL demo_ivfpqfs.cpp)
# target_link_libraries(demo_ivfpqfs PRIVATE faiss_avx2)

# add_executable(demo_sep_ivfpqdisk EXCLUDE_FROM_ALL demo_sep_ivfpqdisk.cpp)
# target_link_libraries(demo_sep_ivfpqdisk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_ivfpqdisk2 EXCLUDE_FROM_ALL demo_sep_ivfpqdisk2.cpp)
# target_link_libraries(demo_sep_ivfpqdisk2 PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivf_pq_fs_disk EXCLUDE_FROM_ALL demo_ivf_pq_fs_disk.cpp)
# target_link_libraries(demo_ivf_pq_fs_disk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_2 EXCLUDE_FROM_ALL demo_sep_2.cpp)
# target_link_libraries(demo_sep_2 PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_refine_disk EXCLUDE_FROM_ALL demo_refine_disk.cpp)
# target_link_libraries(demo_refine_disk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_refine_disk EXCLUDE_FROM_ALL demo_sep_refine_disk.cpp)
# target_link_libraries(demo_sep_refine_disk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_ivfpqfsdisk EXCLUDE_FROM_ALL demo_sep_ivfpqfsdisk.cpp)
# target_link_libraries(demo_sep_ivfpqfsdisk PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_ivfpq_10m EXCLUDE_FROM_ALL demo_ivfpq_10m.cpp)
# target_link_libraries(demo_ivfpq_10m PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_ivfpqdisk10m EXCLUDE_FROM_ALL demo_sep_ivfpqdisk10m.cpp)
# target_link_libraries(demo_sep_ivfpqdisk10m PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_ivfpqdisk10m_beam_search EXCLUDE_FROM_ALL demo_sep_ivfpqdisk10m_beam_search.cpp)
# target_link_libraries(demo_sep_ivfpqdisk10m_beam_search PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_sep_ivfpqdisk2_beam_search EXCLUDE_FROM_ALL demo_sep_ivfpqdisk2_beam_search.cpp)
# target_link_libraries(demo_sep_ivfpqdisk2_beam_search PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_multi_ivfpqdisk2_10m EXCLUDE_FROM_ALL demo_multi_ivfpqdisk2_10m.cpp)
# target_link_libraries(demo_multi_ivfpqdisk2_10m PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_hier_train_10m EXCLUDE_FROM_ALL demo_hier_train_10m.cpp)
# target_link_libraries(demo_hier_train_10m PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_multi_global_ivfpqdisk2_10m EXCLUDE_FROM_ALL demo_multi_global_ivfpqdisk2_10m.cpp)
# target_link_libraries(demo_multi_global_ivfpqdisk2_10m PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_multi_ivfpqdisk2_1B EXCLUDE_FROM_ALL demo_multi_ivfpqdisk2_1B.cpp)
# target_link_libraries(demo_multi_ivfpqdisk2_1B PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_msturing10m_multi_ivfpqdisk2 EXCLUDE_FROM_ALL demo_msturing10m_multi_ivfpqdisk2.cpp)
# target_link_libraries(demo_msturing10m_multi_ivfpqdisk2 PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)


# add_executable(test_sift1B EXCLUDE_FROM_ALL test_sift1B.cpp)
# target_link_libraries(test_sift1B PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(demo_hybrid_deep1B EXCLUDE_FROM_ALL demo_hybrid_deep1B.cpp)
# target_link_libraries(demo_hybrid_deep1B PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)


# add_executable(test_T2I EXCLUDE_FROM_ALL test_T2I.cpp)
# target_link_libraries(test_T2I PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(test_spacev EXCLUDE_FROM_ALL test_spacev.cpp)
# target_link_libraries(test_spacev PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(create_ground_truth EXCLUDE_FROM_ALL create_ground_truth.cpp)
# target_link_libraries(create_ground_truth PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(test_sift1B4cache EXCLUDE_FROM_ALL test_sift1B4cache.cpp)
# target_link_libraries(test_sift1B4cache PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(test_sift1B_cacheupdate EXCLUDE_FROM_ALL test_sift1B_cacheupdate.cpp)
# target_link_libraries(test_sift1B_cacheupdate PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(test_sift1B_shard_cache EXCLUDE_FROM_ALL test_sift1B_shard_cache.cpp)
# target_link_libraries(test_sift1B_shard_cache PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

# add_executable(test_sift1B_shard_cacheupdate EXCLUDE_FROM_ALL test_sift1B_shard_cacheupdate.cpp)
# target_link_libraries(test_sift1B_shard_cacheupdate PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

add_executable(demo_script EXCLUDE_FROM_ALL demo_script.cpp)
target_link_libraries(demo_script PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)

add_executable(demo_script_hybrid EXCLUDE_FROM_ALL demo_script_hybrid.cpp)
target_link_libraries(demo_script_hybrid PRIVATE faiss_avx2 OpenMP::OpenMP_CXX)
