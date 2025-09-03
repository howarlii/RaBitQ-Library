

datasets=("msmarco10M")
# datasets=("laion100m")
B_values=(4 2 8 1)

data_dir=/workspace/dev/SACQ/data/

for dataset in "${datasets[@]}"; do
    # Loop through each B value
    data_path=$data_dir/$dataset/$dataset
    base_data_path=${data_path}_base.fvecs
    # base_data_path=/data/share/users/pqyin/data/laion400m/base.100M.fbin
    for B in "${B_values[@]}"; do
        # ./bin/hnsw_rabitq_indexing $base_data_path ${data_path}_centroid_16.fvecs  ${data_path}_cluster_id_16.ivecs 64 128 ${B} $data_dir/$dataset/hnsw16_b${B}_rbq.index l2
        # ./bin/symqg_indexing $base_data_path 64 128 $data_dir/$dataset/sympg_rbq.index

        numactl -N 0 -l ./bin/hnsw_rabitq_querying ../SACQ/data/${dataset}/hnsw16_b${B}_rbq.index ../SACQ/data/${dataset}/${dataset}_query.fvecs ../SACQ/data/${dataset}/${dataset}_groundtruth.ivecs l2 -num_threads=0 -output_csv_path=../SACQ/results/saq/qps_${dataset}_hnswrbq_b${B}.csv

        if [ "$DEBUG" == 1 ]; then
            echo "Debug flag is set. Breaking the loop."
            exit 0
        fi
    done
done

# ./bin/hnsw_rabitq_indexing /workspace/dev/SACQ/data/gist/gist_base.fvecs /workspace/dev/SACQ/data/gist/gist_centroid_16.fvecs  /workspace/dev/SACQ/data/gist/gist_cluster_id_16.ivecs 64 128 4 /workspace/dev/SACQ/data/gist/hnsw16_b4_rbq.index l2
# ./bin/hnsw_rabitq_indexing /workspace/dev/SACQ/data/gist/gist_base.fvecs /workspace/dev/SACQ/data/gist/gist_centroid_16.fvecs  /workspace/dev/SACQ/data/gist/gist_cluster_id_16.ivecs 64 128 8 /workspace/dev/SACQ/data/gist/hnsw16_b8_rbq.index l2

# ./bin/hnsw_rabitq_querying ../SACQ/data/gist/hnsw16_b3_rbq.index ../SACQ/data/gist/gist_query.fvecs ../SACQ/data/gist/gist_groundtruth.ivecs l2