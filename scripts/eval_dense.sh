export NCCL_DEBUG=WARN
# export HIP_VISIBLE_DEVICES=0,1  

# BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2/
BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2/

TRAIN_NAME=test_dense

encode_query() {
    CUDA_VISIBLE_DEVICES=1 python eval_dense.py \
        --encode_query True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/cotrain_exp1226/checkpoint-10000 \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 32 \
        --q_max_len 32 \
        --embedding_output_dir $BASE_DIR/osv2/embeddings/$TRAIN_NAME
}

encode_corpus() {
    python eval_dense.py \
        --encode_corpus True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb \
        --idmapping_path $BASE_DIR/data/training_data/id2id_test.json \
        --model_name_or_path runs/marcows/cotrain_exp1226/checkpoint-10000 \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --per_device_eval_batch_size 1024 \
        --dataloader_num_workers 32 \
        --k_max_len 128 \
        --embedding_output_dir $BASE_DIR/osv2/embeddings/$TRAIN_NAME \
        --shards_num 8
}

search() {
    python eval_dense.py \
        --search True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb \
        --idmapping_path $BASE_DIR/data/training_data/id2id_test.json \
        --model_name_or_path runs/marcows/cotrain_exp1226/checkpoint-10000 \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --embedding_dir embeddings/$TRAIN_NAME \
        --retrieve_topk 100 \
        --retrieve_result_output_dir dense_results/$TRAIN_NAME \
        --eval_gt_path $BASE_DIR/data/qrels_test.tsv \
        --shards_num 8
}

# encode_query
# encode_corpus
search
