export NCCL_DEBUG=WARN
# export HIP_VISIBLE_DEVICES=0,1  # 根据你的 GPU 设置
# export NCCL_DEBUG=INFO
export ROCM_LOG_LEVEL=5   

# BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2/
BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2/

TRAIN_NAME=warmup_splade

splade_encode_query() {
    CUDA_VISIBLE_DEVICES=1 python eval_splade.py \
        --do_query_encode True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/warmup_splade \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/warmup \
        --retrieve_result_output_dir splade_results/warmup \
        --save_name test.query.json \
        --q_max_len 32 \
        --qterm_num 32 \
        --sparse_pooler_type max \
        --vocab_size 105879
}

splade_build_index() {
    python eval_splade.py \
        --do_corpus_index True \
        --force_build_index True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb \
        --idmapping_path $BASE_DIR/data/training_data/id2id_test.json \
        --model_name_or_path runs/marcows/dep_warmup_splade/checkpoint-30000 \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 256 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/warmup \
        --index_filename splade_index.bin \
        --k_max_len 128 \
        --kterm_num 150 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 5
}


splade_search() {
    python eval_splade.py \
        --do_retrieve_from_json True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path $BASE_DIR/warehouse/warmup_splade  \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 32 \
        --index_dir $BASE_DIR/warehouse/splade_index/cotrain_exp0117 \
        --index_filename splade_index.bin \
        --retrieve_result_output_dir splade_results/warmup \
        --query_json_path $BASE_DIR/warehouse/splade_index/cotrain_exp0117/test.query.json \
        --q_max_len 32 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 5 \
        --retrieve_topk 100 \
        --eval_gt_path $BASE_DIR/data/qrels_test.tsv        
}


# splade_encode_query
# splade_build_index
splade_search
