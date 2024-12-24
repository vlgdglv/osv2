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
        --model_name_or_path models/init_SimANS_ckpt_36k \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 32 \
        --q_max_len 32 \
        --embedding_output_dir embeddings/$TRAIN_NAME
}

encode_query