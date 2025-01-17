export NCCL_DEBUG=WARN
# export HIP_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO
export ROCM_LOG_LEVEL=5   

# BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2/
BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2
TRAIN_NAME=cotrain_exp0116
train(){
    # python -m torch.distributed.launch --nproc_per_node=16 \
    torchrun --nproc_per_node=16 \
        train_tasks.py \
        --in_train True \
        --output_dir runs/marcows/$TRAIN_NAME \
        --overwrite_output_dir  \
        --train_name $TRAIN_NAME \
        --fp16  \
        --model_name_or_path models/init_cotrain \
        --tokenizer_name bert-base-multilingual-uncased \
        --train_example_dirs $BASE_DIR/data/training_data/training_easy0108 \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/train_ids_lmdb \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/train_queries \
        --save_steps 1000 \
        --learning_rate 1e-5 \
        --num_train_epochs 1 \
        --num_neg 15 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --dataloader_num_workers 32 \
        --logging_steps 100 \
        --warmup_ratio 0.2 \
        --use_dense_pooler False \
        --task_list sent,sparse \
        --sparse_loss_weight 0.01 \
        --dense_loss_weight 1.0 \
        --reg_T 10000 \
        --q_reg_lambda 0.002 \
        --k_reg_lambda 0.001
}

splade_encode_query() {
    CUDA_VISIBLE_DEVICES=1 python eval_splade.py \
        --do_query_encode True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/$TRAIN_NAME \
        --retrieve_result_output_dir splade_results/$TRAIN_NAME \
        --save_name test.query.json \
        --q_max_len 32 \
        --sparse_pooler_type max \
        --vocab_size 105879
}

splade_search() {
    python eval_splade.py \
        --do_retrieve_from_json True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/$TRAIN_NAME \
        --retrieve_result_output_dir splade_results/$TRAIN_NAME \
        --index_filename splade_index.bin \
        --query_json_path splade_index/$TRAIN_NAME/test.query.json \
        --q_max_len 32 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 1 \
        --start_shard 3 \
        --retrieve_topk 100 \
        --eval_gt_path $BASE_DIR/data/qrels_test.tsv        
}


encode_query() {
    CUDA_VISIBLE_DEVICES=1 python eval_dense.py \
        --encode_query True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 32 \
        --q_max_len 32 \
        --embedding_output_dir $BASE_DIR/embeddings/$TRAIN_NAME
}

encode_corpus() {
    python eval_dense.py \
        --encode_corpus True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb_v2 \
        --idmapping_path $BASE_DIR/data/training_data/id2id_test.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sent \
        --fp16 \
        --per_device_eval_batch_size 2048 \
        --dataloader_num_workers 32 \
        --k_max_len 128 \
        --embedding_output_dir $BASE_DIR/embeddings/$TRAIN_NAME \
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

splade_build_index() {
    python eval_splade.py \
        --do_corpus_index True \
        --force_build_index True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb_v2 \
        --idmapping_path $BASE_DIR/data/training_data/id2id_test.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 256 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/$TRAIN_NAME \
        --index_filename splade_index.bin \
        --k_max_len 128 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 5 \
        --start_shard 0
}

train
encode_query  
encode_corpus
splade_build_index
# search