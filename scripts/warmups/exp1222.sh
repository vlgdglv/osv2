export NCCL_DEBUG=WARN
# export HIP_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO
export ROCM_LOG_LEVEL=5   

# BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2/
BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2/

TRAIN_NAME=wu_exp1222
train(){
    # python -m torch.distributed.launch --nproc_per_node=16 \
    torchrun --nproc_per_node=16 \
        train_tasks.py \
        --in_train True \
        --output_dir runs/marcows/ \
        --overwrite_output_dir  \
        --train_name $TRAIN_NAME \
        --fp16  \
        --model_name_or_path models/init_SimANS_ckpt_36k \
        --tokenizer_name bert-base-multilingual-uncased \
        --train_example_dirs $BASE_DIR/data/training_data/training_mid \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/train_ids_lmdb \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/train_queries \
        --save_steps 5000 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --num_neg 7 \
        --per_device_train_batch_size 16 \
        --dataloader_num_workers 32 \
        --logging_steps 100 \
        --warmup_ratio 0.2 \
        --use_dense_pooler False \
        --task_list sparse \
        --reg_T 10000 \
        --q_reg_lambda 0.5 \
        --k_reg_lambda 0.4
}

splade_encode_query() {
    CUDA_VISIBLE_DEVICES=1 python eval_tasks.py \
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
        --index_dir splade_index/warmup \
        --retrieve_result_output_dir splade_results/$TRAIN_NAME \
        --save_name test.query.json \
        --q_max_len 32 \
        --qterm_num 32 \
        --sparse_pooler_type max \
        --vocab_size 105879
}

splade_search() {
    python eval_tasks.py \
        --do_retrieve_from_json True \
        --output_dir runs/encode_corpus \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/test_queries \
        --idmapping_path $BASE_DIR/data/test_qid_lookup.json \
        --model_name_or_path runs/marcows/$TRAIN_NAME  \
        --tokenizer_name bert-base-multilingual-uncased  \
        --task_list sparse \
        --fp16 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 32 \
        --index_dir splade_index/warmup \
        --index_filename splade_index.bin \
        --retrieve_result_output_dir splade_results/$TRAIN_NAME  \
        --query_json_path splade_index/warmup/test.query.json \
        --q_max_len 32 \
        --qterm_num 32 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 1 \
        --retrieve_topk 100 \
        --eval_gt_path $BASE_DIR/data/qrels_test.tsv        
}

splade_build_index() {
    python eval_tasks.py \
        --do_corpus_index True \
        --force_build_index True \
        --output_dir runs/encode_corpus \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/test_ids_lmdb \
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
        --kterm_num 150 \
        --sparse_pooler_type max \
        --vocab_size 105879 \
        --shards_num 5
}

train
splade_build_index
# test_splade
# splade_encode_query
# splade_search