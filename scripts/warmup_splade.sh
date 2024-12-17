export NCCL_DEBUG=WARN
BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2/
TRAIN_NAME=warmup_splade
train(){
    CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 \
        train_tasks.py \
        --in_train True \
        --output_dir runs/marcows/ \
        --overwrite_output_dir  \
        --train_name $TRAIN_NAME \
        --fp16  \
        --model_name_or_path models/init_SimANS_ckpt_36k \
        --tokenizer_name bert-base-multilingual-uncased \
        --train_dir  dataset/data/marco-hn/exp1028_init_retriver \
        --train_example_dirs $BASE_DIR/data/training_data/training_mid \
        --passage_lmdb_dir $BASE_DIR/data/lmdb_data/train_ids_lmdb \
        --query_lmdb_dir $BASE_DIR/data/lmdb_data/train_queries \
        --save_steps 10000 \
        --learning_rate 1e-5 \
        --num_train_epochs 1 \
        --num_neg 7 \
        --per_device_train_batch_size 8 \
        --dataloader_num_workers 32 \
        --logging_steps 100 \
        --warmup_ratio 0.1 \
        --use_dense_pooler False \
        --task_list sparse \
        --reg_T 10000 \
        --q_reg_lambda 0.5 \
        --k_reg_lambda 0.4
}

train