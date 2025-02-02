export NCCL_DEBUG=WARN
export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

NAME=$1

BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2
search() {
  python search_spann.py \
    --query_emb_path $BASE_DIR/embeddings/cotrain_exp0129/query_embeddings.fbin \
    --qlookup_path $BASE_DIR/embeddings/cotrain_exp0129/query_ids.ibin \
    --plookup_path $BASE_DIR/data/training_data/id2id_test.json \
    --spann_index ${NAME} \
    --depth 100 \
    --gt_path $BASE_DIR/data/qrels_test.tsv
}

search