export NCCL_DEBUG=WARN
export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

NAME=$1

BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2
search() {
  python search_spann.py \
    --query_emb_path test_query.bin \
    --qlookup_path qlookup.ibin \
    --plookup_path $BASE_DIR/data/training_data/id2id_test.json \
    --spann_index ${NAME} \
    --depth 100 \
    --gt_path $BASE_DIR/data/qrels_test.tsv
}

search