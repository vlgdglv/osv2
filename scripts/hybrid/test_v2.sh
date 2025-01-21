export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2

convert(){
    python auxiliary.py --cvt_query \
        --query_path $BASE_DIR/warehouse/splade_index/cotrain_exp0117/test.query.json \
        --output_path $BASE_DIR/warehouse/splade_index/cotrain_exp0117/splade_query.bin
}


resort_query(){
    python auxiliary.py --resort_query \
        --query_path $BASE_DIR/embeddings/cotrain_exp0114/query_embeddings.fbin \
        --lookup_path $BASE_DIR/embeddings/cotrain_exp0114/query_ids.ibin \
        --output_dir $BASE_DIR/embeddings/cotrain_exp0114
}

build_spann_ii(){
    python hybrids.py --build_spann_ii \
        --cluster_file $BASE_DIR/spann/cotrain_exp0114_8/output.txt.part1 \
        --spann_index_dir $BASE_DIR/spann/cotrain_exp0114_8 \
        --spann_index_name inverted_index.bin \
        --force_rebuild
}

search_bs(){
    python hybrids.py \
        --hybrid_search \
        --splade_index_dir $BASE_DIR/index/bm25_test_cut0.1 \
        --splade_index_name bm25_inverted_index.bin \
        --spann_index_dir $BASE_DIR/spann/bs_SimANS_8_invertedIndex \
        --spann_index_name inverted_index.bin \
        --sptags_index_path $BASE_DIR/spann/bs_SimANS_8/HeadIndex \
        --query_text_path $BASE_DIR/index/bm25_test/query_ids.test.json \
        --query_emb_path $BASE_DIR/embeddings/bs_SimANS_36k/test_query.bin \
        --doc_emb_path  $BASE_DIR/embeddings/bs_SimANS_36k/test_corpus.bin \
        --qlookup_path $BASE_DIR/embeddings/bs_SimANS_36k/qlookup.pkl \
        --plookup_path $BASE_DIR/data/training_data/id2id_test.json \
        --gt_path $BASE_DIR/data/qrels_test.tsv \
        --splade_weight 1 --spann_weight 1 \
        --cluster_num 256 \
        --depth 100 
}

save_posting(){
    python hybrids.py \
        --splade_index_dir $BASE_DIR/spann/cotrain_exp0114_8/ \
        --splade_index_name inverted_index.bin \
        --spann_index_dir $BASE_DIR/spann/cotrain_exp0114_8 \
        --spann_index_name inverted_index.bin \
        --sptags_index_path $BASE_DIR/spann/cotrain_exp0114_8/HeadIndex \
        --query_text_path $BASE_DIR/warehouse/splade_index/cotrain_exp0117/test.query.json \
        --query_emb_path $BASE_DIR/embeddings/cotrain_exp0114/query_embeddings.fbin \
        --doc_emb_path  $BASE_DIR/embeddings/cotrain_exp0114/test_corpus.fbin \
        --qlookup_path $BASE_DIR/embeddings/cotrain_exp0114/query_ids.ibin \
        --plookup_path $BASE_DIR/data/training_data/id2id_test.json \
        --gt_path $BASE_DIR/data/qrels_test.tsv \
        --output_path $BASE_DIR/spann/cotrain_exp0114_8 \
        --splade_weight 1 --spann_weight 10000 \
        --cluster_num 256 \
        --depth 100 \
        --only_spann_list
}

# build_spann_ii
# search_bs
# save_posting
convert
# resort_query