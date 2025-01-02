export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

BASE_DIR=/datacosmos/local/User/baoht/onesparse2/marcov2

build_spann_ii(){
    python hybrids.py --build_spann_ii \
        --cluster_file $BASE_DIR/spann/bs_SimANS_8/output.txt \
        --spann_index_dir $BASE_DIR/spann/bs_SimANS_8_invertedIndex \
        --spann_index_name inverted_index.bin \
        --force_rebuild
}

search_bs(){
    
    python hybrids.py \
        --hybrid_search \
        --splade_index_dir $BASE_DIR/index/bm25_test \
        --splade_index_name bm25_inverted_index.bin \
        --spann_index_dir $BASE_DIR/spann/bs_SimANS_8_invertedIndex \
        --spann_index_name inverted_index.bin \
        --sptags_index_path $BASE_DIR/spann/bs_SimANS_8/HeadIndex \
        --query_text_path $BASE_DIR/index/bm25_test/query_ids.test.json \
        --query_emb_path $BASE_DIR/embeddings/bs_SimANS_36k/test_query.bin \
        --doc_emb_path  $BASE_DIR/embeddings/bs_SimANS_36k/test_corpus.bin \
        --qlookup_path $BASE_DIR/embeddings/bs_SimANS_36k/qlookup.pkl \
        --plookup_path $BASE_DIR/data/training_data/id2id_test.json \
        --gt_path data/msmarco/qrels.dev.tsv \
        --splade_weight 1 --spann_weight 1 \
        --cluster_num 32 \
        --depth 100 
}

# build_spann_ii
search_bs