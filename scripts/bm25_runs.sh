

BASE_DIR=/datacosmos/User/baoht/onesparse2/marcov2

build(){
   # Build for train corpus
   python bm25_tests.py --build_index --passage_path $BASE_DIR/data/lmdb_data/train_ids_lmdb \
      --index_path index/bm25_train --index_name bm25_inverted_index.bin --force_rebuild

   # python bm25_tests.py --build_index --passage_path $BASE_DIR/data/lmdb_data/test_ids_lmdb  \
   #    --idmap_path $BASE_DIR/data/training_data/id2id_test.json \
   #    --index_path index/bm25_test --index_name bm25_inverted_index.bin --force_rebuild
}

search() {
   python bm25_tests.py --do_retrieve  --query_path $BASE_DIR/index/bm25_test/query_ids.test.json   \
      --gt_path $BASE_DIR/data/qrels_test.tsv \
      --idmap_path $BASE_DIR/data/training_data/id2id_test.json \
      --index_path $BASE_DIR/index/bm25_test_cut0.1 --index_name bm25_inverted_index.bin
}

search