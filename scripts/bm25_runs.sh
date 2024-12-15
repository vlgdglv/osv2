

build(){
   # Build for train corpus
   python bm25_tests.py --build_index --passage_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/lmdb_data/train_ids_lmdb \
      --index_path index/bm25_train --index_name bm25_inverted_index.bin --force_rebuild

   # python bm25_tests.py --build_index --passage_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/lmdb_data/test_ids_lmdb  \
   #    --idmap_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/training_data/id2id_test.json \
   #    --index_path index/bm25_test --index_name bm25_inverted_index.bin --force_rebuild
}

search() {
   python bm25_tests.py --do_retrieve --do_tokenize --query_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/queries_test.tsv   \
      --gt_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/qrels_test.tsv \
      --idmap_path /datacosmos/local/User/baoht/onesparse2/marcov2/data/training_data/id2id_test.json \
      --index_path index/bm25_test --index_name bm25_inverted_index.bin
}

search