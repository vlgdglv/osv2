export NCCL_DEBUG=WARN
export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH


do_transform(){
    python transform.py \
        --corpus_path ../embeddings/bs_SimANS_36k/test_corpus.bin \
        --corpus_output_path ../embeddings/bs_SimANS_36k/test_corpus_769.bin 
}

do_build() {
    python build_spann.py \
        --name bs_SimANS \
        --corpus_path ../embeddings/bs_SimANS_36k/test_corpus.bin
}

# do_transform
do_build