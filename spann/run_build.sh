export NCCL_DEBUG=WARN
export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

NAME=$1
do_transform(){
    python transform.py \
        --corpus_path ../embeddings/bs_SimANS_36k/test_corpus.bin \
        --corpus_output_path ../embeddings/bs_SimANS_36k/test_corpus_769.bin 
}

do_build() {
    python build_spann.py \
        --name bs_SimANS_36k \
        --corpus_path /datacosmos/local/User/baoht/onesparse2/marcov2/embeddings/bs_SimANS_36k/test_corpus.bin
}

extract_centroids() {
    ./ParsePostings -i ${NAME} -v float -f SPTAGFullList.bin -h SPTAGHeadVectorIDs.bin -o output.txt
}

# do_transform
# do_build
extract_centroids