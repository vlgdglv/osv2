

do_transform(){
    python transform.py \
        --corpus_path \
        --corpus_output_path \
}

do_build() {
    python build_spann.py \
        --name \
        --corpus_path
}

do_transform
do_build