import os.path

import lmdb
import time
from tqdm import tqdm
import re
import pickle

import threading
from queue import Queue
from transformers import (
    BertTokenizerFast
)

def text_clean(text):
    return " ".join(
        text.replace("#n#", " ")
            .replace("#N#", " ")
            .replace("<sep>", " ")
            .replace("#tab#", " ")
            .replace("#r#", " ")
            .replace("\t", " ")
            .split()
    )

digit_pattern = re.compile(r"\d")
def url_clean(raw_url):

    if not raw_url:
        return ''
    url = raw_url.lower().replace("https", "").replace("http", "")
    url_list = url.replace("www", "").replace("com", "").replace("html", "").replace("htm", "").split()
    if url_list:
        last_item = url_list[-1]
        if len(last_item) >= 8 and digit_pattern.search(last_item):
            url_list = url_list[:-1]
    return " ".join(url_list)

def file_reader_thread(filename_list, line_queue, batch_size=100000):
    batch = []
    cnt = 0
    for filename in filename_list:
        with open(filename, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch.append(line)
                    cnt += 1
                if cnt >= batch_size:
                    line_queue.put(batch)
                    batch = []
                    cnt = 0
    if batch:
        line_queue.put(batch)
    line_queue.put(None)

def process_batch(lines, tokenizer, max_seq_length):
    doc_ids = []
    texts = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 5:
            continue
        url, language, doc_id, title, body = parts
        url = url_clean(url)
        title = text_clean(title)
        body = text_clean(' '.join(body.strip().split(' ')[:500]))
        combined_text = url + " [SEP] " + title + " [SEP] " + body
        doc_ids.append(doc_id)
        texts.append(combined_text)

    encoded = tokenizer(
        texts, 
        truncation=True,
        max_length=max_seq_length, 
        return_attention_mask=False, 
        return_token_type_ids=False
    )
    return doc_ids, encoded['input_ids']


def dump_lmdb(filename, output_dir):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
    total_docs = 101070374
    max_seq_length = 128

    env = lmdb.open(output_dir, map_size=1099511627776*2, readonly=False, meminit=False, map_async=True)
    txn = env.begin(write=True)
    write_frequency = 500000 

    filename_list = [os.path.join(filename, f'part-{i}') for i in range(0, 10)]

    line_queue = Queue(maxsize=10)
    reader_thread = threading.Thread(target=file_reader_thread, args=(filename_list, line_queue))
    reader_thread.start()
    start_time = time.time()
    write_i = 0
    while True:
        batch_lines = line_queue.get()
        if batch_lines is None:
            break
        doc_ids, input_ids_list = process_batch(batch_lines, tokenizer, max_seq_length)

        for doc_id, input_ids in zip(doc_ids, input_ids_list):
            txn.put(doc_id.encode(), pickle.dumps(input_ids))
            write_i += 1
            if write_i % write_frequency == 0:
                txn.commit()
                txn = env.begin(write=True)
        cur_time = time.time()
        avg_time = (cur_time - start_time) / write_i
        eta = (total_docs - write_i) * avg_time
        print("Batch processed in {} s, {} written, eta: {} s".format(cur_time - start_time, write_i, eta))
    txn.commit()
    txn = env.begin(write=True)
    txn.put(b'__len__', pickle.dumps(write_i))
    txn.commit()
    env.sync()
    env.close()
    reader_thread.join()

def dump_lmdb_single(filename, output_dir):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
    max_seq_length = 128
    env = lmdb.open(output_dir, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)

    txn = env.begin(write=True)
    write_frequency = 10000

    write_i = 0
    for i in range(10):
        filename_local = os.path.join(filename, 'part-'+str(i))
        f = open(filename_local, encoding='utf-8')
        for line in tqdm(f):
            lines = line.split('\t')
            url, language, doc_id, title, body = lines
            url = url_clean(url)
            title = text_clean(title)
            body = text_clean(' '.join(body.strip().split(' ')[:500]))

            prev_tokens = ['[CLS]']  + tokenizer.tokenize(url)[:42] + ['[SEP]'] + tokenizer.tokenize(title)[:41] + ['[SEP]']
            body_tokens = tokenizer.tokenize(body)[:(max_seq_length - len(prev_tokens) - 1)]
            passage = prev_tokens + body_tokens + ['[SEP]']
            passage = tokenizer.convert_tokens_to_ids(passage)[:max_seq_length]
            txn.put(doc_id.encode(), pickle.dumps(passage))
            write_i += 1
            if write_i % write_frequency == 0:
                txn.commit()
                txn = env.begin(write=True)
    

    txn.commit()
    txn = env.begin(write=True)
    # txn.put(b'q_id_to_index', pickle.dumps(qid2index))
    # txn.put(b'__key__', pickle.dumps(key_q2d))
    txn.put(b'__len__', pickle.dumps(write_i))
    txn.commit()
    env.sync()
    env.close()

if __name__=='__main__':
    dump_lmdb('data/collection_test', 'data/lmdb_data/test_ids_lmdb_new')
    #dump_lmdb('/kun_data/Jena/100m/collection_test.tsv', '/kun_data/Jena/100m/test_lmdb')