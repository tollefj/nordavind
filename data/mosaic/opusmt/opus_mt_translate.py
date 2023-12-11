# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from collections import defaultdict
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-gmq")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-gmq").to(2)

# %%
dfs = {}
# instr_index_map, instr_sentences = map_and_index(instr_map)
# res_index_map, res_sentences = map_and_index(res_map)
# print(instr_index_map[:10])
# print(instr_sentences[:10])
from util import dataprep, map_and_index

# %%
from tqdm import tqdm
import jsonlines

DEVICE = 2
batch_size = 64

for _split in ["train", "test"]:
    print(_split)
    print("prepping data")
    instr_map, res_map = dataprep(_split, prefix=">>nob<<")
    print("mapping and indexing")
    instr_index_map, instr_sentences = map_and_index(instr_map)
    res_index_map, res_sentences = map_and_index(res_map)
    print("batch translating")

    for sent_type, sentences in zip(["instruction", "response"], [instr_sentences, res_sentences]):
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        filename = f"mosaic_{_split}_{sent_type}_translated.jsonl"

        with open(filename, "w") as f:
            writer = jsonlines.Writer(f)
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sentences))
                batch_sentences = sentences[start_idx:end_idx]

                inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                
                translated = model.generate(**inputs)
                for orig, t in zip(batch_sentences, translated):
                    trans = tokenizer.decode(t, skip_special_tokens=True)
                    # write the translated sentence to the jsonlines file
                    writer.write({"original": orig, "translation": trans})

