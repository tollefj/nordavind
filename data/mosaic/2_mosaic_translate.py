import nltk
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-gmq", device="cuda")

train_df = pd.read_csv("mosaic_train.csv")
test_df = pd.read_csv("mosaic_test.csv")

train_instr = train_df["instruction"].tolist()
train_res = train_df["response"].tolist()

test_instr = test_df["instruction"].tolist()
test_res = test_df["response"].tolist()

# 1. iterate zipped instructions and responses:
prefix = ">>nob<<"
c = 0

for _split, _df in zip("train test".split(), [train_df, test_df]):
    translated = {
        "instruction": [],
        "response": [],
    }
    _instr = _df["instruction"].tolist()
    _res = _df["response"].tolist()
    for instr, res in tqdm(zip(_instr, _res)):
        # 2. create chunks of up to 512 tokens, based on sentence segmentation from nltk
        instr_sents = nltk.sent_tokenize(instr, language="english")
        res_sents = nltk.sent_tokenize(res, language="english")
        # if there's math and latex-style formulas, ignore translation and just use the original
        if "$" in instr or "$" in res:
            instr_translated = instr
            res_translated = res
        else:
            instr_sents = [
                f"{prefix} {s}" for s in instr_sents if len(s.split()) <= 512
            ]
            res_sents = [f"{prefix} {s}" for s in res_sents if len(s.split()) <= 512]
            instr_translated = pipe(instr_sents)
            instr_translated = [t["translation_text"] for t in instr_translated]
            instr_translated = " ".join(instr_translated)
            res_translated = pipe(res_sents)
            res_translated = [t["translation_text"] for t in res_translated]
            res_translated = " ".join(res_translated)
        translated["instruction"].append(instr_translated)
        translated["response"].append(res_translated)

    translated_df = pd.DataFrame(translated)
    translated_df.to_csv(f"mosaic_translated_{_split}.csv", index=False)
