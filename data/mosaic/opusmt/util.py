import pandas as pd
import nltk

def dataprep(_split, prefix=">>nob<<"):
    _df = pd.read_csv(f"mosaic_{_split}.csv")

    _instr = _df["instruction"].tolist()
    _res = _df["response"].tolist()

    chunked_instr = {}
    chunked_res = {}

    max_len = 256

    for i, (instr, res) in enumerate(zip(_instr, _res)):
        # 2. create chunks of up to 512 tokens, based on sentence segmentation from nltk
        instr_sents = nltk.sent_tokenize(instr, language="english")
        res_sents = nltk.sent_tokenize(res, language="english")
        # if there's math and latex-style formulas, ignore translation and just use the original
        # if "$" in instr:
        #     chunked_instr[i] = instr
        # elif "$" in res:
        #     chunked_res[i] = res
        if "$" in instr or "$" in res:
            continue
        prefixed_instr = [f"{prefix} {s}" for s in instr_sents]
        prefixed_res = [f"{prefix} {s}" for s in res_sents]
        # check tokenized input_ids length:
        # if too long, chunk it up
        if len(prefixed_instr) != len(instr_sents):
            # skip incompatible examples
            continue

        chunked_instr[i] = prefixed_instr
        chunked_res[i] = prefixed_res
            
    # convert the dict of index -> list of sents
    # to a 1-D mapping of index, sent
    instr_map = []
    for i, sents in chunked_instr.items():
        if type(sents) == str:
            instr_map.append((i, sents))
        else:
            for s in sents:
                instr_map.append((i, s))
    res_map = []
    for i, sents in chunked_res.items():
        if type(sents) == str:
            res_map.append((i, sents))
        else:
            for s in sents:
                res_map.append((i, s))

    return instr_map, res_map

# instr_map, res_map = dataprep("train")

# %%
def map_and_index(_map):
    _index_map = [(sent_num, idx) for sent_num, (idx, _) in enumerate(_map)]
    _sentences = [sent for _, sent in _map]
    return _index_map, _sentences