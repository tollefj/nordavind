from transformers import pipeline
from tqdm import tqdm
import pandas as pd

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-gmq", device="cuda:0")

train_df = pd.read_csv("mosaic_train.csv")
test_df = pd.read_csv("mosaic_test.csv")

train_instr = train_df["instruction"].tolist()
train_res = train_df["response"].tolist()

test_instr = test_df["instruction"].tolist()
test_res = test_df["response"].tolist()


def translate(pipe, datamapper, batch_size=64):
    return pipe(datamapper, batch_size=batch_size)

def translate_df(_df):
    def datamapper(key):
        for _, d in _df.iterrows():
            yield f">>nob<< {d[key]}"

    for sent in ["instruction", "response"]:
        translator = translate(pipe=pipe, datamapper=datamapper(key=sent))
        translated = []
        for x in tqdm(translator):
            translated.append(x[0]["translation_text"])
        _df[sent] = translated
    
    return _df

translated = {
    "train": translate_df(train_df),
    "test": translate_df(test_df)
}

# save the translated data
for k, v in translated.items():
    v.to_csv(f"{k}_OPUSMT.csv", index=False)


