# %%
from datasets import load_dataset

dataset = load_dataset("mosaicml/instruct-v3")
# %%
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# %%
# reformat instr
# Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction How do you know if you have gout? ### Response	
# to:
# How do you know if you have gout?

def format_prompt(example, instr_format="### Instruction", resp_format="### Response"):
    text = example["prompt"]
    _, instruction = text.split(instr_format)
    instruction, response = instruction.split(resp_format)
    instruction = instruction.strip()
    return instruction

train_df["instruction"] = train_df.apply(format_prompt, axis=1)
test_df["instruction"] = test_df.apply(format_prompt, axis=1)

# %%
# use the mistral tokenizer
from transformers import AutoTokenizer

base_model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# %%
train_df.iloc[0]

# %%
def make_prompt(inst, out):
    return f"""<s>[INST] {inst} [/INST] \\n {out} </s>"""

def tknz(example, max_length=768):
    inst = example["instruction"]
    out = example["response"]
    inst = inst if inst is not None else ""
    out = out if out is not None else ""
    # return tokenizer(make_prompt(inst, inp, out), truncation=True, max_length=max_length, padding="max_length")
    return tokenizer(make_prompt(inst, out))

# demo the prompt:
print(make_prompt("Jeg vil at du skal sortere følgende: 6 3 1 2", out="1 2 3 6"))

tokenized_train = train_df.apply(tknz, axis=1)
tokenized_test = test_df.apply(tknz, axis=1)

# %%
# add the length of the input_ids from the tokenizer to the original dfs

train_df["tokenized_len"] = tokenized_train.apply(lambda x: len(x["input_ids"]))
test_df["tokenized_len"] = tokenized_test.apply(lambda x: len(x["input_ids"]))

# %%
print(train_df.shape, test_df.shape)
train_df = train_df[train_df["tokenized_len"] <= 1024]
test_df = test_df[test_df["tokenized_len"] <= 1024]
print(train_df.shape, test_df.shape)

# %%
# keep only instruction, response and source
train_df = train_df[["instruction", "response", "source"]]
test_df = test_df[["instruction", "response", "source"]]
train_df

# %%
print(train_df.shape, test_df.shape)
# remove rows with instruction or responses longer than 512 words
train_df = train_df[train_df["instruction"].apply(lambda x: len(x.split()) <= 512)]
train_df = train_df[train_df["response"].apply(lambda x: len(x.split()) <= 512)]

test_df = test_df[test_df["instruction"].apply(lambda x: len(x.split()) <= 512)]
test_df = test_df[test_df["response"].apply(lambda x: len(x.split()) <= 512)]
print(train_df.shape, test_df.shape)

# %%
from transformers import pipeline

pipe = pipeline("translation", model="facebook/nllb-200-distilled-1.3B", device="cuda:0")

# %%
train_instr = train_df["instruction"].tolist()
train_res = train_df["response"].tolist()

test_instr = test_df["instruction"].tolist()
test_res = test_df["response"].tolist()

# %%
# subset of train_df n=10
sample_df = train_df.sample(10)
sample_df

# %%
from tqdm import tqdm

LANGS = {
    "EN": "eng_Latn",
    "NOB": "nob_Latn"
}

def translate(pipe, datamapper, src_lang=LANGS["EN"], tgt_lang=LANGS["NOB"], batch_size=64):
    return pipe(datamapper, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size)

def translate_df(_df):
    def datamapper(key):
        for _, d in _df.iterrows():
            yield d[key]

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
    v.to_csv(f"{k}.csv", index=False)


