import os

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

train_json = "trizod_2024-05-09/train.json"
train_df = pd.read_json(train_json, dtype={"ID": str})

test_json = "trizod_2024-05-09/test.json"
test_df = pd.read_json(test_json, dtype={"ID": str})

train_esm_outputs_path = "esm2/train"
test_esm_outputs_path = "esm2/test"

os.makedirs(train_esm_outputs_path, exist_ok=True)
os.makedirs(test_esm_outputs_path, exist_ok=True)

print("test set")
for i, row in test_df.iterrows():
    if i % 100 == 0:
        print(i)
    inputs = tokenizer(row["sequence"], return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    torch.save(output, f"{test_esm_outputs_path}/{row['ID']}.pt")


print("train set")
for i, row in train_df.iterrows():
    if i % 100 == 0:
        print(i)
    inputs = tokenizer(row["sequence"], return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    torch.save(output, f"{train_esm_outputs_path}/{row['ID']}.pt")
