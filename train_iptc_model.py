import pandas as pd
import ast
import pickle
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizer,
)

# ------------------------------------
## load data

df = pd.read_csv("iptc_train_data_example.csv")[:40]

# ------------------------------------
## prep data

binarizer = MultiLabelBinarizer()
onehot = binarizer.fit_transform(df["iptc"].tolist())
classes = binarizer.classes_
onehot = [[int(i) for i in ar] for ar in onehot]
df["onehot"] = onehot

with open("iptc_binarizer.pkl", "wb") as f:
    pickle.dump(binarizer, f)


# ------------------------------------
## split data


def split(df, p=0.95):
    """split data in train, eval and test parts by given train percentage"""
    split = int(p * len(df))
    test_split = int(split + (len(df) - split) / 2)

    return {"train": df[:split], "eval": df[split:test_split], "test": df[test_split:]}


data = split(df)
train_data = data["train"]
eval_data = data["eval"]
test_data = data["test"]

print(len(train_data))
print(len(test_data))
print(len(eval_data))
# # ------------------------------------
## tokenize

model_name = "DTAI-KULeuven/robbert-2022-dutch-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name, max_len=256)

train_encodings = tokenizer(
    train_data["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=256,
    return_overflowing_tokens=True,
    stride=6,
)
val_encodings = tokenizer(
    eval_data["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=256,
    return_overflowing_tokens=True,
    stride=6,
)
test_encodings = tokenizer(
    test_data["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=256,
    return_overflowing_tokens=True,
    stride=6,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_data = Dataset(train_encodings, torch.FloatTensor(train_data["onehot"].tolist()))
val_data = Dataset(val_encodings, torch.FloatTensor(eval_data["onehot"].tolist()))
test_data = Dataset(test_encodings, torch.FloatTensor(test_data["onehot"].tolist()))

# ------------------------------------
## load model

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
    num_labels=len(classes),
)

# # ------------------------------------
## train

lr = 2e-5
ep = 3
bs = 8

training_args = TrainingArguments(
    output_dir="./results_transformers",  # output directory
    evaluation_strategy="epoch",  # evaluation is done at the end of each epoch
    num_train_epochs=ep,  # total number of training epochs (recommended between 1&5)
    per_device_train_batch_size=bs,  # batch size per device during training
    per_device_eval_batch_size=bs,  # batch size for evaluation
    warmup_steps=0,  # number of warmup steps for learning rate scheduler
    weight_decay=0,  # strength of weight decay
    save_total_limit=1,  # limit the total amount of checkpoints (saves), deletes the older checkpoints
    learning_rate=lr,  # value of learning rate (between 1e-1 and 1e-5 recommended)
    fp16=True,  # whether to use fp16 16-bit (mixed) precision training (faster)
    seed=42,  # random seed that will be set at the beginning of training
    #     no_cuda=True,
)


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_data,  # training dataset
    eval_dataset=val_data,  # evaluation dataset
)

trainer.train()

# ------------------------------------
## save model

trainer.model.save_pretrained(
    "iptc_model_" + model_name + "_" + str(lr) + "_" + str(ep) + "_" + str(bs)
)
