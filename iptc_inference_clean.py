import pandas as pd
import pickle
import time

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TextClassificationPipeline,
)

# ------------------------------------
## build classification pipeline

tokenizer_path = "DTAI-KULeuven/robbert-2022-dutch-base"
model_path = "iptc_roberta_DTAI_lr00002_3ep_8b_256len_large"

tokenizer = RobertaTokenizer.from_pretrained(
    tokenizer_path,
    max_len=256,
    return_overflowing_tokens=True,
    stride=6,
    truncation=True,
    padding="max_length",
)

model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to_bettertransformer()
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)

# ------------------------------------
## class to convert predictions to iptc-labels & load binarizer


class IptcLabeler:
    def __init__(self, iptc_labels):
        self.iptc_labels = iptc_labels

    def pred_to_iptc(self, predictions, threshold=0.25):
        index = [
            int(d["label"].replace("LABEL_", ""))
            for d in predictions
            if d["score"] >= threshold
        ]
        return [self.iptc_labels[i] for i in index]


with open("binarizer.pkl", "rb") as handle:
    binarizer = pickle.load(handle)
# ------------------------------------
if __name__ == "__main__":

    ## load data
    test_data = pd.read_csv("iptc_test_data.csv")

    ## predict
    s = time.time()
    predictions = pipe(test_data["text"].tolist(), top_k=20)
    e = time.time()
    t = e - s
    print(f"TIME == {t}")

    ## check predictions
    labeler = IptcLabeler(binarizer.classes_)
    for i, r in test_data.iterrows():
        iptcs = labeler.pred_to_iptc(predictions[i], threshold=0.25)
        print(f'TEXT= {r["text"]}')
        print(f"PREDICTION= {iptcs}")
        print(f'TRUE LABELS= {r["iptc"]}')
        print("----")
