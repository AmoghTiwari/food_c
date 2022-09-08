import json

LABELS_FILE_PATH = "../data/labels.txt"
LABELS2IDX_FILE = "../data/labels2idx.json"
IDX2LABELS_FILE = "../data/idx2labels.json"

label2idx = {}
idx2label = {}

with open (LABELS_FILE_PATH, "r") as f:
    labels = f.read().splitlines()

for idx, label in enumerate(labels):
    idx2label[idx] = label
    label2idx[label]=idx

with open(LABELS2IDX_FILE, "w") as f:
    json.dump(label2idx, f, indent="")

with open(IDX2LABELS_FILE, "w") as f:
    json.dump(idx2label, f, indent="")