# =========================
# 1. Import Libraries
# =========================
import numpy as np

# =========================
# 2. Load Dataset
# =========================
# X: features for (query, item)
# y: relevance label
# qid: query id (used only for evaluation)


def parse_line(line):
    
    parts = line.strip().split()
    
    y = int(parts[0])                 # relevance
    qid = int(parts[1].split(":")[1]) # query id
    
    features = {}
    for item in parts[2:]:
        idx, val = item.split(":")
        features[int(idx)] = float(val)
    
    return y, qid, features

def read_data(file):
    with open(file) as f:
        lines = f.readlines()
    
    ys, qids, X_dicts = [], [], []

    for line in lines:
        y, qid, feats = parse_line(line)
        ys.append(y)
        qids.append(qid)
        X_dicts.append(feats)
        
    return ys, qids, X_dicts


def sample_data(X, y, qid, n_samples, seed=203):
    np.random.seed(seed)
    
    if n_samples > len(X):
        raise ValueError("n_samples larger than dataset")

    idx = np.random.choice(len(X), size=n_samples, replace=False)
    
    return X[idx], y[idx], qid[idx]


## TRAIN DATA
file = "../data/set1.train.txt"
    
ys, qids, X_dicts = read_data(file)

n_features = max(max(d.keys()) for d in X_dicts) + 1
X = np.zeros((len(X_dicts), n_features))

for i, feats in enumerate(X_dicts):
    for j, val in feats.items():
        X[i, j] = val

y = np.array(ys)
qid = np.array(qids)

X_train_s, y_train_s, qid_train_s = sample_data(X, y, qid, n_samples=5000)


np.savez(
    "../data/set1_train_sample_data.npz",
    X=X_train_s,
    y=y_train_s,
    qid=qid_train_s
)

## VAL DATA
file = "../data/set1.valid.txt"
    
ys, qids, X_dicts = read_data(file)

n_features = max(max(d.keys()) for d in X_dicts) + 1
X = np.zeros((len(X_dicts), n_features))

for i, feats in enumerate(X_dicts):
    for j, val in feats.items():
        X[i, j] = val

y = np.array(ys)
qid = np.array(qids)

X_val_s, y_val_s, qid_val_s = sample_data(X, y, qid, n_samples=1000)


np.savez(
    "../data/set1_val_sample_data.npz",
    X=X_val_s,
    y=y_val_s,
    qid=qid_val_s
)



