
import numpy as np
from collections import defaultdict

# =========================
# FUNCTIONS
# =========================

def parse_line(line):
    '''
    For each line, extract the label (relevance), qid and features
    '''
    parts = line.strip().split()
    
    y = int(parts[0])                 # relevance
    qid = int(parts[1].split(":")[1]) # query id
    
    features = {}
    for item in parts[2:]:
        idx, val = item.split(":")
        features[int(idx)] = float(val)
    
    return y, qid, features

def read_data(file):
    '''
    Read the file, line by line, to get the values for each variable and append them to a list.
    '''

    with open(file) as f:
        lines = f.readlines()
    
    ys, qids, X_dicts = [], [], []

    for line in lines:
        y, qid, feats = parse_line(line)
        ys.append(y)
        qids.append(qid)
        X_dicts.append(feats)
        
    return ys, qids, X_dicts


def sample_data_by_query(X, y, qid, target_n_rows=5000, seed=203):
    """
    Sample data by query id, including all documents for selected queries,
    until approximately target_n_rows is reached.
    """
    np.random.seed(seed)
    
    # Map from qid -> indices of rows
    qid_to_indices = defaultdict(list)
    for i, q in enumerate(qid):
        qid_to_indices[q].append(i)
    
    # Shuffle query IDs
    all_qids = np.array(list(qid_to_indices.keys()))
    np.random.shuffle(all_qids)
    
    selected_indices = []
    total_rows = 0
    
    for q in all_qids:
        idxs = qid_to_indices[q]
        if total_rows + len(idxs) <= target_n_rows:
            selected_indices.extend(idxs)
            total_rows += len(idxs)
        else:
            break  # stop when reaching approx target rows
    
    # Subset arrays
    selected_indices = np.array(selected_indices)
    
    return X[selected_indices], y[selected_indices], qid[selected_indices]

# =========================
## TRAINING DATA
# =========================
# Select max. 5000 rows from the training set

file = "../data/set1.train.txt"
    
ys, qids, X_dicts = read_data(file)

n_features = max(max(d.keys()) for d in X_dicts) + 1
X = np.zeros((len(X_dicts), n_features))

for i, feats in enumerate(X_dicts):
    for j, val in feats.items():
        X[i, j] = val

y = np.array(ys)
qid = np.array(qids)

X_train_s, y_train_s, qid_train_s = sample_data_by_query(X, y, qid, target_n_rows=5000)


# Count the amount of zero values for each feature
non_zero_counts = (X_train_s != 0).sum(axis=0)

# Select features with at least 10 non-zero values
mask = non_zero_counts > 10
X_train_reduced = X_train_s[:, mask]

print("Original features:", X_train_s.shape[1])
print("Reduced features:", X_train_reduced.shape[1])


np.savez(
    "../data/train_sample_data.npz",
    X=X_train_reduced,
    y=y_train_s,
    qid=qid_train_s
)

# =========================
## VALIDATION DATA
# =========================
# Select max. 1000 rows from the training set

file = "../data/set1.valid.txt"
    
ys, qids, X_dicts = read_data(file)

n_features = max(max(d.keys()) for d in X_dicts) + 1
X = np.zeros((len(X_dicts), n_features))

for i, feats in enumerate(X_dicts):
    for j, val in feats.items():
        X[i, j] = val

y = np.array(ys)
qid = np.array(qids)

X_val_s, y_val_s, qid_val_s = sample_data_by_query(X, y, qid, target_n_rows=1000)

X_val_reduced = X_val_s[:, mask]

np.savez(
    "../data/val_sample_data.npz",
    X=X_val_reduced,
    y=y_val_s,
    qid=qid_val_s
)


