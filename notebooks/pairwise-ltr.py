# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from itertools import combinations
from sklearn.metrics import ndcg_score

# =========================
# 2. Load Dataset
# =========================
# X: features for (query, item)
# y: relevance label
# qid: query id (used only for evaluation)


train_data = np.load("../data/set1_train_sample_data.npz")

X_train = train_data["X"]
y_train = train_data["y"]
qid_train = train_data["qid"]


val_data = np.load("../data/set1_val_sample_data.npz")

X_val = val_data["X"]
y_val = val_data["y"]
qid_val = val_data["qid"]

# =========================
# 3. EDA
# =========================
# - Distribution of relevance labels
# - Number of items per query

print("Number of observations:", X_val.shape[0])
print("Number of features:", X_val.shape[1])

for i in range(3):
    print(f"\nObservation {i+1}")
    print("y (relevance):", y_val[i])
    print("qid:", qid_val[i])
    print("features:", X_val[i])


# Relevance Label Distribution
unique, counts = np.unique(y_val, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Relevance {u}: {c} samples")


plt.bar(unique, counts)
plt.title("Relevance Label Distribution")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.show()


# Documents per query
unique_qid, qid_counts = np.unique(qid_train, return_counts=True)
unique_val_qid, qid_val_counts = np.unique(qid_val, return_counts=True)


print("Number of queries:", len(unique_qid))
print("Avg docs per query:", qid_counts.mean())
print("Min docs per query:", qid_counts.min())
print("Max docs per query:", qid_counts.max())

plt.hist(qid_counts, bins=30)
plt.title("Documents per Query")
plt.xlabel("Docs per Query")
plt.ylabel("Frequency")
plt.show()


# =========================
# 4. Preprocessing
# =========================
# - Feature selection applied to remove near-empty or near-constant features.”
# - Each feature has been already normalized before downloading to be in the [0,1] range.

# Count non-zero entries per feature
non_zero_counts = (X_train != 0).sum(axis=0)

# Select features with at least 10 non-zero values
mask = non_zero_counts > 10
X_train_reduced = X_train[:, mask]
X_val_reduced = X_val[:, mask]

print("Original features:", X_train.shape[1])
print("Reduced features:", X_train_reduced.shape[1])


unique_qid, qid_counts = np.unique(qid_train, return_counts=True)
unique_val_qid, qid_val_counts = np.unique(qid_val, return_counts=True)


qid_counts = qid_counts.astype(int).flatten().tolist()
qid_val_counts = qid_val_counts.astype(int).flatten().tolist()
# =========================
# 5. Pairwise Model
# =========================
# - Build Pairwise dataset
# - Loss: MSE or Cross-Entropy

def build_pairwise(X, y, qid):
    X_pairs = []
    y_pairs = []

    unique_qids = np.unique(qid)

    for q in unique_qids:
        idx = np.where(qid == q)[0]
        X_q = X[idx]
        y_q = y[idx]

        # all combinations inside query
        for i, j in combinations(range(len(y_q)), 2):
            if y_q[i] == y_q[j]:
                continue

            diff = X_q[i] - X_q[j]

            if y_q[i] > y_q[j]:
                X_pairs.append(diff)
                y_pairs.append(1)

                X_pairs.append(-diff)
                y_pairs.append(0)
            else:
                X_pairs.append(-diff)
                y_pairs.append(1)

                X_pairs.append(diff)
                y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs)


X_pair, y_pair = build_pairwise(X_train_reduced, y_train, qid_train)

print(X_pair.shape, y_pair.shape)


 

ranker = lgb.LGBMRanker(
    objective="lambdarank",   # pairwise ranking loss
    metric="ndcg",
    learning_rate=0.05,
    n_estimators=100
)

ranker.fit(
    X_train,
    y_train,
    group=qid_counts,
    eval_set=[(X_val, y_val)],
    eval_group=[qid_val_counts],
    eval_at=[5]
)


# =========================
# 6. Ranking Evaluation
# =========================
# Predict scores for all original items

pred_val = ranker.predict(X_val)

def pairwise_accuracy(scores, y, qid):
    correct = 0
    total = 0
    
    for q in np.unique(qid):
        idx = np.where(qid == q)[0]
        y_q = y[idx]
        s_q = scores[idx]
        
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                if y_q[i] == y_q[j]:
                    continue
                    
                total += 1
                
                if (y_q[i] > y_q[j] and s_q[i] > s_q[j]) or \
                   (y_q[i] < y_q[j] and s_q[i] < s_q[j]):
                    correct += 1
    
    return correct / total if total > 0 else 0

acc = pairwise_accuracy(pred_val, y_val, qid_val)
print("Pairwise Accuracy:", acc)


def ndcg_per_query(scores, y, qid, k=5):
    vals = []
    for q in np.unique(qid):
        idx = np.where(qid == q)[0]
        if len(idx) < 2:
            continue
            
        vals.append(
            ndcg_score([y[idx]], [scores[idx]], k=k)
        )
    return np.mean(vals)

print("NDCG@5:", ndcg_per_query(pred_val, y_val, qid_val, k=5))



# =========================
# 7. Visualize Results
# =========================
unique_val_qid, qid_val_counts = np.unique(qid_val, return_counts=True)
valid_qids = unique_val_qid[qid_val_counts == 4]

# Pick 3 random queries
np.random.seed(302)
sample_qids = np.random.choice(np.unique(valid_qids), 3, replace=False)

plt.figure(figsize=(18,5))

for i, qid in enumerate(sample_qids):
    mask = qid_val == qid
    true_rels = y_val[mask]
    pred_scores = pred_val[mask]

    # Sort documents by predicted score descending
    order = np.argsort(-true_rels)
    true_sorted = true_rels[order]
    # predicted rank = position after sorting
    pred_ranking = np.argsort(-pred_scores)

    # Plot rankings
    plt.subplot(1, 3, i+1)
    plt.plot(range(1, len(true_sorted)+1), true_sorted, marker='o', label='True Relevance')
    plt.plot(range(1, len(pred_ranking)+1), pred_ranking, marker='x', label='Predicted Score')
    plt.xlabel('Ranked Documents')
    plt.ylabel('Relevance / Score')
    plt.title(f'Query ID {qid} - {len(true_sorted)} docs')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, max(np.max(true_sorted), np.max(pred_ranking)) + 0.5)
    plt.legend()

plt.tight_layout()
plt.show()