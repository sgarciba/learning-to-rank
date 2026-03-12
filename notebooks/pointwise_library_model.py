# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRanker
from itertools import product

import sys
import os

sys.path.append(os.path.abspath(".."))
from src.functions import dcg_at_k, ndcg_at_k



# =========================
# 2. Load Dataset
# =========================
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

print(f"Number of observations: (train_set, {X_train.shape[0]}), (val_set, {X_val.shape[0]})")
print(f"Number of features: (train_set, {X_train.shape[1]}), (val_set, {X_val.shape[1]})")

for i in range(3):
    print(f"\nObservation {i+1}")
    print("y (relevance):", y_train[i])
    print("qid:", qid_train[i])
    print("features:", X_train[i])


# Relevance Label Distribution
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Relevance {u}: {c} samples")


plt.bar(unique, counts)
plt.title("Relevance Label Distribution (Train Data)")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.savefig("../figures/relevance-label-dist-train-set.png")
plt.show()


# Documents per query
unique_qid, qid_counts = np.unique(qid_train, return_counts=True)

print("Number of queries:", len(unique_qid))
print("Avg docs per query:", qid_counts.mean())
print("Min docs per query:", qid_counts.min())
print("Max docs per query:", qid_counts.max())

plt.hist(qid_counts, bins=30)
plt.title("Documents per Query (Train Data)")
plt.xlabel("Docs per Query")
plt.ylabel("Frequency")
plt.savefig("../figures/documents-per-query-train-set.png")
plt.show()


# =========================
# 4. Preprocessing
# =========================

# Count non-zero entries per feature
non_zero_counts = (X_train != 0).sum(axis=0)

# Select features with at least 10 non-zero values
mask = non_zero_counts > 10
X_train_reduced = X_train[:, mask]
X_val_reduced = X_val[:, mask]

print("Original features:", X_train.shape[1])
print("Reduced features:", X_train_reduced.shape[1])


# =========================
# 5. Data Modelling (Pointwise apporach)
# =========================

_, train_group = np.unique(qid_train, return_counts=True)
_, val_group = np.unique(qid_val, return_counts=True)

best_score = -np.inf
best_params = None

param_grid = {
    "num_leaves": [31, 63],
    "max_depth": [-1, 10],
    "learning_rate": [0.05, 0.1],
    "num_iterations": [100, 300],
    "min_data_in_leaf": [20, 50]
}


for values in product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))

    model = LGBMRanker(
        objective="lambdarank"
        , **params
    )

    model.fit(
        X_train,
        y_train,
        group=train_group
    )

    preds = model.predict(X_val)

    ndcg = ndcg_at_k(y_val, preds, qid_val, k=5)

    if ndcg > best_score:
        best_score = ndcg
        best_params = params

print("Best NDCG@5:", best_score)
print("Best hyperparameters:", best_params)



np.random.seed(42)
sample_queries = np.random.choice(np.unique(qid_val), 3, replace=False)


for q in sample_queries:
    mask = qid_val == q
    true_rels = y_val[mask]
    pred_scores = preds[mask]
    
    # sort by predicted scores
    sorted_indices = np.argsort(-pred_scores)
    true_sorted = true_rels[sorted_indices]
    pred_sorted = pred_scores[sorted_indices]
    
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(true_sorted)+1), true_sorted, marker='o', label='True Relevance')
    plt.plot(range(1, len(pred_sorted)+1), pred_sorted, marker='x', label='Predicted Score')
    plt.title(f'Query ID {q} - True vs Predicted Relevance')
    plt.xlabel('Ranked Documents')
    plt.ylabel('Relevance / Score')
    plt.legend()
    plt.show()