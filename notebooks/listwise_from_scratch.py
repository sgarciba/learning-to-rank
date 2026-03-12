# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

print("Number of observations:", X_train.shape[0])
print("Number of features:", X_train.shape[1])

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
plt.title("Relevance Label Distribution")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.show()


# Documents per query
unique_qid, qid_counts = np.unique(qid_train, return_counts=True)

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

# =========================
# 5. Listwise Model (from scratch)
# =========================
# - Linear regression

# ---- Step 0: Softmax function ----
def softmax(scores):
    exps = np.exp(scores - np.max(scores))  
    return exps / np.sum(exps)


def listwise_scratch(X_train, y_train, l_rate, iter):
# ---- Step 1: Initialize weights ----
    weights = np.zeros(X_train_reduced.shape[1])
    bias = 0.0  # or y_train.mean()

    # Track loss
    loss_hist = []

    # ---- Step 2: Training loop ----
    for i in range(iter):
        # 2.1 Predict scores for all docs in the query
        scores = np.dot(X_train_reduced, weights) + bias  

        # 2.2 Convert predicted scores to probabilities
        pred = softmax(scores)  

        # 2.3 Convert relevance labels to target distribution
        target = softmax(y_train)  

        # 2.4 Compute cross-entropy loss
        loss = -np.sum(target * np.log(pred + 1e-8))  
        loss_hist.append(loss)

        # 2.5 Compute gradients
        grad_w = np.dot(X_train_reduced.T, (pred - target))  
        grad_b = np.sum(pred - target) 

        # 2.6 Update weights
        weights -= l_rate * grad_w
        bias -= l_rate * grad_b


    return weights, bias, loss_hist



iter=100
l_rate=0.01
lambda_=0.1

weights, bias, loss_hist = listwise_scratch(X_train_reduced, y_train, l_rate, iter)

plt.plot(range(0,iter), loss_hist)
plt.title("Training Cross Entropy Loss over Iterations")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


# =========================
# 6. Ranking Evaluation
# =========================


def dcg(rels, k):
    rels = np.array(rels)[:k]
    return np.sum((2**rels - 1) / np.log2(np.arange(2, len(rels)+2)))

def ndcg_at_k(y_true, y_pred, qid, k=5):
    ndcgs = []
    for q in np.unique(qid):
        mask = qid == q
        true_rels = y_true[mask]
        pred_scores = y_pred[mask]
        
        # Sort by predicted score
        sorted_indices = np.argsort(-pred_scores)
        true_sorted = true_rels[sorted_indices]
        
        # Ideal DCG
        idcg = dcg(np.sort(true_rels)[::-1], k)
        if idcg == 0:
            continue  # skip queries with all zero relevance
        ndcgs.append(dcg(true_sorted, k)/idcg)
    return np.mean(ndcgs)



pred_val = np.dot(X_val_reduced, weights) + bias
ndcg_score = ndcg_at_k(y_valid, pred_val, qid_valid, k=5)
print(f"NDCG@5 on validation: {ndcg_score:.4f}")


# pick 3 random queries from validation
unique_val_qid, qid_val_counts = np.unique(qid_val, return_counts=True)
valid_qids = unique_val_qid[qid_val_counts == 4]

for q in valid_qids:
    mask = qid_val == q
    true_rels = y_val[mask]
    pred_scores = pred_val[mask]
    
    # sort by predicted scores
    sorted_indices = np.argsort(-true_rels)
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

