# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations


# =========================
# 2. Load Dataset
# =========================
# X: features for (query, item)
# y: relevance label
# qid: query id (used only for evaluation)

train_data = np.load("../data/train_sample_data.npz")

X_train = train_data["X"]
y_train = train_data["y"]
qid_train = train_data["qid"]

val_data = np.load("../data/val_sample_data.npz")

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
unique_qid, qid_counts = np.unique(qid_val, return_counts=True)

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

print("Original features:", X_val.shape[1])
print("Reduced features:", X_train_reduced.shape[1])


# =========================
# 5. Pairwise Model (from scratch)
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


def logistic_regression(X_train, y_train, l_rate, iter, reg=None, lambda_=None):
    # - Initialize weights and bias
    weights = np.zeros(X_train.shape[1]) 
    bias = 0

    '''
    For logistic regression, its usually safe to initialize bias as 0 (or small value) rather than the mean of y.
    Using y_train.mean() wont break the code, but initializing at 0 is simpler and more standard.
    '''

    n_samples = X_train.shape[0]
    loss_hist = []

    for i in range(0, iter):
        
        y_hat = np.dot(X_train, weights) + bias
        sigmoid = 1/(1+np.exp(-y_hat))
        
        if reg == None:
            loss = -np.mean(y_train * np.log(sigmoid) + (1 - y_train) * np.log(1 - sigmoid))
        elif reg == 'lasso':
            loss = -np.mean(y_train * np.log(sigmoid) + (1 - y_train) * np.log(1 - sigmoid)) \
                + lambda_ * np.sum(np.abs(weights))
        elif reg == 'ridge':    
            loss = -np.mean(y_train * np.log(sigmoid) + (1 - y_train) * np.log(1 - sigmoid)) \
                + lambda_ * np.sum(weights ** 2)
        
        loss_hist.append(loss)
        
        if i > 0 and loss > loss_hist[-2]:
            print(f'Reached Optimal Cross-Entropy Loss {loss:.4f} at iteration {i}')
            break
        
        else:
            if reg == None:
                grad_w = np.dot(X_train.T, (sigmoid - y_train)) / n_samples
            elif reg == 'lasso':
                grad_w = np.dot(X_train.T, (sigmoid - y_train)) / n_samples \
                    + lambda_ * np.sign(weights)
            elif reg == 'ridge':    
                grad_w = np.dot(X_train.T, (sigmoid - y_train)) / n_samples \
                    + 2 * lambda_ * weights 
            
            grad_b = np.mean(sigmoid - y_train)
            
            weights = weights - l_rate * grad_w
            bias = bias - l_rate * grad_b
    
    return sigmoid, loss_hist


iter=100
l_rate=0.01
lambda_=0.1

sigmoid, weights, bias, loss_hist = logistic_regression(X_pair, y_pair, l_rate, iter, reg='ridge', lambda_=lambda_)

plt.plot(range(0,iter), loss_hist)
plt.title("Training Cross Entropy Loss over Iterations")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()



# =========================
# 6. Ranking Evaluation
# =========================
# Predict scores for all original items
scores = np.dot(X_train_reduced, weights) + bias

pairwise_correct = 0
total_pairs = 0

for q in np.unique(qid_train):
    idx = np.where(qid_train == q)[0]
    y_q = y_train[idx]
    s_q = scores[idx]

    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            if y_q[i] == y_q[j]:
                continue
            total_pairs += 1
            if (y_q[i] > y_q[j] and s_q[i] > s_q[j]) or (y_q[i] < y_q[j] and s_q[i] < s_q[j]):
                pairwise_correct += 1

pairwise_accuracy = pairwise_correct / total_pairs
print("Pairwise Accuracy:", pairwise_accuracy)


# Compute scores for individual documents in validation
pred_scores_val = np.dot(X_val_reduced, weights) + bias  # shape: (n_documents,)

# Find queries with more than 1 document
unique_qids, counts = np.unique(qid_val, return_counts=True)

# Pick 3 random queries
np.random.seed(302)
sample_qids = np.random.choice(np.unique(unique_qids), 3, replace=False)

plt.figure(figsize=(18,5))

for i, qid in enumerate(sample_qids):
    mask = qid_val== qid
    true_rels = y_val[mask]
    pred_scores = pred_scores_val[mask]

    # Sort documents by predicted score descending
    order = np.argsort(-true_rels)
    true_sorted = true_rels[order]
    pred_sorted = pred_scores[order]

    # Plot rankings
    plt.subplot(1, 3, i+1)
    plt.plot(range(1, len(true_sorted)+1), true_sorted, marker='o', label='True Relevance')
    plt.plot(range(1, len(pred_sorted)+1), pred_sorted, marker='x', label='Predicted Score')
    plt.xlabel('Ranked Documents')
    plt.ylabel('Relevance / Score')
    plt.title(f'Query ID {qid} - {len(true_sorted)} docs')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, max(np.max(true_sorted), np.max(pred_sorted)) + 0.5)
    plt.legend()

plt.tight_layout()
plt.show()