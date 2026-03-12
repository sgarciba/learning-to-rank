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


valid_data = np.load("../data/set1_val_sample_data.npz")

X_valid = valid_data["X"]
y_valid = valid_data["y"]
qid_valid = valid_data["qid"]


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
X_val_reduced = X_valid[:, mask]

print("Original features:", X_train.shape[1])
print("Reduced features:", X_train_reduced.shape[1])


# =========================
# 5. Pointwise Model (from scratch)
# =========================
# - Linear regression
# - Loss: MSE or Cross-Entropy

# Initialize weights and bias
weights = np.zeros(X_train_reduced.shape[1])
bias = y_train.mean()
n_samples = X_train_reduced.shape[0]

# Hyperparameters
l_rate = 0.01
num_iterations = 100

# Track loss
mse_hist = []

for i in range(num_iterations):
    # 1️ Predict
    pred = np.dot(X_train_reduced, weights) + bias  # vectorized
    
    # 2️ Compute MSE
    mse = np.mean((y_train - pred) ** 2)
    mse_hist.append(mse)
    
    # 3️ Compute gradients (vectorized)
    grad_w = (2 / n_samples) * np.dot(X_train_reduced.T, (pred - y_train))
    grad_b = 2 * np.mean(pred - y_train)
    
    # 4️ Update weights
    weights -= l_rate * grad_w
    bias -= l_rate * grad_b
    
    if i % 10 == 0:
        print(f"Iteration {i}: MSE={mse:.4f}")



# =========================
# 6. Ranking Evaluation
# =========================
# - Sort predictions within each query
# - Compute ranking metric (e.g. NDCG@k)


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
np.random.seed(42)
sample_queries = np.random.choice(np.unique(qid_valid), 3, replace=False)


for q in sample_queries:
    mask = qid_valid == q
    true_rels = y_valid[mask]
    pred_scores = pred_val[mask]
    
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

# =========================
# 8. Insights
# =========================



# Parameters
k_max = 10  # max number of top documents to show per query
sample_queries = np.random.choice(np.unique(qid_valid), 50, replace=False)  # sample 50 queries

heatmap_matrix = np.zeros((len(sample_queries), k_max))

for i, q in enumerate(sample_queries):
    mask = qid_valid == q
    true_rels = y_valid[mask]
    pred_scores = pred_val[mask]
    
    # sort documents by predicted scores descending
    sorted_indices = np.argsort(-pred_scores)
    true_sorted = true_rels[sorted_indices]
    
    # take top-k
    topk = true_sorted[:k_max]
    
    # pad with -1 if fewer than k_max documents
    if len(topk) < k_max:
        topk = np.concatenate([topk, -1*np.ones(k_max - len(topk))])
        
    heatmap_matrix[i, :] = topk

# Plot heatmap
plt.figure(figsize=(12,8))
plt.imshow(heatmap_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='True Relevance')
plt.xlabel('Ranked Document Position (by predicted score)')
plt.ylabel('Queries')
plt.title('Heatmap of True Relevance for Top-K Predicted Documents')
plt.show()
