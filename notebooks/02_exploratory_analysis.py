
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================

train_data = np.load("../data/train_sample_data.npz")

X_train = train_data["X"]
y_train = train_data["y"]
qid_train = train_data["qid"]

val_data = np.load("../data/val_sample_data.npz")

X_val = val_data["X"]
y_val = val_data["y"]
qid_val = val_data["qid"]


# =========================
# 2. EDA
# =========================
# - Distribution of relevance labels
# - Number of items per query

print(f"Number of observations: ({X_train.shape[0]}, train_set), ({X_val.shape[0]}, val_set)")
print(f"Number of features: ({X_train.shape[1]}, train_set), ({X_val.shape[1]}, val_set)")


print(f"\nObservation {1}")
print("Y (relevance):", y_train[0])
print("qid:", qid_train[0])
print("Document id_0:", X_train[0])


# Relevance Label Distribution (TRAIN SET)
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Relevance {u}: {c} samples")


plt.bar(unique, counts)
plt.title("Relevance Label Distribution Train Set")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.savefig('../figures/rel_label_dist_train_set.png')
plt.show()

# Relevance Label Distribution (VAL SET)
unique, counts = np.unique(y_val, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Relevance {u}: {c} samples")


plt.bar(unique, counts)
plt.title("Relevance Label Distribution Validation Set")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.savefig('../figures/rel_label_dist_val_set.png')
plt.show()



# Documents per query (TRAIN SET)
unique_qid, qid_counts = np.unique(qid_train, return_counts=True)

print("Number of queries:", len(unique_qid))
print("Avg docs per query:", qid_counts.mean())
print("Min docs per query:", qid_counts.min())
print("Max docs per query:", qid_counts.max())

plt.hist(qid_counts, bins=30)
plt.title("Documents per Query Train Set")
plt.xlabel("Docs per Query")
plt.ylabel("Frequency")
plt.savefig('../figures/docs_per_query_train_set.png')
plt.show()


# Documents per query (VAL SET)
unique_qid, qid_counts = np.unique(qid_val, return_counts=True)

print("Number of queries:", len(unique_qid))
print("Avg docs per query:", qid_counts.mean())
print("Min docs per query:", qid_counts.min())
print("Max docs per query:", qid_counts.max())

plt.hist(qid_counts, bins=30)
plt.title("Documents per Query Validation Set")
plt.xlabel("Docs per Query")
plt.ylabel("Frequency")
plt.savefig('../figures/docs_per_query_val_set.png')
plt.show()