import numpy as np
from itertools import combinations

def pointwise_from_scratch(X_train, y_train, lr, iter, **kwargs):
    
    # Initialize weights and bias
    weights = np.zeros(X_train.shape[1])
    bias = y_train.mean()
    n_samples = X_train.shape[0]

    # Track loss and best params
    lossi = []

    for i in range(iter):
        # 1️ Predict
        pred = np.dot(X_train, weights) + bias  # vectorized
        
        # 2️.1 Loss Function
        mse = np.mean((y_train - pred) ** 2)
        lossi.append(mse)
        
        # 2.2 Track best iteration and store values
        if (i > 0) & (mse <= lossi[-1]):
            best_w = weights
            best_b = bias
        
        # 3️ Compute gradients
        grad_w = (2 / n_samples) * np.dot(X_train.T, (pred - y_train))
        grad_b = 2 * np.mean(pred - y_train)
        
        # 4️ Update weights and bias
        weights += -lr * grad_w
        bias += -lr * grad_b
    
    return best_w, best_b


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



def logistic_regression(X, y, lr, iter, qid):
    
    X_pair, y_pair = build_pairwise(X, y, qid)
    
    # - Initialize weights and bias
    weights = np.zeros(X_pair.shape[1]) 
    bias = 0
    '''
    For logistic regression, its usually safe to initialize bias as 0 (or small value) rather than the mean of y.
    Using y_train.mean() wont break the code, but initializing at 0 is simpler and more standard.
    '''
    n_samples = X_pair.shape[0]
    lossi = []

    for i in range(0, iter):
        
        y_hat = np.dot(X_pair, weights) + bias
        sigmoid = 1/(1+np.exp(-y_hat))
        
        loss = -np.mean(y_pair * np.log(sigmoid) + (1 - y_pair) * np.log(1 - sigmoid))
        lossi.append(loss)
        
        # 2.2 Track best iteration and store values
        if (i > 0) & (loss <= lossi[-1]):
            best_w = weights
            best_b = bias
        
        grad_w = np.dot(X_pair.T, (sigmoid - y_pair)) / n_samples
        grad_b = np.mean(sigmoid - y_pair)
        
        weights += -lr * grad_w
        bias += -lr * grad_b
        
    return best_w, best_b


def softmax(scores):
    exps = np.exp(scores - np.max(scores))  
    return exps / np.sum(exps)


def listwise_scratch(X_train, y_train, lr, iter, qid):
# ---- Step 1: Initialize weights ----
    weights = np.zeros(X_train.shape[1])
    bias = 0.0  # or y_train.mean()
    unique_qids = np.unique(qid)
    n_queries = len(unique_qids)
    
    # Track loss
    lossi = []

    # ---- Step 2: Training loop ----
    for i in range(iter):
        total_loss = 0

        for q in unique_qids:
            mask = (qid == q)
            X_q = X_train[mask]
            y_q = y_train[mask]
        
            # 2.1 Predict scores for all docs in the query
            y_hat = np.dot(X_q, weights) + bias  

            # 2.2 Convert predicted scores to probabilities
            pred = softmax(y_hat)  

            # 2.3 Convert relevance labels to target distribution
            target = softmax(y_q)  
            
            # 2.4 Compute cross-entropy loss
            loss = -np.sum(target * np.log(pred + 1e-8))  
            total_loss += loss
            
            # 2.5 Compute gradients
            grad_w = np.dot(X_q.T, (pred - target))  
            grad_b = np.sum(pred - target) 

            # 2.6 Update weights
            weights += -lr * grad_w
            bias += -lr * grad_b
            
        
        norm_loss = total_loss/n_queries    
        lossi.append(norm_loss) # Normalise Loss
        
        # 2.2 Track best iteration and store values
        if (i > 0) & (norm_loss <= lossi[-1]):
            best_w = weights
            best_b = bias
    

    return best_w, best_b


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k(y_true, y_pred, qids, k=5):
    ndcgs = []

    for q in np.unique(qids):
        mask = qids == q
        true_q = y_true[mask]
        pred_q = y_pred[mask]

        # sort by predicted score
        order = np.argsort(pred_q)[::-1]
        true_sorted = true_q[order]

        dcg = dcg_at_k(true_sorted, k)
        idcg = dcg_at_k(sorted(true_q, reverse=True), k)

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)
