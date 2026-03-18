import numpy as np
from itertools import combinations

def pointwise_from_scratch(X_train, y_train, lr, iter, **kwargs):

    # Initialize weights and bias
    weights = np.zeros(X_train.shape[1])
    bias = y_train.mean()
    n_samples = X_train.shape[0]

    # Track loss
    lossi = []

    for i in range(iter):
        # 1️ Predict
        pred = np.dot(X_train, weights) + bias  # vectorized
        
        # 2️ Loss Function
        mse = np.mean((y_train - pred) ** 2)
        lossi.append(mse)
        
        # 3️ Compute gradients
        grad_w = (2 / n_samples) * np.dot(X_train.T, (pred - y_train))
        grad_b = 2 * np.mean(pred - y_train)
        
        # 4️ Update weights and bias
        weights += -lr * grad_w
        bias += -lr * grad_b
    
    return pred, lossi


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



def logistic_regression(X_train, y_train, lr, iter, qid, reg=None, lambda_=None):
    
    X_train, y_train = build_pairwise(X_train, y_train, qid)
    
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
            
            weights += -lr * grad_w
            bias += -lr * grad_b
    
    return sigmoid, loss_hist


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
    loss_hist = []

    # ---- Step 2: Training loop ----
    for i in range(iter):
        total_loss = 0

        for q in unique_qids:
            mask = (qid == q)
            X_q = X_train[mask]
            y_q = y_train[mask]
        
            # 2.1 Predict scores for all docs in the query
            scores = np.dot(X_q, weights) + bias  

            # 2.2 Convert predicted scores to probabilities
            pred = softmax(scores)  

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
            
        loss_hist.append(total_loss/n_queries) # Normalise Loss

    return pred, loss_hist


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
