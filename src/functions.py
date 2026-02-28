import numpy as np

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
