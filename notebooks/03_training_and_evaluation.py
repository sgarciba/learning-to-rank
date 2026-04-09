# =========================
# 1. Import Libraries
# =========================
import sys
import os
sys.path.append(os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.functions import *
from lightgbm import LGBMRegressor, LGBMRanker
from sklearn.svm import LinearSVC

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


valid_data = np.load("../data/val_sample_data.npz")

X_valid = valid_data["X"]
y_valid = valid_data["y"]
qid_valid = valid_data["qid"]

# =========================
# 3. Model Training
# =========================

class LtrModelSelection:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.weights = {}
        self.bias = {}
        self.predictions = {}
        self.loss = {}
 
    def fit(self, X, y, qid):
        
        for name, model_fn in self.models.items():
            print('Training:', name)
            param = self.params[name]
            self.loss[name] = {'Train': [], 'Val': []}
            _, qid_counts = np.unique(qid, return_counts=True)
            qid_counts = qid_counts.astype(int).flatten().tolist()
            
            if 'Scratch' in name:
                model_output = model_fn(
                    X,
                    y,
                    lr=param['lr'],
                    iter=param['iter'],
                    qid=qid
                )
                self.weights[name] = model_output[0]
                self.bias[name] = model_output[1]
                
                preds = X @ self.weights[name] + self.bias[name]
            
            else:
                model_fn.set_params(**param) 
                if 'Pointwise' in name:
                    model_output = model_fn.fit(X,y)
                    preds = model_output.predict(X)
                else:
                    model_output = model_fn.fit(X,y,group=qid_counts)
                    
            self.loss[name]['Train'].append(ndcg_at_k(y, preds, qid))
                    
    def predict(self, X, y, qid):
        
        for name, model_fn in self.models.items():
            print('Predicting:', name)

            if 'Scratch' in name:
                preds = X @ self.weights[name] + self.bias[name]
            else:
                preds = model_fn.predict(X)
        
            self.predictions[name] = preds
            self.loss[name]['Val'].append(ndcg_at_k(y, preds, qid))
      
            

models = {
    'PointwiseFromScratch': pointwise_from_scratch,
    'PairwiseFromScratch': logistic_regression,
    'ListwiseFromScratch': listwise_scratch,
    'PointwiseLGBMRegressor': LGBMRegressor(objective='regression'),
    'LGBM_lambdarank': LGBMRanker(objective='lambdarank'),# pairwise+listwise
    'LGBM_xendcg': LGBMRanker(objective='rank_xendcg')   # listwise
}

params = {
    'PointwiseFromScratch': {'lr': 0.01, 'iter': 100},
    'PairwiseFromScratch': {'lr': 0.01, 'iter': 100},    
    'ListwiseFromScratch': {'lr': 0.01, 'iter': 100},
    'PointwiseLGBMRegressor': {'learning_rate':0.1, 'n_estimators':100, 'num_leaves':10, 'max_depth':3},
    'LGBM_lambdarank': {'learning_rate':0.1, 'n_estimators':100, 'num_leaves':10, 'max_depth':3},
    'LGBM_xendcg': {'learning_rate':0.1, 'n_estimators':100, 'num_leaves':10, 'max_depth':3}
}


ltr = LtrModelSelection(models, params)

ltr.fit(X_train, y_train, qid_train)


# =========================
# 4. Evaluation
# =========================

ltr.predict(X_valid, y_valid, qid_valid)


models = list(ltr.loss.keys())
train_scores = [ltr.loss[m]['Train'][0] for m in models]
val_scores = [ltr.loss[m]['Val'][0] for m in models]

y = np.arange(len(models))
bar_height = 0.35
plt.figure(figsize=(11, 6))

bars_train = plt.barh(y - bar_height/2, train_scores, height=bar_height, label='Train')
bars_val = plt.barh(y + bar_height/2, val_scores, height=bar_height, label='Validation')

plt.yticks(y, models)
plt.xlabel('NDCG Score')
plt.title('Model Comparison (Train vs Validation)')
plt.legend()

# Add score labels on the right side of bars
for bar in bars_train:
    width = bar.get_width()
    plt.text(width + 0.003, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center')

for bar in bars_val:
    width = bar.get_width()
    plt.text(width + 0.003, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center')

plt.xlim(0, 1)  # optional but makes layout cleaner
plt.tight_layout()
plt.savefig('../figures/model_evaluation_ncdg.png')
plt.show()



# pick 3 random queries from validation
unique_val_qid = np.unique(qid_valid)
np.random.seed(302)
sample_qids = np.random.choice(unique_val_qid, 3, replace=False)

model_names = list(ltr.predictions.keys())
n_models = len(model_names)
n_queries = len(sample_qids)

plt.figure(figsize=(5 * n_queries, 4 * n_models))

for row, model_name in enumerate(model_names):
    preds = ltr.predictions[model_name]
    
    for col, qid in enumerate(sample_qids):
        mask = qid_valid == qid
        true_rels = y_valid[mask]
        pred_scores = preds[mask]
        
        # sort by true relevance (consistent baseline)
        sorted_indices = np.argsort(-true_rels)
        true_sorted = true_rels[sorted_indices]
        pred_sorted = pred_scores[sorted_indices]
        
        ax = plt.subplot(n_models, n_queries, row * n_queries + col + 1)
        
        ax.plot(
            range(1, len(true_sorted) + 1),
            true_sorted,
            marker='o',
            label='True Relevance'
        )
        
        ax.plot(
            range(1, len(pred_sorted) + 1),
            pred_sorted,
            marker='x',
            linestyle='--',
            label='Predicted'
        )
        
        # titles only on top row
        if row == 0:
            ax.set_title(f'Query ID {qid}')
        
        # y-label only on first column
        if col == 0:
            ax.set_ylabel(f'{model_name}\nRelevance / Score')
        
        ax.set_xlabel('Ranked Documents')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../figures/ranking_qid_valid_per_model.png')
plt.show()