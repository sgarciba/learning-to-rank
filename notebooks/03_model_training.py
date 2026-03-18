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
# 5. Pointwise Model (from scratch)
# =========================

class LtrModelSelection:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.predictions = {}
        self.loss = {} 
 
    def train_all(self, X_train, y_train, qid_train):
        
        for name, model_fn in self.models.items():
            
            param = self.params[name]
            
            model_output = model_fn(
                X_train,
                y_train,
                lr=param['lr'],
                iter=param['iter'],
                qid=qid_train
            )
            
            self.predictions[name] = model_output[0]
            self.loss[name] = model_output[1]

models = {
    'PointwiseFromScratch': pointwise_from_scratch,
    'PairwiseFromScratch': logistic_regression,
    'ListwiseFromScratch': listwise_scratch
}

params = {
    'PointwiseFromScratch': {'lr': 0.01, 'iter': 100},
    'PairwiseFromScratch': {'lr': 0.01, 'iter': 100},    
    'ListwiseFromScratch': {'lr': 0.01, 'iter': 100}
}


ltr = LtrModelSelection(models, params)

ltr.train_all(X_train, y_train, qid_train)

for m in ltr.models:
    print(m)
    pred = ltr.predictions[m]
    loss = ltr.loss[m]
    plt.plot(range(100), loss)
    

# -------------------------
# Using Pytorch
# -------------------------

