# 📌 Learning-to-Rank – Yahoo! LTR Challenge Sample

## 🎯 Overview
This project explores **learning-to-rank models** using a subset of the **Yahoo! Learning to Rank Challenge (C14, version 1.0)** dataset.  

Learning-to-rank is widely used in search engines, recommendation systems, and marketplaces to **order items so that the most relevant appear first**.  

The focus of this project is **modeling methodology and experimentation**, rather than producing actionable recommendations, due to anonymized features.

---

## Dataset

**Original dataset**: Yahoo! LTR Challenge C14 (version 1.0)  
- Training queries: 19,944  
- Validation queries: 2,994  
- Test queries: 6,983  
- URLs: 473,134 (train)  
- features: 519–596  

**My sample**:  
- Randomly selected **5,000 rows** on training set for faster experimentation and **1,000 rows** on validation set.
- Limitation: sampled by rows, not by query, so some queries only have 1–4 documents. This reduces model performance but is sufficient for experimentation.

**Important:** Feature names are anonymized. This project is therefore **focused on methodology, model training, and evaluation metrics**, rather than feature-level analysis or business recommendations.

---

## 🧠 Models

### From Scratch
- Pointwise ranking model
- Pairwise ranking model
- Listwise ranking model

### Baselines / Industry Models
- LightGBM Regressor  
- LightGBM LambdaRank  
- LightGBM XendCG  

---

## ⚙️ LTR Approaches

### Pointwise
- Predicts relevance independently per item
- Ignores relationships within query groups
- Simplest but weakest ranking formulation

### Pairwise
- Learns relative ordering between item pairs
- Better aligned with ranking objectives
- Improves over pointwise baseline

### Listwise
- Optimizes the full ranked list
- Strongest theoretical formulation among scratch models
- Best performance among custom implementations

---

## 📊 Results (NDCG)

| Model | Train | Validation |
|------|------|------------|
| Pointwise (Scratch) | 0.656 | 0.679 |
| Pairwise (Scratch) | 0.622 | 0.682 |
| Listwise (Scratch) | 0.749 | 0.722 |
| LightGBM Regressor | 0.809 | 0.762 |
| LightGBM LambdaRank | 0.809 | 0.721 |
| LightGBM XendCG | 0.809 | 0.753 |

---

## 📈 Key Insights

- Pointwise is the weakest due to ignoring item interactions
- Pairwise improves ranking by learning relative preferences
- Listwise performs best among custom implementations
- LightGBM models consistently outperform neural/scratch approaches
- Gradient boosting remains very strong for tabular ranking tasks

---

## 🧪 Takeaways

- LTR performance improves as models move from pointwise → pairwise → listwise
- Classical boosting methods remain highly competitive
- Neural/scratch models require more tuning to match performance
- Query-group structure is essential for ranking tasks


------------------------------------------------------------------

C14 Yahoo! Learning to Rank Challenge, version 1.0

Machine learning has been successfully applied to web search ranking and the goal of this dataset to benchmark such maw
There are two datasets in this distribution: a large one and a small one. Each dataset is divided in 3 sets: training, validation, and test. Statistics are as follows: Set 1 Set 2 Train Val Test Train Val Test # queries 19,944 2,994 6,983 1,266 1,266 3,798 # urls 473,134 71,083 165,660 34,815 34,881 103,174 # features 519 596

Number of features in the union of the two sets: 700; in the intersection: 415. Each feature has been normalized to be in the [0,1] range.

Each url is given a relevance judgment with respect to the query. There are 5 levels of relevance from 0 (least relevant) to 4 (most relevant). 
