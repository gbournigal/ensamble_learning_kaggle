# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:47:09 2022

@author: georg
"""

import pickle
import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

train = pd.read_csv('train.csv', index_col='id')
train = train.clip(0, 1)

param_distribs = {
    "objective": ["binary"],
    "num_leaves": range(10, 120, 10),
    "n_estimators": range(200, 600, 50),
    "learning_rate": uniform(0.01, 0.05),
    "max_depth": range(3, 5),
    "feature_fraction": uniform(0.65, 0.1),
    "subsample": [0.7],
    "subsample_freq": [8],
    "n_jobs": [-2],
    "reg_alpha": [1, 2],
    'lambda': [1, 2],
    "min_child_samples": range(50, 140, 10),  
}


lgbm = LGBMClassifier()
rnd_search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_distribs,
    n_iter=100,
    cv=3,
    scoring="neg_log_loss",
    verbose=2,
    random_state=1,
    n_jobs=1,
    return_train_score=True
)
rnd_search.fit(X=train.drop(columns=['label']),
               y=train['label'])

results_rnd = pd.DataFrame(rnd_search.cv_results_)
pickle.dump(results_rnd, open(f'rnd_search_lightgbm.pickle', 'wb'))