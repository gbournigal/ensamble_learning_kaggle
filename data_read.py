# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:37:56 2022

@author: gbournigal
"""

import pickle
import os
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


submission = pd.read_csv('sample_submission.csv', index_col='id')

train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')

path = Path('submission_files/')

subs = sorted(os.listdir(path))


s0 = pd.read_csv(path / subs[0], index_col='id')


def get_train_full_data():
    train = pd.read_csv('train_labels.csv', index_col='id')
    test = pd.read_csv('sample_submission.csv', index_col='id').drop(columns=['pred'])
    path = Path('submission_files/')
    subs = sorted(os.listdir(path))
    
    for file in subs:
        sub = pd.read_csv(path / file, index_col='id').rename(columns={
            'pred': file})[[file]]
        if len(sub.columns) != 1:
            break
        train = train.merge(sub,
                            how='left',
                            left_index=True,
                            right_index=True)
        test = test.merge(sub,
                          how='left',
                          left_index=True,
                          right_index=True)
        print(file)
        
    train.to_csv('train.csv')
    test.to_csv('test.csv')
        

train = train.clip(0, 1)
test = test.clip(0, 1)

model = LGBMClassifier(n_jobs=-1,
                       max_depth=3)
model.fit(X=train.drop(columns=['label']),
          y=train['label'])
submission['pred'] = model.predict_proba(test)[:, 1]
submission.to_csv('submission.csv')

submission.drop(columns=['pred'], inplace=True)
submission['pred'] = test.mean(axis=1)
submission.to_csv('submission.csv')




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


params_opt = {'feature_fraction': 0.7403293155638789, 'lambda': 1, 'learning_rate': 0.012102657303897132, 'max_depth': 3, 'min_child_samples': 60, 'n_estimators': 550, 'n_jobs': -2, 'num_leaves': 50, 'objective': 'binary', 'reg_alpha': 1, 'subsample': 0.7, 'subsample_freq': 8}


model = LGBMClassifier(**params_opt)
model.fit(X=train.drop(columns=['label']),
          y=train['label'])
submission['pred'] = model.predict_proba(test)[:, 1]
submission.to_csv('submission.csv')

