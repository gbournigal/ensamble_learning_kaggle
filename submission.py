# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:39:43 2022

@author: gbournigal
"""

import pickle
import pandas as pd

submission = pd.read_csv('submission.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
model = pickle.load(open('results/final_models.pickle', 'rb'))

submission['pred'] = model.predict_proba(test)[:, 1]
submission.to_csv('submission.csv')