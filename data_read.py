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
        







