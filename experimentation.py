# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:39:31 2022

@author: georg
"""

import pandas as pd


train = pd.read_csv('train.csv', index_col='id')

train = train.clip(0, 1)
 