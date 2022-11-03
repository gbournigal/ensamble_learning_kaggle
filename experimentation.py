# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:39:31 2022

@author: georg
"""


import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression


def evaluation_model(X,
                     y,
                     model, 
                     model_name, 
                     params, 
                     comment=""):
    
    cross_result = cross_validate(model, 
                                  X, 
                                  y,
                                  cv=5,
                                  return_train_score=True,
                                  scoring='neg_log_loss')

    
    model_results = {
        'model_name': model_name,
        'params': params,
        'comment': comment,
        'cross_result': cross_result,
        }
    
    pickle.dump(model_results, open(f'results/{model_name}.pickle', 'wb'))
    

def selection_by_lightgbm(X, y, model):
    
    loss = cross_validate(model,
                          X, 
                          y,
                          cv=5,
                          scoring='neg_log_loss',
                          verbose=2)['test_score'].mean()
    print(f'Initial loss of: {loss}')
    model.fit(
        X=X,
        y=y
        )
    
    flag = 0
    iteration = 0
    while flag == 0:
        iteration += 1
        print(f'Iteration {iteration}')
        best_features = pd.DataFrame({'importance':model.booster_.feature_importance(importance_type='gain'), 'features':model.feature_name_})
        columns = list(best_features[best_features['importance'] > np.percentile(best_features['importance'], 10)]['features'])
        X_new = X[columns]
        
        new_loss = cross_validate(model,
                                  X_new, 
                                  y,
                                  cv=5,
                                  scoring='neg_log_loss')['test_score'].mean()
        
        if new_loss > loss:
            print('El modelo mejoró')
            print(f'Log_loss: {new_loss}')
            print(f'Predicciones utilizadas: {len(X_new.columns)}')
            loss = new_loss
            model.fit(
                X=X_new,
                y=y
                )
            
        else:
            print('El modelo no mejoró')
            print(f'Log_loss: {new_loss}')
            print(f'Log_loss mejor modelo: {loss}')
            X_new = X[best_features['features']]
            columns = best_features['features']
            flag = 1
            
    return X_new, loss, columns
    
    
    
    
    
    
if __name__ == '__main__':
    train = pd.read_csv('train.csv', index_col='id')
    params = {'colsample_bytree': 0.7403293155638789, 
              'reg_lambda': 1,
              'learning_rate': 0.012102657303897132,
              'max_depth': 3, 
              'min_child_samples': 60, 
              'n_estimators': 550, 
              'n_jobs': -2,
              'num_leaves': 50, 
              'objective': 'binary', 
              'reg_alpha': 1, 
              'subsample': 0.7, 
              'subsample_freq': 8,
              'verbose':-1}
    
    
    model = LGBMClassifier(**params)
    X = train.drop(columns=['label'])
    # X = X.clip(0.001, 0.999)
    y = train['label']
    evaluation_model(X,
                     y,
                     model,
                     'baseline_hyper',
                     params)
    
    X_new, loss, columns = selection_by_lightgbm(X, y, model)
    
    evaluation_model(X_new,
                     y,
                     model,
                     'selection_lightgbm_gain',
                     params)
    
    baseline_result = pickle.load(open('results/baseline_hyper.pickle', 'rb'))
    light_sel_result_gain = pickle.load(open('results/selection_lightgbm_gain.pickle', 'rb'))
    rnd_search = pickle.load(open('rnd_search_lightgbm.pickle', 'rb'))

    final_model = LGBMClassifier(**params)
    final_model.fit(X_new,
                    y)
    
    
    submission = pd.read_csv('submission.csv', index_col='id')
    test = pd.read_csv('test.csv', index_col='id')
    test = test[columns]
    
    submission['pred'] = model.predict_proba(test)[:, 1]
    submission.to_csv('submission.csv')
    
    
    valid_columns = []
    highest_logloss = 0.693
    for i in X.columns:
        if log_loss(y, X[i]) < highest_logloss:
            valid_columns.append(i)
            print(f'{i} added')
            
            
    X_low = X[valid_columns]

    X_new, loss, columns = selection_by_lightgbm(X_low, y, model)        
        
    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    