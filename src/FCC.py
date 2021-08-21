# -*- coding: utf-8 -*-
"""
Created on 2021/08/21 16:05:01

@File -> FCC.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import pandas as pd
import numpy as np
import sys
import os 

sys.path.append('../')

from src.settings import proj_dir
X_df = pd.read_csv(os.path.join(proj_dir, 'data/FCC/X.csv'))
Y_df = pd.read_csv(os.path.join(proj_dir, 'data/FCC/Y.csv'))
print(Y_df.columns)

from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, interpret_model

y_col = 'dry_gas'
data = pd.concat([X_df, Y_df[y_col]], axis = 1)

task = setup(
    data,
    target = y_col,
    numeric_features = X_df.columns,
    verbose = False,
    remove_multicollinearity = False,
    # multicollinearity_threshold = 0.6,
    ignore_low_variance = False,
    silent = True,
    n_jobs=2
)

best_model =  compare_models(
    whitelist = ['rf', 'lightgbm', 'lasso', 'ridge', 'en', 'xgboost', 'knn', 'mlp', 'lr', 'dt'],
    sort = 'R2',
    verbose = True,
    fold=5,
    round=4,
)

# params = {'max_features': 'auto'}
# clf = create_model('rf', verbose = True, **params)

# params = {
#     "n_estimators": np.random.randint(10, 150, 20),
#     "min_samples_leaf": [10, 15, 20, 30, 40 , 50],
#     "min_samples_split": [20, 30, 40],
#     }
# clf_tuned = tune_model(
#     clf, optimize = 'R2', n_iter = 2, fold = 5, round = 2, custom_grid = params
#     )

# print(clf_tuned.get_params)