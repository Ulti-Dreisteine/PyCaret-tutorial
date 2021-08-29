# -*- coding: utf-8 -*-
"""
Created on 2021/08/29 21:45:00

@File -> FCC_regressive_modeling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: FCC过程回归建模
"""

from pycaret.regression import setup, compare_models, create_model, tune_model, evaluate_model, interpret_model
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 2))
sys.path.append(BASE_DIR)

from src.settings import PROJ_DIR

if __name__ == '__main__':
    
    # ---- 载入数据 ---------------------------------------------------------------------------------
    
    X_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/FCC/X.csv'))
    Y_df = pd.read_csv(os.path.join(PROJ_DIR, 'data/FCC/Y.csv'))
    
    print('Y columns: {}'.format(Y_df.columns))
    
    # ---- 建立任务 ---------------------------------------------------------------------------------
    
    x_cols = X_df.columns.tolist()
    y_col = 'gasoline'
    data = pd.concat([X_df, Y_df[y_col]], axis = 1).astype(np.float)
    
    task = setup(
        data,
        target = y_col,
        numeric_features = x_cols,
        verbose = False,
        remove_multicollinearity = False,
        # multicollinearity_threshold = 0.6,
        ignore_low_variance = False,
        silent = True,
        n_jobs=2
    )
    
    # ---- 模型选择 ---------------------------------------------------------------------------------
    
    best_model =  compare_models(
        include = ['rf', 'lightgbm', 'lasso', 'ridge', 'xgboost', 'en', 'knn', 'mlp', 'lr', 'dt'],
        sort = 'R2',
        verbose = True,
        fold=3,
        round=5,
    )
    
    # ---- 模型调参 ---------------------------------------------------------------------------------
    
    # 初始化模型, 固定参数.
    params = {'max_features': 'auto'}
    rgsr = create_model('rf', verbose = False, **params)
    
    # 模型调参.
    params4tuning = {
        "n_estimators": np.arange(30, 250, 30),
        "min_samples_leaf": [10, 15, 20, 30, 40 , 50],
        "min_samples_split": [20, 30, 40],
        }
    rgsr_tuned = tune_model(
        rgsr, optimize = 'R2', n_iter = 2, fold = 5, round = 2, custom_grid = params4tuning
        )
    
    
    # ---- 模型训练和预测 ---------------------------------------------------------------------------
    
    # evaluate_model(rgsr_tuned)
    
    # ---- 模型可解释性 -----------------------------------------------------------------------------
    
    interpret_model(rgsr_tuned, plot = 'summary')
    
    
    