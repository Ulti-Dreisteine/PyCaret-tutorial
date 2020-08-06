# -*- coding: utf-8 -*-
"""
Created on 2020/8/6 5:20 下午

@File: tune_model_params.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 模型调参
"""

import logging

logging.basicConfig(level = logging.INFO)

from pycaret.classification import setup, create_model, tune_model, plot_model, evaluate_model, \
	interpret_model
import pandas as pd
import sys, os

sys.path.append('../..')

from src import proj_dir

if __name__ == '__main__':
	# ---- 载入数据 ---------------------------------------------------------------------------------
	numeric_cols = [
		'AGE', 'ISS', 'CAPRINI_SCORE', 'T', 'P', 'R', 'MBP', 'SHOCK_INDEX', 'HEIGHT', 'WEIGHT',
		'BMI', 'RBC', 'HGB', 'PLT', 'WBC', 'ALB', 'CRE', 'UA', 'AST', 'ALT', 'GLU', 'TG', 'CHO',
		'CA', 'MG', 'LDL', 'NA', 'K', 'CL', 'GFR', 'PT', 'FIB', 'DD', 'CK', 'INR'
	]
	
	data = pd.read_csv(os.path.join(proj_dir, 'data/raw/patient_data.csv'))
	data.drop('INPATIENT_ID', axis = 1, inplace = True)
	
	for col in data.columns:
		if col not in numeric_cols:
			data[col] = data[col].astype(int)

	# ---- 模型训练和测试 ---------------------------------------------------------------------------

	setup(data, target = 'VTE', numeric_features = numeric_cols, verbose = False)

	# 创建模型和训练调参.
	# ID          Name
	# --------    ----------
	# 'lr'        Logistic Regression
	# 'knn'       K Nearest Neighbour
	# 'nb'        Naive Bayes
	# 'dt'        Decision Tree Classifier
	# 'svm'       SVM - Linear Kernel
	# 'rbfsvm'    SVM - Radial Kernel
	# 'gpc'       Gaussian Process Classifier
	# 'mlp'       Multi Level Perceptron
	# 'ridge'     Ridge Classifier
	# 'rf'        Random Forest Classifier
	# 'qda'       Quadratic Discriminant Analysis
	# 'ada'       Ada Boost Classifier
	# 'gbc'       Gradient Boosting Classifier
	# 'lda'       Linear Discriminant Analysis
	# 'et'        Extra Trees Classifier
	# 'xgboost'   Extreme Gradient Boosting
	# 'lightgbm'  Light Gradient Boosting
	# 'catboost'  CatBoost Classifier
	
	clf = create_model('rf', verbose = False)
	# clf_tuned = tune_model(clf)

	evaluate_model(clf)
	# interpret_model(clf)
	# interpret_model(clf, plot = 'correlation')
	


