# -*- coding: utf-8 -*-
"""
Created on 2020/8/6 4:07 下午

@File: patient_VTE_classification.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from pycaret.classification import setup, compare_models
import pandas as pd
import sys, os

sys.path.append('../..')

from src import proj_dir

if __name__ == '__main__':
	# ---- 载入数据 ---------------------------------------------------------------------------------
	
	data = pd.read_csv(os.path.join(proj_dir, 'data/raw/patient_data.csv'))
	data.drop('INPATIENT_ID', axis = 1, inplace = True)
	
	# ---- 模型训练和测试 ---------------------------------------------------------------------------
	
	numeric_cols = [
		'AGE', 'ISS', 'CAPRINI_SCORE', 'T', 'P', 'R', 'MBP', 'SHOCK_INDEX', 'HEIGHT', 'WEIGHT', 'BMI',
		'RBC', 'HGB', 'PLT', 'WBC', 'ALB', 'CRE', 'UA', 'AST', 'ALT', 'GLU', 'TG', 'CHO', 'CA', 'MG',
		'LDL', 'NA', 'K', 'CL', 'GFR', 'PT', 'FIB', 'DD', 'CK', 'INR'
	]
	
	clf = setup(data, target = 'VTE', numeric_features = numeric_cols)
	report = compare_models(fold = 10, sort = 'AUC', turbo = False, verbose = False)  # type: str
	



