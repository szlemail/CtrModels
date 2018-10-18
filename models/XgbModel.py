# -*- coding:utf8 -*-

import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.sparse import coo_matrix
import gc

def square_f1_score(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    samples = len(y_true)
    classes = int(len(y_pred)/samples)
    df = pd.DataFrame()
    for i in range(0, classes):
        df['c_%d'%i] =  y_pred[samples * i: samples*(i+1)]
    pred = np.argmax(np.array(df), axis = 1)
    score = np.square(f1_score(y_true, pred, average = 'macro'))
    return 'square_f1_score', score, True

class XgbModel():
    
    def __init__(self, X, y):
    
        self.xgbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
                   colsample_bytree=0.3, gamma=1e-4, learning_rate=0.1, max_delta_step=0,
                   max_depth=11, min_child_weight=2, missing=None, n_estimators=300,
                   n_jobs=2, nthread=2, objective='multi:softmax', random_state=0,
                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                   silent=True, subsample=0.6)
        self.X = X
        self.y = y
        self.param_bit_length = 43
        self.chromosome = None
        with open("xgb_lm_log.txt", "w+") as f:
            f.writelines("start trainning \n")
        
        
    def getCross(self, cross_point):
        cross_point = cross_point%self.param_bit_length
        return cross_point
    
#         if cross_point<10:
#             return 3
#         elif cross_point<17:
#             return 10
#         elif cross_point<20:
#             return 17
#         elif cross_point<24:
#             return 20
#         elif cross_point<28:
#             return 24
#         elif cross_point<32:
#             return 28
#         elif cross_point<35:
#             return 32
#         elif cross_point<35:
#             return 35
#         else:
#             return 39

        
    def decodeParam(self, chromosome):
        self.chromosome = chromosome
        num_leaves = chromosome&0x7 #3位
        chromosome = chromosome>>3
        reg_alpha = chromosome&0x7f #7位
        chromosome = chromosome>>7
        reg_lambda = chromosome&0x7f #7位
        chromosome = chromosome>>7
        max_depth = chromosome&0x7 #3位
        chromosome = chromosome>>3
        subsample = chromosome&0xf #4位
        chromosome = chromosome>>4
        colsample_bytree = chromosome&0xf #4位
        chromosome = chromosome>>4
        colsample_bylevel = chromosome&0xf #4位
        chromosome = chromosome>>4
        learning_rate = chromosome&0x7 #3位
        chromosome = chromosome>>3
        min_child_weight = chromosome&0xf #4位
        chromosome = chromosome>>4
        gamma = chromosome&0xf #4位

        self.xgbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=max(0.2, colsample_bylevel/16),
                   colsample_bytree=max(0.2, colsample_bytree/16), gamma=1e-4*(gamma*5+1), learning_rate=0.05*learning_rate + 0.05,
                   max_delta_step=0, max_depth=min(2 + max_depth, num_leaves + 3), min_child_weight=min_child_weight+1,
                   missing=None, n_estimators=2000,
                   n_jobs=-1, nthread=-1, objective='multi:softmax', random_state=10,
                   reg_alpha=0.9**reg_alpha, reg_lambda=0.9**reg_lambda, scale_pos_weight=1, seed=20,
                   silent=True, subsample=max(0.2, subsample/16),)

        
    def fit(self, X, y, eval_set = None):
        self.xgbc.fit(self.X, self.y, eval_set = eval_set, eval_metric = square_f1_score, early_stopping_rounds = 100, verbose = True)

    def predict(self, X):
        ypred = self.xgbc.predict(X)
        return ypred
    
    def predict_proba(self, X):
        ypred = self.xgbc.predict_proba(X)
        return ypred
    
    def evalScore(self, train_X, train_y, eval_x, eval_y):
        train_X = coo_matrix(train_X)
        eval_x = coo_matrix(eval_x)
        edata = [(eval_x, eval_y)]
        self.xgbc.fit(train_X, train_y, eval_set = edata, eval_metric = square_f1_score, early_stopping_rounds = 100, verbose = True)
        pred = self.xgbc.predict(eval_x)
        score = (np.square(f1_score(eval_y, pred, average='macro')))
        print("gene:0x%x score:%.4f"%(self.chromosome, score))
        with open("xgb_lm_log.txt", "a+") as f:
            f.writelines("....gene:0x%x score:%.4f\n"%(self.chromosome, score))
        return score
        
    
    def evalModel(self):
        print("evalModel")
        skf = StratifiedKFold(n_splits=3, random_state = 2018)
        scores = []
        for train_index, test_index in skf.split(self.X, self.y):
            score = self.evalScore(self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index])
            scores.append(score)
        avg_score = np.array(scores).mean()
        print("gene:0x%x avg_score:%.4f"%(self.chromosome, avg_score))
        with open("xgb_lm_log.txt", "a+") as f:
            f.writelines("gene:0x%x avg_score:%.4f\n"%(self.chromosome, avg_score))
        return avg_score
    
    
    def crossTrainPredict(self, test_x, n_splits = 5):
        skf = StratifiedKFold(n_splits = n_splits, random_state = 2018)
        scores = []
        test_results = []
        test_x = coo_matrix(test_x)
        for train_index, test_index in skf.split(self.X, self.y):
            score = self.evalScore(self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index])
            test_proba = self.predict_proba(test_x)
            scores.append(score)
            test_results = [*test_results, test_proba]
        avg_score = np.array(scores).mean()
        pred = np.mean(test_results, axis = 0)
        print("gene:0x%x avg_score:%.4f"%(self.chromosome, avg_score))
        return pred, avg_score
    
    def printParams(self, score = None):
        print(self.xgbc.get_params())
        with open("xgb_lm_log.txt", "a+") as f:
            if score == None:
                f.writelines("best params:\n%s\n\n"%(self.xgbc.get_params()))
            else:
                f.writelines("best score :%f, best params:\n%s\n\n"%(score, self.xgbc.get_params()))



