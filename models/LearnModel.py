# -*- coding:utf8 -*-

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
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

class LearnModel():
    
    def __init__(self, X, y):
    
        self.lgbc = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
            learning_rate=0.1, max_depth=-1, min_child_samples=20,
            min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
            n_jobs=-1, num_leaves=31, objective="multiclass", random_state=None,
            reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
            subsample_for_bin=200000, subsample_freq=1)
        self.X = X
        self.y = y
        self.param_bit_length = 53
        self.chromosome = None
        with open("lgb_lm_log.txt", "w+") as f:
            f.writelines("start trainning \n")
        
        
    def getCross(self, cross_point):
        cross_point = cross_point%self.param_bit_length
        return cross_point
        if cross_point<13:
            return 6
        elif cross_point<20:
            return 13
        elif cross_point<23:
            return 20
        elif cross_point<27:
            return 23
        elif cross_point<31:
            return 27
        elif cross_point<34:
            return 31
        elif cross_point<37:
            return 34
        elif cross_point<45:
            return 37
        elif cross_point<49:
            return 45
        else:
            return 49
        
    def decodeParam(self, chromosome):
        self.chromosome = chromosome
        num_leaves = chromosome&0x3f #6位
        chromosome = chromosome>>6
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
        subsample_freq = chromosome&0x7 #3位
        chromosome = chromosome>>3
        learning_rate = chromosome&0x7 #3位
        chromosome = chromosome>>3
        min_child_samples = chromosome&0xff #8位
        chromosome = chromosome>>8
        min_child_weight = chromosome&0xf #4位
        chromosome = chromosome>>4
        min_gain_to_split = chromosome&0xf #4位

        
        self.lgbc = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=max(0.2, colsample_bytree/16),
            learning_rate=0.05*learning_rate + 0.05, max_depth=min(2 + max_depth, num_leaves + 3), min_child_samples= 1 + min_child_samples,
            min_child_weight=min_child_weight+1, min_split_gain=0.1/(min_gain_to_split+1), n_estimators=2000,
            n_jobs = -1, num_leaves = num_leaves + 3, objective="multiclass", random_state=10,
            reg_alpha=0.9**reg_alpha, reg_lambda=0.9**reg_lambda, silent=True, subsample=max(0.2, subsample/16),
            subsample_for_bin=200000, subsample_freq=subsample_freq+1)

        
    def fit(self, X, y, eval_set = None):
        self.lgbc.fit(self.X, self.y, eval_set = eval_set, eval_metric = square_f1_score, early_stopping_rounds = 100, verbose = True)

    def predict(self, X):
        ypred = self.lgbc.predict(X, num_iteration = self.lgbc.best_iteration_)
        return ypred
    
    def predict_proba(self, X):
        ypred = self.lgbc.predict_proba(X, num_iteration = self.lgbc.best_iteration_)
        return ypred
            
    def evalScore(self, train_X, train_y, eval_x, eval_y):
        train_X = coo_matrix(train_X)
        eval_x = coo_matrix(eval_x)
        edata = [(eval_x, eval_y)]
        self.lgbc.fit(train_X, train_y, eval_set = edata, eval_metric = square_f1_score, early_stopping_rounds = 100, verbose = True)
        pred = self.lgbc.predict(eval_x, num_iteration = self.lgbc.best_iteration_)
        score = (np.square(f1_score(eval_y, pred, average='macro')))
        print("gene:0x%x score:%.4f"%(self.chromosome, score))
        with open("lgb_lm_log.txt", "a+") as f:
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
        with open("lgb_lm_log.txt", "a+") as f:
            f.writelines("gene:0x%x avg_score:%.4f\n"%(self.chromosome, avg_score))
        return avg_score
    
    def crossTrainPredict(self, test_x, n_splits = 5):
        skf = StratifiedKFold(n_splits = n_splits, random_state = 2018)
        scores = []
        test_results = []
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
        print(self.lgbc.get_params())
        with open("lgb_lm_log.txt", "a+") as f:
            if score == None:
                f.writelines("best params:\n%s\n\n"%(self.lgbc.get_params()))
            else:
                f.writelines("best score :%f, best params:\n%s\n\n"%(score, self.lgbc.get_params()))


