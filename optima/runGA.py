
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import pickle
from scipy.sparse import hstack
import lightgbm as lgb
import argparse
import GA
from sklearn.model_selection import StratifiedKFold
import LearnModel as lm
import XgbModel as xm

def loadPickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def savePickle(target, filename):
    with open(filename, "wb") as f:
        pickle.dump(target, f)

def loadTestData(model = 'tree'):
    test_x = loadPickle("./data/originaldata/test_x.pkl")
    TestResult = loadPickle("./data/originaldata/TestResult.pkl")
    return np.array(test_x), TestResult

def loadData():
    train_x = loadPickle("./data/originaldata/train_x.pkl")
    train_y = loadPickle("./data/originaldata/train_y.pkl")
    label_dict = loadPickle("./data/originaldata/label_dict.pkl")
    train_y = np.argmax(np.array(train_y), axis = 1)
    return np.array(train_x), np.array(train_y), label_dict
    
def loadNormalizedTestData(model = 'tree'):
    test_x = loadPickle("./data/normaldata/test_x.pkl")
    TestResult = loadPickle("./data/normaldata/TestResult.pkl")
    return np.array(test_x), TestResult
    
def loadNormalizedData(model = 'tree'):
    train_x = loadPickle("./data/normaldata/train_x.pkl")
    train_y = loadPickle("./data/normaldata/train_y.pkl")
    label_dict = loadPickle("./data/normaldata/label_dict.pkl")
    return np.array(train_x), np.argmax(np.array(train_y), axis = 1), label_dict

    
def predictAll():
    #lgb 0x309057f3b237 825, 0x18309057ffe1e5 823
    #xgb nor 0x1abfdbe67de 8225  ori 0x28bddbee47e 8221
    gene_dict = {"original":{"lgb":0x18309057ffe1e5, "xgb":0x28bddbee47e}, "normal":{"lgb":0x18309057ffe1e5, "xgb":0x1abfdbe67de}}
    ntrain_x, ntrain_y, nlabel_dict = loadNormalizedData(model = 'tree')
    ntest_x, nTestResult = loadNormalizedTestData(model = 'tree')
    otrain_x, otrain_y, olabel_dict = loadData()
    otest_x, oTestResult = loadTestData(model = 'tree')
    co_pred =[]
#     otrain_x = otrain_x[:3000]
#     ntrain_x = ntrain_x[:3000]
#     ntrain_y = ntrain_y[:3000]
#     otrain_y = otrain_y[:3000]
    
    if gene_dict['original']['lgb']:
        model = lm.LearnModel(otrain_x, otrain_y)
        model.decodeParam(gene_dict['original']['lgb'])
        pred_ol, avg_score = model.crossTrainPredict(otest_x)
        oTestResult['predict'] = np.argmax(pred_ol, axis = 1)
        oTestResult['predict'] = oTestResult['predict'].apply(lambda x:olabel_dict[x])
        oTestResult.to_csv("./result/lgb_ol_%s_%.4f.csv"%(gene_dict['original']['lgb'], avg_score), index = None)
        co_pred = [*co_pred, pred_ol]
        
    if gene_dict['original']['xgb']:
        model = xm.XgbModel(otrain_x, otrain_y)
        model.decodeParam(gene_dict['original']['xgb'])
        pred_ox, avg_score = model.crossTrainPredict(otest_x)
        oTestResult['predict'] = np.argmax(pred_ox, axis = 1)
        oTestResult['predict'] = oTestResult['predict'].apply(lambda x:olabel_dict[x])
        oTestResult.to_csv("./result/xgb_ox_%s_%.4f.csv"%(gene_dict['original']['xgb'], avg_score), index = None)
        co_pred = [*co_pred, pred_ox]
        
    if gene_dict['normal']['lgb']:
        model = lm.LearnModel(ntrain_x, ntrain_y)
        model.decodeParam(gene_dict['original']['lgb'])
        pred_nl, avg_score = model.crossTrainPredict(ntest_x)
        nTestResult['predict'] = np.argmax(pred_nl, axis = 1)
        nTestResult['predict'] = nTestResult['predict'].apply(lambda x:nlabel_dict[x])
        nTestResult.to_csv("./result/lgb_nl_%s_%.4f.csv"%(gene_dict['normal']['lgb'], avg_score), index = None)
        co_pred = [*co_pred, pred_nl]
        
    if gene_dict['normal']['xgb']:
        model = xm.XgbModel(ntrain_x, ntrain_y)
        model.decodeParam(gene_dict['original']['xgb'])
        pred_nx, avg_score = model.crossTrainPredict(ntest_x)
        nTestResult['predict'] = np.argmax(pred_nx, axis = 1)
        nTestResult['predict'] = nTestResult['predict'].apply(lambda x:nlabel_dict[x])
        nTestResult.to_csv("./result/xgb_nx_%s_%.4f.csv"%(gene_dict['normal']['xgb'], avg_score), index = None)
        co_pred = [*co_pred, pred_nx]
    
    co_pred_enable = True
    for key in olabel_dict.keys():
        if olabel_dict.get(key) != nlabel_dict.get(key):
            print("label_dict not equal! no co_pred!")
            co_pred_enable = False
        
    if len(co_pred) > 1 and co_pred_enable:
        pred = np.mean(co_pred, axis = 0)
        oTestResult['predict'] = np.argmax(pred, axis = 1)
        oTestResult['predict'] = oTestResult['predict'].apply(lambda x:olabel_dict[x])
        oTestResult.to_csv("./result/co_pred_20181007.csv", index = None)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data", "-d", type = str, help = "'original' or 'normalized', select the data source")
    parser.add_argument("--model","-m", type = str, help = "'lgb','xgb' select the ml model" )
    parser.add_argument("--predict","-p", type = str, help = "if predict is set, predict all model and save it to result dir. ignor all ther args" )
    return parser.parse_args()

if __name__ == '__main__':
    args = getArgs()
    if args.predict == "all":
        predictAll()
    else:
        ga = None
        print(args)
        if args.data[:3] == "nor":
            if args.model == "lgb" or args.model == "xgb":
                train_x, train_y, label_dict = loadNormalizedData(model = 'tree')
                print("GA on normalized data")
                ga = GA.GA(30, np.array(train_x)[:], train_y[:], big_is_better = True, model = args.model)
            else:
                print("model %s not ready now!"%args.model)
        else:
            if args.model == "lgb" or args.model == "xgb":
                train_x, train_y, label_dict = loadData()
                print("GA on original data")
                ga = GA.GA(30, np.array(train_x)[:], np.array(train_y).astype(int)[:], big_is_better = True, model = args.model)
            else:
                print("model %s not ready now!"%args.model)

        if ga != None:
            last_score = 0 
            tol = 3
            for i in range(100):
                ga.evolve()
                score = ga.printParam()
                if score > last_score:
                    last_score = score
                    tol = 3
                else:
                    tol = tol -1
                    if tol <= 0:
                        print("early stop!")
                        break

        