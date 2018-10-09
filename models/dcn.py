######################################################################
###  Please do not delete this info 
###  Author: Leo Shen
###  Email: szlemail@tom.com
###  Create Date 2018-10-05
###  Version v1.0
###  Last Modify Date 2018-10-05
###  Last Modify Author Leo Shen

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import f1_score
import math
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
class DeepCrossNet():
    
    sess = None
    learning_rate_decay = 1
    learning_rate = None
    classed = None
    batch_size = None
    embed_dim_multiple = None
    n_cross_layers = None
    n_dnn_layers = None
    losses = []
    sparse_dim = None
    embed_dim = None
    dense_dim = None
    dnn_dim = None
    early_stop = None
    tol = None
    score = 0
    
    def __init__(self, batch_size = 64, classes = 11, learning_rate = 0.001,
                 learning_rate_decay = 0.95, embed_dim_multiple = 6, n_cross_layers = 6, n_dnn_layers = 6):
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.embed_dim_multiple = embed_dim_multiple
        self.n_cross_layers = n_cross_layers
        self.n_dnn_layers = n_dnn_layers
        
        
        # input x should be 1 dim array. with is flattened
    def cross_layer(self, xn, x0):
        #weight
        stddev = 1/np.sqrt(self.sparse_dim + self.embed_dim)
        b = tf.Variable(tf.truncated_normal([self.embed_dim + self.dense_dim, 1], stddev = 1), name = 'b')
        w = tf.Variable(tf.truncated_normal([self.embed_dim + self.dense_dim, 1], stddev = stddev), name = 'w')
        wx0 = tf.expand_dims(x0, 2)
        wx = tf.expand_dims(xn, 2)
        dot = tf.matmul(wx0, tf.transpose(wx, [0,2,1]))
        x_out = tf.tensordot(dot, w, 1) + b
        return tf.squeeze(x_out, 2)

    # input x should be 1 dim array. with is flattened
    def dnn(self, x0, nlayers = 3, outdim = 64):
        x_out = x0
        stddev1 = 1/np.sqrt(self.sparse_dim + self.embed_dim)
        stddev2 = 1/np.sqrt(outdim)
        for i in range(nlayers):
            b = tf.Variable(0.0, name = "b")
            if i == 0:
                w = tf.Variable(tf.truncated_normal([self.embed_dim + self.dense_dim, outdim], stddev = stddev1), name = 'w')
            else:
                w = tf.Variable(tf.truncated_normal([outdim, outdim], stddev = stddev2), name = 'w')
            x_out = tf.add(tf.matmul(x_out, w), b)
            x_out = tf.add(x_out, x0)
            x_out = tf.nn.relu(x_out)
        return x_out
        
    def buildGraph(self, xs, xd, y, lr):
        
        #embedding layer
        with tf.variable_scope("embed"):
            stddev = 1/np.sqrt(self.sparse_dim + self.embed_dim)
            w0 = tf.Variable(tf.truncated_normal([self.sparse_dim, self.embed_dim], stddev = stddev), name = 'w0')
            xe = tf.matmul(xs, w0)

        with tf.variable_scope("dcn"):
            x0 = tf.concat([xd, xe], axis = 1) 
            xdc = self.cross_layer(x0, x0)
            for i in range(self.n_cross_layers):
                xdc = self.cross_layer(xdc, x0)

            xdnn = self.dnn(x0, outdim = self.dnn_dim, nlayers = self.n_dnn_layers)
            x1 = tf.concat([xdc, xdnn], axis = 1) 

            stddev = 1/np.sqrt(self.embed_dim + self.dense_dim + self.dnn_dim)
            b1 = tf.Variable(tf.truncated_normal([self.classes], stddev = 1), name = 'b1')
            w1 = tf.Variable(tf.truncated_normal([self.embed_dim + self.dense_dim + self.dnn_dim, self.classes], stddev = stddev), name = 'w1')
            logits = tf.add(tf.matmul(x1, w1), b1)
            #logits = tf.squeeze(x_out, 1)
    
        y_out = tf.nn.softmax(logits)
        logloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
        optima = tf.train.AdamOptimizer(learning_rate = lr).minimize(logloss)
        return y_out, logloss, optima
        
    def earlyStop(self, eval_loss):
        self.losses.append(eval_loss)
        if len(self.losses) < self.tol + 1:
            return False
        else:
            return eval_loss > self.losses[-self.tol-1]
        
        
    def fit(self, X_sparse, X_dense, train_y, eval_set = None, early_stop = True, tolerance = 5, max_batches = 2000, eval_batches = 1):
        #params
        self.sparse_dim = X_sparse.shape[1]
        self.embed_dim = self.embed_dim_multiple * round(math.pow(self.sparse_dim, 0.25))
        self.dense_dim = X_dense.shape[1]
        self.dnn_dim = self.embed_dim + self.dense_dim
        self.early_stop = early_stop
        self.tol = tolerance
        self.losses = []
        
        #placeholder
        xs = tf.placeholder(tf.float32, shape = [None, self.sparse_dim], name = 'xs')
        xd = tf.placeholder(tf.float32, shape = [None, self.dense_dim], name = 'xd')
        y = tf.placeholder(tf.float32, shape = [None, self.classes], name = 'y')
        lr = tf.placeholder(tf.float32, name = 'lr')

        y_out, logloss, optima = self.buildGraph(xs, xd, y, lr)
        if self.sess != None:
            self.sess.close()
        self.sess = tf.Session()
        sess = self.sess
        print("start Train session dense_dim: %d sparse_dim:%d embed_dim:%d ... "%(self.dense_dim, self.sparse_dim, self.embed_dim))
        sess.run(tf.global_variables_initializer())
        samples = len(X_sparse)
        batches = samples // self.batch_size + 1
        cur_lr = self.learning_rate
        for i in range(max_batches):
            for batch in range(batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, samples)
                xs_batch = X_sparse[start:end]
                xd_batch = X_dense[start:end]
                sess.run(optima, feed_dict = {xs:xs_batch, xd:xd_batch, y:train_y[start:end], lr:cur_lr})

            cur_lr = cur_lr * self.learning_rate_decay
            train_loss = sess.run(logloss, feed_dict = {xs:X_sparse[:1024], xd:X_dense[:1024], y:train_y[:1024]})
            if eval_set != None:
                eval_loss = sess.run(logloss, feed_dict = {xs:eval_set[0], xd:eval_set[1], y:eval_set[2]})
                print("train_loss:%f eval_loss:%f"%(train_loss, eval_loss))
                if self.earlyStop(eval_loss):
                    break
            else:
                print("train_loss:%f"%(train_loss))
                
        if eval_set != None:
            pred = sess.run(y_out, feed_dict = {xs:eval_set[0], xd:eval_set[1], y:eval_set[2]})
            pred_label = np.argmax(pred, axis = 1)
            true_label = np.argmax(np.array(eval_set[2]), axis = 1)
            score = np.square(f1_score(true_label, pred_label, average = 'macro'))
            print("f1-score:",score)
            self.score = score
            
        self.sess = sess
        self.xs = xs
        self.xd = xd
        self.y_out = y_out
        
        
    def predict(self, X_sparse, X_dense):
        pred = self.sess.run(self.y_out, feed_dict = {self.xs:X_sparse, self.xd:X_dense})
        return pred

    def __del__(self):
        if self.sess != None:
            self.sess.close()



    
# example 
#import data:
def savePickle(target, filename):
    with open(filename, "wb") as f:
        pickle.dump(target, f)
        
def loadPickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
train_x_continuous = np.array(loadPickle("./data/normaldata/train_x_continuous.pkl"))
train_x_onehot = np.array(loadPickle("./data/normaldata/train_x_onehot.pkl"))
test_x_continous = np.array(loadPickle("./data/normaldata/test_x_continous.pkl"))
test_x_onehot = np.array(loadPickle("./data/normaldata/test_x_onehot.pkl"))
train_y = np.array(loadPickle("./data/normaldata/train_y.pkl"))
label_dict = loadPickle("./data/normaldata/label_dict.pkl")
labelTestResult = loadPickle("./data/normaldata/TestResult.pkl")
print("data loading finished!")
with open("dcn.log", "w+") as f:
    f.writelines("start dcn\n")
    
skf = StratifiedKFold(n_splits=3, random_state = 2018)

for lr in [0.005, 0.002, 0.001]:
    for deeplayers in [5,6,7]:
        for crosslayers in [1,2,3,4,5]:
            scores = []
            
            
            for train_index, test_index in skf.split(train_x_continuous, np.argmax(train_y, axis = 1)):
                dcn = DeepCrossNet(batch_size = 64, classes = 11, learning_rate = lr,
                             learning_rate_decay = 0.95, embed_dim_multiple = 12,
                             n_cross_layers = crosslayers, n_dnn_layers = deeplayers)
                edata = (train_x_onehot[test_index], train_x_continuous[test_index], train_y[test_index])
                dcn.fit(train_x_onehot[train_index], train_x_continuous[train_index], train_y[train_index], 
                        eval_set = edata, early_stop = True, tolerance = 5, max_batches = 1000, eval_batches = 1)
                scores.append(dcn.score)
                del dcn
            print("lr:%f, deeplayer:%d, crosslayser:%d score:%.4f\n"%(lr, deeplayers, crosslayers, np.mean(scores)))
            with open("dcn.log", "a+") as f:
                f.writelines("lr:%f, deeplayer:%d, crosslayser:%d score:%.4f\n"%(lr, deeplayers, crosslayers, np.mean(scores)))
            print(scores)
                
                
# pred = dcn.predict(test_x_onehot, test_x_continous)
# pred_label = np.argmax(pred, axis = 1)
# pred_label = [label_dict[i] for i in pred_label]
# labelTestResult['predict'] = pred_label
# labelTestResult.to_csv("./result/dcn20181006.csv", index = None)

