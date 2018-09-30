import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
train_x = pd.read_csv("train_x", dtype = np.float32)
print("train_x load done! shape:", train_x.shape)
eval_x = pd.read_csv("eval_x", dtype = np.float32)
print("eval_x load done! shape:", eval_x.shape)
train_y = pd.read_csv("train_y", header = None, dtype = np.float32)
print("train_y load done! shape:", train_y.shape)
eval_y = pd.read_csv("eval_y", header = None, dtype = np.float32)
print("eval_y load done! shape:", eval_y.shape)
eval_y = np.array(eval_y).reshape(-1)
train_y = np.array(train_y).reshape(-1)
import tensorflow as tf

#variable
feature_len = train_x.shape[1]
latent_depth = 32
batch_size = 256
early_stop = True
tol = 3
l2 = 0.001
x = tf.placeholder(tf.float32, shape = [None, feature_len], name = 'x')
y = tf.placeholder(tf.float32, shape = [None], name = 'y')
dropout = tf.placeholder(tf.float32, name = 'dropout')
regularizer = tf.contrib.layers.l2_regularizer(l2)
def fm(x):
    #weight
    with tf.variable_scope("fm", reuse = tf.AUTO_REUSE, regularizer = regularizer):
        b = tf.Variable(0.0, name = "b")
        w = tf.Variable(tf.truncated_normal([feature_len], stddev = 0.1), name = 'w')
        v = tf.get_variable(name = "v",
                            shape = [feature_len, latent_depth],
                            initializer=tf.truncated_normal_initializer(mean = 0,stddev = 0.01))
    #network
    linenear = tf.add(tf.reduce_sum(tf.multiply(x, w),1), b)
    inference = tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(x,v), 2), tf.matmul(tf.pow(x, 2), tf.pow(v, 2))), 1)
    logits = tf.add(linenear, tf.multiply(inference, 0.5))
    return logits

def deep(x):

    with tf.variable_scope("fm", reuse = tf.AUTO_REUSE, regularizer = regularizer):
        v = tf.get_variable(name = "v", shape = [feature_len, latent_depth], initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.01))

    x0 = tf.matmul(x, v)
    xb = x0
    for i in range(16):
        with tf.variable_scope("deep_layer%d"%i, regularizer = regularizer):
            b = tf.Variable(0.0, name = "b")
            w = tf.Variable(tf.truncated_normal([latent_depth, latent_depth], stddev = 0.1), name = 'w')
    
        x0 = tf.add(tf.matmul(x0, w), b)
        x0 = tf.add(x0, xb)
        x0 = tf.nn.relu(x0)
        x0 = tf.nn.dropout(x0, dropout)
    
    with tf.variable_scope("deep_out", regularizer = regularizer):
        b1 = tf.Variable(0.0, name = "b")
        w1 = tf.Variable(tf.truncated_normal([latent_depth, 1], stddev = 0.1), name = 'w')

    x0 = tf.add(tf.matmul(x0, w1), b1)
    x0 = tf.reshape(x0, [-1])
    return x0

alpha = tf.Variable(1.0)
bias = tf.Variable(0.0)
logits = fm(x) + tf.multiply(alpha, deep(x)) + bias
y_out = tf.sigmoid(logits)
#loss = tf.reduce_mean(tf.square(tf.subtract(y_out, y)))
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y)) + regularization_loss
optima = tf.train.AdamOptimizer(learning_rate = 0.00002).minimize(logloss)
auc_value, auc_op = tf.metrics.auc(y, y_out)
pred_eval = None
with tf.Session() as sess:
    print("start Train session ... ")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    samples = len(train_x)
    batches = samples // batch_size + 1
    cur_tol = 0
    last_auc = 0
    stop = False;
    for i in range(4):
        if stop:
            break
        for batch in range(batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, samples)
            sess.run(optima, feed_dict = {x:train_x[start:end], y:train_y[start:end], dropout:0.5})
            if  batch%1000 == 0:
                print(" logloss:", sess.run(logloss, feed_dict = {x:train_x[start:end], y:train_y[start:end], dropout:1}), end = " ")
                print(" evalloss:", sess.run(logloss, feed_dict = {x:eval_x[:10240], y:eval_y[0:10240], dropout:1}))
                #sess.run(auc_op, feed_dict = {x:eval_x[:10240], y:eval_y[:10240], dropout:1})
                #auc = sess.run(auc_value, feed_dict = {x:eval_x[:10240], y:eval_y[:10240], dropout:1})
                #print("eval_auc:", auc)
                #if early_stop:
                #    if auc > last_auc:
                #        cur_tol = 0
                #        last_auc = auc
                #    else:
                #        cur_tol = cur_tol + 1
                #        if cur_tol > tol:
                #            stop = True
                #            break
    print("logloss:", sess.run(logloss, feed_dict = {x:eval_x, y:eval_y, dropout:1}))
    sess.run(auc_op, feed_dict = {x:eval_x, y:eval_y, dropout:1})
    print("auc:", sess.run(auc_value, feed_dict = {x:eval_x, y:eval_y, dropout:1}))
    pred = sess.run(y_out, feed_dict = {x:eval_x, y:eval_y, dropout:1})
    print("roc_auc:", roc_auc_score(eval_y, pred))
