#ARIMA auto regression integreted mean avreage--y based on the previous y 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 

#X = tf.constat(range(10), dtype =tf.float32)
#Y = 2 * X + 10 
X = range(10)
Y = 2 * X + 10 

X_test  = range(10,20)
Y_test = 2 * X_test + 10

y_mean = Y.numpy().mean()
def predict_mean(X):
    y_hat = [y_mean] * len(X)
    return y_hat

Y_hat = predict_mean(X_test)

errors = (Y_hat - Y) **2 
loss = tf.reduce_mean(errors) f ### MSE in the tensorflow
loss.numpy()

def loss_se(X,Y,wo,w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y) ** 2 
    return tf.reduce_mean(errors)


## Tensorflow has the automatic differentiation capabilities we don't have to do the partial derivatives 





