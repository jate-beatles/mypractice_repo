from calendar import EPOCH
import json
import math
import os
from pprint import pprint


import numpy as np

# x.assign(new_value)
# x.assign_add(value_to_be_added)
# x.assign_sub(value_to_be_subtracted)

# tf.add allows to add the components of a tensor
# tf.multiply allows us to multiply the components of a tensor
# tf.subtract allow us to substract the components of a tensor
# tf.math.* contains the usual math operations to be applied on the components of a tensor

import tensorflow as tf
print(tf.version.VERSION)

N_Point = 10
X = tf.constant(range(N_Point), dtype=tf.float32)
Y = 2 * X + 10

#using the batch to extact the data, discard teh last batch by the setting:
#  note that: THE LAST BATCH may not contain the exxact number of elemets you specified
#   dataset = dataset. batch( batch_size, drop_remainder = True)

def create_dataset (X,Y, epochs, batch_size): 
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remaider = True)
    return dataset  


def loss_mse(X,Y,w0, w1): 
        Y_hat = w0 * X + w1
        errors = (Y_hat - Y) ** 2
        return tf.reduce_mean(errors)

def compute_gradients (X,Y,w0,w1):
    with tf.GradientTape() as tape: 
        loss = loss_mse(X,Y,w0,w1)
    return tape.gradient( loss, [w0, w1])


## Hereby, to eiterate 250 tines over synthetic dataset
EPOCH = 250
BATCH_SIZE = 2 
LEARNING_RATE = 0.02

MSG = "STPE{step} - loss: {loss}, w0: {w0}, w1 : {w1}\n"

w0  = tf.Variable(0.0)
w1  = tf.Variable(0.0)
dataset = create_dataset(X,Y, epochs = EPOCH, batch_size = BATCH_SIZE)

for step, (X_batch, Y_batch) in enumerate(dataset):
    dw0, dw1 = compute_gradients(X_batch, Y_batch, w0, w1)
    print(MSG.format(step=step, loss = loss, w0 = w0 . numpy(), w1 = w1.numpy()))

assert loss < 0.0001
assert abs(w0 -2) < 0.001 
assert abs(w1 - 10) < 0.001



