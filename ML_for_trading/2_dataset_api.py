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


CSV_COLUMNS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], ['na'], [0.0], [0.0],[0.0],[0.0],[0.0],['na']]
UNWATED_COLS = ['pickup_datetime', 'key']

def create_dateset(pattern):
    return tf.data.experimental.make_csv_dataset(
        pattern, 1, CSV_COLUMNS, DEFAULTS) 

#https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset

###tf.data.experimental.make_csv_dataset(
#     file_pattern,--path   
#     batch_size, 
#     column_names=None,
#     column_defaults=None, Default
#     label_name=None,
#     select_columns=None,
#     field_delim=',',
#     use_quote_delim=True,
#     na_value='',
#     header=True,
#     num_epochs=None,
#     shuffle=True,
#     shuffle_buffer_size=10000,
#     shuffle_seed=None,
#     prefetch_buffer_size=None,
#     num_parallel_reads=None,
#     sloppy=False,
#     num_rows_for_inference=100,
#     compression_type=None,
#     ignore_errors=False

tempds = create_dataset("../data/taxi-train")
pprint(tempds)

##iterate over first two element of this dataset using dataset.take(2)
##Then convert python dictionary with numpy array as values for more readablitly 
for data in tempds.take(2):
    pprint({k: v.numpy() for k,v in data.items()}) ##pprint is pretty print for JSON 
    print("\n") 

### Transforming the features 
UNWANTED_COLS = ['pickup_datetime', 'key']

def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data

    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)
    
    return features, label

# for data in tempds.take(2):
#     pprint({k: v.numpy() for k,v in data.items()}) ##pprint is pretty print for JSON 
#     print("\n") 






def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN) ###removed the NULL data as DEFAULTS  
    features = row_data

    for unwated_col in UNWATED_COLS:
        features.pop(unwated_col)
    return features, label 

###tf.data.experimetal.make_csv_dataset() method reads CSV files into a dataset
def create_dataset(pattern, batch_size=1, mode="eval"):
    dataset =tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS 
    )####data.experimental.make_csv_dataset() method reads CSV files 
    #The map() function executes a specified function for each item in an iterable

    dataset = dataset.map(features_and_labels).cache()


    if mode =="train":
        dataset = dataset.shuflle(buffer_size = 1000).repeat()
    dataset = dataset.prefetch(1) ##take advantage of multi-threading; 1 = autotune
    return dataset 






