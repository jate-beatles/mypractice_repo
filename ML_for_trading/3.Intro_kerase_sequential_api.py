import datetime
import os
import shutil
from termios import VERASE
import numpy as np 
import pandas as pd 
import tesorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from tensorflow.keras.callbacks import tensorboard 
from tensorflow.keras. layers import Dense, DeseFeatures 
from tensorflow.keras.models import Sequential

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

def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN) ##removed the NULL data as DEFAULTS  
    features = row_data

    for unwated_col in UNWATED_COLS:
        features.pop(unwated_col)
    return features, label 

##tf.data.experimetal.make_csv_dataset() method reads CSV files into a dataset
def creat_dateset(pattern, batch_size=1, mod="eval"):
    dataset =tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS 
    )
    dataset = dataset.map(features_and_labels)

    if mode =="train":
        dataset = dataset.shuflle(buffer_size = 1000).repeat()
    dataset = dataset.prefetch(1)
    return dataset 







