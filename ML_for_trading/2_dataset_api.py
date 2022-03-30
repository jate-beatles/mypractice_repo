import json
import math
import os
from pprint import pprint

import numpy as np

# x.assign(new_value)
# x.assign_add(value_to_be_added)
# x.assign_sub(value_to_be_subtracted

# tf.add allows to add the components of a tensor
# tf.multiply allows us to multiply the components of a tensor
# tf.subtract allow us to substract the components of a tensor
# tf.math.* contains the usual math operations to be applied on the components of a tensor

import tensorflow as tf
print(tf.version.VERSION)

N_Point = 10
X = tf.constant(range(N_Point), dtype=tf.float32)
Y = 2 * X + 10
