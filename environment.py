import platform
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print(platform.architecture())
print("numpy "+np.__version__)
print("pandas "+pd.__version__)
print("Keras "+keras.__version__)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
