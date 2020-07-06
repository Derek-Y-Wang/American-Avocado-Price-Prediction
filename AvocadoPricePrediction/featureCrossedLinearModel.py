import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import dataProcessor as data
import numpy as np
from tensorflow import feature_column
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float32')

print("Imported the modules.")

# Load the dataset
training_df = data.AvocadoCleaner().process_data()
test_df = data.AvocadoCleaner().process_data()


# Shuffle the examples
train_df = training_df.reindex(np.random.permutation(training_df.index))

# Create an empty list that will eventually hold all feature columns.
feature_columns = []
