import numpy as np
import pandas as pd

from keras.layers import LSTM,Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Read Data

bodyswing_df  = pd.read_csv("SWING.txt")
handswing_df  = pd.read_csv("HANDSWING.txt")
