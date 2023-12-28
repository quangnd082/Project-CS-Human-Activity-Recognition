import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
bodyswing_df = pd.read_csv("../data/SWING.txt")
handswing_df = pd.read_csv("../data/HANDSWING.txt")
doze_df = pd.read_csv("../data/DOZE.txt")
love_df = pd.read_csv("../data/LOVE.txt")
clap_df = pd.read_csv("../data/CLAP.txt")
X = []
y = []
no_of_timesteps = 10

dataset_body = bodyswing_df.iloc[:,1:].values
n_sample = len(dataset_body)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset_body[i-no_of_timesteps:i,:])
    y.append(0)

dataset_hand = handswing_df.iloc[:,1:].values
n_sample = len(dataset_hand)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset_hand[i-no_of_timesteps:i,:])
    y.append(1)


dataset_doze = doze_df.iloc[:,1:].values
n_sample = len(dataset_doze)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset_doze[i-no_of_timesteps:i,:])
    y.append(2)

dataset_love = love_df.iloc[:,1:].values
n_sample = len(dataset_love)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset_love[i-no_of_timesteps:i,:])
    y.append(3)

dataset_clap = clap_df.iloc[:,1:].values
n_sample = len(dataset_clap)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset_clap[i-no_of_timesteps:i,:])
    y.append(4)

# # Define paths to CSV files
# body_swing_path = "SWING.txt"
# hand_swing_path = "HANDSWING.txt"
# doze_path = "DOZE.txt"
# love_path = "LOVE.txt"
# clap_path = "LOVE.txt"
#
# # Initialize empty lists for features and labels
# X = []
# y = []
#
# # Specify number of timesteps
# no_of_timesteps = 10

# Read all data
# datasets = [pd.read_csv(path) for path in [body_swing_path, hand_swing_path, doze_path, love_path, clap_path]]
#
# # Split data into features and labels
# for dataset in datasets:
#     dataset_features = dataset.iloc[:, 1:].values
#     dataset_labels = dataset.iloc[:, 0].values
#
#     # Loop through each sample
#     for i in range(no_of_timesteps, len(dataset_features)):
#         # Extract features
#         features = dataset_features[i - no_of_timesteps:i, :]
#         # Append features to X
#         X.append(features)
#         # Append label to y
#         y.append(dataset_labels[i])
X, y = np.array(X), np.array(y)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 132 = 4 * 33
# 10 = no_of_number


# one hot encoding


y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)
print(y_train_onehot.shape , y_test_onehot.shape)
n_features = X.shape[2]

# model  = Sequential()
# model.add(LSTM(units = 50, return_sequences = True, input_shape = (no_of_timesteps, n_features)))
# model.add(Dropout(0.2))
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(LSTM(units = 50))
# model.add(Dropout(0.2))
# model.add(Dense(units = 5, activation="softmax"))
# model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")
#
# model.fit(X_train, y_train_onehot, epochs=20, batch_size=32,validation_data=(X_test, y_test_onehot))
# model.save("model_19.h5")


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(no_of_timesteps, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))



early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
# Nếu validation loss không giảm trong 10 epoch liên tiếp, huấn luyện sẽ dừng.
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

lstm_training_history = model.fit(x = X_train, y = y_train_onehot, epochs = 50, batch_size = 32,
                                                     shuffle = True, validation_split = 0.2,
                                                     callbacks = [early_stopping_callback])



model.save("model_29.h5")
