import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils

df = pd.read_csv('data/datos_proyecto.csv', header=0)
scaled, scaler = utils.normalize_data(df.values)
n_features = scaled.shape[1]

# hyper parameters
batch_size = 100
lr = 1e-3
n_epochs = 300
n_hidden = 100
n_lags = 3

train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
train_y = np.array(train_y > train_X[:, -1, 0]).astype(np.int32)
val_y = np.array(val_y > val_X[:, -1, 0]).astype(np.int32)
test_y = np.array(test_y > test_X[:, -1, 0]).astype(np.int32)


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import regularizers

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

history = model.fit(train_X, train_y, validation_data=(val_X, val_y), verbose=0, batch_size=batch_size, epochs=n_epochs)

plt.plot(history.history['loss'])
plt.show()

preds = (model.predict(test_X) > 0.5).astype(np.int32)

plt.plot(test_y)
plt.plot(preds)
plt.show()

from sklearn.metrics import accuracy_score

print('test accuracy score: ', accuracy_score(test_y, preds))