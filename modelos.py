class Model_predictor():
	def __init__(self, lr, n_hidden, n_lags, n_features, scaler):
		from keras.layers import Dense, Activation, Dropout, LSTM
		from keras.models import Sequential
		from keras.optimizers import Adam

		self.scaler = scaler
		self.n_features = n_features

		# Ensamble model
		self.model = Sequential()
		self.model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
		self.model.add(Dense(1, ))

		# Define optimizer
		self.opt = Adam(lr=lr, decay=0.0)

		# Compile model
		self.model.compile(loss='mse', optimizer=self.opt)

	def train(self, train_X, val_X, train_y, val_y, batch_size, n_epochs):
		self.model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=0, shuffle=False)

	def eval(self, test_X, test_y):
		import math
		from sklearn.metrics import mean_squared_error
		import numpy as np
		import utils

		preds = self.model.predict(test_X)

		# inverse the scaling
		preds_inv = utils.inverse_transform(self.scaler, preds, self.n_features)

		# inverse the scaling
		test_y_inv = utils.inverse_transform(self.scaler, test_y, self.n_features)

		rmse = math.sqrt(mean_squared_error(test_y_inv, preds_inv))

		return rmse, preds, test_y

	def predict(self, values):
		import numpy as np

		preds = self.model.predict(values)
		
		# inverse the scaling
		# preds = utils.inverse_transform(self.scaler, preds, self.n_features)

		return preds