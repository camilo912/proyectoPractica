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












class Model_decisor():
	def __init__(self, lr, n_features, scaler, n_classes, gamma):
		from keras.models import Sequential
		from keras.layers import Dense, Flatten
		from keras.optimizers import Adam

		self.scaler = scaler
		self.n_features = n_features
		self.n_classes = n_classes
		self.gamma = gamma

		# Ensemble model
		self.model = Sequential()
		self.model.add(Dense(20, input_shape=(1, n_features), kernel_initializer='uniform', activation='tanh'))
		self.model.add(Flatten())       # Flatten input so as to have no problems with processing
		self.model.add(Dense(18, kernel_initializer='uniform', activation='tanh'))
		self.model.add(Dense(10, kernel_initializer='uniform', activation='tanh'))
		self.model.add(Dense(n_classes, kernel_initializer='uniform', activation='linear'))

		# Define optimizer
		self.opt = Adam(lr=lr, decay=0.0)

		# Compile model
		#self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
		self.model.compile(loss='mse', optimizer=self.opt, metrics=['accuracy'])
		#self.model.compile(loss=loss_func, optimizer=self.opt, metrics=['accuracy'])

	def train(self, train_X, val_X, train_y, val_y, batch_size, n_epochs, variations):
		import random
		import numpy as np
		from matplotlib import pyplot as plt
		# self.model.train(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=0, shuffle=True)

		evals = []
		for epoch in range(n_epochs):
			batch_idxs = random.sample(set(np.arange(len(train_X))), batch_size)
			inputs = train_X[batch_idxs]
			targets = np.zeros((batch_size, self.n_classes))
			
			for i, idx in enumerate(batch_idxs):
				state = train_X[idx].reshape(1, -1)
				#action = np.argmax(self.model.predict(np.expand_dims(state, axis=0)))
				#action = 1 if(state[-1, -1] > state[-1, -2]) else 0
				#reward = variations[idx] if action else -variations[idx]
				#state_new = np.append(state[:, 1:], train_y[idx].reshape(1, -1), axis=1)

				inputs[i] = state # np.expand_dims(state, axis=0)
				# targets[i] = self.model.predict(np.expand_dims(state, axis=0))
				# targets[i] = np.array([-variations[idx], variations[idx]])
				targets[i] = train_y[idx]
				#Q_sa = self.model.predict(np.expand_dims(state_new, axis=0))

				#targets[i, action] = reward + self.gamma * np.max(Q_sa)

			self.model.train_on_batch(np.expand_dims(inputs, axis=1), targets)
			# self.model.fit(np.expand_dims(inputs, axis=1), targets, epochs=1, batch_size=batch_size, validation_data=(np.expand_dims(val_X, axis=1), val_y), verbose=0, shuffle=False)

			evals.append(self.model.evaluate(np.expand_dims(val_X, axis=1), val_y))

		evals = np.array(evals).T
		plt.plot(evals[0], label=self.model.metrics_names[0])
		plt.plot(evals[1]*100, label=self.model.metrics_names[1])
		plt.legend()
		plt.show()



	def eval(self, test_X, test_y):
		import math
		# from sklearn.metrics import mean_squared_error
		from sklearn.metrics import accuracy_score
		import numpy as np
		import utils

		preds = np.argmax(self.model.predict(np.expand_dims(test_X, axis=1)), axis=1)
		obs = np.argmax(test_y, axis=1)

		acc = accuracy_score(obs, preds)

		# inverse the scaling
		# preds_inv = utils.inverse_transform(self.scaler, preds, self.n_features)

		# inverse the scaling
		# test_y_inv = utils.inverse_transform(self.scaler, test_y, self.n_features)

		# rmse = math.sqrt(mean_squared_error(test_y_inv, preds_inv))


		return acc, preds, test_y

	def predict(self, values):
		import numpy as np

		preds = self.model.predict(np.expand_dims(values, axis=1))
		
		# inverse the scaling
		# preds = utils.inverse_transform(self.scaler, preds, self.n_features)

		return preds

def loss_func(y_true, y_pred):
	from sklearn.metrics import mean_squared_error
	from keras import backend as K

	mse = K.mean(K.square(y_pred - y_true), axis=-1)

	return mse





