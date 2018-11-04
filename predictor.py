def build_expert(input_shape, n_hidden, n_layers, optimizer, loss):
	from keras.models import Sequential
	from keras import regularizers
	from keras.layers import Dense, LSTM

	model = Sequential()
	
	model.add(LSTM(n_hidden, input_shape=input_shape, return_sequences=n_layers>1))
	for layer in range(1, n_layers):
		model.add(LSTM(n_hidden, return_sequences=layer < n_layers-1))

	model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

	model.compile(loss=loss, optimizer=optimizer)

	return model

def init_comite(n, input_shape, n_hidden, n_layers, optimizer, loss):
	experts = [build_expert(input_shape, n_hidden, n_layers, optimizer, loss) for _ in range(n)]
	return experts

def train(model, train_X, train_y, val_X, val_y, batch_size, n_epochs, verbose):
	import time

	start = time.time()
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose)
	elapsed = time.time() - start
	print('trained model, time elapsed: %f seconds (%f minutes)' % (elapsed, elapsed/60))

	return model

class Predictor():
	def __init__(self, optimizer, loss, input_shape, train_X, train_y, val_X, val_y, verbose):
		self.models = []
		#self.models.append(train(build_expert(input_shape, 100, 1, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 200, 1, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		self.models.append(train(build_expert(input_shape, 300, 1, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		self.models.append(train(build_expert(input_shape, 400, 1, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 500, 1, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 100, 2, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 200, 2, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 300, 2, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 400, 2, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		#self.models.append(train(build_expert(input_shape, 500, 2, optimizer, loss), train_X, train_y, val_X, val_y, 32, 120, verbose))
		self.n_models = len(self.models)
		self.gamma = 0.1
		self.weights = []

	def predict(self, X, y, gamma=None, weights=None):
		import numpy as np

		if(gamma):
			self.gamma = gamma

		if weights == None: 
			self.weights = np.ones(len(self.models))
		else:
			self.weights = weights
		
		preds = []
		models_hist = []
		regret_hist = []
		
		for i in range(len(X)):
			p = self.get_model_probabilities()
			model_idx = np.random.choice(np.arange(self.n_models), p=p)
			models_hist.append(model_idx)
			pred = int(self.models[model_idx].predict(np.expand_dims(X[i], axis=0)) > 0.5)
			preds.append(pred)
			reward = 1 if pred == y[i] else 0
			regret_hist.append(1 - reward)
			self.weights[model_idx] = self.weights[model_idx]*np.exp((self.gamma*reward/(p[model_idx]*self.n_models)))

		return preds, weights, models_hist, regret_hist

			

	def get_model_probabilities(self):
		import numpy as np
		p = np.array([(1-self.gamma)*weight/sum(self.weights) + self.gamma/self.n_models for weight in self.weights])
		return p




