def build_expert(input_shape, n_hidden, n_layers, optimizer, loss, lambda_term):
	from keras.models import Sequential
	from keras import regularizers
	from keras.layers import Dense, LSTM

	model = Sequential()
	
	model.add(LSTM(n_hidden, input_shape=input_shape, return_sequences=n_layers>1))
	for layer in range(1, n_layers):
		model.add(LSTM(n_hidden, return_sequences=layer < n_layers-1))

	model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_term)))

	model.compile(loss=loss, optimizer=optimizer)

	return model

#def init_comite(n, input_shape, n_hidden, n_layers, optimizer, loss):
#	experts = [build_expert(input_shape, n_hidden, n_layers, optimizer, loss) for _ in range(n)]
#	return experts

def train(model, train_X, train_y, val_X, val_y, batch_size, n_epochs, verbose, lr):
	import time

	start = time.time()
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose)
	elapsed = time.time() - start
	print('trained model, time elapsed: %f seconds (%f minutes)' % (elapsed, elapsed/60))

	return model

class Predictor():
	def __init__(self, optimizer, loss, input_shape, train_X, train_y, val_X, val_y, verbose):
		self.models = []
		self.models.append(train(build_expert(input_shape, 150, 1, optimizer, loss, 0), train_X, train_y, val_X, val_y, 250, 70, verbose, 0.019))
		self.models.append(train(build_expert(input_shape, 150, 1, optimizer, loss, 0), train_X, train_y, val_X, val_y, 250, 70, verbose, 0.019))
		self.models.append(train(build_expert(input_shape, 150, 1, optimizer, loss, 0.), train_X, train_y, val_X, val_y, 250, 70, verbose, 0.019))
		self.n_models = len(self.models)

	def predict_and_train(self, X, y):
		import numpy as np

		preds = []
		models_hist = []
		regret_hist = []
		
		for i in range(len(X)):
			tmp = [self.models[i].predict(np.expand_dims(X[i], axis=0)) for i in range(len(self.models))]
			distances = [max(p, 1-p) for p in tmp]
			model_idx = np.argmax(distances)
			pred = int(tmp[model_idx] > 0.5)
			models_hist.append(model_idx)
			preds.append(pred)
			reward = 1 if pred == y[i] else 0
			regret_hist.append(1 - reward)
			model = self.models[model_idx]
			model.fit(np.expand_dims(X[i], axis=0), np.expand_dims(y[i], axis=0), epochs=1, verbose=0)
			self.models[model_idx] = model
			#self.weights[model_idx] = self.weights[model_idx]*np.exp((self.gamma*reward/(p[model_idx]*self.n_models)))
			#self.weights[model_idx] = self.weights[model_idx]*np.exp((reward/(p[model_idx]*self.n_models)))

		return preds, models_hist, regret_hist

			

	def get_model_probabilities(self):
		import numpy as np
		#p = np.array([(1-self.gamma)*weight/sum(self.weights) + self.gamma/self.n_models for weight in self.weights])
		p = np.array([weight/sum(self.weights) for weight in self.weights])
		return p




