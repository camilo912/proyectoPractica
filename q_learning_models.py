import pandas as pd
import numpy as np

def get_next_state(state, next_ob, action):
	#print(state)
	#if(state.ndim==1):

	# 	for i in range(1, len(state.ravel()) - 1):
	# 		state[i] = state[i+1]
	# 	state[0] = action
	# 	state[-1] = next_ob
	#elif(state.ndim==2):
	if(state.ndim==2):
		state = np.roll(state, -1, axis=0)
		state[-1] = [action] + list(next_ob.ravel())
	# 	state = np.roll(state, -1, axis=0)
	# 	for i in range(1, state.shape[1]-1):
	# 		state[-1, i] = state[-2, i+1]
	# 	state[-1, 0] = action
	# 	state[-1, -1] = next_ob
	#print(state)
	#raise Exception('Debug')
	return state

class Model():
	def __init__(self, n_features, n_lags, lr, n_hidden, refresh_rate, n_classes):
		from keras.models import Sequential
		from keras.layers import Dense, Dropout, MaxPooling1D, Reshape, Flatten, Conv1D, LSTM
		from keras.optimizers import Adam

		self.n_features = n_features
		self.n_hidden = n_hidden
		self.n_lags = n_lags
		self.lr = lr
		self.refresh_rate = refresh_rate
		self.n_classes = n_classes

		self.model = Sequential()
		self.model.add(LSTM(n_hidden, input_shape=(n_lags, n_features+1), return_sequences=True))
		#self.model.add(Dropout(0.5))
		#self.model.add(LSTM(n_hidden, return_sequences=True))
		self.model.add(Dropout(0.5))
		self.model.add(LSTM(n_hidden))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(n_hidden, activation='tanh'))
		self.model.add(Dropout(0.5))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		self.model.add(Dense(n_classes))

		# self.model = Sequential()
		# self.model.add(Dense(n_hidden*3, activation='relu'))
		# #self.model.add(Dropout(0.5))
		# #self.model.add(Dense(n_hidden, activation='tanh'))
		# #self.model.add(Dropout(0.5))
		# #self.model.add(Reshape((n_hidden, 1)))
		# #self.model.add(Conv1D(64, 5))
		# #self.model.add(MaxPooling1D())
		# #self.model.add(Conv1D(32, 5))
		# #self.model.add(Conv1D(32, 5))
		# #self.model.add(Flatten())
		# self.model.add(Dense(n_hidden, activation='relu'))
		# #self.model.add(Dropout(0.5))
		# self.model.add(Dense(n_hidden*2, activation='relu'))
		# #self.model.add(Dropout(0.5))
		# self.model.add(Dense(int(n_hidden/2), activation='tanh'))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_classes))

		self.opt=Adam(lr=lr)
		self.model.compile(loss='mse', optimizer=self.opt)


	def run(self, X, Y, rewards, epsilon, gamma, n_epochs):
		last_position=0
		if(not hasattr(self, 'future_model')): 
			self.future_model = Model(self.n_features, self.n_lags, self.lr, self.n_hidden, self.refresh_rate, self.n_classes)
			print('creado modelo futuro')
		for epoch in range(n_epochs):
			preds = []
			state = np.insert(X[0], 0, last_position, axis=X[0].ndim-1)
			for i in range(len(X) - 1):
				action, target = self.choose_action(state, epsilon)
				next_state = get_next_state(state, X[i+1, -1], action)
				q = rewards[i, action] + gamma*(max(self.future_model.predict(next_state).ravel()))
				#target = self.model.predict() # rewards[i].squeeze()
				target[action] = q
				self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target, axis=0), epochs=1, verbose=0)

				if(i % self.refresh_rate == 0):
					self.future_model.set_weights(self.model.get_weights())

				preds.append(action)
				last_position = action
				state = next_state

		return preds


	def set_weights(self, weights):
		self.model.set_weights(weights)

	def choose_action(self, state, epsilon):
		# make predictions
		preds = self.model.predict(np.expand_dims(state, axis=0)).ravel()
		# get best q
		action_max = np.argmax(preds)
		# create action selection probabilities
		P = [(1-epsilon)*(action==action_max) + epsilon/self.n_classes for action in range(self.n_classes)]
		# select an action
		chosen_action = np.random.choice(np.arange(self.n_classes), p=P)
		#pred = preds[chosen_action]
		return chosen_action, preds

	def predict(self, x):
		return self.model.predict(np.expand_dims(x, axis=0))