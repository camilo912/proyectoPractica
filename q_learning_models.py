import pandas as pd
import numpy as np

def get_next_state(state, next_ob, action):
	for i in range(1, len(state.ravel()) - 1):
		state[i] = state[i+1]
	state[0] = action
	state[-1] = next_ob
	return state

class Model():
	def __init__(self, n_features, n_lags, lr, n_hidden, refresh_rate, n_classes):
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.optimizers import Adam

		self.n_features = n_features
		self.n_hidden = n_hidden
		self.n_lags = n_lags
		self.lr = lr
		self.refresh_rate = refresh_rate
		self.n_classes = n_classes

		self.model = Sequential()
		self.model.add(Dense(n_hidden, input_dim=(n_features + 1), activation='tanh'))
		self.model.add(Dense(n_classes))

		self.opt=Adam(lr=lr)
		self.model.compile(loss='mse', optimizer=self.opt)


	def run(self, X, Y, rewards, epsilon, gamma, n_epochs):
		last_position=0
		if(not hasattr(self, 'future_model')): self.future_model = Model(self.n_features, self.n_lags, self.lr, self.n_hidden, self.refresh_rate, self.n_classes)
		for epoch in range(n_epochs):
			preds = []
			state = np.insert(X[[0]], len(X[[0]])-1, last_position, axis=1).reshape(-1)
			for i in range(len(X)):
				action, _ = self.choose_action(state, epsilon)
				next_state = get_next_state(state, Y[i], action)
				q = rewards[i, action] + gamma*(max(self.future_model.predict(next_state).ravel()))
				target = rewards[i].squeeze()
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
		pred = preds[chosen_action]
		return chosen_action, pred

	def predict(self, x):
		return self.model.predict(np.expand_dims(x, axis=0))
