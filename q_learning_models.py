import pandas as pd
import numpy as np

def get_next_state(state, next_ob, action):

	if(state.ndim==2):
		state = np.roll(state, -1, axis=0)
		state[-1] = [action] + list(next_ob.ravel())
	return state

class Model():
	def __init__(self, n_features, n_lags, lr, n_hidden, refresh_rate, n_classes):
		from keras.models import Sequential, Model
		from keras.layers import Dense, Dropout, MaxPooling1D, Reshape, Flatten, Conv1D, LSTM, Input, Lambda, BatchNormalization, Activation
		from keras.optimizers import Adam
		from keras import backend as K

		self.n_features = n_features
		self.n_hidden = n_hidden
		self.n_lags = n_lags
		self.lr = lr
		self.refresh_rate = refresh_rate
		self.n_classes = n_classes

		self.model = Sequential()
		self.model.add(LSTM(n_hidden, return_sequences=True))
		self.model.add(LSTM(n_classes))
		

		self.opt=Adam(lr=lr)
		self.model.compile(loss='mse', optimizer=self.opt)


	def run(self, X, Y, rewards, gamma, n_epochs, training, init_explore, min_explore, decay_rate_explore):
		n_samples = len(X)
		if(training):
			ori_X = X.copy()
			ori_Y = Y.copy()
			ori_rew = rewards.copy()
			batch_size = int(n_samples*0.1)
			explore = init_explore + decay_rate_explore
		else:
			explore = min_explore
		actions = np.zeros((len(X)+X.shape[1]))
		if(not hasattr(self, 'future_model')): 
			self.future_model = Model(self.n_features, self.n_lags, self.lr, self.n_hidden, self.refresh_rate, self.n_classes)
		self.future_model.model.set_weights(self.model.get_weights())
		for epoch in range(n_epochs):
			if(training):
				if explore > min_explore: explore -= decay_rate_explore
				train_idxs = np.random.permutation(n_samples-1)[:batch_size]
				X, Y, rewards = ori_X[train_idxs], ori_Y[train_idxs], ori_rew[train_idxs]
			else:
				state = np.insert(X[0], 0, 0, axis=X[0].ndim-1)
			preds = []
			for i in range(len(X) - int(not training)):
				if(training):
					state = np.insert(X[i], 0, actions[train_idxs[i]:train_idxs[i]+X.shape[1]], axis=X[0].ndim-1)
				action, target = self.choose_action(state, explore)
				if(training):
					next_state = get_next_state(state, ori_X[train_idxs[i]+1, -1], action)
				else:
					next_state = get_next_state(state, X[i+1, -1], action)
				
				q = rewards[i, action] + gamma*(self.future_model.model.predict(np.expand_dims(next_state, axis=0)).ravel()[np.argmax(self.model.predict(np.expand_dims(next_state, axis=0)).ravel())])
				target[action] = q
				self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target, axis=0), epochs=1, verbose=0)

				if(i % self.refresh_rate == 0):
					self.future_model.model.set_weights(self.model.get_weights())

				preds.append(action)
				if(training): 
					actions[train_idxs[i]+X.shape[1]] = action
				else:
					state = next_state

		return preds

	def choose_action(self, state, explore):
		# make predictions
		preds = self.model.predict(np.expand_dims(state, axis=0)).ravel()
		# get best q
		action_max = np.argmax(preds)
		# create action selection probabilities
		P = [(1-explore)*(action==action_max) + explore/self.n_classes for action in range(self.n_classes)]
		# select an action
		chosen_action = np.random.choice(np.arange(self.n_classes), p=P)
		return chosen_action, preds

