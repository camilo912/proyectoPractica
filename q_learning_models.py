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

		# self.model = Sequential()
		# self.model.add(LSTM(n_hidden, input_shape=(n_lags, n_features+1), return_sequences=True))
		# #self.model.add(Dropout(0.5))
		# self.model.add(LSTM(n_hidden, return_sequences=True))
		# #self.model.add(Dropout(0.5))
		# self.model.add(LSTM(n_hidden))
		# #self.model.add(Dropout(0.5))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# #self.model.add(Dropout(0.5))
		# self.model.add(Dense(n_hidden, activation='tanh'))
		# self.model.add(Dense(n_classes))

		self.model = Sequential()
		#self.model.add(Flatten())
		#self.model.add(Dense(n_hidden*3, activation='relu'))
		#self.model.add(Dropout(0.5))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		#self.model.add(Dropout(0.5))
		#self.model.add(Reshape((n_hidden, 1)))
		self.model.add(Conv1D(64, 5))
		self.model.add(MaxPooling1D())
		self.model.add(Conv1D(32, 4))
		#self.model.add(MaxPooling1D())
		self.model.add(Conv1D(32, 4))
		self.model.add(Flatten())
		self.model.add(Dense(n_hidden, activation='relu'))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(n_hidden*2, activation='relu'))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(int(n_hidden/2), activation='tanh'))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		#self.model.add(Dense(n_hidden, activation='tanh'))
		self.model.add(Dense(n_classes))

		# a0 = Input(shape=(n_lags, n_features+1))
		# a1 = Conv1D(32, 8, strides=2)(a0)
		# a2 = BatchNormalization()(a1)
		# a3 = Activation('elu')(a2)
		# #a2 = MaxPooling1D()(a1)
		# a4 = Conv1D(64, 4, strides=2)(a3)
		# a5 = BatchNormalization()(a4)
		# a6 = Activation('elu')(a5)
		# a7 = Conv1D(128, 4, strides=2)(a6)
		# a8 = BatchNormalization()(a7)
		# a = Activation('elu')(a8)
		# b = Flatten()(a)
		# c0 = Dense(n_hidden, activation='elu')(b)
		# c = Dense(n_hidden*2, activation='elu')(c0)
		# d = Dense(int(n_hidden/2), activation='tanh')(c)
		# fc_value = Dense(n_hidden, activation='tanh')(d)
		# value = Dense(1, activation='tanh')(fc_value)
		# fc_advantage = Dense(n_hidden, activation='tanh')(d)
		# advantage = Dense(n_classes, activation='tanh')(fc_advantage)
		# avg = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
		# out = Lambda(lambda a: a[0] + (a[1] - a[2]))([value, advantage, avg])
		# self.model = Model(inputs=a0, outputs=out)




		self.opt=Adam(lr=lr)
		self.model.compile(loss='mse', optimizer=self.opt)


	def run(self, X, Y, rewards, epsilon, gamma, n_epochs, training):
		last_position=0
		if(not hasattr(self, 'future_model')): 
			self.future_model = Model(self.n_features, self.n_lags, self.lr, self.n_hidden, self.refresh_rate, self.n_classes)
			print('creado modelo futuro')
		if(training):
			ori_X = X.copy()
			ori_Y = Y.copy()
			ori_rew = rewards.copy()
			batch_size = int(len(X)*0.8)
		for epoch in range(n_epochs):
			if(training):
				train_idxs = np.random.permutation(len(ori_X))[:batch_size]
				X, Y, rewards = ori_X[train_idxs], ori_Y[train_idxs], ori_rew[train_idxs]
			preds = []
			state = np.insert(X[0], 0, last_position, axis=X[0].ndim-1)
			#inputs = []
			#outputs = []
			for i in range(len(X) - 1):
				action, target = self.choose_action(state, epsilon)
				next_state = get_next_state(state, X[i+1, -1], action)
				# normal q learning
				q = rewards[i, action] + gamma*(max(self.future_model.predict(next_state).ravel()))
				# double q learning
				# q = rewards[i, action] + gamma*(self.future_model.predict(next_state).ravel()[np.argmax(self.model.predict(np.expand_dims(state, axis=0)).ravel())])
				#target = self.model.predict() # rewards[i].squeeze()
				#if(action != state[-1, 0]): q -= np.abs(q*0.1)
				target[action] = q
				self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target, axis=0), epochs=1, verbose=0)
				#inputs.append(state)
				#outputs.append(target)

				if(i % self.refresh_rate == 0):
					self.future_model.set_weights(self.model.get_weights())
					# self.future_model.model.fit(np.array(inputs), np.array(outputs), epochs=self.refresh_rate, verbose=0)

				preds.append(action)
				last_position = action
				state = next_state

		return preds


	def set_weights(self, weights):
		self.model.set_weights(weights)

	def choose_action(self, state, epsilon):
		# make predictions
		preds = self.model.predict(np.expand_dims(state, axis=0)).ravel()
		# print(preds.shape)
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
