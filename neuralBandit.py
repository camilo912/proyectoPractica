import numpy as np


def build_experts(n, input_shape, n_hidden, n_layers):
	
	def build_expert():
		from keras.models import Sequential
		from keras import regularizers
		from keras.layers import Dense

		model = Sequential()

		for layer in range(n_layers):
			model.add(Dense(n_hidden, kernel_initializer='glorot_uniform', activation='relu', 
									input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))

		# ouput layer
		model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
		return model

	experts = [build_expert() for _ in range(n)]
	return experts

def build_experts_lstm(n, input_shape, n_hidden, n_layers):

	def build_expert():
		from keras.models import Sequential
		from keras import regularizers
		from keras.layers import Dense, LSTM, Dropout

		model = Sequential()
		
		model.add(LSTM(n_hidden, input_shape=input_shape, return_sequences=n_layers>1))
		model.add(Dropout(0.2))
		for layer in range(1, n_layers):
			model.add(LSTM(n_hidden, return_sequences=layer < n_layers-1))
			model.add(Dropout(0.2))

		model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))#, kernel_regularizer=regularizers.l2(0.01)))
		return model

	experts = [build_expert() for _ in range(n)]
	return experts

def compile_experts(experts, optimizer, loss):
	# compiles a commitee of experts
	n_arms = len(experts)
	
	def compile_expert(expert):
		expert.compile(optimizer=optimizer,
					  loss=loss)
		
		return expert
	
	compiled_experts = [compile_expert(expert) for expert in experts]
	return compiled_experts

def choose_arm(x, experts, explore):
	n_arms = len(experts)
	# make predictions
	preds = [expert.predict(x) for expert in experts]
	# get best arm
	arm_max = np.nanargmax(preds)
	# create arm selection probabilities
	P = [(1-explore)*(arm==arm_max) + explore/n_arms for arm in range(n_arms)]
	# select an arm
	chosen_arm = np.random.choice(np.arange(n_arms), p=P)
	pred = preds[chosen_arm]
	return chosen_arm, pred

def train_bandit_1(X, Y, explore, exp_annealing_rate=1, min_explore=.005, **kwargs):
	import time

	n, n_arms = Y.shape
	input_shape = X.shape[1:]
	experts = build_experts(n_arms, input_shape, 32, 1)
	experts = compile_experts(experts, **kwargs)

	chosen_arms = []
	regrets = []
	true_rewards = []
	
	start_time = time.time()
	message_iteration = 10
	print('Starting bandit\n----------\nN_arms: %d\n----------\n' % (n_arms))
	for i in range(n):
		context = X[[i]]
		chosen_arm, pred = choose_arm(context, experts, explore)
		reward = Y[i, chosen_arm]
		max_reward = np.max(Y[i])
		max_arm = np.argmax(Y[i])
		true_rewards.append(max_arm)
		expert = experts[chosen_arm]
		expert.fit(context, np.expand_dims(reward, axis=0), epochs=1, verbose=0)
		experts[chosen_arm] = expert
		chosen_arms.append(chosen_arm)
		regret = max_reward - reward
		regrets.append(regret)
		
		if explore > min_explore:
			explore *= exp_annealing_rate
		
		if (i % message_iteration == 0) and (i > 0):
			if message_iteration <= 1e4:
				message_iteration *= 10
			elapsed = time.time() - start_time
			remaining = (n*elapsed/i - elapsed)/60
			print('''Completed iteration: %d
			Elapsed time: %.2f seconds
			Estimated time remaining: %.2f minutes
			--------------------''' % (i, elapsed, remaining))
	
	elapsed = (time.time() - start_time)/60
	print('Finished in: %.2f minutes' % (elapsed))
	
	return experts, chosen_arms, true_rewards, regrets


############################################## Bandit 2 ##################################################

def init_models(n_arms, input_shape):
	# init models
	# 32 hidden units, 1 hidden layer, explore = .005
	model_1 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
	model_1 = compile_experts(model_1, loss='binary_crossentropy', optimizer='adam')

	# 64 hidden units, 1 hidden layer, explore = .005
	model_2 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
	model_2 = compile_experts(model_2, loss='binary_crossentropy', optimizer='adam')

	# 128 hidden units, 1 hidden layer, explore = .005
	model_3 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
	model_3 = compile_experts(model_3, loss='binary_crossentropy', optimizer='adam')


	# 64 hidden units, 2 hidden layers, explore = .005
	model_4 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
	model_4 = compile_experts(model_4, loss='binary_crossentropy', optimizer='adam')


	# 64 hidden units, 2 hidden layers, explore = .005
	model_5 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
	model_5 = compile_experts(model_5, loss='binary_crossentropy', optimizer='adam')


	# 32 hidden units, 1 hidden layer, annealing_explore
	model_6 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
	model_6 = compile_experts(model_6, loss='binary_crossentropy', optimizer='adam')

	# 64 hidden units, 1 hidden layer, annealing_explore
	model_7 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
	model_7 = compile_experts(model_7, loss='binary_crossentropy', optimizer='adam')

	# 128 hidden units, 1 hidden layer, annealing_explore
	model_8 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
	model_8 = compile_experts(model_8, loss='binary_crossentropy', optimizer='adam')


	# 64 hidden units, 2 hidden layers, annealing_explore
	model_9 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
	model_9 = compile_experts(model_9, loss='binary_crossentropy', optimizer='adam')


	# 64 hidden units, 2 hidden layers, annealing_explore
	model_10 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
	model_10 = compile_experts(model_10, loss='binary_crossentropy', optimizer='adam')

	# 512 hidden units, 3 hidden layers, explore = 0.005
	model_11 = build_experts(n_arms, input_shape, n_hidden=512, n_layers=3)
	model_11 = compile_experts(model_11, loss='binary_crossentropy', optimizer='adam')

	# 512 hidden units, 3 hidden layers, annealing_explore
	model_12 = build_experts(n_arms, input_shape, n_hidden=512, n_layers=3)
	model_12 = compile_experts(model_12, loss='binary_crossentropy', optimizer='adam')

	return [model_1, model_2, model_3, model_4, model_5, model_11, model_6, model_7, model_8, model_9, model_10, model_12]
	# return [model_2, model_4, model_7, model_9]
	#return [model_4, model_5, model_12, model_9, model_10, model_12]

def init_models_recurrent(n_arms, input_shape):
	# init models
	# 32 hidden units, 1 hidden layer, explore = .005
	model_1 = build_experts_lstm(n_arms, input_shape, n_hidden=150, n_layers=1)
	model_1 = compile_experts(model_1, loss='mse', optimizer='adam')

	# 64 hidden units, 1 hidden layer, explore = .005
	model_2 = build_experts_lstm(n_arms, input_shape, n_hidden=150, n_layers=2)
	model_2 = compile_experts(model_2, loss='mse', optimizer='adam')

	# 128 hidden units, 1 hidden layer, explore = .005
	model_3 = build_experts_lstm(n_arms, input_shape, n_hidden=150, n_layers=1)
	model_3 = compile_experts(model_3, loss='mse', optimizer='adam')


	# 64 hidden units, 2 hidden layers, explore = .005
	model_4 = build_experts_lstm(n_arms, input_shape, n_hidden=150, n_layers=2)
	model_4 = compile_experts(model_4, loss='mse', optimizer='adam')

	return [model_1, model_2, model_3, model_4]

def get_model_probabilities(weights, gamma_model):
	n_models = len(weights)
	# get probabilites of choosing each model
	p = np.array([(1-gamma_model)*weight/sum(weights) + gamma_model/n_models for weight in weights])
	return p

def choose_model(weights, model_probabilities):
	n_models = len(weights)
	# choose a model based on weights
	model = np.random.choice(np.arange(n_models), p=model_probabilities)
	return model

def train_bandit_2(X, Y, explore, exp_annealing_rate, min_explore=0.005, recurrent=False):
	import time

	n, n_arms = Y.shape
	input_shape = X.shape[1:]
	n_steps = len(X)
	idxs = np.random.choice(range(len(X)), n_steps)

	# exploration paramater
	gamma_model = explore

	# init models
	if(recurrent):
		models = init_models_recurrent(n_arms, input_shape)
	else:
		models = init_models(n_arms, input_shape)
	n_models = len(models)

	# init weights
	weights = np.ones(n_models)

	middle = int(n_models/2)

	# init model explore parameters
	explores = np.array([.005]*middle + [.5]*(n_models - middle))
	anneal = np.array([False]*middle + [True]*(n_models - middle))
	annealing_rate = exp_annealing_rate
	min_explore = min_explore

	# init histories
	arm_hist_2 = []
	model_hist_2 = []
	regret_hist_2 = []
	weight_hist_2 = []
	true_reward_hist_2 = []

	# init timing vars
	start_time = time.time()
	next_check = 1

	# train the models
	for step in range(n_steps):
		# store weights
		weight_hist_2.append(weights)
		# get probs and choose a model
		p = get_model_probabilities(weights, gamma_model)
		chosen_model = choose_model(weights, p)
		# store model choice
		model_hist_2.append(chosen_model)
		# get a random data point
		# i = np.random.randint(X.shape[0])
		# get a non random data point
		#i = idxs[step]
		# get a serial step
		i = step % len(X)
		context = X[[i]]
		# choose an arm
		chosen_arm, pred = choose_arm(context, models[chosen_model], explores[chosen_model])
		# store arm selection
		arm_hist_2.append(chosen_arm)
		# observe reward and max reward
		reward = Y[i, chosen_arm]
		max_reward = np.max(Y[i])
		max_arm = np.argmax(Y[i])
		true_reward_hist_2.append(max_arm)
		# calculate and store regret
		regret = max_reward - reward
		regret_hist_2.append(regret)
		# update the chosen arm for each model
		for m, model in enumerate(models):
			expert = model[chosen_arm]
			expert.fit(context, np.expand_dims(reward, axis=0), epochs=1, verbose=0)
			model[chosen_arm] = expert
			# anneal explore param if necessary
			if (anneal[m]) and (explores[m] > min_explore):
				explores[m] *= annealing_rate
		# update weights
		weights[chosen_model] = weights[chosen_model]*np.exp((gamma_model*reward/(p[chosen_model]*n_models)))
		# print progress
		if step == next_check:
			elapsed = time.time()-start_time
			print('Step %d complete in %f seconds.' % (step, elapsed))
			next_check *= 2

	return models, arm_hist_2, true_reward_hist_2, regret_hist_2, weights, model_hist_2

def predict_with_weights(X, Y, models, weights, explore=0):
	preds = []
	regret_hist = []
	true_reward_hist = []
	models_hist = []

	n_models = len(models)
	for i in range(len(X)):
		model_probabilities = np.array([weight/sum(weights) for weight in weights])
		model_idx = np.random.choice(np.arange(n_models), p=model_probabilities)
		experts = models[model_idx]
		chosen_expert, pred = choose_arm(X[[i]], experts, explore)
		
		# historics
		regret_hist.append(1 - Y[i, chosen_expert])
		true_reward_hist.append(np.argmax(Y[i]))
		models_hist.append(model_idx)
		preds.append(chosen_expert)

	return preds, regret_hist, true_reward_hist, models_hist

def predict_and_train_with_weights(X, Y, models, weights, gamma, explore=0):
	preds = []
	regret_hist = []
	true_reward_hist = []
	models_hist = []

	n_models = len(models)
	for i in range(len(X)):
		model_probabilities = np.array([weight/sum(weights) for weight in weights])
		model_idx = np.random.choice(np.arange(n_models), p=model_probabilities)
		experts = models[model_idx]
		chosen_expert, pred = choose_arm(X[[i]], experts, explore)

		# historics
		regret_hist.append(1 - Y[i, chosen_expert])
		true_reward_hist.append(np.argmax(Y[i]))
		models_hist.append(model_idx)
		preds.append(chosen_expert)
		reward = Y[i, chosen_expert]

		# update weights
		weights[model_idx] = weights[model_idx]*np.exp((gamma*reward/(model_probabilities[model_idx]*n_models)))

		# re train
		model = models[model_idx][chosen_expert]
		model.fit(X[[i]], np.expand_dims(reward, axis=0), epochs=1, verbose=0)
		models[model_idx][chosen_expert] = model

	return preds, weights, models, regret_hist, true_reward_hist, models_hist

def predict(X, Y, experts, explore=0):
	preds = []
	regret_hist = []
	true_reward_hist = []

	for i in range(len(X)):
		chosen_expert, pred = choose_arm(X[[i]], experts, explore)


		# historics
		regret_hist.append(1 - Y[i, chosen_expert])
		true_reward_hist.append(np.argmax(Y[i]))
		preds.append(chosen_expert)

	return preds, regret_hist, true_reward_hist

def predict_and_train(X, Y, experts, explore=0):
	preds = []
	regret_hist = []
	true_reward_hist = []

	for i in range(len(X)):
		chosen_expert, pred = choose_arm(X[[i]], experts, explore)


		# historics
		regret_hist.append(1 - Y[i, chosen_expert])
		true_reward_hist.append(np.argmax(Y[i]))
		preds.append(chosen_expert)

		expert = experts[chosen_expert]
		expert.fit(X[[i]], np.expand_dims(Y[i, chosen_expert], axis=0), epochs=1, verbose=0)
		experts[chosen_expert] = expert

	return preds, experts, regret_hist, true_reward_hist