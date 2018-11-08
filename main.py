import numpy as np
import pandas as pd
import modelos
import utils
import neuralBandit
import time

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	# ######################################## only predictions model ####################################################

	# df = pd.read_csv('data/datos_proyecto.csv', header=0)
	# scaled, scaler = utils.normalize_data(df.values)
	# n_features = scaled.shape[1]
	# max_evals = 100

	# # hyper parameters
	# batch_size = 64
	# lr = 1e-3
	# n_epochs = 150
	# n_hidden = 30
	# n_lags = 30

	# # best = utils.bayes_optimization(max_evals, scaled, n_features, scaler)
	# # batch_size, lr, n_epochs, n_hidden, n_lags = int(best['batch_size']), best['lr'], int(best['n_epochs']), int(best['n_hidden']), int(best['n_lags'])

	# train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)

	# start = time.time()

	# model = modelos.Model_predictor(lr, n_hidden, n_lags, n_features, scaler)
	# model.train(train_X, val_X, train_y, val_y, batch_size, n_epochs)
	# preds_train = model.predict(train_X)
	# preds_val = model.predict(val_X)
	# preds_test = model.predict(test_X)

	# print('predictor mdoel finished in: ', time.time()-start, ' seconds')
	
	# ######## start for continue with other model ########

	# actuals_train, actuals_val, actuals_test = train_X[:, -1, 0], val_X[:, -1, 0], test_X[:, -1, 0]
	# nexts_train, nexts_val, nexts_test = train_y.copy(), val_y.copy(), test_y.copy()

	# train_y = (train_X[:, -1, 0] <= train_y).astype(np.int32)
	# val_y = (val_X[:, -1, 0] <= val_y).astype(np.int32)
	# test_y = (test_X[:, -1, 0] <= test_y).astype(np.int32)

	# train_X = np.append(train_X[:, :, 0], preds_train, axis=1).reshape(-1, n_lags+1, 1)
	# val_X = np.append(val_X[:, :, 0], preds_val, axis=1).reshape(-1, n_lags+1, 1)
	# test_X = np.append(test_X[:, :, 0], preds_test, axis=1).reshape(-1, n_lags+1, 1)
	# n_features = 1
	# n_lags += 1

	############# finish for continue with other model ###################

	# rmse_train, y_hat_train, y_trainset = model.eval(train_X, train_y)
	# rmse_val, y_hat_val, y_valset = model.eval(val_X, val_y)
	# rmse, y_hat, y = model.eval(test_X, test_y)

	# run_time = time.time() - start
	
	# # print results
	# print('rmse: %f' % rmse, 'run_time: %f' % run_time)

	# # # plot results
	# # plt.plot(y_hat, label='predictions')
	# # plt.plot(y, label='observations')
	# # plt.suptitle('Predictions vs observations')
	# # plt.legend()
	# # plt.show()



	# train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data(df.values, n_lags, n_features)
	
	# variations = train_y_inv - train_X_inv[:, -1, 0]
	# variations_val = val_y_inv - val_X_inv[:, -1, 0]
	# variations_test = test_y_inv - test_X_inv[:, -1, 0]

	# train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
	
	# train_X = np.append(train_X[:, :, 0], model.predict(train_X).reshape(-1, 1), axis=1)
	# train_y = utils.to_one_hot((train_X[:, -2] < train_y).astype(np.int32))
	# # train_y = np.array([variations.ravel()*-1, variations.ravel()]).T

	# val_X = np.append(val_X[:, :, 0], model.predict(val_X).reshape(-1, 1), axis=1)
	# val_y = utils.to_one_hot((val_X[:, -2] < val_y).astype(np.int32))
	# # val_y = np.array([variations_val.ravel()*-1, variations_val.ravel()]).T
	
	# test_X = np.append(test_X[:, :, 0], model.predict(test_X).reshape(-1, 1), axis=1)
	# test_y = utils.to_one_hot((test_X[:, -2] < test_y).astype(np.int32))
	# # test_y = np.array([variations_test.ravel()*-1, variations_test.ravel()]).T

	# # train_y, val_y, test_y = train_y.astype(np.int32), val_y.astype(np.int32), test_y.astype(np.int32)

	# print(train_X[:4])
	# print(train_y[:4])
	# print(train_X.shape)
	# print(train_y.shape)

	# start = time.time()
	# model2 = modelos.Model_decisor(lr, n_features, scaler, n_classes, gamma)
	# model2.train(train_X, val_X, train_y, val_y, batch_size, n_epochs, variations)
	# acc, y_hat, y = model2.eval(val_X, val_y)

	# run_time = time.time() - start
	
	# # print results
	# print('acc: %f' % acc, 'run_time: %f' % run_time)

	# # ########################################### neural bandits Model #################################################################

	# df = pd.read_csv('data/datos_proyecto.csv', header=0)
	# scaled, scaler = utils.normalize_data(df.values)
	# n_features = scaled.shape[1]
	# #max_evals = 100

	# # hyper parameters
	# # batch_size = 64
	# # lr = 1e-3
	# # n_epochs = 150
	# # n_hidden = 30
	# n_lags = 30

	# n_classes = 2
	# recurrent = True
	# bandit_id = 1
	# re_train = False # True
	# show_graphs = False

	# # hyper parameters
	# #batch_size = 100
	# #lr = 1
	# #n_epochs = 1
	# #gamma = 0.9
	
	# train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
	# actuals_train, actuals_val, actuals_test = train_X[:, -1, 0], val_X[:, -1, 0], test_X[:, -1, 0]
	# nexts_train, nexts_val, nexts_test = train_y.copy(), val_y.copy(), test_y.copy()
	# train_y = utils.to_one_hot((train_X[:, -1, 0] <= train_y).astype(np.int32))
	# val_y = utils.to_one_hot((val_X[:, -1, 0] <= val_y).astype(np.int32))
	# test_y = utils.to_one_hot((test_X[:, -1, 0] <= test_y).astype(np.int32))

	# if(not recurrent):
	# 	train_X, val_X, test_X, _, _, _ = utils.split_data_without_lags(scaled, n_lags, n_features)
	# # recurrent
	# # no recurrent
	# #train_y = (train_X[:, -1, 0] <= train_y).astype(np.int32)
	# #val_y = (val_X[:, -1, 0] <= val_y).astype(np.int32)
	# #test_y = (test_X[:, -1, 0] <= test_y).astype(np.int32)

	# if(bandit_id == 0):
	# 	fit_models, arm_hist, true_reward_hist, regret_hist = neuralBandit.train_bandit_1(train_X, train_y, optimizer='adam', 
	# 																				loss='binary_crossentropy', 
	# 																				explore=.005, exp_annealing_rate=1)
	# else:
	# 	fit_models, arm_hist, true_reward_hist, regret_hist, weights, models_hist = neuralBandit.train_bandit_2(train_X, train_y,
	# 																				explore=0.1, exp_annealing_rate=0.99995, recurrent=recurrent)
	

	# # plt.figure()
	# # plt.plot(np.cumsum(regret_hist), label='actual regret')
	# # plt.plot(np.arange(len(regret_hist)), label='100 %% regret')
	# # plt.plot([0.0, len(regret_hist)], [0.0, len(regret_hist)/2.0], label='50 %% regret')
	# # plt.title('cumulative training regret')
	# # plt.xlabel('$t$')
	# # plt.ylabel('regret')
	# # plt.legend()

	# # print(pd.value_counts(true_reward_hist))
	# # print(pd.value_counts(arm_hist))

	# # plt.figure()
	# # sns.barplot(pd.value_counts(arm_hist).index, pd.value_counts(arm_hist).values)
	# # plt.title('chosen train arm distribution')
	# # plt.ylabel('count')
	# # plt.xlabel('arm')

	# # plt.figure()
	# # sns.barplot(pd.value_counts(true_reward_hist).index, pd.value_counts(true_reward_hist).values)
	# # plt.title('true train reward distribution')
	# # plt.ylabel('reward')
	# # plt.xlabel('arm')

	# # if('models_hist' in locals()):
	# # 	plt.figure()
	# # 	sns.barplot(pd.value_counts(models_hist).index, pd.value_counts(models_hist).values)
	# # 	plt.title('train model selection distribution')
	# # 	plt.ylabel('count')
	# # 	plt.xlabel('model')
	# # plt.show()


	# if(bandit_id == 1):
	# 	if(re_train):
	# 		preds_val, weights, fit_models, regret_hist_val, true_reward_hist_val, models_hist_val = neuralBandit.predict_and_train_with_weights(val_X, val_y, fit_models, weights, gamma=0.1, explore=0.2)

	# 		preds_test, weights, fit_models, regret_hist_test, true_reward_hist_test, models_hist_test = neuralBandit.predict_and_train_with_weights(test_X, test_y, fit_models, weights, gamma=0.1, explore=0.2)
	# 	else:
	# 		preds_val, regret_hist_val, true_reward_hist_val, models_hist_val = neuralBandit.predict_with_weights(val_X, val_y, fit_models, weights, explore=0.2)

	# 		preds_test, regret_hist_test, true_reward_hist_test, models_hist_test = neuralBandit.predict_with_weights(test_X, test_y, fit_models, weights, explore=0.2)
	# else:
	# 	if(re_train):
	# 		preds_val, fit_models, regret_hist_val, true_reward_hist_val = neuralBandit.predict_and_train(val_X, val_y, fit_models, explore=0.2)

	# 		preds_test, fit_models, regret_hist_test, true_reward_hist_test = neuralBandit.predict_and_train(test_X, test_y, fit_models, explore=0.2)
	# 	else:
	# 		preds_val, regret_hist_val, true_reward_hist_val = neuralBandit.predict(val_X, val_y, fit_models, explore=0.2)

	# 		preds_test, regret_hist_test, true_reward_hist_test = neuralBandit.predict(test_X, test_y, fit_models, explore=0.2)			

	# from sklearn.metrics import accuracy_score

	# print('\n\n train score: ', accuracy_score(arm_hist, true_reward_hist), end='\n\n')
	# print('\n\n val score: ', accuracy_score(preds_val, np.argmax(val_y, axis=1)), end='\n\n')
	# print('\n\n test score: ', accuracy_score(preds_test, np.argmax(test_y, axis=1)), end='\n\n')
	
	# if(show_graphs):
	# 	if('models_hist' in locals()):
	# 		utils.show_graphs(regret_hist=regret_hist, arm_hist=arm_hist, true_reward_hist=true_reward_hist, models_hist=models_hist, mode='training')
	# 	else:
	# 		utils.show_graphs(regret_hist=regret_hist, arm_hist=arm_hist, true_reward_hist=true_reward_hist, mode='training')

	# 	if('models_hist_val' in locals()):
	# 		utils.show_graphs(regret_hist=regret_hist_val, arm_hist=preds_val, true_reward_hist=true_reward_hist_val, models_hist=models_hist_val, mode='validation')
	# 	else:
	# 		utils.show_graphs(regret_hist=regret_hist_val, arm_hist=preds_val, true_reward_hist=true_reward_hist_val, mode='validation')

	# 	if('models_hist_test' in locals()):
	# 		utils.show_graphs(regret_hist=regret_hist_test, arm_hist=preds_test, true_reward_hist=true_reward_hist_test, models_hist=models_hist_test, mode='testing')
	# 	else:
	# 		utils.show_graphs(regret_hist=regret_hist_test, arm_hist=preds_test, true_reward_hist=true_reward_hist_test, mode='testing')

	# limit = -1
	# #acum_reward, historic_reward = utils.get_total_reward(actuals_val[:limit], nexts_val[:limit], preds_val[:limit])
	# acum_reward, historic_reward = utils.get_total_reward(actuals_test[:limit], nexts_test[:limit], preds_test[:limit])
	# plt.plot(historic_reward)
	# plt.show()
	# print('total reward: ', acum_reward)
	



	####################################### no weights or gamma project ###################################

	# df = pd.read_csv('data/datos_proyecto.csv', header=0)
	# scaled, scaler = utils.normalize_data(df.values)
	# #n_features = scaled.shape[1]

	# #n_lags = 30

	# #train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
	# #actuals_train, actuals_val, actuals_test = train_X[:, -1, 0], val_X[:, -1, 0], test_X[:, -1, 0]
	# #nexts_train, nexts_val, nexts_test = train_y.copy(), val_y.copy(), test_y.copy()
	# #train_y = (train_X[:, -1, 0] <= train_y).astype(np.int32)
	# #val_y = (val_X[:, -1, 0] <= val_y).astype(np.int32)
	# #test_y = (test_X[:, -1, 0] <= test_y).astype(np.int32)

	# import predictor

	# predictor_instance = predictor.Predictor('adam', 'mse', (n_lags, n_features), train_X, train_y, val_X, val_y, 0)

	# predictor_instance.predict_and_train(train_X, train_y)
	# preds_test, models_hist, regret_hist = predictor_instance.predict_and_train(test_X, test_y)

	# plt.figure()
	# plt.plot(np.cumsum(regret_hist), label='actual regret')
	# plt.plot(np.arange(len(regret_hist)), label='100 %% regret')
	# plt.plot([0.0, len(regret_hist)], [0.0, len(regret_hist)/2.0], label='50 %% regret')
	# plt.title('cumulative training regret')
	# plt.xlabel('$t$')
	# plt.ylabel('regret')
	# plt.legend()
	# plt.show()

	# sns.barplot(pd.value_counts(models_hist).index, pd.value_counts(models_hist).values)
	# plt.title('model selection distribution')
	# plt.ylabel('count')
	# plt.xlabel('model')
	# plt.show()

	# from sklearn.metrics import accuracy_score

	# print('\n\n test score: ', accuracy_score(preds_test, test_y), end='\n\n')

	# limit = -1
	# #acum_reward, historic_reward = utils.get_total_reward(actuals_val[:limit], nexts_val[:limit], preds_val[:limit])
	# acum_reward, historic_reward = utils.get_total_reward(actuals_test[:limit], nexts_test[:limit], preds_test[:limit])
	# plt.plot(historic_reward)
	# plt.show()
	# print('total reward: ', acum_reward)


	################################################# q-learning model ###################################################

	from q_learning_models import Model

	# df = pd.read_csv('data/datos_proyecto.csv', header=0)
	df = pd.read_csv('data/datos_liquidez_nuevos.csv', header=0, index_col=0)
	df = df[['Merval', 'Spread', 'Valor transado (promedio)', 'LIX']]
	# df = df[['Merval', 'Spread', 'Valor transado (suma)', 'LIX']]

	scaled, scaler = utils.normalize_data(df.values)

	n_features = scaled.shape[1]
	n_lags = 30
	lr= 1e-3
	n_hidden = 150
	epsilon = 0.1
	gamma = 0.9
	refresh_rate = 30
	n_classes = 2
	n_epochs = 1

	# train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data_without_lags(df.values, n_lags, n_features)
	# train_rewards = (train_y_inv - train_X_inv[:, -n_features]) / train_X_inv[:, -n_features]
	# val_rewards = (val_y_inv - val_X_inv[:, -n_features]) / val_X_inv[:, -n_features]
	# test_rewards = (test_y_inv - test_X_inv[:, -n_features]) / test_X_inv[:, -n_features]
	# train_rewards = np.array([-train_rewards, train_rewards]).T
	# val_rewards = np.array([-val_rewards, val_rewards]).T
	# test_rewards = np.array([-test_rewards, test_rewards]).T
	# train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data_without_lags(scaled, n_lags, n_features)

	train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data(df.values, n_lags, n_features)
	train_rewards = (train_y_inv - train_X_inv[:, -1, 0]) / train_X_inv[:, -1, 0]
	val_rewards = (val_y_inv - val_X_inv[:, -1, 0]) / val_X_inv[:, -1, 0]
	test_rewards = (test_y_inv - test_X_inv[:, -1, 0]) / test_X_inv[:, -1, 0]
	train_rewards = np.array([-train_rewards, train_rewards]).T
	val_rewards = np.array([-val_rewards, val_rewards]).T
	test_rewards = np.array([-test_rewards, test_rewards]).T
	train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)

	

	# rewards = []

	# for _ in range(10):
	# 	model = Model(n_features, n_lags, lr, n_hidden, refresh_rate, n_classes)

	# 	preds_train = model.run(train_X, train_y, train_rewards, epsilon, gamma, n_epochs)

	# 	preds_val = model.run(val_X, val_y, val_rewards, epsilon, gamma, 1)

	# 	preds_test = model.run(test_X, test_y, test_rewards, epsilon, gamma, 1)

	# 	#print(preds_train)

	# 	#acum_reward_train, historic_reward_train = utils.get_total_reward(train_X_inv[:, -n_features], train_y_inv, preds_train)
	# 	#acum_reward_val, historic_reward_val = utils.get_total_reward(val_X_inv[:, -n_features], val_y_inv, preds_val)
	# 	#acum_reward_test, historic_reward_test = utils.get_total_reward(test_X_inv[:, -n_features], test_y_inv, preds_test)
	# 	acum_reward_train, historic_reward_train = utils.get_total_reward(train_X_inv[:, -1, 0], train_y_inv, preds_train)
	# 	acum_reward_val, historic_reward_val = utils.get_total_reward(val_X_inv[:, -1, 0], val_y_inv, preds_val)
	# 	acum_reward_test, historic_reward_test = utils.get_total_reward(test_X_inv[:, -1, 0], test_y_inv, preds_test)

	# 	rewards.append(acum_reward_test)

	# print(rewards)
	# print(np.mean(rewards))


	total_X = np.append(val_X, test_X, axis=0)
	total_y = np.append(val_y, test_y, axis=0)
	total_rewards = np.append(val_rewards, test_rewards, axis=0)
	total_X_inv = np.append(val_X_inv, test_X_inv, axis=0)
	total_y_inv = np.append(val_y_inv, test_y_inv, axis=0)
	
	# total_X = np.append(np.append(train_X, val_X, axis=0), test_X, axis=0)
	# total_y = np.append(np.append(train_y, val_y, axis=0), test_y, axis=0)
	# total_rewards = np.append(np.append(train_rewards, val_rewards, axis=0), test_rewards, axis=0)
	# total_X_inv = np.append(np.append(train_X_inv, val_X_inv, axis=0), test_X_inv, axis=0)
	# total_y_inv = np.append(np.append(train_y_inv, val_y_inv, axis=0), test_y_inv, axis=0)

	# model = Model(n_features*n_lags, n_lags, lr, n_hidden, refresh_rate, n_classes)
	model = Model(n_features, n_lags, lr, n_hidden, refresh_rate, n_classes)



	# preds_train = model.run(train_X, train_y, train_rewards, epsilon, gamma, n_epochs)

	# preds_val = model.run(val_X, val_y, val_rewards, epsilon, gamma, 1)

	# preds_test = model.run(test_X, test_y, test_rewards, epsilon, gamma, 1)

	preds_train = model.run(train_X, train_y, train_rewards, epsilon, gamma, 1)

	preds_total = model.run(total_X, total_y, total_rewards, epsilon, gamma, n_epochs)

	#print(preds_train)

	#acum_reward_train, historic_reward_train = utils.get_total_reward(train_X_inv[:, -n_features], train_y_inv, preds_train)
	#acum_reward_val, historic_reward_val = utils.get_total_reward(val_X_inv[:, -n_features], val_y_inv, preds_val)
	#acum_reward_test, historic_reward_test = utils.get_total_reward(test_X_inv[:, -n_features], test_y_inv, preds_test)
	
	#acum_reward_train, historic_reward_train = utils.get_total_reward(train_X_inv[:, -1, 0], train_y_inv, preds_train)
	#acum_reward_val, historic_reward_val = utils.get_total_reward(val_X_inv[:, -1, 0], val_y_inv, preds_val)
	#acum_reward_test, historic_reward_test = utils.get_total_reward(test_X_inv[:, -1, 0], test_y_inv, preds_test)
	
	acum_reward_total, historic_reward_total = utils.get_total_reward(total_X_inv[:, -1, 0], total_y_inv, preds_total)
	
	#print('training final reward: ', acum_reward_train)
	#print('validation final reward: ', acum_reward_val)
	#print('test final reward: ', acum_reward_test)
	print('total final reward: ', acum_reward_total)
	#print('signals in test: ', utils.count_signals(preds_test))
	print('signals in total: ', utils.count_signals(preds_total))

	from matplotlib import pyplot as plt

	#plt.subplot(2,1,1)
	#plt.plot(historic_reward_val)
	#plt.suptitle('validation results')

	#plt.subplot(2,1,2)
	#plt.plot(historic_reward_test)
	#plt.suptitle('testing results')

	#plt.show()

	plt.plot(historic_reward_total)
	plt.show()



