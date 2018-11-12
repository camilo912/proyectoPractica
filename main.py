import numpy as np
import pandas as pd
import modelos
import utils
import time

from matplotlib import pyplot as plt


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

	# rmse_train, y_hat_train, y_trainset = model.eval(train_X, train_y)
	# rmse_val, y_hat_val, y_valset = model.eval(val_X, val_y)
	# rmse, y_hat, y = model.eval(test_X, test_y)

	# run_time = time.time() - start
	
	# # print results
	# print('rmse: %f' % rmse, 'run_time: %f' % run_time)

	# # plot results
	# plt.plot(y, label='observations')
	# plt.plot(y_hat, label='predictions')
	# plt.suptitle('Predictions vs observations')
	# plt.legend()
	# plt.show()




	################################################# q-learning model ###################################################

	from q_learning_models import Model

	df = pd.read_csv('data/datos_liquidez_nuevos.csv', header=0, index_col=0)#, nrows=300)
	df = df[['Merval', 'Spread', 'Valor transado (promedio)', 'LIX']]
	# df = df.loc[:1250, :]

	scaled, scaler = utils.normalize_data(df.values)

	n_features = scaled.shape[1]
	n_lags = 15
	lr = 1e-4
	n_hidden = 150
	min_explore = 0.1
	init_explore = 1.0
	gamma = 0.9
	refresh_rate = 20
	n_classes = 2
	n_epochs = 120
	decay_rate_explore = 0.05

	train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data(df.values, n_lags, n_features)
	train_rewards = (train_y_inv - train_X_inv[:, -1, 0]) / train_X_inv[:, -1, 0]
	val_rewards = (val_y_inv - val_X_inv[:, -1, 0]) / val_X_inv[:, -1, 0]
	test_rewards = (test_y_inv - test_X_inv[:, -1, 0]) / test_X_inv[:, -1, 0]
	train_rewards = np.array([-train_rewards, train_rewards]).T
	val_rewards = np.array([-val_rewards, val_rewards]).T
	test_rewards = np.array([-test_rewards, test_rewards]).T
	train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)


	total_X = np.append(val_X, test_X, axis=0)
	total_y = np.append(val_y, test_y, axis=0)
	total_rewards = np.append(val_rewards, test_rewards, axis=0)
	total_X_inv = np.append(val_X_inv, test_X_inv, axis=0)
	total_y_inv = np.append(val_y_inv, test_y_inv, axis=0)

	model = Model(n_features, n_lags, lr, n_hidden, refresh_rate, n_classes)

	preds_train = model.run(train_X, train_y, train_rewards, gamma, n_epochs, True, init_explore, min_explore, decay_rate_explore)

	preds_total = model.run(total_X, total_y, total_rewards, gamma, 1, False, init_explore, min_explore, decay_rate_explore)
	
	acum_reward_total, historic_reward_total = utils.get_total_reward(total_X_inv[:, -1, 0], total_y_inv, preds_total)
	acum_reward_buy_and_hold_total, historic_buy_and_hold_reward_total = utils.get_buy_and_hold_reward(total_rewards[:, 1])
	
	print('total final reward: ', acum_reward_total)
	print('total buy and hold reward: ', acum_reward_buy_and_hold_total)
	print('signals in total: ', utils.count_signals(preds_total))

	from matplotlib import pyplot as plt

	plt.plot(historic_reward_total, label='algorithm reward')
	plt.plot(historic_buy_and_hold_reward_total, label='buy and hold reward')
	plt.legend()
	plt.show()



