import numpy as np
import pandas as pd
import modelos
import utils
import neuralBandit

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
from timeit import default_timer as timer


if __name__ == '__main__':
	df = pd.read_csv('data/datos_proyecto.csv', header=0)
	scaled, scaler = utils.normalize_data(df.values)
	n_features = scaled.shape[1]
	MAX_EVALS = 100

	# hyper parameters
	batch_size = 64
	lr = 1e-3
	n_epochs = 150
	n_hidden = 30
	n_lags = 30

	#best = utils.bayes_optimization(MAX_EVALS, scaled, n_features, scaler)
	#batch_size, lr, n_epochs, n_hidden, n_lags = int(best['batch_size']), best['lr'], int(best['n_epochs']), int(best['n_hidden']), int(best['n_lags'])

	# train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)

	# start = timer()

	# model = modelos.Model_predictor(lr, n_hidden, n_lags, n_features, scaler)
	# model.train(train_X, val_X, train_y, val_y, batch_size, n_epochs)

	# rmse_train, y_hat_train, y_trainset = model.eval(train_X, train_y)
	# rmse_val, y_hat_val, y_valset = model.eval(val_X, val_y)
	# rmse, y_hat, y = model.eval(test_X, test_y)

	# run_time = timer() - start
	
	# # print results
	# print('rmse: %f' % rmse, 'run_time: %f' % run_time)

	# # # plot results
	# # plt.plot(y_hat, label='predictions')
	# # plt.plot(y, label='observations')
	# # plt.suptitle('Predictions vs observations')
	# # plt.legend()
	# # plt.show()

	# # ########################################### Q-learning Model #################################################################

	# train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data(df.values, n_lags, n_features)
	
	# variations = train_y_inv - train_X_inv[:, -1, 0]
	# variations_val = val_y_inv - val_X_inv[:, -1, 0]
	# variations_test = test_y_inv - test_X_inv[:, -1, 0]
	
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

	# # print(train_X[:4])
	# # print(train_y[:4])
	# # print(train_X.shape)
	# # print(train_y.shape)

	# # raise Exception('Debug')

	# n_classes = 2

	# # hyper parameters
	# #batch_size = 100
	# #lr = 1
	# #n_epochs = 1
	# #gamma = 0.9

	# # fit_models, arm_hist, true_reward_hist, regret_hist = neuralBandit.run_bandit_1(train_X, train_y, optimizer='adam', 
	# # 																			loss='binary_crossentropy', 
	# # 																			explore=.005, exp_annealing_rate=1)
	train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
	train_y = (train_X[:, -1, 0] <= train_y).astype(np.int32)
	val_y = (val_X[:, -1, 0] <= val_y).astype(np.int32)
	test_y = (test_X[:, -1, 0] <= test_y).astype(np.int32)
	#train_y = utils.to_one_hot((train_X[:, -1, 0] <= train_y).astype(np.int32))
	#val_y = utils.to_one_hot((val_X[:, -1, 0] <= val_y).astype(np.int32))
	#test_y = utils.to_one_hot((test_X[:, -1, 0] <= test_y).astype(np.int32))
	#train_X, val_X, test_X, _, _, _ = utils.split_data_without_lags(scaled, n_lags, n_features)

	#fit_models, arm_hist, true_reward_hist, regret_hist, weights, models_hist = neuralBandit.run_bandit_2(train_X, train_y,
	#																			explore=0.1, exp_annealing_rate=0.99995, recurrent=True)

	import predictor

	predictor_instance = predictor.Predictor('adam', 'mse', (n_lags, n_features), train_X, train_y, val_X, val_y, 0)

	_, weights, _, _ = predictor_instance.predict(val_X, val_y, gamma=0.1)
	preds_test, weights, models_hist, regret_hist = predictor_instance.predict(test_X, test_y, weights=weights)

	plt.plot(np.cumsum(regret_hist))
	plt.title('cumulative regret')
	plt.xlabel('$t$')
	plt.ylabel('regret')
	plt.show()

	sns.barplot(pd.value_counts(models_hist).index, pd.value_counts(models_hist).values)
	plt.title('model selection distribution')
	plt.ylabel('count')
	plt.xlabel('model')
	plt.show()

	# start = timer()
	# model2 = modelos.Model_decisor(lr, n_features, scaler, n_classes, gamma)
	# model2.train(train_X, val_X, train_y, val_y, batch_size, n_epochs, variations)
	# acc, y_hat, y = model2.eval(val_X, val_y)

	# run_time = timer() - start
	
	# # print results
	# print('acc: %f' % acc, 'run_time: %f' % run_time)

	from sklearn.metrics import accuracy_score

	#print('\n\n train score: ', accuracy_score(arm_hist, true_reward_hist), end='\n\n')

	#preds_val = neuralBandit.predict_with_models(val_X, fit_models, weights)

	#print('\n\n val score: ', accuracy_score(preds_val, np.argmax(val_y, axis=1)), end='\n\n')

	#preds_test = neuralBandit.predict_with_models(test_X, fit_models, weights)

	print('\n\n test score: ', accuracy_score(preds_test, test_y), end='\n\n')

