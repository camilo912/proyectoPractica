import numpy as np
import pandas as pd
import modelos
import utils

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from timeit import default_timer as timer


if __name__ == '__main__':
	df = pd.read_csv('data/datos_proyecto.csv', header=0)
	scaled, scaler = utils.normalize_data(df.values)
	n_features = scaled.shape[1]
	MAX_EVALS = 100

	# hyper parameters
	batch_size = 32
	lr = 1e-3
	n_epochs = 300
	n_hidden = 30
	n_lags = 3

	#best = utils.bayes_optimization(MAX_EVALS, scaled, n_features, scaler, 0)
	#batch_size, lr, n_epochs, n_hidden, n_lags = int(best['batch_size']), best['lr'], int(best['n_epochs']), int(best['n_hidden']), int(best['n_lags'])

	train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)

	# start = timer()

	# model = modelos.Model_predictor(lr, n_hidden, n_lags, n_features, scaler)
	# model.train(train_X, val_X, train_y, val_y, batch_size, n_epochs)

	# rmse_train, y_hat_train, y_trainset = model.eval(train_X, train_y)
	# rmse_val, y_hat_val, y_valset = model.eval(val_X, val_y)
	# rmse, y_hat, y = model.eval(test_X, test_y)

	# run_time = timer() - start
	
	# # save the model
	# model.save()
	# del model

	# # print results
	# print('rmse: %f' % rmse, 'run_time: %f' % run_time)

	# # plot results
	# plt.plot(y_hat, label='predictions')
	# plt.plot(y, label='observations')
	# plt.suptitle('Predictions vs observations')
	# plt.legend()
	# plt.show()

	# ########################################### Q-learning Model #################################################################

	from keras.models import load_model

	model = load_model('model_predictor.h5')

	train_X_inv, val_X_inv, test_X_inv, train_y_inv, val_y_inv, test_y_inv = utils.split_data(df.values, n_lags, n_features)
	
	#train_X_inv = np.array([[scaler.inverse_transform(train_X[j, i, :].reshape(1, -1)).ravel() for i in range(n_lags)] for j in range(len(train_X))])
	#val_X_inv = np.array([[scaler.inverse_transform(val_X[j, i, :].reshape(1, -1)).ravel() for i in range(n_lags)] for j in range(len(val_X))])
	#test_X_inv = np.array([[scaler.inverse_transform(test_X[j, i, :].reshape(1, -1)).ravel() for i in range(n_lags)] for j in range(len(test_X))])

	# print(train_X[0])
	# print(val_X.shape)
	# print(test_X.shape)



	train_X = np.append(train_X[:, :, 0], model.predict(train_X).reshape(-1, 1), axis=1)
	observations = train_y
	train_y = train_X[:, -2] < train_y
	val_X = np.append(val_X[:, :, 0], model.predict(val_X).reshape(-1, 1), axis=1)
	val_y = val_X[:, -2] < val_y
	test_X = np.append(test_X[:, :, 0], model.predict(test_X).reshape(-1, 1), axis=1)
	test_y = test_X[:, -2] < test_y
	variations = train_y_inv - train_X_inv[:, -1, 0]

	train_y, val_y, test_y = train_y.astype(np.int32), val_y.astype(np.int32), test_y.astype(np.int32)

	# print(train_X[:4])
	# print(train_y[:4])
	# print(train_X.shape)
	# print(train_y.shape)

	# raise Exception('Debug')

	n_classes = 2

	# hyper parameters
	batch_size = 5
	lr = 0.5
	n_epochs = 2
	gamma = 0.9
	MAX_EVALS = 100

	#best = utils.bayes_optimization(MAX_EVALS, scaled, n_features, scaler, 1, train_X=train_X, val_X=val_X, test_X=test_X, train_y=train_y, val_y=val_y, test_y=test_y, n_classes=n_classes, variations=variations)
	#batch_size, lr, n_epochs, gamma = int(best['batch_size']), best['lr'], int(best['n_epochs']), best['gamma']

	start = timer()
	model2 = modelos.Model_decisor(lr, n_features, scaler, n_classes, gamma)
	model2.train(train_X, val_X, train_y, val_y, batch_size, n_epochs, variations, observations)
	# acc_val, y_hat_val, y_val = model2.eval(val_X, val_y)
	acc, y_hat, y, history = model2.eval(test_X, test_y)

	plt.plot(history)
	plt.show()

	run_time = timer() - start

	# print results
	print('acc: %f' % acc, 'run_time: %f' % run_time)