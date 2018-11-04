import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils

def execute_model(scaled, n_features, scaler, batch_size, lambda_term, loss, lr, n_epochs, n_hidden, n_lags, show):
	train_X, val_X, test_X, train_y, val_y, test_y = utils.split_data(scaled, n_lags, n_features)
	train_y = np.array(train_y > train_X[:, -1, 0]).astype(np.int32)
	val_y = np.array(val_y > val_X[:, -1, 0]).astype(np.int32)
	test_y = np.array(test_y > test_X[:, -1, 0]).astype(np.int32)


	from keras.models import Sequential
	from keras.layers import LSTM, Dense
	from keras import regularizers

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_term)))

	model.compile(optimizer='adam', loss=loss)

	history = model.fit(train_X, train_y, validation_data=(val_X, val_y), verbose=0, batch_size=batch_size, epochs=n_epochs)

	preds = model.predict(test_X)
	preds_classes = (preds  > 0.5).astype(np.int32)
	
	from sklearn.metrics import accuracy_score

	acc = accuracy_score(test_y, preds_classes)
	
	if(show):
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.show()

		plt.plot(test_y)
		plt.plot(preds)
		plt.show()

		print('test accuracy score: ', acc)

	return acc

def objective(params, scaled, n_features, scaler, losses):
	from hyperopt import STATUS_OK
	import time
	import csv

	global ITERATION
	ITERATION += 1
	print(ITERATION, params)

	# Make sure parameters that need to be integers are integers
	for parameter_name in ['n_hidden', 'batch_size', 'n_epochs', 'n_lags', 'loss_idx']:
		params[parameter_name] = int(params[parameter_name])

	# Make sure parameters that need to be float are float
	for parameter_name in ['lr', 'lambda_term']:
		params[parameter_name] = float(params[parameter_name])
	
	out_file = 'trials/gbm_trials.csv'

	# hyper parameters
	batch_size = params['batch_size']
	lambda_term = params['lambda_term']
	loss = losses[params['loss_idx']]
	lr = params['lr']
	n_epochs = params['n_epochs']
	n_hidden = params['n_hidden']
	n_lags = params['n_lags']


	start = time.time()

	acc = execute_model(scaled, n_features, scaler, batch_size, lambda_term, loss, lr, n_epochs, n_hidden, n_lags, 0)

	error = 1 - acc

	run_time = time.time() - start

	print('error: ', error, 'time elapsed: ', run_time, end='\n\n\n')

	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([error, params, ITERATION, run_time])
	of_connection.close()

	# Dictionary with information for evaluation
	return {'loss': error, 'params': params, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}


def bayes_optimization(max_evals, scaled, n_features, scaler, losses):
	from hyperopt import fmin
	from hyperopt import tpe
	from hyperopt import Trials
	from hyperopt import hp
	import csv

	global ITERATION
	ITERATION = 0

	# space
	space = {'batch_size': hp.quniform('batch_size', 1, 500, 1),
			'lambda_term': hp.uniform('lambda_term', 0.001, 0.5),
			'loss_idx': hp.quniform('loss_idx', 0, 1, 1),
			'lr': hp.uniform('lr', 0.001, 1.0),
			'n_epochs': hp.quniform('n_epochs', 50, 150, 1),
			'n_hidden': hp.quniform('n_hidden', 5, 500, 1),
			'n_lags': hp.quniform('n_lags', 2, 30, 1)}

	# Keep track of results
	bayes_trials = Trials()

	# File to save first results
	out_file = 'trials/gbm_trials.csv'
	of_connection = open(out_file, 'w')
	writer = csv.writer(of_connection)

	# Write the headers to the file
	writer.writerow(['no-score', 'params', 'iteration', 'train_time'])
	of_connection.close()

	# Run optimization
	best = fmin(fn = lambda x: objective(x, scaled, n_features, scaler, losses), space = space, algo = tpe.suggest, max_evals = max_evals, trials = bayes_trials, rstate = np.random.RandomState(0))

	# store best results
	of_connection = open('trials/bests.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	writer.writerow([bayes_trials_results[0]['loss'], int(best['batch_size']), best['lambda_term'], int(best['loss_idx']), best['lr'], int(best['n_epochs']), int(best['n_hidden']), int(best['n_lags']), max_evals])
	of_connection.close()

	return best

def run(data):
	scaled, scaler = utils.normalize_data(df.values)
	n_features = scaled.shape[1]

	losses = ['mse', 'binary_crossentropy']
	
	# hyper parameters
	batch_size = 100
	lambda_term = 0.01
	loss_idx = 0
	lr = 1e-3
	n_epochs = 300
	n_hidden = 500
	n_lags = 10

	max_evals = 100
	best = bayes_optimization(max_evals, scaled, n_features, scaler, losses)
	batch_size, lambda_term, loss_idx, lr, n_epochs, n_hidden, n_lags  = int(best['batch_size']), best['lambda_term'], int(best['loss_idx']), best['lr'], int(best['n_epochs']), int(best['n_hidden']), int(best['n_lags'])

	execute_model(scaled, n_features, scaler, batch_size, lambda_term, losses[loss_idx], lr, n_epochs, n_hidden, n_lags, 1)


if __name__ == '__main__':
	df = pd.read_csv('data/datos_proyecto.csv', header=0)
	run(df.values)