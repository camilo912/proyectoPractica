import numpy as np
import pandas as pd
import csv
import modelos

from sklearn.preprocessing import MinMaxScaler

from timeit import default_timer as timer

def inverse_transform(scaler, data, n_features):
	data = data.copy()
	assert type(data) == np.ndarray
	if(data.ndim == 1): data = data.reshape(-1, 1)
	assert data.ndim == 2
	for i in range(data.shape[1]):
		tmp = np.zeros((data.shape[0], n_features))
		tmp[:, 0] = data[:, i]
		data[:, i] = scaler.inverse_transform(tmp)[:, 0]
	return data

def to_one_hot(data):
	data = np.array(data).ravel()
	one_hot = np.zeros((len(data), len(np.unique(data))))
	one_hot[np.arange(len(data)), data] = 1
	return one_hot

def normalize_data(data, scale=(0,1)):
	scaler = MinMaxScaler(feature_range=scale)
	scaled = scaler.fit_transform(data)
	return scaled, scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def split_data(values, n_lags, n_features):
	reframed = series_to_supervised(values, n_lags, 1).values
	wall = int(reframed.shape[0]*0.6)
	wall_val = int(reframed.shape[0]*0.2)
	n_obs = n_features*n_lags

	train_X, train_y = reframed[:wall, :n_obs].reshape(-1, n_lags, n_features), reframed[:wall, -n_features]
	val_X, val_y = reframed[wall:wall+wall_val, :n_obs].reshape(-1, n_lags, n_features), reframed[wall:wall+wall_val, -n_features]
	test_X, test_y = reframed[wall+wall_val:, :n_obs].reshape(-1, n_lags, n_features), reframed[wall+wall_val:, -n_features]

	return train_X, val_X, test_X, train_y, val_y, test_y

def split_data_without_lags(values, n_lags, n_features):
	reframed = series_to_supervised(values, n_lags, 1).values
	wall = int(reframed.shape[0]*0.6)
	wall_val = int(reframed.shape[0]*0.2)
	n_obs = n_features*n_lags

	train_X, train_y = reframed[:wall, :n_obs].reshape(-1, n_lags*n_features), reframed[:wall, -n_features]
	val_X, val_y = reframed[wall:wall+wall_val, :n_obs].reshape(-1, n_lags*n_features), reframed[wall:wall+wall_val, -n_features]
	test_X, test_y = reframed[wall+wall_val:, :n_obs].reshape(-1, n_lags*n_features), reframed[wall+wall_val:, -n_features]

	return train_X, val_X, test_X, train_y, val_y, test_y

def objective(params, scaled, n_features, scaler):
	from hyperopt import STATUS_OK

	global ITERATION
	ITERATION += 1
	print(ITERATION, params)

	# Make sure parameters that need to be integers are integers
	for parameter_name in ['n_hidden', 'batch_size', 'n_epochs', 'n_lags']:
		params[parameter_name] = int(params[parameter_name])

	# Make sure parameters that need to be float are float
	for parameter_name in ['lr']:
		params[parameter_name] = float(params[parameter_name])
	
	out_file = 'trials/gbm_trials.csv'

	# hyper parameters
	batch_size = params['batch_size']
	lr = params['lr']
	n_epochs = params['n_epochs']
	n_hidden = params['n_hidden']
	n_lags = params['n_lags']


	start = timer()

	train_X, val_X, test_X, train_y, val_y, test_y = split_data(scaled, n_lags, n_features)

	model = modelos.Model_predictor(lr, n_hidden, n_lags, n_features, scaler)
	model.train(train_X, val_X, train_y, val_y, batch_size, n_epochs)
	rmse, _, _ = model.eval(test_X, test_y)

	run_time = timer() - start

	print(rmse, run_time, end='\n\n\n')

	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([rmse, params, ITERATION, run_time])
	of_connection.close()

	# Dictionary with information for evaluation
	return {'loss': rmse, 'params': params, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}

def bayes_optimization(MAX_EVALS, scaled, n_features, scaler):
	from hyperopt import fmin
	from hyperopt import tpe
	from hyperopt import Trials
	from hyperopt import hp

	global ITERATION
	ITERATION = 0

	# space
	space = {'batch_size': hp.quniform('batch_size', 2, 150, 1),
			'lr': hp.uniform('lr', 0.00001, 1.0),
			'n_epochs': hp.quniform('n_epochs', 5, 500, 1),
			'n_hidden': hp.quniform('n_hidden', 5, 200, 1),
			'n_lags': hp.quniform('n_lags', 2, 50, 1)}

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
	best = fmin(fn = lambda x: objective(x, scaled, n_features, scaler), space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

	# store best results
	of_connection = open('trials/bests.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	writer.writerow([bayes_trials_results[0]['loss'], bayes_trials_results[0]['params']['n_hidden'], bayes_trials_results[0]['params']['batch_size'], bayes_trials_results[0]['params']['n_epochs'], bayes_trials_results[0]['params']['lr'], MAX_EVALS])
	of_connection.close()

	return best