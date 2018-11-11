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

def split_data(values, n_lags, n_features, n_out=1):
	reframed = series_to_supervised(values, n_lags, n_out).values
	wall = int(reframed.shape[0]*0.6)
	wall_val = int(reframed.shape[0]*0.2)
	n_obs = n_features*n_lags

	train_X, train_y = reframed[:wall, :n_obs].reshape(-1, n_lags, n_features), reframed[:wall, -n_features] # [-n_features*i for i in range(1, n_out+1)]
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

def show_graphs(regret_hist=None, true_reward_hist=None, arm_hist=None, models_hist=None, mode='train'):
	from matplotlib import pyplot as plt
	import seaborn as sns

	if(regret_hist):
		plt.figure()
		plt.plot(np.cumsum(regret_hist), label='actual ' + mode + ' regret')
		plt.plot(np.arange(len(regret_hist)), label='100 %% regret')
		plt.plot([0.0, len(regret_hist)], [0.0, len(regret_hist)/2.0], label='50 %% regret')
		plt.title('cumulative ' + mode + ' regret')
		plt.xlabel('$t$')
		plt.ylabel('regret')
		plt.legend()

	if(arm_hist):
		print(pd.value_counts(arm_hist))
		plt.figure()
		sns.barplot(pd.value_counts(arm_hist).index, pd.value_counts(arm_hist).values)
		plt.title('chosen ' + mode + ' arm distribution')
		plt.ylabel('count')
		plt.xlabel('arm')

	if(true_reward_hist):
		print(pd.value_counts(true_reward_hist))
		plt.figure()
		sns.barplot(pd.value_counts(true_reward_hist).index, pd.value_counts(true_reward_hist).values)
		plt.title('true ' + mode + ' reward distribution')
		plt.ylabel('reward')
		plt.xlabel('arm')

	if(models_hist):
		plt.figure()
		sns.barplot(pd.value_counts(models_hist).index, pd.value_counts(models_hist).values)
		plt.title(mode + ' model selection distribution')
		plt.ylabel('count')
		plt.xlabel('model')
	
	plt.show()

def get_total_reward(actuals, nexts, preds):
	variations = (nexts - actuals)/actuals

	reward = 1
	historic_reward = [1]

	for i in range(len(preds)):
		if(preds[i]):
			reward += reward*(variations[i])
		else:
			reward -= reward*(variations[i])
		historic_reward.append(reward)

	return reward, historic_reward

def count_signals(serie):
	cont = 0
	for i in range(len(serie)-1):
		if(serie[i] != serie[i+1]): cont += 1

	return cont

def get_rewards(X, y):
	return (y - X[:, -1, 0]) / X[:, -1, 0]

def get_buy_and_hold_reward(rewards):
	rode = 1
	historic_rode = []
	for i in range(len(rewards)):
		rode = rode*(1+rewards[i])
		historic_rode.append(rode)

	return rode, historic_rode

class SumTree(object):
    """
    This SumTree code is copied from: 
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node





class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This class is copied from:
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)