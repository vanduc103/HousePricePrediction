from random import seed, random, randrange
from math import exp
from csv import reader
import numpy as np
import time
import sys

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Preprocessing
from sklearn.metrics import mean_squared_error as MSE


# Rescale dataset to [0-1]
def rescale_dataset(dataset):
	dataset_normed = dataset / dataset.max()
	return dataset_normed

# Accuracy metric
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 

class MLP_Classifier(object):

  # Evaluate an algorithm using a cross validation split
  def evaluate_algorithm(self, dataset, target, algorithm, n_epochs, *args):
        # split training and validation set
	accuracy_max = 0.0
        for epoch in range(n_epochs):
		print('Epoch %d' %(epoch+1))
		# split training and validation set
	        X_train, X_validate, y_train, y_validate \
               		 = train_test_split(dataset, target, test_size=0.2, random_state=np.random.randint(1,128))
		X_train = rescale_dataset(X_train)
		X_validate = rescale_dataset(X_validate)

                y_pred_validate = algorithm(X_train, y_train, X_validate, *args)
                accuracy = accuracy_metric(y_validate, y_pred_validate)
                print("Validate accuracy %.3f\n" % (accuracy))
		if (accuracy > accuracy_max):
			accuracy_max = accuracy
			np.savetxt('weights_wi.out1', self.wi, fmt='%.5f')
			np.savetxt('weights_wo.out1', self.wo, fmt='%.5f')
	print('Max accuracy: %.3f' %accuracy_max)

  # Initialize a network
  def __init__(self, n_input, n_hidden, n_output):
	self.input = n_input
	self.hidden = n_hidden
	self.output = n_output
	self.action = 'show'

  # Initialize weights
  def init_weights(self, test=False):
	# set up array of 1s for activations
	self.ai = [1.0] * self.input
	self.ah = [1.0] * self.hidden
	self.ao = [1.0] * self.output

	# create randomized weights
	self.wi = np.random.rand(self.input, self.hidden)
	self.wo = np.random.rand(self.hidden, self.output)
	if test == True or self.action == 'show':
	  self.wi = np.loadtxt('weights_wi.out1')
	  self.wo = np.loadtxt('weights_wo.out1')

	# create zeros biases
	self.bh = np.zeros(self.hidden)
	self.bo = np.zeros(self.output)

	# create arrays of 0 for changes
	self.ci = np.zeros((self.input, self.hidden))
	self.co = np.zeros((self.hidden, self.output))


  # Transfer neuron activation
  def sigmoid(self, activation):
	return 1.0 / (1.0 + np.exp(-activation))

  # Calculate the derivative of an neuron output
  def sigmoid_derivative(self, output):
	return output * (1.0 - output)

  # ReLU transfer function
  def relu(self, activation):
	return np.maximum(activation, 0)

  def relu_derivative(self, output):
	return (output > 0).astype(float)

  # Tanh function
  def tanh(self, z):
	return np.tanh(z)

  def tanh_derivative(self, output):
	return 1.0 - (output ** 2)

  # Forward propagate input to a network output
  def feedForward(self, inputs):
	# input activations
        self.ai = inputs

	# hidden activations
	activation = np.dot(self.ai, self.wi)
        self.ah = self.sigmoid(activation)

	# output activations
	activation = np.dot(self.ah, self.wo)
        self.ao = self.sigmoid(activation)

	return self.ao[:]

  # Backpropagation
  def backPropagate(self, targets, N):
    """
    :param targets: y values
    :param N: learning rate
    :return: updated weights and current error
    """

    # calculate error terms for output
    # the delta tell you which direction to change the weights
    output_deltas = np.zeros(self.output)
    error_out = -(targets - self.ao)
    output_deltas = self.sigmoid_derivative(self.ao) * error_out

    # calculate error terms for hidden
    # delta tells you which direction to change the weights
    hidden_deltas = np.zeros(self.hidden)
    error_hidden = np.dot(self.wo, output_deltas)
    hidden_deltas = self.sigmoid_derivative(self.ah) * error_hidden

    # update the weights connecting hidden to output
    change = np.dot(self.ah.reshape(self.hidden,1), output_deltas.reshape(1,self.output))
    self.wo -= N*change + self.co
    self.co = change

    # update the weights connecting input to hidden
    change = np.dot(self.ai.reshape(self.input, 1), hidden_deltas.reshape(1,self.hidden))
    self.wi -= N*change + self.ci
    self.ci = change

    # calculate error
    error = 0.5 * np.sum((targets - self.ao) ** 2)
    return error


  # Train a network for a fixed number of epochs
  def train(self, x_train, y_train, iterations = 1000, N = 0.001):
    # N: learning rate
    dataset_length = x_train.shape[0]
    batch_size = 100
    rng = np.random.RandomState(128)
    for i in range(iterations):
        error = 0.0
        batch_mask = rng.choice(dataset_length, batch_size)
	batch_x = x_train[[batch_mask]]
	batch_y = y_train[[batch_mask]]
	for j in range(batch_x.shape[0]):
            inputs = x_train[j]
            targets = np.zeros(2)
	    targets[int(y_train[j])] = 1
            self.feedForward(inputs)
            error = self.backPropagate(targets, N)
        if i % 500 == 0:
            print('iteration %d, error %-.5f' % (i+1, error))


  # Make a prediction with a network
  def predict(self, X):
    """
    return list of predictions after training algorithm
    """
    predictions = []
    for p in X:
	outputs = self.feedForward(p).tolist()
        predictions.append(outputs.index(max(outputs)))
    return predictions


  # Backpropagation Algorithm With Stochastic Gradient Descent
  def back_propagation(self, x_train, y_train, x_validate):
	if self.action == 'train':
		self.train(x_train, y_train)
	predictions = self.predict(x_validate)
	return(predictions)


# pre-processing for testing data
def preprocessing(df_data, cols_to_remove, categorical_features):
  print('Testing set shape: (%d,%d)' %df_data.shape)

  # remove cols as in training data
  for col in cols_to_remove:
    df_data.drop(col, axis=1, inplace=True)

  print("New shape of the test set: (%d,%d)" %df_data.shape)
  print("The removed columns are:")
  for column in cols_to_remove:
    print(column)

  null_values_per_col = np.sum(df_data.drop(["Id","SalePrice"], axis=1).isnull(), axis=0)
  print(r"There are %d"  %np.sum(null_values_per_col != 0)  + " columns to impute.\n")

  cols_to_impute = []
  for col_index, val in null_values_per_col.iteritems():
    if val != 0: cols_to_impute.append(col_index)

  # Imputing missing values
  imputation_val_for_na_cols = dict()
  for col in cols_to_impute:
    if (df_data[col].dtype == 'float64' ) or  (df_data[col].dtype == 'int64'):
        imputation_val_for_na_cols[col] = np.nanmedian(df_data[col])
    else:
        imputation_val_for_na_cols[col] = df_data[col].value_counts().argmax()

  for key, val in imputation_val_for_na_cols.iteritems():
    df_data[key].fillna(value= val, inplace = True)

  print("Checking if everything went well ...")
  print("Number of missing values in data set after imputation and cleaning ",
          np.sum(np.sum(df_data.isnull())))

  del imputation_val_for_na_cols, cols_to_impute

  # Prepare data for testing
  X, y = df_data.drop(["Id","SalePrice"], axis = 1), (df_data["SalePrice"])
  print("Test set X shape: (%d,%d)" %X.shape)
  # convert y to (0,1) type
  y = (y >= 160000).astype(float)
  y = y.values.reshape(-1, 1)

  for feature in categorical_features:
    #Label categorical values
    le = LabelEncoder()
    X[feature]  = le.fit_transform(X[feature])
    # Drop original feature
    X.drop(feature, axis=1, inplace=True)

  print("\nAfter remove categorical features")
  print("Test set X shape: (%d,%d)\n" %X.shape)

  # normalize
  select_X = X.values
  select_X = rescale_dataset(select_X)

  return select_X, y

### main program
def demo():
  # Get program argument
  action='show'  # default to show the result
  file_path = 'ml_project_train.csv'
  if len(sys.argv) >= 2:
    action = sys.argv[1]
  if len(sys.argv) >= 3:
    file_path = sys.argv[2]

  df_data = pd.read_csv('../../data/ml_project_train.csv')
  print('Training data shape: (%d,%d)' %df_data.shape)

  # Missing values columns
  null_values_per_col = np.sum(df_data.drop(["Id", "SalePrice"], axis=1).isnull(), axis=0)

  # remove if has more than 20% missing values
  max_na = int(1*df_data.shape[0]/5.0)
  cols_to_remove = []

  for col in df_data.drop(["Id","SalePrice"],axis=1).columns.tolist():
    if null_values_per_col[col] > max_na: 
        cols_to_remove.append(col)
        df_data.drop(col, axis=1, inplace=True)
        
  print("New shape of the training set is: (%d,%d)" %df_data.shape)
  print("The removed columns are:")
  for column in cols_to_remove:
    print(column, "Dropped because it has %d missing values" %null_values_per_col[column])
    
  null_values_per_col = np.sum(df_data.drop(["Id","SalePrice"], axis=1).isnull(), axis=0)
  print(r"There are %d"  %np.sum(null_values_per_col != 0)  + " columns to impute.\n")

  cols_to_impute = []
  for col_index, val in null_values_per_col.iteritems():
    if val != 0: cols_to_impute.append(col_index)

  # Imputing missing values
  imputation_val_for_na_cols = dict()
  for col in cols_to_impute:
    if (df_data[col].dtype == 'float64' ) or  (df_data[col].dtype == 'int64'):
        imputation_val_for_na_cols[col] = np.nanmedian(df_data[col])
    else:
        imputation_val_for_na_cols[col] = df_data[col].value_counts().argmax()

  for key, val in imputation_val_for_na_cols.iteritems():
    df_data[key].fillna(value= val, inplace = True)

  print("Checking if everything went well ...")    
  print("Number of missing values in data set after imputation and cleaning ",
          np.sum(np.sum(df_data.isnull())))    

  del imputation_val_for_na_cols, cols_to_impute

  # Prepare data for training
  X, y = df_data.drop(["Id","SalePrice"], axis = 1), (df_data["SalePrice"])
  print("Training set X shape: (%d,%d)" %X.shape)
  # convert y to (0,1) type
  y = (y >= 160000).astype(float)
  y = y.values.reshape(-1, 1)

  # Determine categorical features
  categorical_features = []
  onehot_encode_features = {}
  is_categorical = X.dtypes == 'object'
  for col in X.columns.tolist():
    if is_categorical[col]: categorical_features.append(col)

  for feature in categorical_features:    
    #Label categorical values
    le = LabelEncoder()
    X[feature]  = le.fit_transform(X[feature])
    # Drop original feature
    X.drop(feature, axis=1, inplace=True)

  print("\nAfter removing categorical features")
  print("Training set X shape: (%d,%d)\n" %X.shape)

  # normalize
  select_X = X.values

  # evaluate algorithm
  n_epochs = 1
  if action == 'train': n_epochs = 5
  n_hidden = 10
  start = time.time()
  NN = MLP_Classifier(select_X.shape[1], n_hidden, 2)
  NN.action = action
  NN.init_weights()
  if action == 'train':
	NN.evaluate_algorithm(select_X, y, NN.back_propagation, n_epochs)
  else:
	X_train, X_validate, y_train, y_validate \
               		 = train_test_split(select_X, y, test_size=0.2, random_state=np.random.randint(1,128))
	select_X_train = rescale_dataset(X_train)
	select_X_validate = rescale_dataset(X_validate)

	NN.init_weights(test=True)
        y_pred_train = NN.predict(select_X_train)
	y_pred_validate = NN.predict(select_X_validate)

        accuracy_train = accuracy_metric(y_train, y_pred_train)
	accuracy_validate = accuracy_metric(y_validate, y_pred_validate)
        print("In-sample error %.3f\n" % (100.0-accuracy_train))
	print("Validate accuracy %.3f\n" % (accuracy_validate))
  print("Training time: %d seconds"  %(time.time() - start))

  ### Testing with test data
  if action == 'test':
	  df_test = pd.read_csv(file_path)
	  # pre-processing data
	  X_test, y_test = preprocessing(df_test, cols_to_remove, categorical_features)
	  # test set accuracy
	  NN.init_weights(test=True)
	  y_pred_test = NN.predict(X_test)
	  accuracy = accuracy_metric(y_test, y_pred_test)
	  print("Test accuracy %.3f\n" % (accuracy))

if __name__ == '__main__':
  demo()

