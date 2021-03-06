from random import seed, random, randrange
from math import exp
from csv import reader
import numpy as np
import time
import sys

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # Preprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.decomposition import PCA, TruncatedSVD


# Rescale dataset to [0-1]
def rescale_dataset(dataset):
	dataset_normed = dataset / dataset.max()
	return dataset_normed

# Error metric
def error_metric(y_test, y_predict):
	print('y_pred_val	y_val')
	for i in range(10):
  		print('%d		%d' % (y_predict[i], y_test[i]))
	# Bias error
	print("\nBias error: %.2f" % np.mean(y_predict - y_test))

	# Maximum deviation
	print("Maximum deviation error: %.2f" % np.amax(np.absolute(y_predict - y_test)))

	# Mean absolute deviation
	print("Mean absolute deviation error: %.2f" % np.mean(np.absolute(y_predict - y_test)))

	# The mean squared error
	print("Mean squared error: %.2f\n" % np.mean(np.square(y_predict - y_test)))


class MLP_Regressor(object):

  # Evaluate an algorithm using a cross validation split
  def evaluate_algorithm(self, dataset, target, algorithm, n_epochs, *args):
        # split training and validation set
	train_set, validate_set, y_train, y_validate \
		= train_test_split(dataset, target, test_size=0.20, random_state=100)
        for epoch in range(n_epochs):
                y_pred_validate = algorithm(train_set, y_train, validate_set, *args)
		if epoch % 1 == 0:
			print('> Epoch %d:' % (epoch+1))
			#np.savetxt('weights_reg.out2', self.w, fmt='%.5f')
                	error_metric(y_validate, y_pred_validate)

  def __init__(self, n_input, n_output):
	# weight
	self.w = np.zeros([n_input, n_output])
	self.w = np.loadtxt('weights_reg.out2')


  # Train a network for a fixed number of epochs
  def train(self, x_train, y_train, iterations = 3000, N = 0.0002):
    # N: learning rate
    dataset_length = x_train.shape[0]
    batch_size = 100
    rng = np.random.RandomState(np.random.randint(1,128))
    for i in range(iterations):
        cost = 0.0
        batch_mask = rng.choice(dataset_length, batch_size)
	batch_x = x_train[[batch_mask]]
	batch_y = y_train[[batch_mask]]
	for j in range(batch_x.shape[0]):
            inputs = x_train[j]
	    inputs = inputs.reshape(-1, 1)
            targets = y_train[j]

	    # feed forward
            outputs = np.dot(self.w.T, inputs)

	    # compute gradient by back propagate
	    error = outputs - targets
	    cost = 0.5 * np.sum((outputs - targets) ** 2)
	    gradient = inputs * error

	    # update weight by gradient and learning rate
            self.w = self.w - N * gradient

        if i % 500 == 0:
    	    print('error cost %-.5f' % cost)


  # Make a prediction with a network
  def predict(self, X):
    """
    return list of predictions after training algorithm
    """
    predictions = []
    for p in X:
	p = p.reshape(-1, 1)
	outputs = np.dot(self.w.T, p)
        predictions.append(outputs)
    return predictions


  # Backpropagation Algorithm With Stochastic Gradient Descent
  def back_propagation(self, x_train, y_train, x_test):
	#self.train(x_train, y_train)
	predictions = self.predict(x_test)
	return predictions

# pre-processing for testing data
def preprocessing(df_data, cols_to_remove, categorical_features):
  print('Testing set shape: (%d,%d)' %df_data.shape)  

  # remove cols as in training data
  for col in cols_to_remove:
    df_data.drop(col, axis=1, inplace=True)

  print("New shape of the testing set: (%d,%d)" %df_data.shape)
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
  print("Testing set X shape: (%d,%d)" %X.shape)
  y = y.values.reshape(-1, 1)

  for feature in categorical_features:
    #Label categorical values
    le = LabelEncoder()
    X[feature]  = le.fit_transform(X[feature])
    # Perform one hot encoding
    ohe = OneHotEncoder(sparse = False)
    # Name columns
    columns = [feature + '_' + str(class_) for class_ in le.classes_]
    #Drop first column to avoid dummy variable trap
    X_dummies = pd.DataFrame( ohe.fit_transform(X[feature].values.reshape(-1,1))[:, 1:] ,
                            columns = columns[1:])
    # Drop original feature
    X.drop(feature, axis=1)
    X = pd.concat([X, X_dummies], axis=1)

  print("\nAfter One Hot Encoding")
  print("Testing set X shape: (%d,%d)\n" %X.shape)

  # Dimensionality Reduction
  model = TruncatedSVD(n_components=40)
  model.fit(X)
  select_X = model.transform(X)

  print('After reduction: (%d,%d)\n' %select_X.shape)

  # normalize
  select_X = rescale_dataset(select_X)

  return select_X, y

# Main program
def demo():
  df_data = pd.read_csv('../../data/ml_project_train.csv')
  print(df_data.shape)

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
  print("Training set shape (%d,%d)" %X.shape)
  y = y.values.reshape(-1, 1)

  # Determine categorical features
  categorical_features = []
  is_categorical = X.dtypes == 'object'
  for col in X.columns.tolist():
    if is_categorical[col]: categorical_features.append(col)

  for feature in categorical_features:    
    #Label categorical values
    le = LabelEncoder()
    X[feature]  = le.fit_transform(X[feature])
    # Perform one hot encoding
    ohe = OneHotEncoder(sparse = False)
    # Name columns
    columns = [feature + '_' + str(class_) for class_ in le.classes_]
    #Drop first column to avoid dummy variable trap
    X_dummies = pd.DataFrame( ohe.fit_transform(X[feature].values.reshape(-1,1))[:, 1:] ,
                            columns = columns[1:])
    # Drop original feature
    X.drop(feature, axis=1)
    X = pd.concat([X, X_dummies], axis=1)

  print("\n After One Hot Encoding")
  print("Training set shape (%d,%d)" %X.shape)

  # Dimensionality Reduction
  model = TruncatedSVD(n_components=40)
  model.fit(X)
  select_X = model.transform(X)

  print('After reduction: (%d,%d)\n' %select_X.shape)

  #normalize
  select_X = rescale_dataset(select_X)

  # evaluate algorithm
  n_epochs = 1
  start = time.time()
  NN = MLP_Regressor(select_X.shape[1], 1)
  NN.evaluate_algorithm(select_X, y, NN.back_propagation, n_epochs)
  print("Training time: %d seconds"  %(time.time() - start))

  # Get program parameter
  action = 'show'
  file_path = 'ml_project_train.csv'
  if len(sys.argv) >= 2:
    action = sys.argv[1]
  if len(sys.argv) >= 3:
    file_path = sys.argv[2]

  if action == 'test':
    print('\n--------   Testing --------\n')
    df_test = pd.read_csv(file_path)
    X_test, y_test = preprocessing(df_test, cols_to_remove, categorical_features)
    y_pred_test = NN.predict(X_test)
    error_metric(y_test, y_pred_test)

if __name__ == '__main__':
  demo()

