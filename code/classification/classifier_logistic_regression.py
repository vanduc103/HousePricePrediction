import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Preprocessing
from sklearn.metrics import mean_squared_error as MSE

import math
import sys


# Rescale dataset to [0-1]
def rescale_dataset(dataset):
	dataset_normed = dataset / dataset.max()
	return dataset_normed

# Pre-processing with test data
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
  # convert y to (0,1) type
  y = (y >= 160000).astype(float)
  y = y.values.reshape(-1, 1)

  for feature in categorical_features:
    #Label categorical values
    le = LabelEncoder()
    X[feature]  = le.fit_transform(X[feature])
    X.drop(feature, axis=1, inplace=True)

  print("\nAfter removing categorical features")
  print("Testing set X shape: (%d,%d)\n" %X.shape)

  # normalize
  select_X = X.values
  select_X = rescale_dataset(select_X)

  return select_X, y


#### Main program
# Get program arguments
action='show'	# default to show the result
file_path='ml_project_train.csv'
if len(sys.argv) >= 2:
  action = sys.argv[1]
if len(sys.argv) >= 3:
  file_path = sys.argv[2]


# Read and preprocessing training data
df_data = pd.read_csv('../../data/ml_project_train.csv')

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
X = df_data.drop(["Id","SalePrice"], axis = 1)
y = df_data["SalePrice"]
y = (y >= 160000).astype(float)
y = y.values.reshape(-1, 1)
print("Training set shape (%d,%d)" %X.shape)

# Determine categorical features
categorical_features = []
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
print("Training set shape (%d,%d)\n" %X.shape)

seed = 100
val_split = 0.2
X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size = val_split, random_state = seed)
print('X_train shape: (%d,%d)' %X_train.shape)
print('X_val shape: (%d,%d)' %X_val.shape)

# Delete X and df_data to save memory
del X, df_data

select_X_train = X_train.values
# eval model
select_X_val = X_val.values

# scale data
select_X_train = rescale_dataset(select_X_train)
select_X_val = rescale_dataset(select_X_val)

# Make a prediction with row and coefficients
def predict(row, coefs):
  yhat = np.dot(row[:-1].T, coefs[1:])
  yhat += coefs[0] # b0
  return 1.0 / (1.0 + math.exp(-yhat))

# Stochastic gradient descent update for coefficients
def fit(train, l_rate, iterations):
  # initialize for coefficients
  coefs = np.zeros(train.shape[1])

  # loop for each iteration
  for iteration in range(iterations):
    sum_errors = 0
    for row in train:
      yhat = predict(row, coefs)
      error = row[-1] - yhat
      sum_errors += error**2

      # update coefficients by error and learning rate
      delta = l_rate*error*yhat*(1.0-yhat)
      coefs[0] = coefs[0] + delta
      coefs[1:] += delta*row[:-1]
    
    # print epoch information
    if iteration % 100 == 0:
      print(">Iteration %d: lrate = %.3f, error = %.3f" %(iteration, l_rate, sum_errors))

  # save final coefs
  np.savetxt('weights_coefs.out', coefs, fmt='%.5f')

  return coefs

# Logistic Regression Algorithm with stochastic gradient descent
def logistic_regression_algorithm(test, model):
  predictions = list()
  for row in test:
    yhat = predict(row, model)
    yhat = round(yhat)	# round to 0 or 1
    predictions.append(yhat)
  
  return predictions

# calculate accuracy percentage
def accuracy_cal(predict, actual):
  correct = 0
  for i in range(len(actual)):
    if predict[i] == actual[i]:
      correct += 1
  
  return correct / float(len(actual)) * 100.0

# Training
l_rate = 0.001
n_iteration = 1000

train_data = np.append(select_X_train, y_train, axis=1)
val_data = np.append(select_X_val, y_val, axis=1)

model = np.loadtxt('weights_coefs.out')
if action == 'train':
	model = fit(train_data, l_rate, n_iteration)

y_pred_train = logistic_regression_algorithm(train_data, model)
y_pred_val = logistic_regression_algorithm(val_data, model)

# In-sample error
print("In-sample error: %.3f" % (100.0 - accuracy_cal(y_pred_train, y_train)))

# Accuracy validation
print("Validation accuracy: %.2f" % accuracy_cal(y_pred_val, y_val))

# Testing data
if action == 'test':
	print('\n-------------   Testing   --------------\n')
	df_test = pd.read_csv(file_path)
	X_test, y_test = preprocessing(df_test, cols_to_remove, categorical_features)

	test_data = np.append(X_test, y_test, axis=1)
	y_pred_test = logistic_regression_algorithm(test_data, model)
	print("Testing accuracy: %.2f" % accuracy_cal(y_pred_test, y_test))

