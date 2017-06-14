import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Preprocessing
from sklearn.metrics import mean_squared_error as MSE

from random import randint
from math import exp, sqrt
import sys


### SVM SMO implementation
# Pre-compute the kernel
def pre_cal_gaussian_kernel():
  K_pre = np.zeros([N, N])
  for i in range(N):
    for j in range(N):
      t = np.sum((X_data[i] - X_data[j])**2)
      K_pre[i,j] = np.exp(-t/(2*sigma*sigma))
      K_pre[j, i] = K_pre[i, j]  # the matrix is symetric
  return K_pre

# gaussian kernel
def gaussian_kernel(x1, x2):
  t = np.sum((x1 - x2)**2)
  return np.exp(-t/(2*sigma*sigma))

# Do optimize to find alphas and b
def takeStep(i1, i2, kernel, alphas, b):
  alpha1 = alphas[i1]
  alpha2 = alphas[i2]
  y1 = y_data[i1]
  y2 = y_data[i2]
  E1 = np.sum(alphas * y_data * K[:,i1])
  E2 = np.sum(alphas * y_data * K[:,i2])
  E1 = E1 + b[0] - y1
  E2 = E2 + b[0] - y2
  s = y1 * y2
  # Compute L and H
  if y1 != y2:
    L = max(0, alpha2 - alpha1)
    H = min(C, C + alpha2 - alpha1)
  else:
    L = max(0, alpha2 + alpha1 - C)
    H = min(C, alpha2 + alpha1)
  if L == H:
    return 0
  # Compute eta
  eta = K[i1, i1] + K[i2, i2] - 2*K[i1, i2]
  if eta <= 0:
    return 0
  a2 = alpha2 + y2*(E1-E2)/eta
  # Clip
  a2 = min(H, a2)
  a2 = max(L, a2)
  # Check the change in alpha is significant
  eps = 0.00001 # 10^-5
  if abs(a2 - alpha2) < eps:
    return 0
  # Update a1
  a1 = alpha1 + s*(alpha2 - a2)
  # Update threshold b
  b1 = b[0] - E1 - y1*(a1-alpha1)*K[i1, i1] - y2*(a2-alpha2)*K[i1, i2]
  b2 = b[0] - E2 - y1*(a1-alpha1)*K[i1, i2] - y2*(a2-alpha2)*K[i2, i2]
  if (0 < a1 and a1 < C):
    b[0] = b1
  elif (0 < a2 and a2 < C):
    b[0] = b2
  else:
    b[0] = (b1 + b2) / 2
  # Store a1 and a2 in alphas array
  alphas[i1] = a1
  alphas[i2] = a2
  return 1
  
 
# Examine each training example
def examineExample(i2, kernel, alphas, b):
  y2 = y_data[i2]
  alpha2 = alphas[i2]
  # compute error by svm output
  E2 = np.sum(alphas * y_data * K[:, i2])
  E2 = E2 + b[0] - y2
  r2 = E2 * y2
  if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
    # get random i1
    i1 = randint(0, N-1)
    while i1 == i2:
      i1 = randint(0, N-1) # make sure i1 different i2
    # do optimize to find alphas
    if takeStep(i1, i2, kernel, alphas, b):
      return 1
  return 0

def svm_train(alphas, b):
  numChanged = 0
  examineAll = 1

  while (numChanged > 0 or examineAll == 1):
    numChanged = 0
    if examineAll == 1:
      for i in range(N):
        numChanged += examineExample(i, gaussian_kernel, alphas, b)
    else:
      for i in range(len(alphas)):
	if alphas[i] != 0 and alphas[i] != C:
	  numChanged += examineExample(i, gaussian_kernel, alphas, b)
    if examineAll == 1:
      examineAll = 0
    elif numChanged == 0:
      examineAll = 1

# Prediction by kernel
def predict(testset):
  predicted = np.zeros(testset.shape[0])
  # loop through all testset examples
  for i in range(testset.shape[0]):
    # calculate svm output by kernel
    p = 0
    for j in range(N):
      p += alphas[j] * y_data[j,0] * gaussian_kernel(X_data[j], testset[i])
    p += b
    if p >= 0:
      predicted[i] = 1.0
    else:
      predicted[i] = -1.0
  return predicted

# Calculate accuracy on testset
def cal_accuracy(testset, actual):
  correct = 0
  predicted = predict(testset)
  for i in range(len(actual)):
    if predicted[i] == actual[i]:
      correct += 1
  return correct / float(len(actual)) * 100.0

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
  # convert y to (-1,1) type for svm
  y = (y >= 160000).astype(float)
  y[y == 0] = -1.0
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
if len(sys.argv) >= 3 and action == 'test':
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
y[y == 0] = -1.0
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

### Training SVM
C = 2.0
sigma = 0.1

if action == 'train' and len(sys.argv) >= 3:
  C = float(sys.argv[2])
if action == 'train' and len(sys.argv) >= 4:
  sigma = float(sys.argv[3])

tol = 0.001
X_data = select_X_train
y_data = y_train
N = X_data.shape[0]
alphas = np.zeros(N)
b = np.zeros(1)

# Train
if action == 'train':
    print('Training SVM with C = %.3f, sigma = %.3f' % (C, sigma))
    K = pre_cal_gaussian_kernel()
    svm_train(alphas, b)
else:
    # load pre-training parameters
    b = np.loadtxt('svm_bias.out')
    alphas = np.loadtxt('svm_alphas.out')
    w = np.loadtxt('svm_weights.out')

print('Bias b = %.3f' % b)
print('Alphas=')
print alphas

# Compute w
if action == 'train':
    w = np.zeros(X_data.shape[1])
    for i in range(N):
      w += X_data[i,:] * (y_data[i] * alphas[i])
print('Weights:')
print(w)

# Save training parameters
if action == 'train':
    np.savetxt('svm_bias.out', b, fmt='%.3f')
    np.savetxt('svm_alphas.out', alphas, fmt='%.3f')
    np.savetxt('svm_weights.out', w, fmt='%.3f')

# Accuracy
print("Accuracy on validate set = %.3f" % cal_accuracy(select_X_val, y_val))


# Testing data
if action == 'test':
    print('\n-------------   Testing   --------------\n')
    df_test = pd.read_csv(file_path)
    X_test, y_test = preprocessing(df_test, cols_to_remove, categorical_features)

    print("Accuracy on test set = %.3f" % cal_accuracy(X_test, y_test))

