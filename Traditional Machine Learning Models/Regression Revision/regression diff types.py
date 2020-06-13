import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()
X = dataset['data']
y = dataset['target']
y = y.reshape((-1,1))

# Doing feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)

# Splitting in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Using LinearRegression without any regularisation
from sklearn.linear_model import LinearRegression
slr = LinearRegression(normalize = True)
slr.fit(X_train, y_train)
y_pred_slr = slr.predict(X_test)
y_pred_train = slr.predict(X_train)

# Using Ridge Regression
from sklearn.linear_model import Ridge
rr = Ridge(alpha =1, max_iter = 1, normalize= True)
rr.fit(X_train, y_train)
y_pred_rr = rr.predict(X_test)

# Using Lasso Regression
from sklearn.linear_model import Lasso
Lasso_reg = Lasso(normalize=True, alpha=.1, max_iter= 1000)
Lasso_reg.fit(X_train, y_train)
y_pred_lr = Lasso_reg.predict(X_test)
y_pred_lr = y_pred_lr.reshape((-1,1))

#y_performance_comparison =np.concatenate((y_test, y_pred_slr, y_pred_rr, y_pred_lr),axis =1)
#y_performance_comparison = np.concatenate((np.asarray(['y_test', 'y_pred_slr', 'y_pred_rr','y_pred_lr']).reshape(-1,4),y_performance_comparison))

# Creating pd dataframe from np array
pd_df = pd.DataFrame(data = y_performance_comparison, columns = ['y_test', 'y_pred_slr', 'y_pred_rr','y_pred_lr'], index =None)
pd_df = pd_df.iloc[1:,:]

# Just as a tip, Use F9 to run current line

#  Saving your predictions as a csv file
predictions = open('predictions.txt','w')
pd_df.to_csv(path_or_buf = predictions, index =False)
predictions.close()

#test_dataset = pd.read_csv('predictions.txt')

#Visualising loss (both training as well as test) as we vary alpha

epochs = 1000
alpha_vector = []
test_error = []
train_error = []
for i in range(2000):
       rr = Ridge(normalize=True, alpha=epochs)
       rr.fit(X_train, y_train)
       y_pred_lr = rr.predict(X_test)
       y_pred_lr = y_pred_lr.reshape((-1,1))
       y_pred_train_rr = rr.predict(X_train)
       y_pred_train_rr = y_pred_train_rr.reshape((-1,1))
       # Calculating the norm L2
       test_error.append(np.linalg.norm(y_pred_lr - y_test))
       alpha_vector.append(epochs)
       train_error.append(np.linalg.norm(y_pred_train_rr - y_train))
       epochs /= 2
       
plt.plot(alpha_vector, train_error, 'g-', label = 'Training Error')
plt.plot(alpha_vector, test_error, 'r-', label = 'Test Error')
plt.xlabel('No of Iterations')
plt.ylabel('Error')
plt.legend()
plt.title("Learning Curves")
plt.show()             

# We can say that this simple model was not able to overfit the data since test error decreases continuously on decreasing alpha
