# importing numpy and pandas libraries
import numpy as np
import pandas as pd

# importing linear regressor, train test split and error libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# reading dataset and storing in variable
data = pd.read_csv("Used_Bikes.csv")

# log transformation of the price variable because it is skewed
data['price'] = np.log(data['price'])

# storing the independent variables in x
x = data[['kms_driven', 'owner', 'age', 'power', 'brand']]

# the owner and brand variables are categorical so we are creating dumy variables for them
x = pd.get_dummies(data=x, columns=['owner'])
x = pd.get_dummies(data=x, columns=['brand'])

# storing dependant variable price in y
y = data[['price']]

# randomly spiltting data into train and test with 75% used for training and 25% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)

# calling the regressor and fitting the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)
print("MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
