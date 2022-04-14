from cgi import print_exception
from multiprocessing.spawn import import_main_path
from matplotlib.pyplot import yticks
import numpy as np
import pandas as pd
import numpy as np
from scipy import rand
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = pd.read_csv("Used_Bikes.csv")
data['price'] = np.log(data['price'])
x = data[['kms_driven', 'owner', 'age', 'power', 'brand']]
x = pd.get_dummies(data=x, columns=['owner'])
x = pd.get_dummies(data=x, columns=['brand'])
y = data[['price']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
