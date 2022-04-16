# importing libraries
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
from openpyxl import NUMPY
import pandas as pd
import warnings

# ignoring warnings
warnings.filterwarnings('ignore')

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

# calling the regressor and fitting the model
regressor = LinearRegression()
regressor.fit(x, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(np.e**model.predict([np.random.random(30)]))
