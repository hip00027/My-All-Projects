import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

dataset = pd.read_csv(r"D:\Naresh i Class\Sept 2024\9 Sep 24\House Prediction\House_data.csv")

space=dataset['sqft_living']  #space is extracted as the independent variable (feature), representing the size of the house in square feet.
price=dataset['price'] # price is extracted as the dependent variable (target), representing the price of the house.

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# x = np.array(space) - converts space (which is a pandas Series or list of values) into a 1D numpy array
 # which looks like : space = [1180, 2570, 770, 1960, 1680]

# Purpose of .reshape(-1, 1) - converts the 1D array into a 2D array with one column and as many rows as needed to accommodate all elements.
 # where -1 in the reshape function means "infer the number of rows automatically based on the length of the array. In this case, it will 
 # determine the number of rows needed to keep the elements as individual rows.
# 1 specifies that there should be exactly one column.

#Splitting the data into Train and Test

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)


#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


filename_house = 'linear_regression_model_house.pkl' # entire code is saved in 'filename' which is object. This name will be used for the file in which the trained model will be saved. The extension .pkl is commonly used for files that are saved using the pickle module in Python.
with open(filename_house, 'wb') as file1: # Opens the file in write-binary mode ('wb'). Binary mode is necessary because pickle saves the data in a binary format. 'with' statement is used to open the file, which ensures that the file is properly closed after the block of code is executed, even if an error occurs.

    pickle.dump(regressor, file1) #pickle.dump() is used to serialize the trained model (in this case, the object regressor) and write it to the opened file (file).

print("Opening file for writing...")
with open(filename_house, 'wb') as file1:
    print("Dumping model to file...")
    pickle.dump(regressor, file1)
    print("Model has been pickled and saved as", filename_house)


print("Model has been pickled and saved as linear_regression_model_house.pkl")


import os
print(os.getcwd())