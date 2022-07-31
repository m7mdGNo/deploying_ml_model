from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle
import numpy as np

data = pd.read_csv('data.csv')

X = np.array(data.iloc[:,0]).reshape(-1, 1) 
y = np.array(data.iloc[:,1]).reshape(-1, 1) 

lr = LinearRegression()
lr.fit(X,y)

pickle.dump(lr,open('model.sav','wb'))