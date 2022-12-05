import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('deploy_df')
df.drop('Unnamed 0', axis=1, inplace=True)

x = df.drop('Price', axis=1)
y = df['Price']

# splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, randome_state=50)

# import the catboost model
import pickle
model = pickle.load(open('model.pkl','rb'))
y_pred = model.predict(x_test)
print(y_pred)