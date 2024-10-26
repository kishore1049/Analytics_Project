#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# Load the dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target  # Add target column
print(df.head())  # View the first few rows of data


# In[2]:


# Check the data types and null values
print(df.info())

# Check descriptive statistics
print(df.describe())


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot Price vs. RM (Average number of rooms)
sns.scatterplot(x='RM', y='PRICE', data=df)
plt.xlabel("Average Number of Rooms per Dwelling")
plt.ylabel("Housing Price")
plt.title("House Price vs. Number of Rooms")
plt.show()


# In[4]:


X = df[['RM']]  # Feature
y = df['PRICE']  # Target variable


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


y_pred = model.predict(X_test)


# In[8]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[9]:


plt.scatter(X_test, y_test, color='blue', label="Actual Prices")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predicted Regression Line")
plt.xlabel("Average Number of Rooms per Dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Regression Model: Predicted vs Actual Prices")
plt.legend()
plt.show()


# In[10]:


X = df[['RM', 'LSTAT', 'AGE']]
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[11]:


# # #Interpreting the results
# # Mean Squared Error (MSE): This value indicates the average squared difference between the actual
# #     and predicted values. The lower it is, the better the model has performed.
# # R-squared (𝑅2R 2): This value explains the proportion of the variance in the target variable explained by the feature.
# A value closer to 1 indicates a better fit


# In[ ]:




