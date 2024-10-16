# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preprocessing: Load the dataset, drop unnecessary columns, convert categorical data to numerical using label encoding.

2.Model Initialization: Initialize parameters, define sigmoid and loss functions, and implement gradient descent.

3.Training: Train the model using gradient descent with specified alpha and iterations.

4.Prediction and Evaluation: Predict labels, calculate accuracy, and test the model with new data.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SURYAMALARV
RegisterNumber:  212223230224
*/
```
```
# import modules.
import pandas as pd
import numpy as np
dataset = pd.read_csv("Placement_Data_Full_Class.csv")
dataset.head()
```
![image](https://github.com/user-attachments/assets/ffdf6e37-b4b2-49a3-88ea-4001bd038a1f)
```
dataset.tail()
```
![image](https://github.com/user-attachments/assets/e401e8ff-ab6c-4eaf-a2b0-da8d1f6abba2)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/3e79784b-61bc-4943-9643-70c3348c7d74)
```
# Dropping the serial 
dataset = dataset.drop('sl_no',axis=1)
dataset.info()
```
![image](https://github.com/user-attachments/assets/b7329bdf-a3bc-4f3f-884c-4fc9f271fd58)
```
# Categorizing column for further labelling.
dataset['gender'] = dataset['gender'].astype('category')
dataset['ssc_b'] = dataset['ssc_b'].astype('category')
dataset['hsc_b'] = dataset['hsc_b'].astype('category')
dataset['degree_t'] = dataset['degree_t'].astype('category')
dataset['workex'] = dataset['workex'].astype('category')
dataset['specialisation'] = dataset['specialisation'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset['hsc_s'] = dataset['hsc_s'].astype('category')

# Analysing the datatype of the datase. 
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/e3908998-4961-4a10-8c12-9ac5f80c01d3)
```
# Labelling the columns.
dataset['gender']=dataset['gender'].cat.codes
dataset['ssc_b']=dataset['ssc_b'].cat.codes
dataset['hsc_b']=dataset['hsc_b'].cat.codes
dataset['degree_t']=dataset['degree_t'].cat.codes
dataset['workex']=dataset['workex'].cat.codes
dataset['specialisation']=dataset['specialisation'].cat.codes
dataset['status']=dataset['status'].cat.codes
dataset['hsc_s']=dataset['hsc_s'].cat.codes
```
```
# Display dataset.
dataset.head()
```
![image](https://github.com/user-attachments/assets/fac7a6e3-defd-47db-b99a-f48f044f4652)
```
# Selecting the features and labels.
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
```
```
# Analyse the shape of independent variable.
X.shape
```
![image](https://github.com/user-attachments/assets/0169e102-dce7-42e3-8e72-8247c72929c9)
```
# Analyse the shape of dependent variable.
Y.shape
```
![image](https://github.com/user-attachments/assets/7de0fc89-6dd3-44ff-9e72-8fba6a279644)
```
# Initialize the model parameters.
theta = np.random.randn(X.shape[1])
y=Y

# Define the sigmoid function.
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
```
# Define the loss function.
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```
```
# Define the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
```
```
# Train the model.
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
```
```
# Make Predictions.
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
```
```
# Predict the y value by calling predict function.
y_pred = predict(theta, X)
```
```
# Actual y values.
print(y)
```
![image](https://github.com/user-attachments/assets/719de956-53ee-4c63-af32-de05ad49fe68)
```
# Predicted y values.
print(y_pred)
```
![image](https://github.com/user-attachments/assets/7c77e3d2-d864-4fdd-a963-30a3a26e2009)
```
# Evaluate the model.
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy)
```
![image](https://github.com/user-attachments/assets/50854700-da46-4d3f-963d-b537d0ddc22c)
```
print(theta)
```
![image](https://github.com/user-attachments/assets/fa2a1d66-4cbb-4522-8717-cb7b2554a3ad)
```
# Testing the model with own data.
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/9aad049a-4cf1-4065-87cb-f1ae88afc3f8)
```
# Testing the model with own data.
x_new = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
y_pred_new = predict(theta, x_new)
print(y_pred_new)
```
![image](https://github.com/user-attachments/assets/b27a2440-ca6e-44bb-a798-7c3b744a53f9)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

