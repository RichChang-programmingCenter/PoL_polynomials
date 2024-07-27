# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:34:22 2024

@author: User
"""

# In[] Import Area
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hashlib

# In[] Dataset preprocessing
# split N = 1024 into 32 x 32 np array

import numpy as np
import math

N = 1024
side_N = int(math.sqrt(N)) # 邊長 32，各邊長以 x1 ~ x6、y1 ~ y6 表示

dataset = pd.read_csv(r"C:\\Users\\User\\Desktop\\prog\\dataset_preprocessed{}.csv".format(N))

# 创建一个包含 1024 个元素的示例数据
# data = np.arange(1024)

# 将数据重塑为 32x32 的数组
reshaped_data = dataset['annual_inc'].values.reshape(side_N, side_N)

# 打印结果
print(reshaped_data)

# In[] MLE P

import numpy as np
import time

def index_to_binary_terms(index, prefix):
    terms = []
    for bit in range(6):
        if index & (1 << bit):
            terms.append(f"{prefix}{bit+1}")
        else:
            terms.append(f"(1-{prefix}{bit+1})")
    return "*".join(terms[::-1])

def generate_MLE_polynomial(values):
    terms = []
    size = values.shape[0]
    for i in range(size):
        for j in range(size):
            coeff = values[i, j]
            x_terms = index_to_binary_terms(i, 'x')
            y_terms = index_to_binary_terms(j, 'y')
            terms.append(f"{coeff}*{x_terms}*{y_terms}")
    return " + ".join(terms)

# 记录开始时间
MLE_P_start_time = time.time()

# Note: Only use this for generating the expression to understand the structure
p = generate_MLE_polynomial(reshaped_data)

# 记录结束时间
MLE_P_end_time = time.time()

print("---------------------")
print("MLE - Polynomial:")
print(p)

# 计算并打印执行时间
print("---------------------")
MLE_P_time = MLE_P_end_time - MLE_P_start_time
print(f"MLE - P time: {MLE_P_time} seconds")

# In[] MLE - V time 

import numpy as np
import itertools

def binary_to_decimal(binary_tuple):
    """ Convert a tuple of binary digits to a decimal number ensuring it fits in the grid size. """
    decimal_number = 0
    for i, bit in enumerate(reversed(binary_tuple)):
        decimal_number += bit * (2 ** i)
    return decimal_number

# Define a safe function to access grid values
def get_grid_value(grid, x_index, y_index):
    """ Safely get the value from the grid, ensuring indices are within the valid range. """
    max_index = grid.shape[0] - 1  # Maximum index (31 for a 32x32 grid)
    # Clamp the indices to the maximum index
    safe_x_index = min(x_index, max_index)
    safe_y_index = min(y_index, max_index)
    return grid[safe_x_index, safe_y_index]

def validate_grid(values, vrfy_cnt):
    for bin_index_x in itertools.product([0, 1], repeat=5):
        for bin_index_y in itertools.product([0, 1], repeat=5):
            x_index = binary_to_decimal(bin_index_x)
            y_index = binary_to_decimal(bin_index_y)
            value = get_grid_value(values, x_index, y_index)            
            vrfy_cnt = vrfy_cnt + 1
            # print(f"Value at ({x_index}, {y_index}) = {value}, cnt = {vrfy_cnt}")
    return vrfy_cnt

# vrfy_cnt = 0
# validate_grid(reshaped_data, vrfy_cnt)
# print(vrfy_cnt)

# 记录开始时间
MLE_V_start_time = time.time()

vrfy_cnt = 0
validate_grid(reshaped_data, vrfy_cnt)

# 记录结束时间
MLE_V_end_time = time.time()

# 计算并打印执行时间
MLE_V_time = MLE_V_end_time - MLE_V_start_time
print(f"V time of MLE: {MLE_V_time} seconds")


# In[] ML - P 

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assume x_points are the flattened indices for a 32x32 matrix
rows, cols = np.indices((32, 32))
X = np.c_[rows.ravel(), cols.ravel()]  # Convert indices to two features

# Flatten y_points to create a single vector of target values
y = reshaped_data.flatten()


# # 记录开始时间
ML_P_start_time = time.time()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline that includes polynomial feature transformation and linear regression
model = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=True)),  # Adjust degree as needed
    ('linear', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

# Access the polynomial feature transformer and linear regression objects
poly_transformer = model.named_steps['poly']
linear_model = model.named_steps['linear']

# Get the feature names generated by PolynomialFeatures
feature_names = poly_transformer.get_feature_names_out(['row', 'col'])

# Get coefficients from the linear regression model
coefficients = linear_model.coef_
intercept = linear_model.intercept_

# 记录结束时间
ML_P_end_time = time.time()

# Output the polynomial coefficients and their corresponding terms
print("---------------------")
print("Intercept:", intercept)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("---------------------")
print("Mean Squared Error:", mse)


print("---------------------")
print("MLE - Polynomial:")
for coef, name in zip(coefficients, feature_names):
    print(f"{coef:.4f} * {name}")


# 计算并打印执行时间
print("---------------------")
ML_P_time = ML_P_end_time - ML_P_start_time
print(f"ML - P time: {ML_P_time} seconds")


# In[] ML - V
# Perform check calculation for the first few points

# 记录开始时间
ML_V_start_time = time.time()

print("\nCheck Calculation:")
print("Predicted vs Actual")
for actual, predicted, x in zip(y, y_pred, X):
    print(f"Input {x}: Predicted = {predicted:.2f}, Actual = {actual:.2f}")

# 记录结束时间
ML_V_end_time = time.time()

# 计算并打印执行时间
ML_V_time = ML_V_end_time - ML_V_start_time
print(f"V time of MLE: {ML_V_time} seconds")


# In[]