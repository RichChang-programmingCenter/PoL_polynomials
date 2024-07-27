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

#########################################################################################################
# The followings are preprocessing of Naive Protocol                                                    #
# Meaning an exchange P generates 2 columns: 'pseudo_ID' and 'pseudo_bin_ID' for each user i (verifier) #
#########################################################################################################

# Load dataset with assigned number of columns

dataset = pd.read_csv(r"C:\Users\User\Desktop\prog\dataset.csv")

N = 1024
dataset = dataset[['annual_inc', 'address', 'loan_amnt']].head(N)

def generate_pseudo_id(data, N):
    hash_object = hashlib.sha256(data.encode())
    hash_digest = hash_object.hexdigest()
    return int(hash_digest, 16) % N       # 注意這裡 hash digest bit 呼應下面 generate_large_prime 皆設為 16 bit (2~36間 N=1024 32會溢位)

# 初始化一个集合来跟踪已使用的 ID
used_ids = set()

# 生成虚拟 ID
pseudo_ids = []
pseudo_bin_ids = []

for _, row in dataset.iterrows():
    combined_data = f"{row['annual_inc']}-{row['address']}"
    pseudo_id = generate_pseudo_id(combined_data, N)
    
    # 检查碰撞并分配新 ID
    while pseudo_id in used_ids:
        # 碰撞处理，简单递增（这只是一个基本例子，可能需要更复杂的逻辑）
        pseudo_id = (pseudo_id + 1) % N
    used_ids.add(pseudo_id)
    pseudo_ids.append(pseudo_id)
    pseudo_bin_ids.append(format(pseudo_id, 'b'))

# 将生成的虚拟 ID 添加到 DataFrame
dataset['pseudo_ID'] = pseudo_ids
dataset['pseudo_bin_ID'] = pseudo_bin_ids

# place in order
dataset.sort_values('pseudo_ID', ascending=True,inplace=True, na_position='first')

# 檢查重複
# dataset = dataset.drop_duplicates(subset=['pseudo_ID'])
# print(dataset['pseudo_ID'].duplicated().sum())

# 確保數據是數字型態
dataset['pseudo_ID'] = pd.to_numeric(dataset['pseudo_ID'], errors='coerce')
dataset['loan_amnt'] = pd.to_numeric(dataset['loan_amnt'], errors='coerce')


###############################################################################################
# The followings are Pedersen Commitments Preprocessing                                       #
# Meaning an exchange P generates 1 column of 128 bits commitments for each user i (verifier) #
###############################################################################################
import random
from sympy import isprime

def generate_large_prime(bits=16):
    """Generate a large prime number."""
    while True:
        num = random.getrandbits(bits)
        if isprime(num):
            return num

def generate_generators(p):
    """Generate two generators for the group of order p."""
    g = random.randint(2, p - 1)
    h = random.randint(2, p - 1)
    # Ensure that g and h are likely to be generators of the entire group
    if pow(g, 2, p) == 1 or pow(h, 2, p) == 1:
        return generate_generators(p)  # Recursively generate if not suitable
    return g, h

def commit(value, randomness, p, g, h):
    """Create a Pedersen commitment."""
    return (pow(g, value, p) * pow(h, randomness, p)) % p

p = generate_large_prime(24)  # Generate a large prime (搭配N=1024筆且 hash_digest=16最佳)
g, h = generate_generators(p)  # Generate generators g and h

# Generate 64 random values for R and r
R = [random.randint(1, p - 1) for _ in range(N)]
r = [random.randint(1, p - 1) for _ in range(N)]

dataset['commitments'] = [commit(R[i], r[i], p, g, h) for i in range(N)]
dataset['commitments'] = pd.to_numeric(dataset['commitments'], errors='coerce')

# 保存到新的 CSV 文件
dataset.to_csv(r"C:\\Users\\User\\Desktop\\prog\\dataset_preprocessed{}.csv".format(N), index=False)


# In[] Naive P
# Lagrange of Naive Protocol (referred previous poly gen: relu modulus)

import numpy as np
import time
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from sympy import nextprime

# 確保以整數實作。lagrange 函數庫帶 x 得出的 y 值是 'function' 物件
def mod_pow(base, exponent, modulus):
    base = int(base)
    exponent = int(exponent)
    modulus = int(modulus)
    return pow(base, exponent, modulus)

# 費馬：a ^ (p - 2) 為 a ^ -1 (mod p)
# 這裡的 mod 「必須大於 x point 的總長度」
def mod_inv(a, mod=1031):
    return mod_pow(a, mod - 2, mod)

# 計算拉格朗日多項式的係數，結果是 mod 
def lagrange_interpolation_coefficients(x_points, y_points, mod_value):
    coefficients = [0] * len(x_points)
    n = len(x_points)
    
    for i in range(n):
        # 確保 xi, yi 在有限域
        xi, yi = x_points[i] % mod_value, y_points[i] % mod_value
        
        # Start with the coefficient for the Lagrange basis polynomial L_i(x)
        # 從 yi 開始，初始化第 i 個 Lagrange polynomial basis 的係數
        term_coefficients = [yi]
        
        for j in range(n):
            if i != j:
                xj = x_points[j] % mod_value
                
                # Each term in the Lagrange basis polynomial contributes (x - xj)/(xi - xj)
                # We need to adjust our current coefficients to account for this new term.
                # 計算 Lagrange polynomial basis 的係數，並逐項更新
                new_term_coefficients = [0] * (len(term_coefficients) + 1)
                inv = mod_inv((xi - xj + mod_value) % mod_value, mod_value) # 計算 mod inverse
                
                for k in range(len(term_coefficients)):
                    new_term_coefficients[k] = (new_term_coefficients[k] - term_coefficients[k] * xj * inv) % mod_value
                    new_term_coefficients[k + 1] = (new_term_coefficients[k + 1] + term_coefficients[k] * inv) % mod_value
                term_coefficients = new_term_coefficients
        
        # Add the current Lagrange basis polynomial's contribution to the overall coefficients
        # 將目前的 Lagrange polynomial basis 累加到總係數中
        for k in range(len(term_coefficients)):
            coefficients[k] = int((coefficients[k] + term_coefficients[k]) % mod_value)
    
    # 回傳低到高次的係數
    return coefficients[::-1]

def polynomial_to_string_high_to_low(coefficients):
    terms = []
    for i, coef in reversed(list(enumerate(coefficients))):  # Reverse to start from the highest degree
        if coef == 0:
            continue  # Skip terms with a coefficient of 0
        elif i == 0:
            terms.append(f"{coef}")  # The constant term
        elif i == 1:
            terms.append(f"{coef}*x")  # The linear term
        else:
            terms.append(f"{coef}*x^{i}")  # Higher degree terms
    return " + ".join(terms)

# 初始化 f(0), ..., f(2000), 共 2001 個點
# x_points = list(range(2001))
# y_points = [relu_custom(x) for x in x_points]

# 记录开始时间
naive_P_start_time = time.time()

x_points = dataset['pseudo_ID'].values
y_points = dataset['loan_amnt'].values

# 設大於 y_points 的質數 modulus
new_modulus= nextprime(y_points.max())

# 計算在 mod 值為 new_modulus 下的拉格朗日插值的函數的係數
coefficients = lagrange_interpolation_coefficients(x_points, y_points, new_modulus)

# 印出多項式
# print("---------------------")
# print("Coefficients of the polynomial: (from the highest to the lowest degree)")
# print(list(map(int, coefficients[::-1])))  # 係數由高次到低次

print("---------------------")
polynomial_str_high_to_low = polynomial_to_string_high_to_low(coefficients)
print("Naive - Polynomial:")
print(polynomial_str_high_to_low)

# 记录结束时间
naive_P_end_time = time.time()

# 计算并打印执行时间
naive_P_time = naive_P_end_time - naive_P_start_time
print(f"Naive - P time: {naive_P_time} seconds")


# In[] Naive V 測試部分 (全部驗證)
# 代入計算出的多項式係數，驗證 0 ~ N-1 所有整數點的插值皆正確
def evaluate_polynomial(coefficients, mod_value, x_range = range(N)):
    coefficients = coefficients[::-1]  # 係數由高次到低次
    evaluated_points = []
    
    for x in x_range:
        y = 0
        
        # Evaluate using Horner's method for numerical stability and efficiency
        # 用 Horner's method 快速計算多項式的 mod_value
        for coef in reversed(coefficients):
            y = (y * x + coef) % mod_value
        
        evaluated_points.append(y)
    
    return evaluated_points

# V start 個別驗證所有
naive_V_start_time = time.time()

# 由產生的多項式，驗算後的值
evaluated_points = evaluate_polynomial(coefficients, new_modulus)

# 檢查代入的值相同，全部正確則 correct_values = true
correct_values = all(original == evaluated for original, evaluated in zip(y_points, evaluated_points))

# 记录结束时间
naive_V_end_time = time.time()

print("---------------------")
print("(Test) Correctness of the coefficients:", correct_values)

print("---------------------")
print("(Test) The result after interpolation should be the input of 'loan_amnt' column")
print(evaluated_points)

# 计算并打印执行时间
print("---------------------")
naive_V_time = naive_V_end_time - naive_V_start_time
print(f"Naive - V time: {naive_V_time} seconds")


# In[] Commit P
# Lagrange of Pedersen Commitment Protocol (referred previous poly gen: relu modulus)

import numpy as np
import time
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from sympy import nextprime

# 確保以整數實作。lagrange 函數庫帶 x 得出的 y 值是 'function' 物件
def mod_pow(base, exponent, modulus):
    base = int(base)
    exponent = int(exponent)
    modulus = int(modulus)
    return pow(base, exponent, modulus)

# 費馬：a ^ (p - 2) 為 a ^ -1 (mod p)
# 這裡的 mod 「必須大於 x point 的總長度」
def mod_inv(a, mod=67):
    return mod_pow(a, mod - 2, mod)

# 計算拉格朗日多項式的係數，結果是 mod 
def lagrange_interpolation_coefficients(x_points, y_points, mod_value):
    coefficients = [0] * len(x_points)
    n = len(x_points)
    
    for i in range(n):
        # 確保 xi, yi 在有限域
        xi, yi = x_points[i] % mod_value, y_points[i] % mod_value
        
        # Start with the coefficient for the Lagrange basis polynomial L_i(x)
        # 從 yi 開始，初始化第 i 個 Lagrange polynomial basis 的係數
        term_coefficients = [yi]
        
        for j in range(n):
            if i != j:
                xj = x_points[j] % mod_value
                
                # Each term in the Lagrange basis polynomial contributes (x - xj)/(xi - xj)
                # We need to adjust our current coefficients to account for this new term.
                # 計算 Lagrange polynomial basis 的係數，並逐項更新
                new_term_coefficients = [0] * (len(term_coefficients) + 1)
                inv = mod_inv((xi - xj + mod_value) % mod_value, mod_value) # 計算 mod inverse
                
                for k in range(len(term_coefficients)):
                    new_term_coefficients[k] = (new_term_coefficients[k] - term_coefficients[k] * xj * inv) % mod_value
                    new_term_coefficients[k + 1] = (new_term_coefficients[k + 1] + term_coefficients[k] * inv) % mod_value
                term_coefficients = new_term_coefficients
        
        # Add the current Lagrange basis polynomial's contribution to the overall coefficients
        # 將目前的 Lagrange polynomial basis 累加到總係數中
        for k in range(len(term_coefficients)):
            coefficients[k] = int((coefficients[k] + term_coefficients[k]) % mod_value)
    
    # 回傳低到高次的係數
    return coefficients[::-1]

def polynomial_to_string_high_to_low(coefficients):
    terms = []
    for i, coef in reversed(list(enumerate(coefficients))):  # Reverse to start from the highest degree
        if coef == 0:
            continue  # Skip terms with a coefficient of 0
        elif i == 0:
            terms.append(f"{coef}")  # The constant term
        elif i == 1:
            terms.append(f"{coef}*x")  # The linear term
        else:
            terms.append(f"{coef}*x^{i}")  # Higher degree terms
    return " + ".join(terms)

# 记录开始时间
commit_P_start_time = time.time()


# 初始化 f(0), ..., f(2000), 共 2001 個點
# x_points = list(range(2001))
# y_points = [relu_custom(x) for x in x_points]
x_points = dataset['pseudo_ID'].values
y_points = dataset['commitments'].values

# 設大於 y_points 的質數 modulus
new_modulus= nextprime(p)

# 計算在 mod 值為 new_modulus 下的拉格朗日插值的函數的係數
coefficients = lagrange_interpolation_coefficients(x_points, y_points, new_modulus)

# 印出多項式
# print("---------------------")
# print("Coefficients of the polynomial: (from the highest to the lowest degree)")
# print(list(map(int, coefficients[::-1])))  # 係數由高次到低次

# 记录结束时间
commit_P_end_time = time.time()

print("---------------------")
polynomial_str_high_to_low = polynomial_to_string_high_to_low(coefficients)
print("Commit - Polynomial:")
print(polynomial_str_high_to_low)


# 计算并打印执行时间
commit_P_time = commit_P_end_time - commit_P_start_time
print(f"Commit - P time: {commit_P_time} seconds")


# In[] commit V 個別驗證所有

# 測試部分
# 代入計算出的多項式係數，驗證 0 ~ N-1 所有整數點的插值皆正確
def evaluate_polynomial(coefficients, mod_value, x_range = range(N)):
    coefficients = coefficients[::-1]  # 係數由高次到低次
    evaluated_points = []
    
    for x in x_range:
        y = 0
        
        # Evaluate using Horner's method for numerical stability and efficiency
        # 用 Horner's method 快速計算多項式的 mod_value
        for coef in reversed(coefficients):
            y = (y * x + coef) % mod_value
        
        evaluated_points.append(y)
    
    return evaluated_points

# 记录开始时间
commit_V_start_time = time.time()

# 由產生的多項式，驗算後的值
evaluated_points = evaluate_polynomial(coefficients, new_modulus)

# 檢查 0 ~ 2001 代入的值相同，全部正確則 correct_values = true
correct_values = all(original == evaluated for original, evaluated in zip(y_points, evaluated_points))

# 记录结束时间
commit_V_end_time = time.time()

print("---------------------")
print("(Test) Correctness of the coefficients:", correct_values)

print("---------------------")
print("(Test) The result after interpolation should be the input of 'commitments' column")
print(evaluated_points)


# 计算并打印执行时间
print("---------------------")
commit_V_time = commit_V_end_time - commit_V_start_time
print(f"Commit - V time: {commit_V_time} seconds")



# In[] 



