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

N = 512
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














