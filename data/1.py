import os
import pickle
import numpy as np


cache_file1 = 'train.pkl'
cache_file2 = 'train1.pkl'

with open(cache_file1,'rb') as f1:
    gt_labels = pickle.load(f1)

train_labels = []

for i in range(len(gt_labels)):
    path = gt_labels[i]['path']
    a = path.split('/')
    

