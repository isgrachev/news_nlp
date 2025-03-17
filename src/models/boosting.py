import os
print(os.getcwd())

import numpy as numpy
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

from sklearn.metrics import confusion_matrix, classification_report


# load data 
emb_train = pd.read_parquet('misc/embedded/tfidf_train.parquet')
label_train = pd.read_parquet('misc/data/labels_train.parquet')
emb_val = pd.read_parquet('misc/embedded/tfidf_val.parquet')
label_val = pd.read_parquet('misc/data/labels_val.parquet')

# train and make estimates
train_data = lgb.Dataset(emb_train.values, label=label_train.values, feature_name=list(emb_train.columns))
validation_data = lgb.Dataset(emb_val.values, label=label_val.values, feature_name=list(emb_val.columns), reference=train_data)

params = {'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss', 'num_threads': -1, 'seed': 124, 'early_stopping_round': 5}

boosting = lgb.train(params=params, train_set=train_data, valid_sets=(validation_data, ))
boosting_val = np.argmax(boosting.predict(emb_val), axis=1)

# return results
print('\n')
print('Gradient Boosting on TF-IDF embeddings validation result:')
print('\n')
print(classification_report(label_val, boosting_val))
print('\n')
