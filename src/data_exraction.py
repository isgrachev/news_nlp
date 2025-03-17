import os
# os.chdir('..')
print(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_parquet('misc/data/raw.parquet')

labels = data[['label', 'label_text']].drop_duplicates('label').sort_values(by='label')
with open('misc/data/txt_labels.txt', 'w') as file:
    for row in labels.values:
        file.write(f'{str(row[0])} {row[1]}')
    del labels

data.drop(columns=['label_text'], inplace=True)

X_train, X_val, y_train, y_val = train_test_split(data[['text']], data[['label']], train_size=0.8, random_state=235)

X_train.to_parquet('misc/data/text_train.parquet', index=False)
X_val.to_parquet('misc/data/text_val.parquet', index=False)
y_train.to_parquet('misc/data/labels_train.parquet', index=False)
y_val.to_parquet('misc/data/labels_val.parquet', index=False)
