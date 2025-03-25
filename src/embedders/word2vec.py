import numpy as np
import pandas as pd
import os
os.chdir('..')
print(os.getcwd())
from src.embedders import CustomTokenizer
from gensim.models import Word2Vec


# upload training sample
data = list(pd.read_parquet('misc/data/text_train.parquet').loc[:, 'text'])
val = list(pd.read_parquet('misc/data/text_val.parquet').loc[:, 'text'])

# preprocess training documents
tokenizer = CustomTokenizer()
train_tokenized = tokenizer.fit_transform(data, lemmatize=True, cutoff=5, pad=True)
val_tokenized = tokenizer.transform(val)

# embed based on training documents
embedder = Word2Vec(train_tokenized, vector_size=64, seed=125)
embeddings = pd.DataFrame(data=embedder.wv.vectors, index=embedder.wv.index_to_key, columns=[str(x) for x in range(64)])

train_embedded = []
for corpus in train_tokenized:
    train_embedded.append([embeddings.loc[token, :].values for token in corpus])
train_embedded = np.array(train_embedded)
np.save('misc/embedded/w2v_train.npy', train_embedded)
del train_embedded

val_embedded = []
for corpus in val_tokenized:
    val_embedded.append([embeddings.loc[token, :].values for token in corpus])
val_embedded = np.array(val_embedded)
np.save('misc/embedded/w2v_val.npy', val_embedded)

# save embeddings in a parquet with vectors as row values
embeddings.to_parquet('misc/embedded/w2v.parquet')
