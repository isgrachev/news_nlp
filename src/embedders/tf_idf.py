import os
# print(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow
from custom_tokenizer import CustomTokenizer


# retreive text
train_corpus = list(pd.read_parquet('misc/data/text_train.parquet').loc[:, 'text'])
val_corpus = list(pd.read_parquet('misc/data/text_val.parquet').loc[:, 'text'])

# apply tokenization, remove stop-words and non-character elements, apply stemming
tokenizer = CustomTokenizer()
train_corpus = [' '.join(article) for article in tokenizer.fit_transform(train_corpus, clean=True, stem=True)]
val_corpus = [' '.join(article) for article in tokenizer.transform(val_corpus)]
vocab = tokenizer.vocabulary()

# embed tokenized documents with tf-idf
tfidf = TfidfVectorizer(lowercase=False, vocabulary=vocab)
tfidf_embeddings = tfidf.fit_transform(train_corpus)
tfidf_embeddings_val = tfidf.transform(val_corpus)

# save embeddings
pd.DataFrame(tfidf_embeddings.toarray(), columns=list(vocab.keys())).to_parquet('misc/embedded/tfidf_train.parquet', index=False)
pd.DataFrame(tfidf_embeddings_val.toarray(), columns=list(vocab.keys())).to_parquet('misc/embedded/tfidf_val.parquet', index=False)
