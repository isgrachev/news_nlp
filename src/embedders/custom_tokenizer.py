import os
print(os.getcwd())
from collections import Counter

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class CustomTokenizer():
    """
    
    """

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = PorterStemmer()
        self.counter = Counter()
        self.lemmatizer = WordNetLemmatizer()

    def _normalize(self, article, clean, stem, lemmatize):
        temp = self.tokenizer.tokenize(article)
        
        if clean:
            temp = [token.lower() for token in temp if (token.isalpha() and token.lower() not in stop_words)]
        
        if stem:
            temp = [self.stemmer.stem(token) for token in temp]
        elif lemmatize:
            temp = [self.lemmatizer.lemmatize(token) for token in temp]
        
        return temp

    def fit(self, arr, clean=True, stem=False, lemmatize=False):
        """
        
        """

        assert not (stem and lemmatize), "Can't stem and lemmatize simultanously, choose 1 method"
        self.clean = clean
        self.stem = stem
        self.lemmatize = lemmatize
        for article in arr:
            temp = self._normalize(article, clean, stem, lemmatize)
            self.counter.update(temp)
    
    def fit_transform(self, arr, clean=True, stem=False, lemmatize=False):
        """
        
        """
        
        assert not (stem and lemmatize), "Can't stem and lemmatize simultanously, choose 1 method"
        self.clean = clean
        self.stem = stem
        self.lemmatize = lemmatize

        tokenized = []
        for article in arr:
            temp = self._normalize(article, clean, stem, lemmatize)
            tokenized.append(temp)
            self.counter.update(temp)
        
        return tokenized

    def transform(self, arr):
        """
        
        """

        try:
            self.stem
        except:
            raise NameError('The instance of the object must be fit to the data. Use .fit or .fit_transform methods before applying .transform')
        
        tokenized = []
        for article in arr:
            temp = self._normalize(article, self.clean, self.stem, self.lemmatize)
            tokenized.append(temp)
        
        return tokenized
    
    def vocabulary(self):
        """
        
        """

        assert len(self.counter) > 0, 'Counter is empty, fit the tokenizer to the data first'

        return dict(zip(self.counter.keys(), list(range(0, len(self.counter)))))
