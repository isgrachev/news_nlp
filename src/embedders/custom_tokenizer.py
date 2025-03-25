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

    def fit(self, arr, clean=True, stem=False, lemmatize=False, cutoff=0):
        """
        
        """

        assert not (stem and lemmatize), "Can't stem and lemmatize simultanously, choose 1 method"
        self.clean = clean
        self.stem = stem
        self.lemmatize = lemmatize
        self.max_len = 0
        self.counter = Counter()

        for article in arr:
            temp = self._normalize(article, clean, stem, lemmatize)
            self.max_len = max(self.max_len, len(temp))
            self.counter.update(temp)

        self.vocabulary = self._get_vocab(self.counter, cutoff)
        self.vocabulary['<UNK>'] = len(self.vocabulary)
        self.vocabulary['<PAD>'] = len(self.vocabulary)
    
    def fit_transform(self, arr, clean=True, stem=False, lemmatize=False, cutoff=0, pad=False):
        """
        
        """
        
        self.fit(arr, clean, stem, lemmatize, cutoff)
        tokenized = self.transform(arr, pad)
        
        return tokenized

    def transform(self, arr, pad=False):
        """
        
        """

        try:
            self.vocabulary
        except:
            raise NameError('The instance of the object must be fit to the data. Use .fit or .fit_transform methods before applying .transform')
        
        tokenized = []
        
        for article in arr:
            temp = self._normalize(article, self.clean, self.stem, self.lemmatize)
            temp = [token if token in self.vocabulary.keys() else '<UNK>' for token in temp]
            while (pad and len(temp) < self.max_len):
                temp.append('<PAD>')
            tokenized.append(temp)
        
        return tokenized

    def _get_vocab(self, counter, cutoff):
        """
        
        """

        assert len(counter) > 0, 'Counter is empty, fit the tokenizer to the data first'

        vocab = [token[0] for token in counter.items() if token[1] > cutoff]

        return dict(zip(vocab, list(range(0, len(vocab)))))
