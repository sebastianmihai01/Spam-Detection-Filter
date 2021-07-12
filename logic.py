import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
import nltk
import matrix_tokeniaztion as mt

# load the dataset if doen with google colab
# dataset = files.upload()


""" structure: text | (1/0 <- spam) """
# read file
# df = data frame
df = pd.read_csv('dataset-spam.csv')

n = 10
df.head(n)  # print the first n words

# return the shape in the form of ('rows', 'columns')
df.shape

# get the column names
df.columns  # => text & spam

# delete duplicates
df.drop_duplicates(inplace=True)
df.shape

# for every column, show the number of data types like NAN, NaN, na
df.isnull.sum()

# download the stopwords package
nltk.download('stopwords')

"""
first remove punctuation
secondly remove stopwords like ("I like reading, so I read" => text without stopwords: like, reading, read)
finally: return the neatly formatted text
"""

# text = email
def process_text(text):

    # the text without punctuation is now in 'nopunc'
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

    # show the tokenization (list of tokens, also called lemmas)
    # tokenize (separate the words by commas)
    # df['text'] = text column

    df['text'].head(n).apply(process_text)  # get the first n rows

    # Convert a collection of text to a matrix of tokens
    mt.tokenize(df)
    # print(mt.tokenize(df))
