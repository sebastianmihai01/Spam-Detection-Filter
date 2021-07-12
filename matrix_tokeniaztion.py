# we want all our data to be divided into token counts
import logic
from sklearn.model_selection import train_test_split
# convert the text to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
import classifier as cl

# bag_of_words = CountVectorizer(analyzer= logic.process_text).fit_transform([[message1], [message2]])

""" 
(a, b)  nr
 
(0, 0)  3
(0, 4)  2
(0, 2)  1

a1, a2, a3 (the first zero from every tuple (x,y)) is message1
3 tuples starting with 0 => we have 3 UNIQUE words => 6-3 words are the same
3, 2, 1 (second column) => how many times a certain word appears (one word 3 times, one 2, one once)
"""

# print(bag_of_words.shape) => (2, 5), 2 rows and 5 columns
#                              which are the 5 words per 2 messages


def tokenize(df):
    # create dataset
    messages_bow = CountVectorizer(analyzer=logic.process_text).fit_transform(df['text'])

    # split the data into 80% training and 20% testing
    X_train, X_test = 0,0 # FEATURE training set
    y_train, y_test = 0,0 # TARGET training set

    # random_state = randomizer coefficient
    X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size=0.2, random_state=0)
    cl.train(X_train=X_train, y_train=y_train)
