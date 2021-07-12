# Our model for predictions
# Create and train the Naive Bayes classifier
# Multinomial - classification for discrete features (word count for text)

# NB = naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# fit = train
def train(X_train, y_train):
    classifier = MultinomialNB().fit(X_train, y_train)

    # print the predictions
    print(classifier.predict(X_train))  # [0,0 .. 0]

    # print the actual values
    # (.values = the val from key-val dictioary)
    print(y_train.values)  # [0,0 .. 0]

    # Evaluate the model on the training data set
    pred = classification_report(X_train)

    # y_train = target/actual values, pred = predicted values
    print(classification_report(y_train, pred))
