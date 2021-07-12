"""
Cross validation is the approach.
To divide your dataset into n consecutive folds, use the KFold( ) method in scikit-learn.

"""
import numpy as np
from sklearn.model_selection import KFold

X = None
y = None

"""
Both the X and Y sets were separated into five folds when the n splits parameter was set to 5. 
(the y sets now shown here). You may have observed that the software always chose two nearby integers
 from the original data sets this time, indicating that the data points were not shuffled 
 (why is the default option of the shuffle parameter different here than in train test split?)
"""

kf = KFold(n_splits=5)
X = np.array(X)
y = np.array(y)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_test: ", X_test)
# X_test: [0 1]
# X_test: [2 3]
# X_test: [4 5]
# X_test: [6 7]
# X_test: [8 9]