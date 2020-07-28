import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target
data.target_names

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def train_and_plot(X_train, y_train, X_test, y_test, penalty="l2"):
    print(f"Penalty {penalty}")
    if penalty == 'elasticnet':
        model = LogisticRegression(solver='saga', penalty=penalty, l1_ratio=0.7)
    else:
        model = LogisticRegression(solver='saga', penalty=penalty)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', model)])

    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
    importance = model.coef_[0]

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

train_and_plot(X_train, y_train, X_test, y_test, "none")
train_and_plot(X_train, y_train, X_test, y_test, "l2")
train_and_plot(X_train, y_train, X_test, y_test, "l1")

X_train_outlier = X_train.append(X_train.iloc[172]*10,  ignore_index=True)
y_train_outlier = y_train.append(pd.Series([1]), ignore_index=True)

train_and_plot(X_train_outlier, y_train_outlier, X_test, y_test, "none")
train_and_plot(X_train_outlier, y_train_outlier, X_test, y_test, "l2")
train_and_plot(X_train_outlier, y_train_outlier, X_test, y_test, "l1")






from sklearn.datasets import load_iris
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf =LogisticRegression()
clf =LogisticRegression()
clf.predict([X_test[0]])
clf.predict_proba([X_test[0]])