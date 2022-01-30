import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


data_path = './../data/cancer_modified.csv'
modified_df = pd.read_csv(data_path)

train_df, test_df = train_test_split(modified_df, test_size = 0.2, random_state = 1)

x = ["perimeter_mean"]
y = ["diagnosis"]
x_train_df = train_df[x]
x_test_df = test_df[x]
y_train_df = train_df[y]
y_test_df = test_df[y]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", C=0.025)]

for classifier in classifiers:
  print("Experimenting with: ", classifier)
  classifier.fit(x_train_df, y_train_df)
  y_prediction = classifier.predict(x_test_df)
  print("Accuracy: ", metrics.accuracy_score(y_test_df, y_prediction))
  print("Precision: ", metrics.precision_score(y_test_df, y_prediction))
  print("Recall: ", metrics.recall_score(y_test_df, y_prediction)) 
  print("—————————")
  print("—————————")