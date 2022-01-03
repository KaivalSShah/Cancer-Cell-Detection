import PIL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = './data/cancer.csv'
dataframe = pd.read_csv(data_path)
dataframe_copy = pd.DataFrame.copy(dataframe)

dataframe_copy = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe_copy['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
dataframe_copy['diagnosis_binary'] = dataframe_copy['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})
print(dataframe_copy.head())