import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

data_path = './../data/cancer_modified.csv'
modified_df = pd.read_csv(data_path)

categories = ['perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']
for category in categories:
    sns.catplot(x = category, y = 'diagnosis_binary', data = modified_df, order=['1 (malignant)', '0 (benign)'])
plt.show()