# %%
import pandas as pd

df = pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')
df.head()
features = []
for feature in df.columns:
    if feature != 'heart_disease':
        features.append(feature)
X = df[features]
y = df['heart_disease']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
import matplotlib.pyplot as plt
count = y_train.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('heart_disease')
plt.show()