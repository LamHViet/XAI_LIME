# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
import numpy as np

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
X_train_float = X_train.to_numpy(dtype=np.float64)
X_test_float = X_test.to_numpy(dtype=np.float64)
print(X_train.shape)
print(X_test.shape)
# print(X_train_float)
# print(X_test)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average ='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Plot the confusion matrix.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['YES', 'NO'],
            yticklabels=['YES', 'NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

# %% Apply lime
# Initilize Lime for Tabular data
lime = LimeTabular(model = rf,
                   feature_names = X_train.columns,
                   data = X_train_float, 
                   random_state = 1)
# Get local explanations
lime_local = lime.explain_local(X_test[300:400], 
                                y_test[300:400], 
                                name='LIME')
show(lime_local)
# %%