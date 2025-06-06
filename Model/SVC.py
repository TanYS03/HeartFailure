import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Local filename assumption when run by HTCondor
LOCAL_FILENAME = 'heart.csv'

if os.path.exists(LOCAL_FILENAME):
    file_path = LOCAL_FILENAME
else:
    # Fallback for local development
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    file_path = os.path.join(BASE_DIR, 'Data', LOCAL_FILENAME)

# Load the dataset
data = pd.read_csv(file_path)

HD0=data[data["HeartDisease"]==0]
HD1=data[data["HeartDisease"]==1]

Sex_Mapper={'M':2,'F':3}
ChestPainType_Mapper={'TA':3,'ATA':3,'NAP':4,'ASY':5}
RestingECG_Mapper={'Normal':2,'ST':3,'LVH':4}
ExerciseAngina_Mapper={'Y':2,'N':3}
ST_Slope_Mapper={'Up':2,'Flat':3,'Down':4}

df=data
df1=df

df1.replace({
    'Sex': Sex_Mapper,
    'ChestPainType': ChestPainType_Mapper,
    'RestingECG': RestingECG_Mapper,
    'ExerciseAngina': ExerciseAngina_Mapper,
    'ST_Slope': ST_Slope_Mapper
}, inplace=True)



df2 = df1[(df1["Cholesterol"] >= 50) & (df1["Oldpeak"] >= 0)]

# Split dataset
X=df2.drop("HeartDisease",axis=1)
Y=df2["HeartDisease"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.35,random_state=0)


SVM=SVC()
svm_params = {
    'C': [1.0,2.0],
    'kernel': ['linear',  'rbf'],
    'degree': [3],
    'gamma': ['scale'],
    'coef0': [0.0],
    'shrinking': [True],
    'probability': [False],
    'cache_size': [200],
    'class_weight': [None],
    'max_iter': [-1],
    'decision_function_shape': ['ovr',"ovo"],
    'break_ties': [False],
    'random_state': [None]
}
SVMG=GridSearchCV(estimator=SVM, param_grid=svm_params, cv=3, scoring='accuracy', n_jobs=-1)
SVMG1=SVMG.fit(X_train,Y_train)
best_paramsSVM=SVMG1.best_params_
print("Best Parameters:", best_paramsSVM)
SVMPARAMS={'C': 2.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True}
SVM1=SVC(**SVMPARAMS)
SVM2=SVM1.fit(X_train,Y_train)
Y_predSVM1=SVM2.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test,Y_predSVM1)*100)
SVMCM = confusion_matrix(Y, SVM1.predict(X))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(SVMCM, cmap='viridis')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Predicted 0s', 'Predicted 1s'], fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Actual 0s', 'Actual 1s'], fontsize=12)

for i in range(2):
    for j in range(2):
        ax.text(j, i, SVMCM[i, j], ha='center', va='center', 
                color='white', 
                fontsize=14, fontweight='bold')

ax.grid(False)
plt.tight_layout()
plt.show()
print(classification_report(Y, SVM1.predict(X)))
