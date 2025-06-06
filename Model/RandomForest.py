# Import necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Local filename assumption when run by HTCondor
LOCAL_FILENAME = 'heart.csv'

if os.path.exists(LOCAL_FILENAME):
    file_path = LOCAL_FILENAME
else:
    # Fallback for local development
    SCRIPT_DIR = os.path.dirname(os.path.abspath(_file_))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    file_path = os.path.join(BASE_DIR, 'Data', LOCAL_FILENAME)

# Load the dataset
data = pd.read_csv(file_path, parse_dates=['Timestamp'])
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


RF=RandomForestClassifier(n_estimators=10,max_depth=3)
RF=RF.fit(X_train,Y_train)
Y_predRF=RF.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test,Y_predRF)*100)

param_gridRF = {
    'n_estimators': [10,20,50],
    'bootstrap': [True,False],
    'max_depth': [ 2, 3, 5, 10,15],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 5],     
    'criterion': ['gini', 'entropy'],
    'verbose': [0,1,2,3],
    'max_features':["sqrt","None","log2"]
}

grid_searchRF = GridSearchCV(estimator=RF, param_grid=param_gridRF, cv=5, scoring='accuracy', n_jobs=-1)
grid_searchRF.fit(X_train, Y_train)
best_paramsRF = grid_searchRF.best_params_
print("Best Parameters:", best_paramsRF)
bp={'bootstrap': False, 'criterion': 'gini', 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 20, 'verbose': 3}
RF1=RandomForestClassifier(**bp)
R1F=RF1.fit(X_train,Y_train)
Y_predRF1=RF1.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test,Y_predRF1)*100)


