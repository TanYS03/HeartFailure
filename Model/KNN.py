import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

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

kknn=20
Acc=np.zeros(kknn)
for i in range (1,kknn+1):
    clfknn=KNeighborsClassifier(algorithm="auto",n_neighbors=i)
    clfknn.fit(X_train,Y_train.to_numpy())
    Y_pred_knn=clfknn.predict(X_test)
    Acc[i-1]=metrics.accuracy_score(Y_test,Y_pred_knn)
Acc

print(max(Acc))

parameters = {"n_neighbors": range(1, 50)}
clfknnCV=KNeighborsClassifier(algorithm="auto")
grid_kn = GridSearchCV(
    estimator=clfknnCV,  # Model
    param_grid=parameters,  # Range of k
    scoring='accuracy',  # Strategy to evaluate performance
    cv=5,  # Cross-validation folds
    verbose=1,  # Verbosity level
    n_jobs=-1  # Use all available CPU cores
)

# Fit the model
grid_kn.fit(X_train, Y_train.to_numpy())
print("Best Parameters:", grid_kn.best_params_)
