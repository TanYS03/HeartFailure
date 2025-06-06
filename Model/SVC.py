import pandas as pd
import os
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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(_file_))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    file_path = os.path.join(BASE_DIR, 'Data', LOCAL_FILENAME)

# Load the dataset
data = pd.read_csv(file_path)

# Mappings
Sex_Mapper = {'M': 2, 'F': 3}
ChestPainType_Mapper = {'TA': 3, 'ATA': 3, 'NAP': 4, 'ASY': 5}
RestingECG_Mapper = {'Normal': 2, 'ST': 3, 'LVH': 4}
ExerciseAngina_Mapper = {'Y': 2, 'N': 3}
ST_Slope_Mapper = {'Up': 2, 'Flat': 3, 'Down': 4}

# Apply mappings
df = data.copy()
df.replace({
    'Sex': Sex_Mapper,
    'ChestPainType': ChestPainType_Mapper,
    'RestingECG': RestingECG_Mapper,
    'ExerciseAngina': ExerciseAngina_Mapper,
    'ST_Slope': ST_Slope_Mapper
}, inplace=True)

# Filter data
df_filtered = df[(df["Cholesterol"] >= 50) & (df["Oldpeak"] >= 0)]

# Split dataset
X = df_filtered.drop("HeartDisease", axis=1)
Y = df_filtered["HeartDisease"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# Grid search for best parameters
SVM = SVC()
svm_params = {
    'C': [1.0, 2.0],
    'kernel': ['linear', 'rbf'],
    'degree': [3],
    'gamma': ['scale'],
    'coef0': [0.0],
    'shrinking': [True],
    'probability': [False],
    'cache_size': [200],
    'class_weight': [None],
    'max_iter': [-1],
    'decision_function_shape': ['ovr', "ovo"],
    'break_ties': [False],
    'random_state': [None]
}
SVMG = GridSearchCV(estimator=SVM, param_grid=svm_params, cv=3, scoring='accuracy', n_jobs=-1)
SVMG1 = SVMG.fit(X_train, Y_train)
best_paramsSVM = SVMG1.best_params_

print("\n=== Best Parameters ===")
print(best_paramsSVM)

# Train with best params
SVM1 = SVC(**best_paramsSVM)
SVM1.fit(X_train, Y_train)
Y_pred = SVM1.predict(X_test)

# Accuracy
accuracy = metrics.accuracy_score(Y_test, Y_pred)
print(f"\n=== Accuracy ===\n{accuracy * 100:.2f}%")

# Confusion Matrix as Table
cm = confusion_matrix(Y, SVM1.predict(X))
cm_df = pd.DataFrame(cm, 
                     index=["Actual 0", "Actual 1"], 
                     columns=["Predicted 0", "Predicted 1"])
print("\n=== Confusion Matrix ===")
print(cm_df)

# Classification Report as Table
report_dict = classification_report(Y, SVM1.predict(X), output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print("\n=== Classification Report ===")
print(report_df.round(2))
