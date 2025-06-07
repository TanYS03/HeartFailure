import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import r2_score,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,ComplementNB,GaussianNB
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.model_selection import cross_val_score
from IPython.core.display import HTML
import warnings
warnings.filterwarnings("ignore")

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

if data.isnull().sum().any():
    print("Warning: Missing values detected. Proceeding to drop them.")
    data = data.dropna()

# Map categorical variables to numerical values
mappings = {
    'Sex': {'M': 2, 'F': 3},
    'ChestPainType': {'TA': 3, 'ATA': 3, 'NAP': 4, 'ASY': 5},
    'RestingECG': {'Normal': 2, 'ST': 3, 'LVH': 4},
    'ExerciseAngina': {'Y': 2, 'N': 3},
    'ST_Slope': {'Up': 2, 'Flat': 3, 'Down': 4}
}

for column, mapping in mappings.items():
    if column in data.columns:
        data[column] = data[column].map(mapping)

# Filter the dataset based on Cholesterol and Oldpeak values
data_filtered = data[(data["Cholesterol"] >= 50) & (data["Oldpeak"] >= 0)]

output_dir = 'EDA'
os.makedirs(output_dir, exist_ok=True)

# Setting aesthetics for the plots
sns.set(style="whitegrid")

# KDE Plots for numerical features
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[feature], fill=True, color="red", label='HeartDisease')
    sns.kdeplot(data[data["HeartDisease"] == 0][feature], fill=True, color="blue", label='HeartDisease=0')
    sns.kdeplot(data[data["HeartDisease"] == 1][feature], fill=True, color="yellow", label='HeartDisease=1')
    plt.legend()
    plt.title(f'KDE Plot for {feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, f'{feature}_kde.png'))
    plt.close()

# Count Plots for categorical features
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=feature, hue='HeartDisease', palette=["#1F509A", "#CC2B52"])
    plt.title(f'Count Plot of {feature} by Heart Disease Class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Heart Disease Class')
    plt.savefig(os.path.join(output_dir, f'{feature}_count.png'))
    plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = data.select_dtypes(include='number').corr()
sns.heatmap(corr, cmap="Purples", annot=True, linewidths=.7)
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# Scatter Plots for selected feature pairs
scatter_pairs = [
    ('ST_Slope', 'Oldpeak'),
    ('ExerciseAngina', 'MaxHR'),
    ('ExerciseAngina', 'Oldpeak'),
    ('Oldpeak', 'HeartDisease'),
    ('Age', 'RestingBP')
]

for x_feature, y_feature in scatter_pairs:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue="HeartDisease", palette='coolwarm')
    plt.title(f'Scatter Plot of {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Heart Disease')
    plt.savefig(os.path.join(output_dir, f'{x_feature}vs{y_feature}_scatter.png'))
    plt.close()
    
    

# Value mapping
Sex_Mapper = {'M': 2, 'F': 3}
ChestPainType_Mapper = {'TA': 3, 'ATA': 3, 'NAP': 4, 'ASY': 5}
RestingECG_Mapper = {'Normal': 2, 'ST': 3, 'LVH': 4}
ExerciseAngina_Mapper = {'Y': 2, 'N': 3}
ST_Slope_Mapper = {'Up': 2, 'Flat': 3, 'Down': 4}

df = data.copy()
df.replace({
    'Sex': Sex_Mapper,
    'ChestPainType': ChestPainType_Mapper,
    'RestingECG': RestingECG_Mapper,
    'ExerciseAngina': ExerciseAngina_Mapper,
    'ST_Slope': ST_Slope_Mapper
}, inplace=True)



# Features and labels
X = data_filtered.drop("HeartDisease", axis=1)
Y = data_filtered["HeartDisease"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.35, random_state=0)

# Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("\n=== GaussianNB Accuracy ===")
print(f"{metrics.accuracy_score(Y_test, Y_pred)*100:.2f}%")

# Check negative values for MultinomialNB
print("\n=== Negative Value Count in X_train ===")
print((X_train < 0).sum())

# Multinomial Naive Bayes
clf1 = MultinomialNB()
clf1.fit(X_train, Y_train)
Y_pred1 = clf1.predict(X_test)

print("\n=== MultinomialNB Accuracy ===")
print(f"{metrics.accuracy_score(Y_test, Y_pred1)*100:.2f}%")

# Confusion matrix and classification report (GaussianNB)
cmNB = confusion_matrix(Y, clf.predict(X))
cm_df = pd.DataFrame(cmNB, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
print("\n=== Confusion Matrix (GaussianNB) ===")
print(cm_df)

report_dict = classification_report(Y, clf.predict(X), output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
print("\n=== Classification Report (GaussianNB) ===")
print(report_df)

# K-Fold Cross-Validation
k_fold = KFold(n_splits=5)
cv_scores = cross_val_score(clf, X, Y.ravel(), cv=k_fold, n_jobs=1)
print("\n=== 5-Fold Cross-Validation Scores (GaussianNB) ===")
print(cv_scores)
print("Average CV Score: {:.2f}%".format(cv_scores.mean() * 100))



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
    'max_features':["sqrt",None,"log2"]
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
