import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
LOCAL_FILENAME = 'heart.csv'

if os.path.exists(LOCAL_FILENAME):
    file_path = LOCAL_FILENAME
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    file_path = os.path.join(BASE_DIR, 'Data', LOCAL_FILENAME)

data = pd.read_csv(file_path)

# Value mapping
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

# Filtering
df2 = df[(df["Cholesterol"] >= 50) & (df["Oldpeak"] >= 0)]

# Features and labels
X = df2.drop("HeartDisease", axis=1)
Y = df2["HeartDisease"]

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
