
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Local filename assumption when run by HTCondor
LOCAL_FILENAME = 'heart.csv'

if os.path.exists(LOCAL_FILENAME):
    data_path = LOCAL_FILENAME
else:
    # Fallback for local development
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    data_path = os.path.join(BASE_DIR, 'Data', LOCAL_FILENAME)

output_dir = 'EDA'
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_csv(data_path)

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
