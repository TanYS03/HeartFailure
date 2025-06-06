
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def perform_eda(data, output_dir='output'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # KDE Plots for numerical features
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[feature], shade=True, color="red", label='HeartDisease')
        sns.kdeplot(data[data["HeartDisease"] == 0][feature], shade=True, color="blue", label='HeartDisease=0')
        sns.kdeplot(data[data["HeartDisease"] == 1][feature], shade=True, color="yellow", label='HeartDisease=1')
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
    corr = data.corr()
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
        plt.savefig(os.path.join(output_dir, f'{x_feature}_vs_{y_feature}_scatter.png'))
        plt.close()
