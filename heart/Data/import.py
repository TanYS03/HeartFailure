import pandas as pd

def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('data/heart.csv')

    # Check for missing values
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

    return data_filtered