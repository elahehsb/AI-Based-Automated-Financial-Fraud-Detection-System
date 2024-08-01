import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    # Handle missing values if any
    df = df.dropna()
    
    # Feature and target variables
    X = df.drop('Class', axis=1)  # Assuming 'Class' is the target variable
    y = df['Class']
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

df = load_data('path_to_transactions.csv')
(X_train, X_test, y_train, y_test), scaler = preprocess_data(df)

# Save the scaler for later use
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
