import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# Function to preprocess the dataset
def preprocess_data(df):
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert date strings to days since values
    current_date = datetime.now()
    data['Days_Since_Last_Service'] = data['Last_Service_Date'].apply(
        lambda x: (current_date - datetime.strptime(x, '%Y-%m-%d')).days)
    data['Days_Until_Warranty_Expiry'] = data['Warranty_Expiry_Date'].apply(
        lambda x: (datetime.strptime(x, '%Y-%m-%d') - current_date).days)
    
    # Drop original date columns
    data = data.drop(['Last_Service_Date', 'Warranty_Expiry_Date'], axis=1)
    
    # Create categorical features list
    categorical_features = [
        'Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 
        'Transmission_Type', 'Owner_Type', 'Tire_Condition', 
        'Brake_Condition', 'Battery_Status'
    ]
    
    # Create numerical features list
    numerical_features = [
        'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size',
        'Odometer_Reading', 'Insurance_Premium', 'Service_History',
        'Accident_History', 'Fuel_Efficiency', 'Days_Since_Last_Service',
        'Days_Until_Warranty_Expiry'
    ]
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(data[categorical_features])
    
    # Create DataFrame with encoded categories
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(data[numerical_features])
    
    # Create DataFrame with scaled numerical features
    scaled_df = pd.DataFrame(
        scaled_nums,
        columns=numerical_features
    )
    
    # Combine processed features
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Add target variable
    processed_data['Need_Maintenance'] = data['Need_Maintenance']
    
    return processed_data, scaler, encoder

# Example usage (you would replace this with your actual data loading)
df = pd.read_csv('vehicle_maintenance_data.csv')
processed_data, scaler, encoder = preprocess_data(df)

# Split data into training and testing sets
X = processed_data.drop('Need_Maintenance', axis=1)
y = processed_data['Need_Maintenance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessors for inference
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Add these lines to save the actual preprocessed data
# Save the full preprocessed dataset
processed_data.to_csv('processed_data.csv', index=False)

# Save the train/test split data
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')

# Alternatively, save all split data in one file
# joblib.dump((X_train, X_test, y_train, y_test), 'train_test_data.pkl')