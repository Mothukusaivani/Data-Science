#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the CSV file
file_path = "calories.csv"  # Ensure correct file path
df = pd.read_csv(file_path)

# Keep only the important features
selected_features = ['Weight', 'Exercise Duration', 'Heart Rate', 'Calories Burnt']
df = df[selected_features]

# Define Features (X) and Target (y)
X = df.drop(columns=['Calories Burnt'])  # Features: Weight, Duration, Heart Rate
y = df['Calories Burnt']  # Target variable

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

# Get User Input
print("\nEnter details to predict calories burnt:")
weight = float(input("Enter Weight (kg): "))
duration = float(input("Enter Exercise Duration (min): "))
heart_rate = float(input("Enter Heart Rate (bpm): "))

# Create New Data Input
new_data = np.array([[weight, duration, heart_rate]])

# Scale and Predict
new_data_scaled = scaler.transform(new_data)
predicted_calories = model.predict(new_data_scaled)

print(f"\nPredicted Calories Burnt: {predicted_calories[0]:.2f}")


# In[ ]:




