# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:25:21 2023

@author: Aoife O'Connor

lung_cancer_data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# Load and preprocess the lung_cancer dataset
lung_cancer_data = pd.read_csv('lung_cancer_data.csv')

# Display the first few rows of the dataset
#print(lung_cancer_data.head())

# Encode the target variable ('LUNG_CANCER') as binary (1 for 'YES' and 0 for 'NO')
lung_cancer_data['LUNG_CANCER'] = lung_cancer_data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 1, 'F': 0})

# Checking the distribution of the target variable
print(lung_cancer_data['LUNG_CANCER'].value_counts())

# Scaling the numerical features
numerical_features = ['AGE']
scaler = StandardScaler()
lung_cancer_data[numerical_features] = scaler.fit_transform(lung_cancer_data[numerical_features])


# Split the dataset into features and the target
X = lung_cancer_data.drop('LUNG_CANCER', axis=1).values
Y = lung_cancer_data['LUNG_CANCER'].values


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)




number_of_features = X.shape[1]
print(number_of_features)
missing_values = lung_cancer_data.isnull().sum()
print("Missing Values\n", missing_values)
#Check the information about the dataset, including data types and missing values
lung_cancer_data.info()


# Create a Keras model
model = Sequential()
#model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, input_dim=15, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use EarlyStopping callback to monitor validation loss and stop training early if needed
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stopping])


print("Distribution of Y_val:")
print(pd.Series(Y_val).value_counts())

# Calculate Y_pred as you did before
Y_pred = model.predict(X_val)
Y_pred = (Y_pred > 0.5)
Y_pred_flat = Y_pred.flatten()
print("Distribution of Y_pred:")
print(pd.Series(Y_pred_flat).value_counts())

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Evaluate the model on the validation or test set
Y_pred = model.predict(X_val)  # Replace with X_test for test set
Y_pred = (Y_pred > 0.5)  # Convert predicted probabilities to binary values

precision = precision_score(Y_val, Y_pred, zero_division=1)
recall = recall_score(Y_val, Y_pred, zero_division=1)
f1 = f1_score(Y_val, Y_pred, zero_division=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ... Plotting code for loss and accuracy ...
# You can plot precision, recall, and F1-score over epochs using your own lists to store these values during training.
# For example:
precision_values = []
recall_values = []
f1_values = []

# Train the model with monitoring precision, recall, and f1-score
for epoch in range(100):
    history = model.fit(X_train, Y_train, epochs=1, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
    
    # Calculate precision, recall, and f1-score on the validation set
    Y_pred = model.predict(X_val)
    Y_pred = (Y_pred > 0.5)
    
    precision = precision_score(Y_val, Y_pred)
    recall = recall_score(Y_val, Y_pred)
    f1 = f1_score(Y_val, Y_pred)
    
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)
    
# Plotting code for precision, recall, and F1-score
plt.plot(precision_values)
plt.plot(recall_values)
plt.plot(f1_values)
plt.title('Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend(['Precision', 'Recall', 'F1-Score'], loc='lower right')
plt.show()

