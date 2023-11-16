# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:05:06 2023

@author: Aoife O'Connor

Diabetes Dataset Keras Test
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Load and preprocess the dataset
diabetes_data = pd.read_csv('diabetes_data.csv')

# Encoding categorical variables 
#diabetes_data = pd.get_dummies(diabetes_data, columns=['gender'], drop_first=True)
diabetes_data['gender'] = diabetes_data['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})


# Define a mapping dictionary for smoking history
smoking_history_mapping = {
    'No Info': 0,
    'never': 0,
    'ever': 1,
    'not current': 2,
    'former': 2,
    'current': 3
}

# Use the map function to replace the values in the "smoking_history" column
diabetes_data['smoking_history'] = diabetes_data['smoking_history'].map(smoking_history_mapping)

# Scaling the numerical features
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
diabetes_data[numerical_features] = scaler.fit_transform(diabetes_data[numerical_features])

# Split the data into features and target
X = diabetes_data.drop(columns='diabetes', axis=1)
Y = diabetes_data['diabetes']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


print(X_train.dtypes,"\n")
print(Y.dtypes,"\n")
print(X_train.isnull().sum(),"\n")
print(Y_train.isnull().sum(), "\n")

number_of_features = X.shape[1]
print(number_of_features)

# Create a Keras model
model = Sequential()
#model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use EarlyStopping callback to monitor validation loss and stop training early if needed
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stopping])

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

precision = precision_score(Y_val, Y_pred)
recall = recall_score(Y_val, Y_pred)
f1 = f1_score(Y_val, Y_pred)

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

