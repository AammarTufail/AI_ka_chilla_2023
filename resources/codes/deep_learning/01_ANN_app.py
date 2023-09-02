import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers

# Load the datasets
@st.cache_resource
def load_data(dataset_name):
    if dataset_name == 'tips':
        data = sns.load_dataset('tips')
    elif dataset_name == 'titanic':
        data = sns.load_dataset('titanic')
    return data

# Split the data
def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Preprocess the data
def preprocess_data(X_train, X_test, imputation_strategy='mean'):
    # Handling missing values
    if imputation_strategy == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        imputer = SimpleImputer(strategy=imputation_strategy)

    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Separating categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Encoding categorical features
    if cat_cols:
        ohe = OneHotEncoder(drop='first', sparse=False)
        X_train_encoded = ohe.fit_transform(X_train[cat_cols])
        X_test_encoded = ohe.transform(X_test[cat_cols])

        # Replace categorical columns with their one-hot encoded values
        X_train = pd.concat([X_train.drop(cat_cols, axis=1), pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(cat_cols), index=X_train.index)], axis=1)
        X_test = pd.concat([X_test.drop(cat_cols, axis=1), pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out(cat_cols), index=X_test.index)], axis=1)

    # Scaling numerical features
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test

# Build the neural network model
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Streamlit app main code
st.title("Streamlit Neural Network App")

dataset_name = st.selectbox("Select Dataset", ['tips', 'titanic'])
data = load_data(dataset_name)

all_columns = data.columns.tolist()

# Letting the user select columns for X and y
selected_X_columns = st.multiselect("Select features (X)", all_columns, default=all_columns[:-1])
selected_y_column = st.selectbox("Select target variable (y)", all_columns, index=len(all_columns)-1)

X = data[selected_X_columns]
y = data[selected_y_column]

imputation_method = st.selectbox("Select imputation method", ['mean', 'median', 'constant'])
test_size = st.slider("Select test size", 0.1, 0.5, 0.2)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = split_data(X, y, test_size)
X_train, X_test = preprocess_data(X_train, X_test, imputation_strategy=imputation_method)

# Train the neural network
if st.button("Train Neural Network"):
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

if st.button("Show Train Data"):
    st.write(X_train)

if st.button("Show Test Data"):
    st.write(X_test)
