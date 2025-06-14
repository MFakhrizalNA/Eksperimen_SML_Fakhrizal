import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Drop kolom yang tidak digunakan
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

     # Tangani missing value
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df

def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    X = df.drop(columns='Survived')
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def build_preprocessor():
    num_columns = ['Age', 'Fare']
    ordinal_columns = []  # Tidak ada ordinal di Titanic
    nominal_columns = ['Sex', 'Embarked']

    num_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=1, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('ord_encoder', OrdinalEncoder())
    ])

    nominal_pipeline = Pipeline([
        ('nom_encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_columns),
        ('ordinal_pipeline', ordinal_pipeline, ordinal_columns),
        ('nominal_pipeline', nominal_pipeline, nominal_columns)
    ]).set_output(transform='pandas')

    return preprocessor

def run_preprocessing_pipeline(input_path, output_dir="titanic_preprocessing"):
    print("🚀 Memulai proses preprocessing...")

    # Load dan bersihkan data
    df_clean = load_and_clean_data(input_path)

     # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df_clean)

    # Preprocessing
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)
    X_test_proc = preprocessor.transform(X_test)

# Eksekusi hanya jika file ini dijalankan langsung
if __name__ == "__main__":
    run_preprocessing_pipeline("../titanic_raw/train.csv")