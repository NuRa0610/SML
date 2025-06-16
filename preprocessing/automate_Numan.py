import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data, target_column, save_path, file_path):
    data = data.drop_duplicates()
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns.drop(target_column)
    pd.DataFrame(columns=column_names).to_csv(file_path, index=False)
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    dump(preprocessor, save_path)

    return X, y, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import os

    # Path dataset dan output
    input_csv = os.path.join("..", "bots_vs_users_raw.csv")
    output_pkl = os.path.join("bots_vs_users_preprocessing.pkl")
    pipeline_path = os.path.join("preprocessor_pipeline.joblib")
    header_csv = os.path.join("data.csv")

    # Baca dataset
    data = pd.read_csv(input_csv)

    # Preprocessing
    X, y, X_train, X_test, y_train, y_test = preprocess_data(
        data, 'target', pipeline_path, header_csv
    )

    # Simpan hasil preprocessing ke pickle
    import pickle
    data_to_export = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_to_export, f)
    print(f"All data exported successfully to {output_pkl}")