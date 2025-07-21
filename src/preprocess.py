import numpy as np
import pandas as pd
import yaml
import os
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib


params = yaml.safe_load(open('params.yaml'))
params_target = yaml.safe_load(open('params.yaml'))['common']['target']
params = params['preprocess']

def split_columns(df, target_col, cat_threshold=10,col_to_drop=None):
    cat_col = []
    num_col = []
    for col in df.columns:
        if col == target_col or col in col_to_drop : 
            continue
        unique_vals = df[col].nunique()

        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            cat_col.append(col)
        elif unique_vals <= cat_threshold:
            cat_col.append(col)
        else:
            num_col.append(col)

    print("Numerical columns:", num_col)
    print("Categorical columns:", cat_col)
        

    return num_col, cat_col


def preprocess_data(target_col,input_path,output_path,artifact_save_path):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    columns_to_drop = ['Gender','Gate location','Departure/Arrival time convenient','Departure Delay in Minutes']
    df.drop(columns = columns_to_drop,inplace=True)
    num_col,cat_col = split_columns(df = df, target_col=target_col, col_to_drop=columns_to_drop)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_pipeline = make_pipeline(
        StandardScaler()
    )
    cat_pipeline = make_pipeline(
        OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_col),
            ("cat", cat_pipeline, cat_col)
        ]
    )

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)

    # Save the preprocesses data as train and test.csv also cat target col in both
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(artifact_save_path), exist_ok=True)
    train_df = pd.DataFrame(X_train, columns=num_col + list(preprocessing.named_transformers_['cat'].named_steps['onehotencoder'].get_feature_names_out(cat_col)))
    train_df[target_col] = y_train.values
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)

    test_df = pd.DataFrame(X_test, columns=num_col + list(preprocessing.named_transformers_['cat'].named_steps['onehotencoder'].get_feature_names_out(cat_col)))    
    test_df[target_col] = y_test.values
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    joblib.dump(preprocessing,artifact_save_path)

    print(f"Preprocessing complete. Train and test data saved to {output_path} folder.")


if __name__ == "__main__":
    preprocess_data(params_target,params['input'],params['output'],params['artifact_path'])


   
