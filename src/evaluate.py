import pandas as pd
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score
import mlflow

params = yaml.safe_load(open('params.yaml'))
params_target = yaml.safe_load(open('params.yaml'))['common']['target']
params = params['evaluate']

def evaluate(data_path,model_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[params_target])
    y = df[params_target]

    model = pickle.load(open(model_path,'rb'))

    y_pred = model.predict(X)
    accuracy = accuracy_score(y,y_pred)
    
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params['data'],params['model'])    
