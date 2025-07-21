
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from urllib.parse import urlparse
from mlflow.models import infer_signature
import mlflow
import warnings
warnings.filterwarnings("ignore")
import pickle

params = yaml.safe_load(open('params.yaml'))
params_target = yaml.safe_load(open('params.yaml'))['common']['target']
params = params['train']

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/devanshprataps6/Airline_Passenger_Satisfaction_Predictor.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "devanshprataps6"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "a16023baea3d90b6fd742c43657d87fa6d0fc8c9"


def hyperparameter_tuning(X_train,y_train,param_grid):
    lr = LogisticRegression()
    grid_search = GridSearchCV(estimator=lr,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

# Load params from yaml
params = yaml.safe_load(open("params.yaml"))
params_target = params['common']['target']
params = params['train']

def train(data_path,model_path,param_grid) :
    df = pd.read_csv(data_path)
    X = df.drop(columns=[params_target])
    y = df[params_target]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=params['random_state'])
    signature = infer_signature(X_train, y_train)


    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():

        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        cm = confusion_matrix(y_valid, y_pred)
        cr = classification_report(y_valid, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(best_params)

            
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type!='file':
            mlflow.sklearn.log_model(best_model,"model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        # Create the directory to save model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)        

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Accuracy : {accuracy}")
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(params['data'],params['model_path'],params['models']['logistic']['param_grid'])
        

