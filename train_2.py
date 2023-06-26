from io import BytesIO 
import os 

import pandas as pd
import boto3


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import mlflow
import optuna


def read_df_from_s3(bucket='datasets', file_name='winequality-red.csv'):
    s3 = boto3.resource(
        's3',
        endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL') 
    )

    obj = s3.Object(bucket, file_name)

    with BytesIO(obj.get()['Body'].read()) as bio:
        df = pd.read_csv(bio)

    return df 


def train(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality'], axis=1),
                                                    df['quality'], test_size=0.30,
                                                    random_state=42)
    
    # Применяем метод oversampling к тренировочной выборке
    os = SMOTE(random_state=1, k_neighbors=2)
    X_train_os, y_train_os = os.fit_resample(X_train, y_train)

    # Теперь X_train_os и y_train_os содержат сбалансированные данные для тренировочной выборки
    # X_test и y_test содержат несбалансированные данные для тестовой выборки

    with mlflow.start_run():  

        def objective(trial):

            n_estimators = trial.suggest_int("n_estimators", 20, 150)
            max_depth = trial.suggest_int("max_depth", 2, 20, log=True)
            max_features = trial.suggest_float("max_features", 0.1, 1.0, step=0.1)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)

            forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                    min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

            return np.mean(cross_val_score(forest, X_train_os, y_train_os, cv=3, n_jobs=-1, scoring='r2'))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        best_params = study.best_params

        model_2 = RandomForestClassifier(**best_params).fit(X_train_os, y_train_os)
        model_2_pred = model_2.predict(X_test)

        balanced_accuracy = balanced_accuracy_score(y_test, model_2_pred)
        recall = recall_score(y_test, model_2_pred, average='weighted')
        precision = precision_score(y_test, model_2_pred, average='weighted')
        f1 = f1_score(y_test, model_2_pred, average='weighted')
        f2 = r2_score(y_test, model_2_pred)

        mlflow.log_metric('balanced_accuracy', balanced_accuracy_score(y_test, model_2_pred))    
        mlflow.log_metric('recall', recall_score(y_test, model_2_pred, average='weighted', zero_division=1))
        mlflow.log_metric('precision', precision_score(y_test, model_2_pred, average='weighted', zero_division=1))
        mlflow.log_metric('f1', f1_score(y_test, model_2_pred, average='weighted', zero_division=1))
        mlflow.log_metric('r2', r2_score(y_test, model_2_pred))

        print('Balanced accuracy: ', balanced_accuracy)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F1-score: ', f1)
        print("r2:", f2)  

        mlflow.sklearn.log_model(model_2, 'classifier_2')

if __name__ == '__main__':


    df = read_df_from_s3()
    print(df.head())

    train(df)