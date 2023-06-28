from io import BytesIO 
import os 
import time
from functools import partial

import pandas as pd
import boto3


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


space = {
  'model': hp.choice(
    'classifier_type', 
    [
      {
        'model_type': 'RandomForestClassifier',
        'n_estimators': hp.uniform('n_estimators', 10, 500),
        'max_depth': hp.choice('max_depth', np.arange(5, 31, dtype=int)),
        'min_samples_split': hp.uniform('min_samples_split', 0.0, 10.0),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0.0, 10.0),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
      },
      {
        'model_type': 'KMeans',
        'n_clusters': hp.uniform('n_clusters', 2, 10),
        'init': hp.choice('init', ['k-means++', 'random']),
        'max_iter': hp.uniform('max_iter', 100, 1000),
        'random_state': hp.uniform('random_state', 0, 100)

      },
      {
        'model_type': 'DBSCAN',
        'eps': hp.uniform('eps', 1.02, 4.3),
        'min_samples': hp.uniform('min_samples', 2, 6)
      },
    ]
  )
}


def read_df_from_s3(bucket='datasets', file_name='winequality-red.csv'):
    s3 = boto3.resource(
        's3',
        endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL') 
    )

    obj = s3.Object(bucket, file_name)

    with BytesIO(obj.get()['Body'].read()) as bio:
        df = pd.read_csv(bio)

    return df 


def objective(params, X_train, y_train):

    classifier_type = params['model']
    with mlflow.start_run():
        mlflow.log_params(params['model'])

        if classifier_type == 'RandomForestClassifier':
            model_hp = RandomForestClassifier(**params['model'])
        elif classifier_type == 'KMeans':
            model_hp = KMeans(**params['model'])
        elif classifier_type == 'DBSCAN':
            model_hp = DBSCAN(**params['model'])
        else:
            return 0

        t0 = time.time()

        accuracy = np.mean(cross_val_score(model_hp, X_train, y_train, cv=3, n_jobs=-1, scoring='accuracy'))
        mlflow.log_metric('time', time.time() - t0)
        mlflow.log_metric('accuracy', accuracy)

    return{'loss': -accuracy, 'status': STATUS_OK}



if __name__ == '__main__':


    df = read_df_from_s3()

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality'], axis=1),
                                                    df['quality'], test_size=0.30,
                                                    random_state=42)
    os = SMOTE(random_state=1, k_neighbors=2)
    X_train_os, y_train_os = os.fit_resample(X_train, y_train)

    trials = Trials()

    best = fmin(
        fn = partial(objective, X_train = X_train_os, y_train = y_train_os), 
        space = space,
        algo = tpe.suggest,
        max_evals = 10,
        trials = trials

    )

    print(df.head())

