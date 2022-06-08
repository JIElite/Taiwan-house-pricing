import os

import mlflow


def environment_setup(experiment='house-price'):
    SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPRIMENT_NAME = 'house-price'
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(EXPRIMENT_NAME)
