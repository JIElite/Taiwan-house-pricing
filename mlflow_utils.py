import os

import mlflow


def environment_setup():
    SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPRIMENT_NAME = 'house_project'
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(EXPRIMENT_NAME)
