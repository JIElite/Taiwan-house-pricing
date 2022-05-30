import os
import joblib
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from utils import split_features_target
from eval import simple_evaluate


def simple_drop(df):
    df = df.drop(columns=['address', 'TDATE', 'Total_price', '編號'])
    return df


TRAIN_DATA_PATH = './merged_data/clean_data_future_train.csv'
TEST_DATA_PATH = './merged_data/clean_data_future_test.csv'
VAL_SIZE = 0.1
MAX_DEPTH = 20
RAMDOM_SEED_SHUFFLE_TRAIN_SET = 0

MLFLOW = True
if MLFLOW:
    import mlflow
    SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPRIMENT_NAME = 'house-price'
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(EXPRIMENT_NAME)
    mlflow.start_run(run_name='DSTR. Baseline')
    mlflow.log_params({'model_type': DecisionTreeRegressor.__name__,
                       'training_data': TRAIN_DATA_PATH,
                       'testing_data': TEST_DATA_PATH,
                       'VAL_SIZE': VAL_SIZE,
                       'MAX_DEPTH': MAX_DEPTH,
                       'RANDOM_SEED_SHUFFLE_TRAIN_SET': RAMDOM_SEED_SHUFFLE_TRAIN_SET})

df_future = pd.read_csv(TRAIN_DATA_PATH)
df_future = simple_drop(df_future)
df_future = df_future.sample(
    frac=1, random_state=RAMDOM_SEED_SHUFFLE_TRAIN_SET)
X_train, y_train = split_features_target(df_future)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE)

df_future_test = pd.read_csv(TEST_DATA_PATH)
df_future_test = simple_drop(df_future_test)
X_test, y_test = split_features_target(df_future_test)


avg_train_r2, avg_train_mae, avg_train_mse = 0, 0, 0
avg_val_r2, avg_val_mae, avg_val_mse = 0, 0, 0
avg_test_r2, avg_test_mae, avg_test_mse = 0, 0, 0

for i in tqdm(range(100)):
    model = DecisionTreeRegressor(max_depth=20)
    model.fit(X_train, y_train)
    r2, mae, mse = simple_evaluate(model, X_train, y_train)
    avg_train_r2 += r2
    avg_train_mae += mae
    avg_train_mse += mse

    r2, mae, mse = simple_evaluate(model, X_val, y_val)
    avg_val_r2 += r2
    avg_val_mae += mae
    avg_val_mse += mse

    r2, mas, mse = simple_evaluate(model, X_test, y_test)
    avg_test_r2 += r2
    avg_test_mae += mae
    avg_test_mse += mse

avg_train_r2 /= 100.
avg_train_mae /= 100.
avg_train_mse /= 100.

avg_val_r2 /= 100.
avg_val_mae /= 100.
avg_val_mse /= 100.

avg_test_r2 /= 100.
avg_test_mae /= 100.
avg_test_mse /= 100.

if MLFLOW:
    mlflow.log_metrics({'train-R-square': avg_train_r2,
                        'train-MAE': avg_train_mae,
                        'train-MSE': avg_train_mse})
    mlflow.log_metrics({'val-R-square': avg_val_r2,
                        'val-MAE': avg_val_mae,
                        'val-MSE': avg_val_mse})
    mlflow.log_metrics({'test-R-square': avg_test_r2,
                        'test-MAE': avg_test_mae,
                        'test-MSE': avg_test_mse})

timestamp = str(datetime.now())
MODEL_DIR = './models/'
MODEL_NAME = f'model_{timestamp}.pkl'
MODEL_PATH = f'{MODEL_DIR}{MODEL_NAME}'
joblib.dump(model, MODEL_PATH)
if MLFLOW:
    mlflow.log_param('model_path', MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.end_run()

print('Training Finished!')
