import os
import joblib
from datetime import datetime

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from utils import split_features_target
from eval import simple_evaluate


def simple_drop(df):
    df = df.drop(columns=['address', 'TDATE', 'Total_price', '編號'])
    return df


TRAIN_DATA_PATH = './merged_data/clean_data_future_train.csv'
VAL_SIZE = 0.1
MAX_DEPTH = 15
RAMDOM_SEED_SHUFFLE_TRAIN_SET = 0

MLFLOW = True
if MLFLOW:
    import mlflow
    SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
    EXPRIMENT_NAME = 'house-price'
    SCRIPT_PATH = os.path.basename(__file__)
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(EXPRIMENT_NAME)
    mlflow.start_run(run_name='DSTR. Baseline')
    mlflow.log_params({'model_type': DecisionTreeRegressor.__name__,
                       'training_data': TRAIN_DATA_PATH,
                       'VAL_SIZE': VAL_SIZE,
                       'MAX_DEPTH': MAX_DEPTH,
                       'RANDOM_SEED_SHUFFLE_TRAIN_SET': RAMDOM_SEED_SHUFFLE_TRAIN_SET})
    # Log current script code
    mlflow.log_artifact(SCRIPT_PATH)


df_future = pd.read_csv(TRAIN_DATA_PATH)
df_future = simple_drop(df_future)
df_future['Month'] = df_future['Month'].astype(int)

df_future_train = df_future.loc[df_future['Month'] <= 202110]
df_future_val = df_future.loc[df_future['Month'] > 202110]

X_train, y_train = split_features_target(df_future_val)
X_val, y_val = split_features_target(df_future_val)

avg_train_r2, avg_train_mae, avg_train_mse = 0, 0, 0
avg_val_r2, avg_val_mae, avg_val_mse = 0, 0, 0
for i in tqdm(range(100)):
    model = DecisionTreeRegressor(max_depth=MAX_DEPTH)
    model.fit(X_train, y_train)
    r2, mae, mse = simple_evaluate(model, X_train, y_train)
    avg_train_r2 += r2
    avg_train_mae += mae
    avg_train_mse += mse

    r2, mae, mse = simple_evaluate(model, X_val, y_val)
    avg_val_r2 += r2
    avg_val_mae += mae
    avg_val_mse += mse

avg_train_r2 /= 100.
avg_train_mae /= 100.
avg_train_mse /= 100.

avg_val_r2 /= 100.
avg_val_mae /= 100.
avg_val_mse /= 100.

if MLFLOW:
    mlflow.log_metrics({'train-R-square': avg_train_r2,
                        'train-MAE': avg_train_mae,
                        'train-MSE': avg_train_mse})
    mlflow.log_metrics({'val-R-square': avg_val_r2,
                        'val-MAE': avg_val_mae,
                        'val-MSE': avg_val_mse})

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
