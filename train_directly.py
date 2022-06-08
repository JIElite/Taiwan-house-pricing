import os

from sklearn.tree import DecisionTreeRegressor

from model_training import train_model


MLFLOW = False

RUN_NAME = 'Decision Tree Regressor'
TRAIN_DATA_PATH = 'merged_data/20220606/clean_data_train_all.csv'
TEST_DATA_PATH = 'merged_data/20220606/clean_data_test_all.csv'
MODEL = DecisionTreeRegressor.__name__
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 20
SCRIPT_PATH = os.path.basename(__file__)
TARGET = 'Total_price'
VAL_SIZE = 0.1


if __name__ == '__main__':
    exp_params = {
        'run_name': RUN_NAME,  # necessary field
        'training_data': TRAIN_DATA_PATH,  # necessary field
        'testing_data': TEST_DATA_PATH,  # necessary field
        'model_type': MODEL,  # necessary field
        'script': SCRIPT_PATH,
        'target': TARGET,
        'val_size': VAL_SIZE,
    }
    model_params = {
        'max_depth': MAX_DEPTH,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
    }
    model = DecisionTreeRegressor(**model_params)
    train_model(model, model_params, exp_params=exp_params, use_mlflow=MLFLOW,
                scoring=['r2', 'neg_mean_absolute_percentage_error'])
