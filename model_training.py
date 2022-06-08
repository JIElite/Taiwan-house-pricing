import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from utils import split_features_target
from mlflow_utils import environment_setup
from eval import evaluate_partitions, default_partitions


def train_model(model, model_params, exp_params, use_mlflow=False, save_model=True):
    assert 'run_name' in exp_params
    assert 'training_data' in exp_params
    assert 'testing_data' in exp_params
    assert 'model_type' in exp_params

    df_train = pd.read_csv(exp_params['training_data'])
    df_test = pd.read_csv(exp_params['testing_data'])
    target = exp_params.get('target', 'Total_price')
    val_size = exp_params.get('val_size', 0.1)
    X_train, y_train = split_features_target(df_train, target_field=target)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size)
    X_test, y_test = split_features_target(df_test, target_field=target)

    if use_mlflow:
        import mlflow
        environment_setup()
        mlflow.start_run(run_name=exp_params['run_name'])
        mlflow.log_params(exp_params)
        mlflow.log_params(model_params)
        # log execution script
        if 'script' in exp_params:
            mlflow.log_artifact(exp_params['script'])

    model.fit(X_train, y_train)
    scores = {}
    train_mape = mean_absolute_percentage_error(
        y_train, model.predict(X_train))
    val_mape = mean_absolute_percentage_error(y_val, model.predict(X_val))
    test_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    scores['train-MAPE'] = train_mape
    scores['val-MAPE'] = val_mape
    scores['test-MAPE'] = test_mape
    print(scores)

    train_df = X_train
    train_df[target] = y_train
    train_maes = evaluate_partitions(model, train_df, default_partitions,
                                     metric=mean_absolute_error, target_field=target,
                                     index_prefix='train-mae-')
    print(train_maes)
    val_df = X_val
    val_df[target] = y_val
    val_maes = evaluate_partitions(model, val_df, default_partitions,
                                   metric=mean_absolute_error, target_field=target,
                                   index_prefix='val-mae-')
    print(val_maes)
    test_df = X_test
    test_df[target] = y_test
    test_maes = evaluate_partitions(model, test_df, default_partitions,
                                    metric=mean_absolute_error, target_field=target,
                                    index_prefix='test-mae-')
    print(test_maes)

    if use_mlflow:
        mlflow.log_metrics(scores)
        mlflow.log_metrics(train_maes)
        mlflow.log_metrics(val_maes)
        mlflow.log_metrics(test_maes)

    # Save model
    if save_model:
        import joblib
        from datetime import datetime
        timestamp = str(datetime.now())
        MODEL_DIR = './models/'
        MODEL_NAME = f'model_{timestamp}.pkl'
        MODEL_PATH = f'{MODEL_DIR}{MODEL_NAME}'
        joblib.dump(model, MODEL_PATH)

        if use_mlflow:
            mlflow.log_param('model_path', MODEL_PATH)
            mlflow.log_artifact(MODEL_PATH)

    if use_mlflow:
        mlflow.end_run()
