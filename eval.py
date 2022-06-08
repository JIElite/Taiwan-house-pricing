from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             check_scoring,
                             )


default_partitions = {
    '0-10': 'Transfer_Total_Ping >= 0 and Transfer_Total_Ping < 10',
    '10-20': 'Transfer_Total_Ping >= 10 and Transfer_Total_Ping < 20',
    '20-30': 'Transfer_Total_Ping >= 20 and Transfer_Total_Ping < 30',
    '30-40': 'Transfer_Total_Ping >= 30 and Transfer_Total_Ping < 40',
    '40-50': 'Transfer_Total_Ping >= 40 and Transfer_Total_Ping < 50',
    '50-60': 'Transfer_Total_Ping >= 50 and Transfer_Total_Ping < 60',
    '60-80': 'Transfer_Total_Ping >= 60 and Transfer_Total_Ping < 80',
    '80-': 'Transfer_Total_Ping >= 80',
}


metrics_brief_name = {
    mean_absolute_error: 'MAE',
    mean_squared_error: 'MSE',
    mean_absolute_percentage_error: 'MAPE',
    r2_score: 'R-square',
    'neg_mean_absolute_error': 'MAE',
    'neg_mean_squared_error': 'MSE',
    'neg_mean_absolute_percentage_error': 'MAPE',
    'r2': 'R-square',
}


def simple_evaluate(model, X, y, verbose=False):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    if verbose:
        print(f'R2 score: {r2}')
        print(f'MAE score: {mae}')
        print(f'MSE score: {mse}')
        print()

    return r2, mae, mse


def evaluate_builtin_metric(model, X, y, scoring, index_predix=''):
    scores = {}

    if callable(scoring):
        scorers = scoring
        metric_name = metrics_brief_name[scoring]
        scores[index_predix + metric_name] = scorers(y, model.predict(X))
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(model, scoring)
        metric_name = metrics_brief_name[scoring]
        scores[index_predix + metric_name] = scorers(y, model.predict(X))
    else:
        for scoring_str in scoring:
            scorers = check_scoring(model, scoring_str)
            metric_name = metrics_brief_name[scorers._score_func]
            scores[index_predix +
                   metric_name] = scorers._score_func(y, model.predict(X))

    return scores


def evaluate_partitions(model, df, partitions, metric, target_field, index_prefix=''):
    scores = {}
    for ping_range, partition_cond in partitions.items():
        df_of_interest = df.query(partition_cond)

        y = df_of_interest[target_field]
        if 'Total_price' in df_of_interest.columns:
            df_of_interest = df_of_interest.drop(columns=['Total_price'])
        if 'Unit_Price_Ping' in df_of_interest.columns:
            df_of_interest = df_of_interest.drop(columns=['Unit_Price_Ping'])
        X = df_of_interest

        y_pred = model.predict(X)
        scores[index_prefix + ping_range] = metric(y, y_pred)

    return scores
